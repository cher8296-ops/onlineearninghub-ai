import json
import logging
import os
import random
import re
import socket
import time
import urllib.error
import urllib.request
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# --------------------------------------------------
# APPLICATION CONFIGURATION
# --------------------------------------------------

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("onlineearninghub")

APP_VERSION = "0.5.0"
API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models/"
REQUEST_TIMEOUT_SECONDS = 25
MAX_PRIMARY_ATTEMPTS = 2
RETRYABLE_HTTP_STATUSES = {500, 502, 503, 504}

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()
GEMINI_FALLBACK_MODEL = os.getenv(
    "GEMINI_FALLBACK_MODEL", "gemini-3.6-flash"
).strip()

app = FastAPI(
    title="OnlineEarningHub AI",
    description=(
        "AI-powered tools for planning, writing, research and quality checking. "
        "Transient AI-provider failures are handled with bounded retries and fallback."
    ),
    version=APP_VERSION,
)


class Task(BaseModel):
    description: str = Field(
        min_length=3,
        max_length=10000,
        description="The task or topic to process.",
    )


def _validate_model_name(model: str) -> bool:
    return bool(model and re.fullmatch(r"[A-Za-z0-9._-]+", model))


def _retry_after_seconds(headers) -> float | None:
    """Read Retry-After as seconds or an HTTP date; cap waits to 8 seconds."""
    value = headers.get("Retry-After") if headers else None
    if not value:
        return None
    try:
        return max(0.0, min(float(value), 8.0))
    except (TypeError, ValueError):
        try:
            retry_at = parsedate_to_datetime(value)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            return max(0.0, min((retry_at - datetime.now(timezone.utc)).total_seconds(), 8.0))
        except (TypeError, ValueError, OverflowError):
            return None


def _make_request(model: str, prompt: str, use_search: bool):
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.4,
            "maxOutputTokens": 1800,
        },
    }
    if use_search:
        payload["tools"] = [{"google_search": {}}]

    request = urllib.request.Request(
        f"{API_BASE_URL}{model}:generateContent",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": GEMINI_API_KEY,
        },
        method="POST",
    )
    return request


def _parse_success(response_body: str) -> dict:
    try:
        result = json.loads(response_body)
    except json.JSONDecodeError as exc:
        logger.exception("Gemini returned malformed JSON.")
        raise HTTPException(
            status_code=502,
            detail="The AI provider returned an unreadable response. Please try again.",
        ) from exc

    candidates = result.get("candidates") or []
    if not candidates:
        feedback = result.get("promptFeedback") or {}
        logger.warning(
            "Gemini returned no candidates; block_reason=%s",
            feedback.get("blockReason", "not provided"),
        )
        raise HTTPException(
            status_code=502,
            detail="The AI could not generate an answer for this request.",
        )

    candidate = candidates[0]
    parts = (candidate.get("content") or {}).get("parts") or []
    answer = "\n".join(
        part.get("text", "") for part in parts if part.get("text")
    ).strip()
    if not answer:
        logger.warning(
            "Gemini returned an empty answer; finish_reason=%s",
            candidate.get("finishReason", "unknown"),
        )
        raise HTTPException(
            status_code=502,
            detail="The AI provider returned an empty answer. Please try again.",
        )

    sources = []
    seen_urls = set()
    grounding = candidate.get("groundingMetadata") or {}
    for chunk in grounding.get("groundingChunks") or []:
        web_source = chunk.get("web") or {}
        source_url = web_source.get("uri")
        if source_url and source_url not in seen_urls:
            seen_urls.add(source_url)
            sources.append({
                "title": web_source.get("title") or "Source",
                "url": source_url,
            })

    return {"answer": answer, "sources": sources}


def _call_model(model: str, prompt: str, use_search: bool) -> dict:
    request = _make_request(model, prompt, use_search)
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        body = response.read().decode("utf-8", errors="replace")
    return _parse_success(body)


def ask_ai(prompt: str, use_search: bool = False) -> dict:
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY is not configured.")
        raise HTTPException(
            status_code=503,
            detail="AI service is not configured. Please contact the administrator.",
        )

    if not _validate_model_name(GEMINI_MODEL):
        logger.error("Invalid GEMINI_MODEL configuration.")
        raise HTTPException(
            status_code=503,
            detail="The primary AI model configuration is invalid.",
        )

    fallback_enabled = (
        _validate_model_name(GEMINI_FALLBACK_MODEL)
        and GEMINI_FALLBACK_MODEL != GEMINI_MODEL
    )
    last_status = None
    last_reason = "temporary provider failure"

    # Try the configured primary model twice, with bounded exponential backoff.
    for attempt in range(1, MAX_PRIMARY_ATTEMPTS + 1):
        logger.info(
            "Gemini request started; model=%s attempt=%s/%s",
            GEMINI_MODEL,
            attempt,
            MAX_PRIMARY_ATTEMPTS,
        )
        try:
            result = _call_model(GEMINI_MODEL, prompt, use_search)
            logger.info("Gemini request succeeded; model=%s", GEMINI_MODEL)
            result["model"] = GEMINI_MODEL
            result["fallback_used"] = False
            return result

        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            last_status = error.code
            # Log a short provider message only; never log the API key or prompt.
            logger.warning(
                "Gemini HTTP failure; model=%s status=%s body=%s",
                GEMINI_MODEL,
                error.code,
                body[:350].replace("\n", " "),
            )

            if error.code == 429:
                last_reason = "rate limit or usage quota"
                wait_seconds = _retry_after_seconds(error.headers)
                if wait_seconds is None:
                    wait_seconds = 3.0
                if attempt < MAX_PRIMARY_ATTEMPTS:
                    logger.info("Rate limited; retrying after %.1f seconds.", wait_seconds)
                    time.sleep(wait_seconds)
                    continue
                raise HTTPException(
                    status_code=503,
                    detail="The AI service has reached a temporary rate or usage limit. Please wait and try again.",
                )

            if error.code in (401, 403):
                raise HTTPException(
                    status_code=502,
                    detail="The AI provider rejected the API credentials or permissions. The administrator must check the server configuration.",
                )

            if error.code in RETRYABLE_HTTP_STATUSES:
                last_reason = f"provider HTTP {error.code}"
                if attempt < MAX_PRIMARY_ATTEMPTS:
                    wait_seconds = (2 ** attempt) + random.uniform(0, 0.5)
                    logger.info("Temporary provider error; retrying after %.1f seconds.", wait_seconds)
                    time.sleep(wait_seconds)
                    continue
                break

            raise HTTPException(
                status_code=502,
                detail=f"The AI provider rejected the request (HTTP {error.code}).",
            )

        except (TimeoutError, socket.timeout) as error:
            last_reason = "provider timeout"
            logger.warning(
                "Gemini request timed out; model=%s attempt=%s/%s",
                GEMINI_MODEL,
                attempt,
                MAX_PRIMARY_ATTEMPTS,
            )
            if attempt < MAX_PRIMARY_ATTEMPTS:
                time.sleep(2 ** attempt)
                continue
            break

        except urllib.error.URLError as error:
            reason = getattr(error, "reason", None)
            if isinstance(reason, (TimeoutError, socket.timeout)):
                last_reason = "provider network timeout"
                logger.warning("Gemini network timeout; model=%s", GEMINI_MODEL)
                if attempt < MAX_PRIMARY_ATTEMPTS:
                    time.sleep(2 ** attempt)
                    continue
                break
            logger.error("Gemini connection failed; reason=%s", reason)
            raise HTTPException(
                status_code=503,
                detail="The AI provider cannot currently be reached. Please try again later.",
            ) from error

    # A separate model can recover from model-specific 503 overload/timeouts.
    # Do not silently switch models for invalid credentials or quota (429).
    if fallback_enabled and (last_status in (503, 504) or last_reason in {
        "provider timeout", "provider network timeout"
    }):
        logger.warning(
            "Primary model unavailable; trying configured fallback model=%s",
            GEMINI_FALLBACK_MODEL,
        )
        try:
            result = _call_model(GEMINI_FALLBACK_MODEL, prompt, use_search)
            logger.info("Gemini fallback succeeded; model=%s", GEMINI_FALLBACK_MODEL)
            result["model"] = GEMINI_FALLBACK_MODEL
            result["fallback_used"] = True
            return result
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            logger.error(
                "Gemini fallback failed; model=%s status=%s body=%s",
                GEMINI_FALLBACK_MODEL,
                error.code,
                body[:350].replace("\n", " "),
            )
            if error.code in (401, 403):
                raise HTTPException(
                    status_code=502,
                    detail="The AI provider rejected the API credentials or permissions. The administrator must check the server configuration.",
                ) from error
            if error.code == 429:
                raise HTTPException(
                    status_code=503,
                    detail="The fallback AI model has reached a temporary rate or usage limit. Please wait and try again.",
                ) from error
        except (TimeoutError, socket.timeout, urllib.error.URLError):
            logger.exception("Gemini fallback request failed; model=%s", GEMINI_FALLBACK_MODEL)

    if last_status == 503:
        raise HTTPException(
            status_code=503,
            detail="The AI models are temporarily overloaded. Please try again in a few minutes.",
        )
    raise HTTPException(
        status_code=504,
        detail="The AI provider did not respond in time. Please try again shortly.",
    )


# --------------------------------------------------
# HEALTH AND INFORMATION ENDPOINTS
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "platform": "OnlineEarningHub AI",
        "status": "running",
        "version": APP_VERSION,
    }


@app.get("/health")
def health():
    # This checks the web service only, not Gemini availability.
    return {"status": "healthy", "version": APP_VERSION}


@app.get("/agents")
def agents():
    return {
        "agents": [
            "JobFinder",
            "TaskPlanner",
            "WritingAssistant",
            "ResearchAssistant",
            "QualityChecker",
        ]
    }


@app.get("/jobs")
def jobs():
    return {
        "notice": "These are demonstration jobs, not verified paid vacancies.",
        "jobs": [
            {"title": "Content Writing", "category": "Writing", "type": "Demo"},
            {"title": "Online Research", "category": "Research", "type": "Demo"},
            {"title": "Data Entry", "category": "Administration", "type": "Demo"},
        ],
    }


# --------------------------------------------------
# AI AGENT ENDPOINTS
# --------------------------------------------------

@app.post("/plan")
def plan(task: Task):
    prompt = (
        "You are TaskPlanner for OnlineEarningHub AI. "
        "Create a realistic, numbered, step-by-step plan. "
        "Identify risks and distinguish confirmed facts from assumptions. "
        "Do not promise income or claim work has been completed.\n\n"
        f"User task: {task.description}"
    )
    result = ask_ai(prompt)
    return {
        "agent": "TaskPlanner",
        "task": task.description,
        "plan": result["answer"],
        "model_used": result["model"],
        "fallback_used": result["fallback_used"],
    }


@app.post("/write")
def write(task: Task):
    prompt = (
        "You are WritingAssistant for OnlineEarningHub AI. "
        "Complete the user's writing request professionally. "
        "Do not invent qualifications, work experience, or achievements.\n\n"
        f"Writing request: {task.description}"
    )
    result = ask_ai(prompt)
    return {
        "agent": "WritingAssistant",
        "task": task.description,
        "content": result["answer"],
        "model_used": result["model"],
        "fallback_used": result["fallback_used"],
    }


@app.post("/research")
def research(task: Task):
    prompt = (
        "You are ResearchAssistant for OnlineEarningHub AI. "
        "Analyze the topic carefully using your available knowledge. "
        "Prioritize reliable information. Distinguish established facts from assumptions. "
        "Do not claim to have searched the live internet. "
        "Do not invent citations, URLs, statistics, or sources. "
        "Explain when information may be outdated and needs verification.\n\n"
        f"Research topic: {task.description}"
    )
    result = ask_ai(prompt, use_search=False)
    return {
        "agent": "ResearchAssistant",
        "topic": task.description,
        "findings": result["answer"],
        "sources": result["sources"],
        "research_mode": "AI knowledge; no live web search",
        "model_used": result["model"],
        "fallback_used": result["fallback_used"],
    }


@app.post("/check")
def check(task: Task):
    prompt = (
        "You are QualityChecker for OnlineEarningHub AI. "
        "Review the submitted work for clarity, completeness, and accuracy. "
        "List errors, missing information, and practical improvements. "
        "Do not claim facts were verified unless you actually verified them.\n\n"
        f"Work to review: {task.description}"
    )
    result = ask_ai(prompt)
    return {
        "agent": "QualityChecker",
        "submitted_work": task.description,
        "review": result["answer"],
        "model_used": result["model"],
        "fallback_used": result["fallback_used"],
    }
