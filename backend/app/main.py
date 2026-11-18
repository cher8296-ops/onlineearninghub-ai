import json
import logging
import os
import random
import re
import socket
import time
import urllib.error
import urllib.request

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ============================================================
# ONLINEEARNINGHUB AI
# Professional FastAPI Backend
# ============================================================

APP_NAME = "OnlineEarningHub AI"
APP_VERSION = "0.7.0"

API_BASE_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
)

REQUEST_TIMEOUT_SECONDS = 25
MAX_PRIMARY_ATTEMPTS = 2

RETRYABLE_HTTP_STATUSES = {
    500,
    502,
    503,
    504,
}


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

logger = logging.getLogger("onlineearninghub")


# ============================================================
# ENVIRONMENT CONFIGURATION
# ============================================================

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY", ""
).strip()

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL", "gemini-3.8-flash"
).strip()

GEMINI_FALLBACK_MODEL = os.getenv(
    "GEMINI_FALLBACK_MODEL", "gemini-3.6-flash"
).strip()


# ============================================================
# SHARED INTERNATIONAL QUALITY STANDARD
# ============================================================

GLOBAL_QUALITY_STANDARD = """
You are a professional AI specialist working for OnlineEarningHub AI.
Apply these standards to every response:
- Understand the user's actual goal and answer the full request.
- Match the language used by the user unless they request another language.
- Use clear structure, precise wording, practical steps, and professional tone.
- Separate verified facts, estimates, assumptions, and recommendations.
- Never fabricate sources, URLs, statistics, credentials, capabilities, or completed actions.
- State important limitations and uncertainty plainly; never overpromise results or income.
- Respect privacy and security. Never request passwords, API keys, or other secrets.
- Prefer actionable, cost-conscious solutions and disclose meaningful costs or prerequisites.
- If essential information is missing, state the assumption or ask one focused question.
- Review the response for relevance, consistency, correctness, and readability before returning it.
"""


DEFAULT_FRONTEND_ORIGINS = [
    "https://onlineearninghub-ai-frontend.onrender.com",
    "http://localhost:5173",
]

configured_origins = os.getenv(
    "FRONTEND_ORIGINS", ""
)

ALLOWED_ORIGINS = list(
    dict.fromkeys(
        origin.strip().rstrip("/")
        for origin in (
            DEFAULT_FRONTEND_ORIGINS
            + configured_origins.split(",")
        )
        if origin.strip()
    )
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title=APP_NAME,
    description=(
        "AI-powered workspace for planning, writing, "
        "research and quality checking."
    ),
    version=APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ============================================================
# CORS CONFIGURATION
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


# ============================================================
# REQUEST MODEL
# ============================================================

class Task(BaseModel):
    description: str = Field(
        min_length=3,
        max_length=10000,
        description="The task or topic to process.",
    )


# ============================================================
# MODEL VALIDATION
# ============================================================

def validate_model_name(model: str) -> bool:
    return bool(
        model
        and re.fullmatch(r"[A-Za-z0-9._-]+", model)
    )


# ============================================================
# RETRY-AFTER HANDLING
# ============================================================

def get_retry_after(headers) -> float | None:

    value = headers.get("Retry-After") if headers else None

    if not value:
        return None

    try:
        return max(0.0, min(float(value), 8.0))

    except (TypeError, ValueError):

        try:
            retry_at = parsedate_to_datetime(value)

            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(
                    tzinfo=timezone.utc
                )

            seconds = (
                retry_at - datetime.now(timezone.utc)
            ).total_seconds()

            return max(0.0, min(seconds, 8.0))

        except (
            TypeError,
            ValueError,
            OverflowError,
        ):
            return None


# ============================================================
# GEMINI REQUEST
# ============================================================

def make_gemini_request(
    model: str,
    prompt: str,
    use_search: bool = False,
):

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.4,
            "maxOutputTokens": 1800,
        },
    }

    if use_search:
        payload["tools"] = [
            {"google_search": {}}
        ]

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


# ============================================================
# GEMINI RESPONSE PARSER
# ============================================================

def parse_gemini_response(response_body: str) -> dict:

    try:
        result = json.loads(response_body)

    except json.JSONDecodeError as exc:

        logger.error(
            "AI provider returned invalid JSON."
        )

        raise HTTPException(
            status_code=502,
            detail="The AI provider returned an invalid response.",
        ) from exc

    candidates = result.get("candidates") or []

    if not candidates:

        feedback = result.get("promptFeedback") or {}

        logger.warning(
            "No AI candidates returned. Reason: %s",
            feedback.get("blockReason", "unknown"),
        )

        raise HTTPException(
            status_code=502,
            detail=(
                "The AI could not generate an answer. "
                "Please try another request."
            ),
        )

    candidate = candidates[0]

    parts = (
        candidate.get("content") or {}
    ).get("parts") or []

    answer = "\n".join(
        part.get("text", "")
        for part in parts
        if part.get("text")
    ).strip()

    if not answer:

        logger.warning(
            "AI provider returned an empty answer."
        )

        raise HTTPException(
            status_code=502,
            detail=(
                "The AI returned an empty answer. "
                "Please try again."
            ),
        )

    sources = []
    seen_urls = set()

    grounding = (
        candidate.get("groundingMetadata") or {}
    )

    for chunk in grounding.get("groundingChunks") or []:

        web_source = chunk.get("web") or {}

        source_url = web_source.get("uri")

        if source_url and source_url not in seen_urls:

            seen_urls.add(source_url)

            sources.append({
                "title": web_source.get("title") or "Source",
                "url": source_url,
            })

    return {
        "answer": answer,
        "sources": sources,
    }


# ============================================================
# CALL AI MODEL
# ============================================================

def call_gemini(
    model: str,
    prompt: str,
    use_search: bool = False,
) -> dict:

    request = make_gemini_request(
        model,
        prompt,
        use_search,
    )

    with urllib.request.urlopen(
        request,
        timeout=REQUEST_TIMEOUT_SECONDS,
    ) as response:

        body = response.read().decode(
            "utf-8",
            errors="replace",
        )

    return parse_gemini_response(body)


# ============================================================
# AI SERVICE WITH RETRY AND FALLBACK
# ============================================================

def ask_ai(
    prompt: str,
    use_search: bool = False,
) -> dict:

    if not GEMINI_API_KEY:

        logger.error(
            "GEMINI_API_KEY is not configured."
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "AI service is not configured. "
                "Please contact the administrator."
            ),
        )

    if not validate_model_name(GEMINI_MODEL):

        raise HTTPException(
            status_code=503,
            detail="The primary AI model configuration is invalid.",
        )

    fallback_enabled = (
        validate_model_name(GEMINI_FALLBACK_MODEL)
        and GEMINI_FALLBACK_MODEL != GEMINI_MODEL
    )

    models_to_try = [GEMINI_MODEL]

    if fallback_enabled:
        models_to_try.append(GEMINI_FALLBACK_MODEL)

    last_status = None
    last_reason = "temporary provider failure"

    for model_index, model in enumerate(models_to_try):

        fallback_used = model_index > 0

        for attempt in range(1, MAX_PRIMARY_ATTEMPTS + 1):

            logger.info(
                "AI request started. Model=%s Attempt=%s/%s Fallback=%s",
                model,
                attempt,
                MAX_PRIMARY_ATTEMPTS,
                fallback_used,
            )

            try:

                result = call_gemini(
                    model,
                    prompt,
                    use_search,
                )

                logger.info(
                    "AI model succeeded. Model=%s Fallback=%s",
                    model,
                    fallback_used,
                )

                result["model"] = model
                result["fallback_used"] = fallback_used

                return result

            except urllib.error.HTTPError as error:

                last_status = error.code

                logger.warning(
                    "AI model HTTP error. Model=%s Status=%s",
                    model,
                    error.code,
                )

                if error.code in (401, 403):

                    raise HTTPException(
                        status_code=502,
                        detail=(
                            "The AI provider rejected its credentials "
                            "or permissions. The administrator must "
                            "check the server configuration."
                        ),
                    ) from error

                if error.code == 404:

                    last_reason = "model not found or unavailable"
                    break

                if error.code == 429:

                    last_reason = "provider quota limit"
                    break

                if error.code in RETRYABLE_HTTP_STATUSES:

                    last_reason = f"provider HTTP {error.code}"

                    if attempt < MAX_PRIMARY_ATTEMPTS:

                        wait_seconds = (
                            (2 ** attempt)
                            + random.uniform(0, 0.5)
                        )

                        time.sleep(wait_seconds)
                        continue

                    break

                raise HTTPException(
                    status_code=502,
                    detail=(
                        "The AI provider rejected the request "
                        f"(HTTP {error.code})."
                    ),
                ) from error

            except (
                TimeoutError,
                socket.timeout,
            ):

                last_reason = "provider timeout"

                logger.warning(
                    "AI model timed out. Model=%s",
                    model,
                )

                if attempt < MAX_PRIMARY_ATTEMPTS:

                    time.sleep(2 ** attempt)
                    continue

                break

            except urllib.error.URLError as error:

                reason = getattr(error, "reason", None)

                if isinstance(
                    reason,
                    (TimeoutError, socket.timeout),
                ):

                    last_reason = "provider network timeout"

                    logger.warning(
                        "AI model network timeout. Model=%s",
                        model,
                    )

                    if attempt < MAX_PRIMARY_ATTEMPTS:

                        time.sleep(2 ** attempt)
                        continue

                    break

                last_reason = "provider connection failure"

                logger.warning(
                    "AI provider connection failed. Model=%s",
                    model,
                )

                break

    if last_status == 404:

        raise HTTPException(
            status_code=502,
            detail=(
                "The configured AI model was not found or is unavailable. "
                "Please check the model names in the server configuration."
            ),
        )

    if last_status == 429:

        raise HTTPException(
            status_code=503,
            detail=(
                "The AI models have reached their temporary usage limits. "
                "Please try again later."
            ),
        )

    if last_status in RETRYABLE_HTTP_STATUSES:

        raise HTTPException(
            status_code=503,
            detail=(
                "The AI models are temporarily overloaded. "
                "Please try again in a few minutes."
            ),
        )

    if last_reason in {
        "provider timeout",
        "provider network timeout",
    }:

        raise HTTPException(
            status_code=504,
            detail=(
                "The AI provider did not respond in time. "
                "Please try again shortly."
            ),
        )

    raise HTTPException(
        status_code=503,
        detail=(
            "The AI provider cannot currently be reached. "
            "Please try again later."
        ),
    )


# ============================================================
# HEALTH AND INFORMATION ENDPOINTS
# ============================================================

@app.get("/")
def home():

    return {
        "platform": APP_NAME,
        "status": "running",
        "version": APP_VERSION,
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "version": APP_VERSION,
        "ai_configured": bool(GEMINI_API_KEY),
        "ai_status": (
            "configured" if GEMINI_API_KEY else "not_configured"
        ),
        "note": (
            "Configuration is present; provider availability is checked "
            "when an AI task is submitted."
        ),
    }


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
        "notice": (
            "These are demonstration jobs, "
            "not verified paid vacancies."
        ),
        "jobs": [
            {
                "title": "Content Writing",
                "category": "Writing",
                "type": "Demo",
            },
            {
                "title": "Online Research",
                "category": "Research",
                "type": "Demo",
            },
            {
                "title": "Data Entry",
                "category": "Administration",
                "type": "Demo",
            },
        ],
    }


# ============================================================
# PLAN AGENT
# ============================================================

@app.post("/plan")
def plan(task: Task):

    prompt = GLOBAL_QUALITY_STANDARD + "\n\n" + (
        "You are TaskPlanner for OnlineEarningHub AI. "
        "Create a realistic, numbered, step-by-step plan. "
        "Identify risks and distinguish confirmed facts "
        "from assumptions. Do not promise income or claim "
        "work has been completed.\n\n"
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


# ============================================================
# WRITING AGENT
# ============================================================

@app.post("/write")
def write(task: Task):

    prompt = GLOBAL_QUALITY_STANDARD + "\n\n" + (
        "You are WritingAssistant for OnlineEarningHub AI. "
        "Complete the user's writing request professionally. "
        "Use clear and useful language. Do not invent "
        "qualifications, work experience, or achievements.\n\n"
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


# ============================================================
# RESEARCH AGENT
# ============================================================

@app.post("/research")
def research(task: Task):

    prompt = GLOBAL_QUALITY_STANDARD + "\n\n" + (
        "You are ResearchAssistant for OnlineEarningHub AI. "
        "Analyze the topic carefully using your available "
        "knowledge. Prioritize reliable information. "
        "Distinguish established facts from assumptions. "
        "Do not claim to have searched the live internet. "
        "Do not invent citations, URLs, statistics, or sources. "
        "Explain when information may be outdated and needs "
        "verification.\n\n"
        f"Research topic: {task.description}"
    )

    result = ask_ai(
        prompt,
        use_search=False,
    )

    return {
        "agent": "ResearchAssistant",
        "topic": task.description,
        "findings": result["answer"],
        "sources": result["sources"],
        "research_mode": "AI knowledge; no live web search",
        "model_used": result["model"],
        "fallback_used": result["fallback_used"],
    }


# ============================================================
# QUALITY CHECK AGENT
# ============================================================

@app.post("/check")
def check(task: Task):

    prompt = GLOBAL_QUALITY_STANDARD + "\n\n" + (
        "You are QualityChecker for OnlineEarningHub AI. "
        "Review the submitted work for clarity, completeness, "
        "and accuracy. List errors, missing information, "
        "and practical improvements. Do not claim facts were "
        "verified unless you actually verified them.\n\n"
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


# ============================================================
# STARTUP INFORMATION
# ============================================================

@app.on_event("startup")
def startup_event():

    logger.info(
        "%s version %s starting.",
        APP_NAME,
        APP_VERSION,
    )

    logger.info(
        "Allowed frontend origins: %s",
        ALLOWED_ORIGINS,
    )

    if not GEMINI_API_KEY:

        logger.warning(
            "GEMINI_API_KEY is not configured."
        )
