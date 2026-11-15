
import os
import re
import json
import socket
import logging
import urllib.request
import urllib.error

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# --------------------------------------------------
# APPLICATION CONFIGURATION
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)

logger = logging.getLogger("onlineearninghub")

app = FastAPI(
    title="OnlineEarningHub AI",
    description="AI-powered tools for planning, writing, research and quality checking.",
    version="0.4.0"
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash"
).strip()

GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
)


# --------------------------------------------------
# REQUEST MODEL
# --------------------------------------------------

class Task(BaseModel):
    description: str = Field(
        min_length=3,
        max_length=10000,
        description="The task or topic to process."
    )


# --------------------------------------------------
# GEMINI AI SERVICE
# --------------------------------------------------

def ask_ai(prompt: str, use_search: bool = False):

    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="AI service is not configured."
        )

    if not GEMINI_MODEL or not re.fullmatch(
        r"[A-Za-z0-9._-]+",
        GEMINI_MODEL
    ):
        logger.error("Invalid GEMINI_MODEL configuration.")
        raise HTTPException(
            status_code=503,
            detail="AI model configuration is invalid."
        )

    url = (
        f"{GEMINI_API_URL}"
        f"{GEMINI_MODEL}:generateContent"
    )

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
            "maxOutputTokens": 1800
        }
    }

    if use_search:
        payload["tools"] = [
            {"google_search": {}}
        ]

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": GEMINI_API_KEY
        },
        method="POST"
    )

    try:
        logger.info(
            "Sending request to Gemini model: %s",
            GEMINI_MODEL
        )

        with urllib.request.urlopen(
            request,
            timeout=60
        ) as response:

            response_data = response.read().decode("utf-8")
            result = json.loads(response_data)

        candidates = result.get("candidates", [])

        if not candidates:
            feedback = result.get("promptFeedback", {})
            block_reason = feedback.get(
                "blockReason",
                "No candidate returned"
            )

            logger.warning(
                "Gemini returned no candidates: %s",
                block_reason
            )

            raise HTTPException(
                status_code=502,
                detail="AI could not generate a response. Please try again."
            )

        candidate = candidates[0]

        parts = candidate.get(
            "content", {}
        ).get("parts", [])

        answer = "\n".join(
            part.get("text", "")
            for part in parts
            if part.get("text")
        ).strip()

        if not answer:
            logger.warning(
                "Gemini returned an empty answer. Finish reason: %s",
                candidate.get("finishReason", "unknown")
            )

            raise HTTPException(
                status_code=502,
                detail="AI returned an empty response. Please try again."
            )

        # Extract Google Search grounding sources when available.
        metadata = candidate.get(
            "groundingMetadata", {}
        )

        sources = []
        seen_urls = set()

        for chunk in metadata.get("groundingChunks", []):
            web_source = chunk.get("web", {})
            source_url = web_source.get("uri")

            if source_url and source_url not in seen_urls:
                seen_urls.add(source_url)

                sources.append({
                    "title": web_source.get(
                        "title", "Source"
                    ),
                    "url": source_url
                })

        logger.info("Gemini request completed successfully.")

        return {
            "answer": answer,
            "sources": sources
        }

    except HTTPException:
        raise

    except urllib.error.HTTPError as error:
        error_body = error.read().decode(
            "utf-8",
            errors="replace"
        )

        logger.error(
            "Gemini HTTP error: status=%s, response=%s",
            error.code,
            error_body[:500]
        )

        if error.code == 429:
            raise HTTPException(
                status_code=503,
                detail="AI service is busy or its usage limit has been reached. Try again later."
            )

        if error.code in (401, 403):
            raise HTTPException(
                status_code=502,
                detail="AI provider rejected the request. Check API credentials and permissions."
            )

        raise HTTPException(
            status_code=502,
            detail=f"AI provider returned HTTP {error.code}."
        )

    except (TimeoutError, socket.timeout):
        logger.exception("Gemini request timed out.")

        raise HTTPException(
            status_code=504,
            detail="AI request timed out. Please try again."
        )

    except urllib.error.URLError as error:
        if isinstance(
            error.reason,
            (TimeoutError, socket.timeout)
        ):
            logger.exception("Gemini network request timed out.")

            raise HTTPException(
                status_code=504,
                detail="AI request timed out. Please try again."
            )

        logger.error(
            "Gemini connection error: %s",
            error.reason
        )

        raise HTTPException(
            status_code=503,
            detail="Could not connect to the AI provider. Try again later."
        )

    except json.JSONDecodeError:
        logger.exception("Gemini returned invalid JSON.")

        raise HTTPException(
            status_code=502,
            detail="AI provider returned an invalid response."
        )

    except Exception:
        logger.exception("Unexpected error in AI service.")

        raise HTTPException(
            status_code=500,
            detail="An unexpected AI service error occurred."
        )


# --------------------------------------------------
# HOME ENDPOINT
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "platform": "OnlineEarningHub AI",
        "status": "running",
        "version": "0.4.0"
    }


# --------------------------------------------------
# AGENTS ENDPOINT
# --------------------------------------------------

@app.get("/agents")
def agents():
    return {
        "agents": [
            "JobFinder",
            "TaskPlanner",
            "WritingAssistant",
            "ResearchAssistant",
            "QualityChecker"
        ]
    }


# --------------------------------------------------
# JOBS ENDPOINT
# --------------------------------------------------

@app.get("/jobs")
def jobs():
    return {
        "notice": "These are demonstration jobs, not verified paid vacancies.",
        "jobs": [
            {
                "title": "Content Writing",
                "category": "Writing",
                "type": "Demo"
            },
            {
                "title": "Online Research",
                "category": "Research",
                "type": "Demo"
            },
            {
                "title": "Data Entry",
                "category": "Administration",
                "type": "Demo"
            }
        ]
    }


# --------------------------------------------------
# TASK PLANNER
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
        "plan": result["answer"]
    }


# --------------------------------------------------
# WRITING ASSISTANT
# --------------------------------------------------

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
        "content": result["answer"]
    }


# --------------------------------------------------
# RESEARCH ASSISTANT
# --------------------------------------------------

@app.post("/research")
def research(task: Task):

    prompt = (
        "You are ResearchAssistant for OnlineEarningHub AI. "
        "Analyze the topic carefully using your available knowledge. "
        "Prioritize reliable information. "
        "Clearly distinguish established facts from assumptions. "
        "Do not claim to have searched the live internet. "
        "Do not invent citations, URLs, statistics, or sources. "
        "Explain when information may be outdated and needs verification.\n\n"
        f"Research topic: {task.description}"
    )

    result = ask_ai(
        prompt,
        use_search=False
    )

    return {
        "agent": "ResearchAssistant",
        "topic": task.description,
        "findings": result["answer"],
        "sources": result["sources"],
        "research_mode": "AI knowledge; no live web search"
    }


# --------------------------------------------------
# QUALITY CHECKER
# --------------------------------------------------

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
        "review": result["answer"]
    }
