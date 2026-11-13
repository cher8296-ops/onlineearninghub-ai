
import os
import json
import urllib.request
import urllib.error

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="OnlineEarningHub AI",
    version="0.3.0"
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


class Task(BaseModel):
    description: str = Field(min_length=3, max_length=10000)


def ask_ai(prompt: str, use_search: bool = False):
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="AI is not configured. Add GEMINI_API_KEY in Render."
        )

    url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        f"models/{GEMINI_MODEL}:generateContent"
    )

    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "temperature": 0.4,
            "maxOutputTokens": 1800
        }
    }

    if use_search:
        payload["tools"] = [{"google_search": {}}]

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
        with urllib.request.urlopen(request, timeout=60) as response:
            result = json.loads(response.read().decode("utf-8"))

        candidates = result.get("candidates", [])
        if not candidates:
            raise HTTPException(
                status_code=502,
                detail="AI returned no response. Please try again."
            )

        parts = candidates[0].get("content", {}).get("parts", [])
        answer = "\n".join(
            part.get("text", "")
            for part in parts
            if part.get("text")
        )

        metadata = candidates[0].get("groundingMetadata", {})
        sources = []

        for chunk in metadata.get("groundingChunks", []):
            web_source = chunk.get("web", {})
            if web_source.get("uri"):
                sources.append({
                    "title": web_source.get("title", "Source"),
                    "url": web_source["uri"]
                })

        return {
            "answer": answer,
            "sources": sources
        }

    except urllib.error.HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise HTTPException(
            status_code=502,
            detail=f"AI provider error ({error.code}): {details[:1000]}"
        )

    except urllib.error.URLError:
        raise HTTPException(
            status_code=503,
            detail="Could not connect to the AI provider. Try again later."
        )

    except TimeoutError:
        raise HTTPException(
            status_code=504,
            detail="AI request timed out. Please try again."
        )


@app.get("/")
def home():
    return {
        "platform": "OnlineEarningHub AI",
        "status": "running",
        "version": "0.3.0"
    }


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


@app.get("/jobs")
def jobs():
    return {
        "notice": "These are demonstration jobs, not verified paid vacancies.",
        "jobs": [
            {"title": "Content Writing", "category": "Writing", "type": "Demo"},
            {"title": "Online Research", "category": "Research", "type": "Demo"},
            {"title": "Data Entry", "category": "Administration", "type": "Demo"}
        ]
    }


@app.post("/plan")
def plan(task: Task):
    result = ask_ai(
        "You are TaskPlanner for OnlineEarningHub AI. "
        "Create a realistic, numbered step-by-step plan. "
        "Identify risks and distinguish confirmed facts from assumptions. "
        "Do not promise income or claim work has been completed.\n\n"
        f"User task: {task.description}"
    )

    return {
        "agent": "TaskPlanner",
        "task": task.description,
        "plan": result["answer"]
    }


@app.post("/write")
def write(task: Task):
    result = ask_ai(
        "You are WritingAssistant for OnlineEarningHub AI. "
        "Complete the user's writing request professionally. "
        "Do not invent qualifications, work experience, or achievements.\n\n"
        f"Writing request: {task.description}"
    )

    return {
        "agent": "WritingAssistant",
        "task": task.description,
        "content": result["answer"]
    }


@app.post("/research")
def research(task: Task):
    result = ask_ai(
        "You are ResearchAssistant for OnlineEarningHub AI. "
        "Research the user's topic using Google Search grounding. "
        "Prioritize reliable and recent sources. "
        "Clearly state uncertainty and never invent sources. "
        "If information cannot be verified, say so.\n\n"
        f"Research topic: {task.description}",
        use_search=True
    )

    return {
        "agent": "ResearchAssistant",
        "topic": task.description,
        "findings": result["answer"],
        "sources": result["sources"]
    }


@app.post("/check")
def check(task: Task):
    result = ask_ai(
        "You are QualityChecker for OnlineEarningHub AI. "
        "Review the submitted work for clarity, completeness, and accuracy. "
        "List errors, missing information, and practical improvements. "
        "Do not claim facts were verified unless you actually verified them.\n\n"
        f"Work to review: {task.description}"
    )

    return {
        "agent": "QualityChecker",
        "submitted_work": task.description,
        "review": result["answer"]
    }
