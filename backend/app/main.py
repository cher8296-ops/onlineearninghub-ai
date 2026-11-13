
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="OnlineEarningHub AI",
    version="0.2.0"
)

class Task(BaseModel):
    description: str

@app.get("/")
def home():
    return {
        "platform": "OnlineEarningHub AI",
        "status": "running",
        "version": "0.2.0"
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
        "jobs": [
            {"title": "Content Writing", "category": "Writing", "type": "Demo"},
            {"title": "Online Research", "category": "Research", "type": "Demo"},
            {"title": "Data Entry", "category": "Administration", "type": "Demo"}
        ]
    }

@app.post("/plan")
def plan(task: Task):
    return {
        "agent": "TaskPlanner",
        "task": task.description,
        "steps": [
            "Understand the task",
            "Prepare a work plan",
            "Complete the work",
            "Review the result"
        ]
    }

@app.post("/write")
def write(task: Task):
    return {
        "agent": "WritingAssistant",
        "task": task.description,
        "message": "Writing demo received your task. AI generation is not yet connected."
    }

@app.post("/research")
def research(task: Task):
    return {
        "agent": "ResearchAssistant",
        "topic": task.description,
        "message": "Research demo received your topic. Live web research is not yet connected."
    }

@app.post("/check")
def check(task: Task):
    return {
        "agent": "QualityChecker",
        "submitted_work": task.description,
        "status": "Received for review",
        "checks": ["Clarity", "Completeness", "Accuracy"]
    }
