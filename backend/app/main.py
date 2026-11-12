from fastapi import FastAPI

app = FastAPI(title="OnlineEarningHub AI")

@app.get("/")
def home():
    return {
        "platform": "OnlineEarningHub AI",
        "status": "running",
        "message": "Welcome!"
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
