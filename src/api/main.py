from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import sys

# Add src to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

app = FastAPI(
    title="IMCs Engineering Platform API",
    version="3.3",
    description="Backend API for Intelligent Materials Catalyst System"
)

# CORS (Allow Frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace with frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health Check
@app.get("/health")
async def health_check():
    return {"status": "ok", "version": "3.3"}

# Include Routers (Lazy import to avoid circular dep issues during init)
from src.api.routers import tasks, ml, theory, experiment
app.include_router(tasks.router, prefix="/tasks", tags=["Tasks"])
app.include_router(ml.router, prefix="/ml", tags=["Machine Learning"])
app.include_router(theory.router, prefix="/theory", tags=["Theory"])
app.include_router(experiment.router, prefix="/experiment", tags=["Experiment"])

try:
    from src.api.routers import auth
    app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
except ImportError:
    pass # Dependencies not installed yet

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
