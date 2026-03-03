from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import sys

from src.core.logger import get_logger
from src.config.config import config

# Add src to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

logger = get_logger(__name__)
APP_VERSION = os.getenv("IMCS_API_VERSION", "4.0")

app = FastAPI(
    title="IMCs Engineering Platform API",
    version=APP_VERSION,
    description="Backend API for Intelligent Materials Catalyst System"
)

# Startup warnings for unsafe defaults
if config.SECRET_KEY == "dev-secret-change-me-in-production":
    logger.warning("IMCS_SECRET_KEY is using the default value; set a secure key in .env.")
if config.MP_API_KEY == "abx7GG5NQg5YncfROEP4vvQi8Tc5Ywqp":
    logger.warning("MP_API_KEY is using the default placeholder; set your real API key in .env.")

# CORS (Allow Frontend)
cors_env = os.getenv("IMCS_CORS_ORIGINS", "*")
origins = [o.strip() for o in cors_env.split(",") if o.strip()]
if not origins:
    origins = ["*"]
if origins == ["*"]:
    logger.warning("IMCS_CORS_ORIGINS is '*'; restrict origins for production.")
allow_credentials = str(os.getenv("IMCS_CORS_ALLOW_CREDENTIALS", "true")).lower() in ("1", "true", "yes")
if origins == ["*"] and allow_credentials:
    logger.warning("CORS allow_credentials=True with wildcard origins is not recommended.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, # In production, replace with frontend URL
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health Check
@app.get("/health")
async def health_check():
    return {"status": "ok", "version": APP_VERSION}

# Include Routers (Lazy import to avoid circular dep issues during init)
from src.api.routers import tasks, ml, theory, experiment, knowledge, robot
app.include_router(tasks.router, prefix="/tasks", tags=["Tasks"])
app.include_router(ml.router, prefix="/ml", tags=["Machine Learning"])
app.include_router(theory.router, prefix="/theory", tags=["Theory"])
app.include_router(experiment.router, prefix="/experiment", tags=["Experiment"])
app.include_router(knowledge.router, prefix="/knowledge", tags=["Knowledge"])
app.include_router(robot.router, prefix="/robot", tags=["Robot/Middleware"])

try:
    from src.api.routers import auth
    app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
except ImportError:
    pass # Dependencies not installed yet

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
