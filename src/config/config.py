import os
from dotenv import load_dotenv

# Load .env file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(BASE_DIR, ".env"))

class Config:
    """
    Central Configuration.
    Reads from environment variables (Priority) or defaults (Dev).
    """
    
    # Security
    SECRET_KEY = os.getenv("IMCS_SECRET_KEY", "dev-secret-change-me-in-production")
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 300
    
    # External APIs
    MP_API_KEY = os.getenv("MP_API_KEY", "abx7GG5NQg5YncfROEP4vvQi8Tc5Ywqp") # Default is risky, user should override
    
    # Paths
    DATA_DIR = os.getenv("DATA_DIR", "data")
    DB_PATH = os.path.join(DATA_DIR, "imcs.db")
    
    # ML
    ML_MODEL_DIR = os.path.join(DATA_DIR, "ml_agent")

config = Config()
