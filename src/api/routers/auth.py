from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from datetime import timedelta
from typing import Optional

from src.services.auth import security
from src.services.db.database import DatabaseService

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")
db = DatabaseService()

# --- Models ---
class UserCreate(BaseModel):
    username: str
    password: str

class User(BaseModel):
    id: int
    username: str
    created_at: Optional[str] = None

class Token(BaseModel):
    access_token: str
    token_type: str

# --- Dependency ---
async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    payload = security.decode_access_token(token)
    if payload is None:
        raise credentials_exception
    
    user_dict = db.get_user_by_username(payload.username)
    if user_dict is None:
        raise credentials_exception
    return User(**user_dict)

# --- Routes ---

@router.post("/register", response_model=User)
async def register(user: UserCreate):
    # Check existing
    existing = db.get_user_by_username(user.username)
    if existing:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    hashed_pw = security.get_password_hash(user.password)
    user_id = db.create_user(user.username, hashed_pw)
    
    if not user_id:
        raise HTTPException(status_code=500, detail="Failed to create user")
        
    return {
        "id": user_id,
        "username": user.username,
        "created_at": None # Computed by DB
    }

@router.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user_dict = db.get_user_by_username(form_data.username)
    if not user_dict or not security.verify_password(form_data.password, user_dict["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=security.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = security.create_access_token(
        data={"sub": user_dict["username"]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/users/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user
