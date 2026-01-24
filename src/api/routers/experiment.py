from fastapi import APIRouter, File, UploadFile, HTTPException
from typing import Dict, Any
import os
import shutil

# Service Instance
from src.agents.core.experiment_agent import ExperimentDataAgent, LSVResult
exp_service = ExperimentDataAgent()

router = APIRouter()

TEMP_UPLOAD_DIR = "data/temp_uploads"
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

@router.post("/upload")
async def upload_experiment_data(file: UploadFile = File(...), method: str = "lsv"):
    """
    Upload and analyze experiment file.
    
    Args:
        file: The uploaded file (CSV).
        method: Analysis method (default: lsv).
    """
    try:
        # Save temp file
        file_path = os.path.join(TEMP_UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Process using Agent
        # Agent will also save to DB
        result = exp_service.process_request(file_path, method)
        
        if not result:
            return {"error": "Analysis failed or returned no result"}
            
        # Serialize result
        # If it's a dataclass, convert to dict
        if hasattr(result, "__dict__"):
            data = result.__dict__
        else:
            data = result
            
        return {
            "message": "File processed successfully",
            "filename": file.filename,
            "analysis": data
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup temp file? 
        # For now keep it as raw_data record needs it. 
        # But in production we should move it to permanent storage.
        # ExperimentAgent uses the path provided.
        pass
