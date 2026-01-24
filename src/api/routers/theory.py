from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

# Service Instance
from src.agents.core.theory_agent import TheoryDataAgent
theory_service = TheoryDataAgent()

router = APIRouter()

class SearchRequest(BaseModel):
    elements: List[str]
    limit: int = 20

@router.post("/search")
async def search_materials(req: SearchRequest):
    """Search materials by elements."""
    # Assuming MPClient exposes search_materials via TheoryAgent
    # TheoryAgent v3.1 exposes query_oqmd, query_aflow etc.
    # It also has download_structures which searches internally.
    # We might need to expose a pure search method in Agent or access Service directly.
    # For now, let's use the underlying Service directly for purity.
    
    docs = theory_service.mp.search_materials(req.elements, limit=req.limit)
    return [
        {
            "material_id": str(d.material_id),
            "formula": str(d.formula_pretty),
            "energy": float(d.formation_energy_per_atom) if d.formation_energy_per_atom else None
        }
        for d in docs
    ]

@router.post("/download/cif")
async def download_cifs(req: SearchRequest, background_tasks: BackgroundTasks):
    """Download CIFs (Async)."""
    background_tasks.add_task(theory_service.download_structures, None, req.limit) # uses config elements if None
    # If we want specific elements, we need to update Agent config or method
    return {"message": "CIF download started"}

@router.get("/status")
async def get_status():
    """Get data status."""
    return theory_service.get_status()

@router.get("/materials")
async def list_materials():
    """List all materials in DB."""
    return theory_service.list_stored_materials()

@router.get("/materials/{material_id}")
async def get_material(material_id: str):
    """Get material details + CIF."""
    data = theory_service.get_material_details(material_id)
    if not data:
        raise HTTPException(status_code=404, detail="Material not found")
    return data
