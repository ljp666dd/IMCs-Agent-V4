from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.services.knowledge import KnowledgeService, KnowledgeRAG
from src.services.db.database import DatabaseService

router = APIRouter()
svc = KnowledgeService()
rag = KnowledgeRAG()
db = DatabaseService()


class EntityCreate(BaseModel):
    entity_type: str
    name: str
    canonical_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class SourceCreate(BaseModel):
    source_type: str
    source_id: Optional[str] = None
    title: Optional[str] = None
    url: Optional[str] = None
    year: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class RelationCreate(BaseModel):
    subject_id: int
    predicate: str
    object_id: int
    confidence: Optional[float] = None
    evidence_source_id: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class RelationEvidenceCreate(BaseModel):
    source_id: int
    score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class RagQuery(BaseModel):
    query: str
    material_id: Optional[str] = None
    entity_id: Optional[int] = None
    top_k: int = 5
    source_type: Optional[str] = "literature"


class AdsorptionCreate(BaseModel):
    material_id: Optional[str] = None
    surface_composition: str
    facet: str
    adsorbate: str
    reaction_energy: Optional[float] = None
    activation_energy: Optional[float] = None
    source: str = "Catalysis-Hub"
    metadata: Optional[Dict[str, Any]] = None


class ActivityMetricCreate(BaseModel):
    material_id: Optional[str] = None
    metric_name: str
    metric_value: Optional[float] = None
    unit: Optional[str] = None
    conditions: Optional[Dict[str, Any]] = None
    source: str = "experiment"
    source_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@router.post("/entities")
def create_entity(req: EntityCreate):
    entity = svc.ensure_entity(
        req.entity_type,
        req.name,
        canonical_id=req.canonical_id,
        metadata=req.metadata,
    )
    if not entity:
        raise HTTPException(status_code=500, detail="Failed to create entity")
    return entity


@router.get("/entities/{entity_id}")
def get_entity(entity_id: int):
    entity = svc.get_entity(entity_id)
    if not entity:
        raise HTTPException(status_code=404, detail="Entity not found")
    return entity


@router.get("/entities/by-name")
def get_entity_by_name(entity_type: str, name: str):
    entity = svc.get_entity_by_name(entity_type, name)
    if not entity:
        raise HTTPException(status_code=404, detail="Entity not found")
    return entity


@router.post("/sources")
def create_source(req: SourceCreate):
    source = svc.create_source(
        req.source_type,
        source_id=req.source_id,
        title=req.title,
        url=req.url,
        year=req.year,
        metadata=req.metadata,
    )
    if not source:
        raise HTTPException(status_code=500, detail="Failed to create source")
    return source


@router.post("/relations")
def create_relation(req: RelationCreate):
    rel = svc.create_relation(
        req.subject_id,
        req.predicate,
        req.object_id,
        confidence=req.confidence,
        evidence_source_id=req.evidence_source_id,
        metadata=req.metadata,
    )
    if not rel:
        raise HTTPException(status_code=500, detail="Failed to create relation")
    return rel


@router.post("/relations/{relation_id}/evidence")
def add_relation_evidence(relation_id: int, req: RelationEvidenceCreate):
    ev = svc.add_relation_evidence(
        relation_id,
        req.source_id,
        score=req.score,
        metadata=req.metadata,
    )
    if not ev:
        raise HTTPException(status_code=500, detail="Failed to add evidence")
    return ev


@router.get("/relations/by-entity/{entity_id}")
def list_relations(entity_id: int, direction: str = "both", limit: int = 200):
    return svc.list_relations(entity_id, direction=direction, limit=limit)


@router.get("/trace/{entity_id}")
def trace_entity(entity_id: int, depth: int = 1, limit: int = 200):
    return svc.trace_entity(entity_id, depth=depth, limit=limit)


@router.post("/rag")
def graph_rag(req: RagQuery):
    if not req.query:
        raise HTTPException(status_code=400, detail="Query is required")

    entity_id = req.entity_id
    if not entity_id and req.material_id:
        ent = svc.get_entity_by_canonical("material", req.material_id)
        if not ent:
            ent = svc.get_entity_by_name("material", req.material_id)
        if ent:
            entity_id = ent["id"]

    candidate_source_ids = []
    mode = "global"
    if entity_id:
        rels = svc.list_relations(entity_id, direction="out", limit=500)
        for rel in rels:
            if rel.get("predicate") != "supported_by":
                continue
            src_id = rel.get("evidence_source_id")
            if src_id:
                candidate_source_ids.append(src_id)
        if candidate_source_ids:
            mode = "graph_filtered"

    results = rag.query(
        req.query,
        candidate_source_ids=candidate_source_ids if candidate_source_ids else None,
        top_k=req.top_k or 5,
        source_type=req.source_type,
    )
    return {
        "mode": mode,
        "entity_id": entity_id,
        "candidate_sources": len(candidate_source_ids),
        "results": results,
    }


@router.post("/adsorption")
def create_adsorption(req: AdsorptionCreate):
    record_id = db.save_adsorption_energy(
        material_id=req.material_id,
        surface_composition=req.surface_composition,
        facet=req.facet,
        adsorbate=req.adsorbate,
        reaction_energy=req.reaction_energy,
        activation_energy=req.activation_energy,
        source=req.source,
        metadata=req.metadata,
    )
    return {"id": record_id}


@router.get("/adsorption/{material_id}")
def list_adsorption(material_id: str):
    return db.list_adsorption_energies(material_id)


@router.post("/activity")
def create_activity_metric(req: ActivityMetricCreate):
    record_id = db.save_activity_metric(
        material_id=req.material_id,
        metric_name=req.metric_name,
        metric_value=req.metric_value,
        unit=req.unit,
        conditions=req.conditions,
        source=req.source,
        source_id=req.source_id,
        metadata=req.metadata,
    )
    return {"id": record_id}


@router.get("/activity/{material_id}")
def list_activity_metrics(material_id: str):
    return db.list_activity_metrics(material_id)


@router.get("/stats")
def get_knowledge_stats():
    """Return evidence coverage and data stats for meta-controller/UI."""
    try:
        from src.agents.core.theory_agent import TheoryDataConfig
        allowed = TheoryDataConfig().elements
    except Exception:
        allowed = None
    return db.get_evidence_stats(allowed_elements=allowed)
