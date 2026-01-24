from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional

class TaskType(Enum):
    """Types of tasks the system can handle."""
    CATALYST_DISCOVERY = "catalyst_discovery"
    PROPERTY_PREDICTION = "property_prediction"
    PERFORMANCE_ANALYSIS = "performance_analysis"
    LITERATURE_REVIEW = "literature_review"
    GENERAL = "general"

@dataclass
class TaskStep:
    """A single step in the execution graph."""
    step_id: str
    agent: str
    action: str
    params: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list) # List of step_ids
    status: str = "pending" # pending, running, completed, failed
    result: Any = None
    error: str = None

@dataclass
class TaskPlan:
    """Plan for executing a task (DAG)."""
    task_id: str
    task_type: TaskType
    description: str
    steps: List[TaskStep] = field(default_factory=list)
    status: str = "pending"
    results: Dict[str, Any] = field(default_factory=dict) # Aggregate results
    created_at: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
            
    def get_step(self, step_id: str) -> Optional[TaskStep]:
        for s in self.steps:
            if s.step_id == step_id:
                return s
        return None
