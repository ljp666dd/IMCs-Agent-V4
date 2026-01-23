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
class TaskPlan:
    """Plan for executing a task."""
    task_id: str
    task_type: TaskType
    description: str
    steps: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "pending"
    results: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
