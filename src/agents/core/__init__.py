"""
Multi-Agent Catalyst Research System
Core Agents Package

This package provides specialized agents for:
- Machine Learning (MLAgent)
- Theoretical Data (TheoryDataAgent)
- Experimental Data (ExperimentDataAgent)
- Literature Research (LiteratureAgent)
- Task Management (TaskManagerAgent)
"""

from .ml_agent import MLAgent, MLAgentConfig, ModelResult, ModelType
from .theory_agent import TheoryDataAgent, TheoryDataConfig
from .experiment_agent import ExperimentDataAgent, ExperimentDataConfig
from .literature_agent import LiteratureAgent, LiteratureConfig
from .task_manager import TaskManagerAgent, TaskPlan, TaskType

__all__ = [
    # ML Agent
    "MLAgent",
    "MLAgentConfig",
    "ModelResult",
    "ModelType",
    
    # Theory Agent
    "TheoryDataAgent",
    "TheoryDataConfig",
    
    # Experiment Agent
    "ExperimentDataAgent",
    "ExperimentDataConfig",
    
    # Literature Agent
    "LiteratureAgent",
    "LiteratureConfig",
    
    # Task Manager
    "TaskManagerAgent",
    "TaskPlan",
    "TaskType",
]

__version__ = "1.0.0"

def create_research_system():
    """Create a complete research system with all agents."""
    return TaskManagerAgent()
