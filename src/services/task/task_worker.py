import time
import json
import threading
from typing import Optional
from src.services.db.database import DatabaseService
from src.agents.orchestrator import AgentOrchestrator
from src.core.logger import get_logger

logger = get_logger(__name__)

class TaskWorker:
    """
    Asynchronous Task Worker that polls the database for queued tasks.
    Supports V5.6 Infrastructure Scaling.
    """
    def __init__(self, check_interval: float = 5.0):
        self.db = DatabaseService()
        self.orchestrator = AgentOrchestrator()
        self.check_interval = check_interval
        self.running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("TaskWorker started.")

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join()
        logger.info("TaskWorker stopped.")

    def _run(self):
        while self.running:
            try:
                tasks = self.db.list_robot_tasks(limit=5)
                queued_tasks = [t for t in tasks if t["status"] == "queued"]
                
                for task in queued_tasks:
                    self._process_task(task)
                    
            except Exception as e:
                logger.error(f"TaskWorker error: {e}")
            
            time.sleep(self.check_interval)

    def _process_task(self, task: dict):
        task_id = task["id"]
        task_type = task["task_type"]
        payload = task.get("payload", {})
        
        logger.info(f"Processing task {task_id} ({task_type})")
        self.db.update_robot_task(task_id, status="processing")
        
        try:
            if task_type == "catalyst_discovery":
                query = payload.get("query")
                if not query:
                    raise ValueError("Missing query in payload")
                
                # Execute orchestration
                result = self.orchestrator.orchestrate(query)
                self.db.update_robot_task(
                    task_id, 
                    status="completed", 
                    result=result.to_dict() if hasattr(result, "to_dict") else str(result)
                )
            else:
                logger.warning(f"Unknown task type: {task_type}")
                self.db.update_robot_task(task_id, status="failed", result={"error": "Unknown type"})
                
        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}")
            self.db.update_robot_task(task_id, status="failed", result={"error": str(e)})

if __name__ == "__main__":
    worker = TaskWorker()
    worker.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        worker.stop()
