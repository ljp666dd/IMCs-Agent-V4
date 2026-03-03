import asyncio
import os
from datetime import datetime

from fastapi import BackgroundTasks, HTTPException

from src.services.db.database import DatabaseService


def _cleanup_db_files(db_path: str) -> None:
    for suffix in ("", "-wal", "-shm"):
        try:
            os.remove(db_path + suffix)
        except Exception:
            pass


def _count_rows(db: DatabaseService, table: str) -> int:
    with db._get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        return int(cursor.fetchone()[0] or 0)


def test_robot_result_callback_idempotent_by_callback_id():
    import src.api.routers.robot as robot

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    db_path = os.path.join("data", f"test_robot_callback_{stamp}.db")
    test_db = DatabaseService(db_path=db_path)

    old_db = robot.db
    robot.db = test_db
    try:
        task_id = test_db.create_robot_task("experiment", payload={"x": 1}, external_id="ext-1")
        req = robot.RobotResultCallback(
            task_id=task_id,
            external_id="ext-1",
            callback_id="cb-1",
            status="succeeded",
            result={"material_id": "mp-1", "metrics": {"exchange_current_density": 1.23}},
            auto_iterate=False,
            auto_ingest=False,
        )

        resp1 = asyncio.run(robot.result_callback(req, BackgroundTasks()))
        assert resp1.get("duplicate") is False
        assert resp1.get("saved_metrics") == 1
        assert _count_rows(test_db, "robot_task_events") == 1
        assert _count_rows(test_db, "activity_metrics") == 1

        events = test_db.list_robot_task_events(task_id=task_id, limit=10)
        assert len(events) == 1
        assert events[0].get("callback_id") == "cb-1"
        assert events[0].get("status") == "completed"
        assert isinstance(events[0].get("payload"), dict)

        resp2 = asyncio.run(robot.result_callback(req, BackgroundTasks()))
        assert resp2.get("duplicate") is True
        assert resp2.get("saved_metrics") == 0
        assert _count_rows(test_db, "robot_task_events") == 1
        assert _count_rows(test_db, "activity_metrics") == 1
    finally:
        robot.db = old_db
        _cleanup_db_files(db_path)


def test_robot_result_callback_idempotent_by_payload_hash_without_callback_id():
    import src.api.routers.robot as robot

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    db_path = os.path.join("data", f"test_robot_callback_hash_{stamp}.db")
    test_db = DatabaseService(db_path=db_path)

    old_db = robot.db
    robot.db = test_db
    try:
        task_id = test_db.create_robot_task("experiment", payload={"x": 1}, external_id="ext-2")
        req = robot.RobotResultCallback(
            task_id=task_id,
            external_id="ext-2",
            status="completed",
            result={"material_id": "mp-2", "metrics": {"exchange_current_density": 2.34}},
            auto_iterate=False,
            auto_ingest=False,
        )

        resp1 = asyncio.run(robot.result_callback(req, BackgroundTasks()))
        assert resp1.get("duplicate") is False
        assert resp1.get("saved_metrics") == 1
        assert _count_rows(test_db, "robot_task_events") == 1
        assert _count_rows(test_db, "activity_metrics") == 1

        events = test_db.list_robot_task_events(task_id=task_id, limit=10)
        assert len(events) == 1
        assert events[0].get("callback_id") is None
        assert events[0].get("status") == "completed"

        resp2 = asyncio.run(robot.result_callback(req, BackgroundTasks()))
        assert resp2.get("duplicate") is True
        assert resp2.get("saved_metrics") == 0
        assert _count_rows(test_db, "robot_task_events") == 1
        assert _count_rows(test_db, "activity_metrics") == 1
    finally:
        robot.db = old_db
        _cleanup_db_files(db_path)


def test_robot_result_callback_conflict_when_callback_id_reused_with_different_payload():
    import src.api.routers.robot as robot

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    db_path = os.path.join("data", f"test_robot_callback_conflict_{stamp}.db")
    test_db = DatabaseService(db_path=db_path)

    old_db = robot.db
    robot.db = test_db
    try:
        task_id = test_db.create_robot_task("experiment", payload={"x": 1}, external_id="ext-3")
        req1 = robot.RobotResultCallback(
            task_id=task_id,
            external_id="ext-3",
            callback_id="cb-dup",
            status="completed",
            result={"material_id": "mp-3", "metrics": {"exchange_current_density": 1.0}},
            auto_iterate=False,
            auto_ingest=False,
        )
        asyncio.run(robot.result_callback(req1, BackgroundTasks()))
        assert _count_rows(test_db, "robot_task_events") == 1
        assert _count_rows(test_db, "activity_metrics") == 1

        events = asyncio.run(robot.task_events(task_id=task_id, limit=10))
        assert events.get("task_id") == task_id
        assert len(events.get("events") or []) == 1

        req2 = robot.RobotResultCallback(
            task_id=task_id,
            external_id="ext-3",
            callback_id="cb-dup",
            status="completed",
            result={"material_id": "mp-3", "metrics": {"exchange_current_density": 9.9}},
            auto_iterate=False,
            auto_ingest=False,
        )
        try:
            asyncio.run(robot.result_callback(req2, BackgroundTasks()))
            assert False, "expected conflict HTTPException"
        except HTTPException as e:
            assert e.status_code == 409

        assert _count_rows(test_db, "robot_task_events") == 1
        assert _count_rows(test_db, "activity_metrics") == 1
    finally:
        robot.db = old_db
        _cleanup_db_files(db_path)
