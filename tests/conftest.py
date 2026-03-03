import pytest

from src.services.db.database import DatabaseService


@pytest.fixture
def db():
    """
    Provide an isolated sqlite database for integration-style tests.

    Notes:
    - We intentionally avoid using the default `data/imcs.db` to keep tests
      deterministic and prevent polluting local user data.
    - We avoid pytest's `tmp_path` on this Windows environment due to sporadic
      WinError 5 PermissionError issues when pytest creates temp dirs with
      restrictive permissions.
    """
    import uuid
    from pathlib import Path

    artifacts_dir = Path("tests") / ".artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    db_path = artifacts_dir / f"imcs_test_{uuid.uuid4().hex}.db"
    service = DatabaseService(db_path=str(db_path))
    yield service
    try:
        db_path.unlink(missing_ok=True)
    except Exception:
        pass
