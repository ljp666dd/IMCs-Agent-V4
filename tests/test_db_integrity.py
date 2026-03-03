import sqlite3

import pytest


def test_foreign_keys_prevent_orphan_evidence(db):
    # No materials exist yet; inserting evidence should fail with FK enforcement.
    with pytest.raises(sqlite3.IntegrityError):
        db.save_evidence(
            material_id="mp-does-not-exist",
            source_type="theory",
            source_id="x",
            score=1.0,
            metadata={"note": "orphan should be rejected"},
        )

    # Once material exists, evidence insert should succeed.
    db.save_material(material_id="mp-1", formula="Pt", energy=-1.0, cif_path=None)
    evid_id = db.save_evidence(
        material_id="mp-1",
        source_type="theory",
        source_id="mp-1",
        score=1.0,
        metadata={"formation_energy": -1.0},
    )
    assert evid_id and evid_id > 0

