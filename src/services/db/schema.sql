-- Database Schema using SQLite

-- 0. Users Table (v4.0 Auth)
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    api_key TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 1. Materials Table (Theoretical Data)
CREATE TABLE IF NOT EXISTS materials (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,                     -- v4.0 Owner (Nullable for legacy)
    material_id TEXT UNIQUE NOT NULL,    -- MP ID (e.g. mp-1234)
    formula TEXT NOT NULL,
    formation_energy REAL,
    cif_path TEXT,                       -- Path to local CIF file
    dos_data TEXT,                       -- JSON string of DOS descriptors
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(user_id) REFERENCES users(id)
);

-- 2. Experiments Table (Experimental Data)
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,                     -- v4.0 Owner
    name TEXT NOT NULL,
    type TEXT NOT NULL,                  -- LSV, CV, EIS, etc.
    raw_data_path TEXT,                  -- Path to original file
    parsed_data_path TEXT,               -- Path to processed CSV
    metadata TEXT,                       -- JSON string of extra params
    results TEXT,                        -- JSON string of analysis results
    material_id TEXT,                    -- Link to materials.material_id
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(user_id) REFERENCES users(id)
);

-- 3. Models Table (Training History)
CREATE TABLE IF NOT EXISTS models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,                     -- v4.0 Owner
    name TEXT NOT NULL,
    type TEXT NOT NULL,                  -- Traditional, DNN, GNN
    target TEXT NOT NULL,                -- e.g. formation_energy
    metrics TEXT,                        -- JSON: {r2_test: 0.95, mae: 0.1}
    params TEXT,                         -- JSON: Hyperparameters
    filepath TEXT,                       -- Path to saved model file
    dataset_hash TEXT,                   -- Training data snapshot hash (M3)
    hyperparameters TEXT,                -- JSON: Hyperparameters (M4)
    feature_cols TEXT,                   -- JSON: Feature columns used (M4)
    training_size INTEGER,               -- Training set size (M4)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(user_id) REFERENCES users(id)
);

-- 3.5 Plans Table (Task Graph)
CREATE TABLE IF NOT EXISTS plans (
    id TEXT PRIMARY KEY,                 -- Plan ID (string/UUID)
    user_id INTEGER,
    task_type TEXT NOT NULL,
    status TEXT NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(user_id) REFERENCES users(id)
);

-- 3.6 Plan Steps Table (Task Graph Steps)
CREATE TABLE IF NOT EXISTS plan_steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plan_id TEXT NOT NULL,
    step_id TEXT NOT NULL,
    agent TEXT NOT NULL,
    action TEXT NOT NULL,
    params TEXT,                         -- JSON params for step
    dependencies TEXT,                   -- JSON list of step_ids
    status TEXT NOT NULL,
    result TEXT,                         -- JSON
    error TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(plan_id) REFERENCES plans(id)
);

-- 3.7 Evidence Table (M3 Evidence Chain)
CREATE TABLE IF NOT EXISTS evidence (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    material_id TEXT NOT NULL,           -- materials.material_id
    source_type TEXT NOT NULL,           -- experiment / literature / theory / ml_prediction / adsorption_energy
    source_id TEXT,
    score REAL,
    metadata TEXT,                       -- JSON
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(material_id) REFERENCES materials(material_id)
);

-- 3.8 Adsorption Energies (Catalysis-Hub proxy for activity)
CREATE TABLE IF NOT EXISTS adsorption_energies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    material_id TEXT,                    -- Nullable if only surface composition is known
    surface_composition TEXT,
    facet TEXT,
    adsorbate TEXT,                      -- H* / OH*
    reaction_energy REAL,
    activation_energy REAL,
    source TEXT,
    metadata TEXT,                       -- JSON
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(material_id) REFERENCES materials(material_id)
);

-- 3.8.1 Activity Metrics (HOR/HER indicators)
CREATE TABLE IF NOT EXISTS activity_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    material_id TEXT,                    -- Nullable if not linked to a known material
    metric_name TEXT NOT NULL,           -- e.g., overpotential_10mA, exchange_current_density
    metric_value REAL,
    unit TEXT,
    conditions TEXT,                     -- JSON string of test conditions
    source TEXT,                         -- experiment / literature / database
    source_id TEXT,                      -- e.g., experiment id or DOI
    metadata TEXT,                       -- JSON
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(material_id) REFERENCES materials(material_id)
);

-- 3.9 Knowledge Entities (Knowledge Core)
CREATE TABLE IF NOT EXISTS knowledge_entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_type TEXT NOT NULL,           -- material / property / adsorbate / reaction / dataset / model / paper
    name TEXT NOT NULL,
    canonical_id TEXT,                   -- optional external ID (e.g., mp-1234, DOI)
    metadata TEXT,                       -- JSON
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(entity_type, name)
);

-- 3.10 Knowledge Sources (Evidence)
CREATE TABLE IF NOT EXISTS knowledge_sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_type TEXT NOT NULL,           -- literature / experiment / theory / ml_prediction / dataset
    source_id TEXT,
    title TEXT,
    url TEXT,
    year INTEGER,
    metadata TEXT,                       -- JSON
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 3.11 Knowledge Relations (Triples)
CREATE TABLE IF NOT EXISTS knowledge_relations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_id INTEGER NOT NULL,
    predicate TEXT NOT NULL,
    object_id INTEGER NOT NULL,
    confidence REAL,
    evidence_source_id INTEGER,          -- optional main evidence link
    metadata TEXT,                       -- JSON
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(subject_id) REFERENCES knowledge_entities(id),
    FOREIGN KEY(object_id) REFERENCES knowledge_entities(id),
    FOREIGN KEY(evidence_source_id) REFERENCES knowledge_sources(id)
);

-- 3.12 Relation Evidence (many-to-many)
CREATE TABLE IF NOT EXISTS knowledge_relation_evidence (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    relation_id INTEGER NOT NULL,
    source_id INTEGER NOT NULL,
    score REAL,
    metadata TEXT,                       -- JSON
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(relation_id) REFERENCES knowledge_relations(id),
    FOREIGN KEY(source_id) REFERENCES knowledge_sources(id)
);

-- 3.13 Dataset Snapshots (Reproducibility)
CREATE TABLE IF NOT EXISTS dataset_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plan_id TEXT,
    name TEXT,
    description TEXT,
    metadata TEXT,                       -- JSON
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(plan_id) REFERENCES plans(id)
);

-- 3.14 Snapshot Items (materials / models / papers)
CREATE TABLE IF NOT EXISTS dataset_snapshot_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id INTEGER NOT NULL,
    item_type TEXT NOT NULL,             -- material / model / paper
    item_id TEXT NOT NULL,
    metadata TEXT,                       -- JSON
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(snapshot_id) REFERENCES dataset_snapshots(id)
);

-- 4. Chat Sessions (v4.0 History)
CREATE TABLE IF NOT EXISTS chat_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    title TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(user_id) REFERENCES users(id)
);

-- 5. Chat Messages (v4.0 History)
CREATE TABLE IF NOT EXISTS chat_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    role TEXT NOT NULL,                  -- 'user' or 'assistant'
    content TEXT NOT NULL,
    artifacts TEXT,                      -- JSON: Linked Plan or Files
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(session_id) REFERENCES chat_sessions(id)
);
