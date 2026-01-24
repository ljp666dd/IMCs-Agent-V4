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
    source_type TEXT NOT NULL,           -- experiment / literature / theory / ml_prediction
    source_id TEXT,
    score REAL,
    metadata TEXT,                       -- JSON
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(material_id) REFERENCES materials(material_id)
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
