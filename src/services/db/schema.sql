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
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(user_id) REFERENCES users(id)
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
