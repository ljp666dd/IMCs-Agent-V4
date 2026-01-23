-- Database Schema using SQLite

-- 1. Materials Table (Theoretical Data)
CREATE TABLE IF NOT EXISTS materials (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    material_id TEXT UNIQUE NOT NULL,    -- MP ID (e.g. mp-1234)
    formula TEXT NOT NULL,
    formation_energy REAL,
    cif_path TEXT,                       -- Path to local CIF file
    dos_data TEXT,                       -- JSON string of DOS descriptors
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Experiments Table (Experimental Data)
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    type TEXT NOT NULL,                  -- LSV, CV, EIS, etc.
    raw_data_path TEXT,                  -- Path to original file
    parsed_data_path TEXT,               -- Path to processed CSV
    metadata TEXT,                       -- JSON string of extra params
    results TEXT,                        -- JSON string of analysis results
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 3. Models Table (Training History)
CREATE TABLE IF NOT EXISTS models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    type TEXT NOT NULL,                  -- Traditional, DNN, GNN
    target TEXT NOT NULL,                -- e.g. formation_energy
    metrics TEXT,                        -- JSON: {r2_test: 0.95, mae: 0.1}
    params TEXT,                         -- JSON: Hyperparameters
    filepath TEXT,                       -- Path to saved model file
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
