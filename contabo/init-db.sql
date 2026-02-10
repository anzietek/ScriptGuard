-- ScriptGuard PostgreSQL Initialization Script

-- IMPORTANT: This script is automatically executed in the database defined
-- in docker-compose as POSTGRES_DB (i.e., 'scriptguard').

-- 1. Enable extensions (requires superuser privileges)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- 2. Table: samples
CREATE TABLE IF NOT EXISTS samples (
    id SERIAL PRIMARY KEY,
    content_hash VARCHAR(64) UNIQUE NOT NULL,
    content TEXT NOT NULL,
    label VARCHAR(20) NOT NULL CHECK (label IN ('malicious', 'benign')),
    source VARCHAR(100) NOT NULL,
    url TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for samples
CREATE INDEX IF NOT EXISTS idx_samples_content_hash ON samples(content_hash);
CREATE INDEX IF NOT EXISTS idx_samples_label ON samples(label);
CREATE INDEX IF NOT EXISTS idx_samples_source ON samples(source);
CREATE INDEX IF NOT EXISTS idx_samples_created_at ON samples(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_samples_metadata_gin ON samples USING GIN(metadata);
-- Full-text search index
CREATE INDEX IF NOT EXISTS idx_samples_content_trgm ON samples USING GIN(content gin_trgm_ops);

-- 3. Table: dataset_versions
CREATE TABLE IF NOT EXISTS dataset_versions (
    id SERIAL PRIMARY KEY,
    version VARCHAR(50) UNIQUE NOT NULL,
    total_samples INTEGER NOT NULL,
    malicious_count INTEGER NOT NULL,
    benign_count INTEGER NOT NULL,
    sources JSONB DEFAULT '{}'::jsonb,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_versions_version ON dataset_versions(version);
CREATE INDEX IF NOT EXISTS idx_versions_created_at ON dataset_versions(created_at DESC);

-- 4. Function and Trigger for updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

DROP TRIGGER IF EXISTS update_samples_updated_at ON samples;
CREATE TRIGGER update_samples_updated_at
    BEFORE UPDATE ON samples
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- 5. Materialized View (Statistics)
CREATE MATERIALIZED VIEW IF NOT EXISTS sample_statistics AS
SELECT
    COUNT(*) as total_count,
    COUNT(*) FILTER (WHERE label = 'malicious') as malicious_count,
    COUNT(*) FILTER (WHERE label = 'benign') as benign_count,
    COUNT(DISTINCT source) as source_count,
    AVG(LENGTH(content)) as avg_content_length,
    MIN(created_at) as first_sample_date,
    MAX(created_at) as last_sample_date
FROM samples
WITH DATA;

CREATE UNIQUE INDEX IF NOT EXISTS idx_sample_statistics ON sample_statistics ((1));

-- Function to refresh statistics
CREATE OR REPLACE FUNCTION refresh_sample_statistics()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY sample_statistics;
END;
$$ LANGUAGE plpgsql;

-- Optional: Initial data (uncomment if needed)
-- INSERT INTO samples (content_hash, content, label, source, url, metadata)
-- VALUES ('test_hash_123', 'print("Hello World")', 'benign', 'manual', NULL, '{}'::jsonb)
-- ON CONFLICT (content_hash) DO NOTHING;

-- Log success message
\echo 'ScriptGuard database initialized successfully!'

