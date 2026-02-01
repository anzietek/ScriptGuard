-- ScriptGuard PostgreSQL Initialization Script
-- This script is automatically executed when PostgreSQL container starts

-- Create database if not exists (usually done by POSTGRES_DB env var)
-- SELECT 'CREATE DATABASE scriptguard' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'scriptguard')\gexec

-- Connect to scriptguard database
\c scriptguard

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For text search

-- Create samples table
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

-- Create indexes for samples
CREATE INDEX IF NOT EXISTS idx_samples_content_hash ON samples(content_hash);
CREATE INDEX IF NOT EXISTS idx_samples_label ON samples(label);
CREATE INDEX IF NOT EXISTS idx_samples_source ON samples(source);
CREATE INDEX IF NOT EXISTS idx_samples_created_at ON samples(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_samples_metadata_gin ON samples USING GIN(metadata);

-- Full-text search index on content
CREATE INDEX IF NOT EXISTS idx_samples_content_trgm ON samples USING GIN(content gin_trgm_ops);

-- Create dataset_versions table
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

-- Create indexes for dataset_versions
CREATE INDEX IF NOT EXISTS idx_versions_version ON dataset_versions(version);
CREATE INDEX IF NOT EXISTS idx_versions_created_at ON dataset_versions(created_at DESC);

-- Create trigger function for updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger
DROP TRIGGER IF EXISTS update_samples_updated_at ON samples;
CREATE TRIGGER update_samples_updated_at
    BEFORE UPDATE ON samples
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Create materialized view for statistics
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

-- Create unique index for materialized view
CREATE UNIQUE INDEX IF NOT EXISTS idx_sample_statistics ON sample_statistics ((1));

-- Create function to refresh statistics
CREATE OR REPLACE FUNCTION refresh_sample_statistics()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY sample_statistics;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO scriptguard;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO scriptguard;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO scriptguard;

-- Insert initial data (optional)
-- INSERT INTO samples (content_hash, content, label, source, url, metadata)
-- VALUES ('example_hash', 'print("Hello World")', 'benign', 'manual', NULL, '{}'::jsonb)
-- ON CONFLICT (content_hash) DO NOTHING;

-- Print success message
\echo 'ScriptGuard database initialized successfully!'
