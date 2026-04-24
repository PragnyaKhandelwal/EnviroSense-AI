-- Pragnya Data Quality + Trust Layer deliverables
-- Safe to run multiple times.

CREATE TABLE IF NOT EXISTS pragnya_quality_metrics (
    id BIGSERIAL PRIMARY KEY,
    window_end TIMESTAMPTZ NOT NULL,
    window_minutes INTEGER NOT NULL,
    device_id TEXT NOT NULL,
    expected_points INTEGER NOT NULL,
    actual_points INTEGER NOT NULL,
    valid_points INTEGER NOT NULL,
    completeness_pct NUMERIC(6,2) NOT NULL,
    uptime_pct NUMERIC(6,2) NOT NULL,
    valid_pct NUMERIC(6,2) NOT NULL,
    source_tag TEXT NOT NULL DEFAULT 'pragnya_analytics',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS pragnya_drift_metrics (
    id BIGSERIAL PRIMARY KEY,
    measured_at TIMESTAMPTZ NOT NULL,
    device_id TEXT NOT NULL,
    ratio_name TEXT NOT NULL,
    ratio_value DOUBLE PRECISION NOT NULL,
    baseline_mean DOUBLE PRECISION NOT NULL,
    baseline_std DOUBLE PRECISION NOT NULL,
    z_score DOUBLE PRECISION NOT NULL,
    drift_alert BOOLEAN NOT NULL,
    source_tag TEXT NOT NULL DEFAULT 'pragnya_analytics',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE model_features
    ADD COLUMN IF NOT EXISTS quality_uptime_pct DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS quality_valid_pct DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS quality_completeness_pct DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS quality_expected_points INTEGER,
    ADD COLUMN IF NOT EXISTS quality_actual_points INTEGER,
    ADD COLUMN IF NOT EXISTS quality_valid_points INTEGER,
    ADD COLUMN IF NOT EXISTS drift_ratio_name TEXT,
    ADD COLUMN IF NOT EXISTS drift_ratio_value DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS drift_baseline_mean DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS drift_baseline_std DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS drift_z_score DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS drift_alert BOOLEAN,
    ADD COLUMN IF NOT EXISTS reliability_score DOUBLE PRECISION;

CREATE UNIQUE INDEX IF NOT EXISTS idx_pragnya_quality_device_window
    ON pragnya_quality_metrics (device_id, window_end);

CREATE INDEX IF NOT EXISTS idx_pragnya_quality_window
    ON pragnya_quality_metrics (window_end);

CREATE INDEX IF NOT EXISTS idx_pragnya_drift_device_time_ratio
    ON pragnya_drift_metrics (device_id, measured_at, ratio_name);

CREATE INDEX IF NOT EXISTS idx_pragnya_drift_time_device
    ON pragnya_drift_metrics (measured_at, device_id);

CREATE UNIQUE INDEX IF NOT EXISTS idx_model_features_device_time
    ON model_features (device_id, time);

CREATE INDEX IF NOT EXISTS idx_model_features_time
    ON model_features (time);

CREATE OR REPLACE VIEW reliability_score_view AS
SELECT
    m.time,
    m.device_id,
    m.quality_completeness_pct,
    m.quality_valid_pct,
    m.quality_uptime_pct,
    m.drift_z_score,
    m.drift_alert,
    ROUND(
        (
            0.35 * COALESCE(m.quality_valid_pct, 0) / 100.0 +
            0.30 * COALESCE(m.quality_uptime_pct, 0) / 100.0 +
            0.25 * COALESCE(m.quality_completeness_pct, 0) / 100.0 +
            0.10 * GREATEST(0.0, LEAST(1.0, 1.0 - (ABS(COALESCE(m.drift_z_score, 0)) / 5.0)))
        )::numeric,
        4
    ) AS reliability_score
FROM model_features m;

