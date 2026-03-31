-- Initialize TimescaleDB for Equipment Utilization Tracking
-- This script runs automatically via Docker's init-scripts mechanism.

CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Main events table
CREATE TABLE IF NOT EXISTS equipment_events (
    recorded_at              TIMESTAMPTZ     NOT NULL,
    frame_id                 BIGINT          NOT NULL,
    equipment_id             TEXT            NOT NULL,
    equipment_class          TEXT            NOT NULL,
    video_timestamp          TEXT,                          -- "HH:MM:SS.mmm" from video
    current_state            TEXT            NOT NULL,      -- ACTIVE | INACTIVE
    current_activity         TEXT            NOT NULL,      -- DIGGING | SWINGING_LOADING | DUMPING | WAITING
    motion_source            TEXT,                          -- full_body | arm_only | tracks_only | none
    total_tracked_seconds    DOUBLE PRECISION DEFAULT 0,
    total_active_seconds     DOUBLE PRECISION DEFAULT 0,
    total_idle_seconds       DOUBLE PRECISION DEFAULT 0,
    utilization_percent      DOUBLE PRECISION DEFAULT 0,
    bbox_x1                  INT,
    bbox_y1                  INT,
    bbox_x2                  INT,
    bbox_y2                  INT
);

-- Convert to hypertable (TimescaleDB magic — partitions by recorded_at)
SELECT create_hypertable('equipment_events', 'recorded_at', if_not_exists => TRUE);

-- Index for fast per-machine queries
CREATE INDEX IF NOT EXISTS idx_equipment_id_time
    ON equipment_events (equipment_id, recorded_at DESC);

-- Continuous aggregate: 1-minute utilization rollup (for dashboard charts)
CREATE MATERIALIZED VIEW IF NOT EXISTS equipment_utilization_1min
WITH (timescaledb.continuous) AS
    SELECT
        time_bucket('1 minute', recorded_at) AS bucket,
        equipment_id,
        AVG(utilization_percent)             AS avg_utilization,
        MODE() WITHIN GROUP (ORDER BY current_state)    AS dominant_state,
        MODE() WITHIN GROUP (ORDER BY current_activity) AS dominant_activity,
        COUNT(*)                             AS sample_count
    FROM equipment_events
    GROUP BY bucket, equipment_id
WITH NO DATA;

-- Keep raw events for 7 days, auto-drop older data
SELECT add_retention_policy('equipment_events', INTERVAL '7 days', if_not_exists => TRUE);
