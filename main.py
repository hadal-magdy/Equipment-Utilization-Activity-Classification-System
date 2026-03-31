"""
Consumer Service

Reads EquipmentEvent messages from Kafka and writes them to
TimescaleDB (PostgreSQL with the timescaledb extension).

The hypertable is partitioned on `recorded_at` (the video timestamp
mapped to wall-clock time), enabling efficient time-range queries from
the UI.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone

import psycopg2
from psycopg2.extras import execute_values
from confluent_kafka import Consumer, KafkaError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("consumer_service")


# ── Config from environment ──────────────────────────────────────────────────

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "kafka:9092")
KAFKA_GROUP     = os.getenv("KAFKA_GROUP",     "consumer-db-writer")
KAFKA_TOPIC     = os.getenv("KAFKA_TOPIC",     "equipment-events")

DB_DSN = (
    f"host={os.getenv('DB_HOST', 'timescaledb')} "
    f"port={os.getenv('DB_PORT', '5432')} "
    f"dbname={os.getenv('DB_NAME', 'equipment_db')} "
    f"user={os.getenv('DB_USER', 'postgres')} "
    f"password={os.getenv('DB_PASS', 'postgres')}"
)

BATCH_SIZE    = int(os.getenv("BATCH_SIZE",    "20"))
COMMIT_EVERY  = int(os.getenv("COMMIT_EVERY",  "5"))   # seconds


# ── DB helpers ───────────────────────────────────────────────────────────────

INSERT_SQL = """
INSERT INTO equipment_events (
    recorded_at, frame_id, equipment_id, equipment_class,
    video_timestamp, current_state, current_activity, motion_source,
    total_tracked_seconds, total_active_seconds, total_idle_seconds,
    utilization_percent, bbox_x1, bbox_y1, bbox_x2, bbox_y2
) VALUES %s
ON CONFLICT DO NOTHING;
"""


def _connect_db(retries: int = 20) -> psycopg2.extensions.connection:
    for attempt in range(1, retries + 1):
        try:
            conn = psycopg2.connect(DB_DSN)
            conn.autocommit = False
            logger.info("Connected to TimescaleDB.")
            return conn
        except psycopg2.OperationalError as e:
            logger.warning(f"DB connect attempt {attempt}/{retries}: {e}")
            time.sleep(3)
    raise RuntimeError("Could not connect to TimescaleDB.")


def _build_row(msg: dict) -> tuple:
    ta   = msg.get("time_analytics", {})
    util = msg.get("utilization", {})
    bbox = msg.get("bbox", [0, 0, 0, 0])
    return (
        datetime.now(timezone.utc),
        msg["frame_id"],
        msg["equipment_id"],
        msg["equipment_class"],
        msg["timestamp"],
        util.get("current_state", "INACTIVE"),
        util.get("current_activity", "WAITING"),
        util.get("motion_source", "none"),
        ta.get("total_tracked_seconds", 0.0),
        ta.get("total_active_seconds", 0.0),
        ta.get("total_idle_seconds", 0.0),
        ta.get("utilization_percent", 0.0),
        bbox[0] if len(bbox) > 0 else 0,
        bbox[1] if len(bbox) > 1 else 0,
        bbox[2] if len(bbox) > 2 else 0,
        bbox[3] if len(bbox) > 3 else 0,
    )


# ── Main loop ────────────────────────────────────────────────────────────────

running = True

def _shutdown(signum, frame):
    global running
    logger.info("Shutdown signal received.")
    running = False

signal.signal(signal.SIGTERM, _shutdown)
signal.signal(signal.SIGINT,  _shutdown)


def main():
    conn = _connect_db()
    cur  = conn.cursor()

    consumer = Consumer({
        "bootstrap.servers":  KAFKA_BOOTSTRAP,
        "group.id":           KAFKA_GROUP,
        "auto.offset.reset":  "earliest",
        "enable.auto.commit": False,
    })
    consumer.subscribe([KAFKA_TOPIC])
    logger.info(f"Subscribed to topic: {KAFKA_TOPIC}")

    batch: list[tuple] = []
    last_commit = time.monotonic()

    while running:
        msg = consumer.poll(timeout=1.0)

        if msg is None:
            # Nothing new — still flush pending batch on timeout
            pass
        elif msg.error():
            if msg.error().code() != KafkaError._PARTITION_EOF:
                logger.error(f"Kafka error: {msg.error()}")
        else:
            try:
                payload = json.loads(msg.value().decode("utf-8"))
                batch.append(_build_row(payload))
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Malformed message skipped: {e}")

        # Write batch when size threshold or time threshold is reached
        now = time.monotonic()
        if batch and (len(batch) >= BATCH_SIZE or now - last_commit >= COMMIT_EVERY):
            try:
                execute_values(cur, INSERT_SQL, batch)
                conn.commit()
                consumer.commit(asynchronous=False)
                logger.debug(f"Wrote {len(batch)} rows to TimescaleDB.")
                batch.clear()
                last_commit = now
            except Exception as e:
                logger.error(f"DB write error: {e}")
                conn.rollback()

    # Flush remaining
    if batch:
        try:
            execute_values(cur, INSERT_SQL, batch)
            conn.commit()
        except Exception as e:
            logger.error(f"Final flush error: {e}")

    consumer.close()
    cur.close()
    conn.close()
    logger.info("Consumer service exited cleanly.")


if __name__ == "__main__":
    main()
