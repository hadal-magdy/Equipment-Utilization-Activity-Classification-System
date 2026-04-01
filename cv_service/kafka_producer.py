"""
Kafka Producer

Thin wrapper around confluent-kafka that serialises EquipmentEvent
dataclasses to JSON and publishes them to a topic.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict
from typing import Optional

from confluent_kafka import Producer, KafkaException

logger = logging.getLogger(__name__)

TOPIC = "equipment-events"


@dataclass
class EquipmentEvent:
    frame_id: int
    equipment_id: str
    equipment_class: str
    timestamp: str                   # "HH:MM:SS.mmm"
    utilization: dict                # current_state, current_activity, motion_source
    time_analytics: dict             # total_tracked, active, idle seconds + utilization%
    bbox: list                       # [x1, y1, x2, y2]


class EquipmentKafkaProducer:
    def __init__(self, bootstrap_servers: str = "kafka:9092", retries: int = 10):
        self._bootstrap = bootstrap_servers
        self._producer: Optional[Producer] = None
        self._connect(retries)

    def _connect(self, retries: int):
        for attempt in range(1, retries + 1):
            try:
                self._producer = Producer({"bootstrap.servers": self._bootstrap,
                                           "queue.buffering.max.ms": 50})
                logger.info(f"Kafka producer connected to {self._bootstrap}")
                return
            except KafkaException as e:
                logger.warning(f"Kafka connect attempt {attempt}/{retries} failed: {e}")
                time.sleep(3)
        raise RuntimeError(f"Could not connect to Kafka at {self._bootstrap}")

    def _delivery_report(self, err, msg):
        if err:
            logger.warning(f"Kafka delivery failed: {err}")

    def send(self, event: EquipmentEvent):
        if self._producer is None:
            logger.error("Producer not initialised.")
            return
        payload = json.dumps(asdict(event)).encode("utf-8")
        self._producer.produce(
            TOPIC,
            key=event.equipment_id.encode("utf-8"),
            value=payload,
            callback=self._delivery_report,
        )
        self._producer.poll(0)   # non-blocking flush trigger

    def flush(self):
        if self._producer:
            self._producer.flush()
