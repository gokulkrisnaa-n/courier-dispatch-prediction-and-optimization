"""Pluggable event bus so the streaming code never hard-depends on a broker.

``InMemoryBus`` is a process-local queue used by unit tests and quick local runs
(no Kafka required). ``KafkaBus`` wraps kafka-python and talks to a real broker
(Redpanda in docker-compose). Both honor the same tiny interface, so the producer
and dispatcher are written once and swapped via ``stream.bus = memory | kafka``.

Events are plain JSON-serializable dicts (see producer.py for the schema).
"""
from __future__ import annotations

import json
import queue
from abc import ABC, abstractmethod
from typing import Any

from ..config import Config, load_config

Event = dict[str, Any]
# Sentinel a producer publishes to signal "no more events" to in-memory consumers.
_DONE = {"event_type": "__done__"}


class Bus(ABC):
    @abstractmethod
    def publish(self, event: Event) -> None: ...

    @abstractmethod
    def poll(self, timeout: float = 1.0) -> Event | None:
        """Return the next event, or None if none arrived within ``timeout``."""

    def close(self) -> None:  # optional override
        pass

    def done(self) -> None:
        """Signal end-of-stream (in-memory). No-op for brokers."""

    def is_done(self) -> bool:
        """True once the stream has been explicitly ended. Brokers never end."""
        return False


class InMemoryBus(Bus):
    def __init__(self) -> None:
        self._q: "queue.Queue[Event]" = queue.Queue()
        self._done = False

    def publish(self, event: Event) -> None:
        self._q.put(event)

    def poll(self, timeout: float = 1.0) -> Event | None:
        try:
            event = self._q.get(timeout=timeout)
        except queue.Empty:
            return None
        if event.get("event_type") == "__done__":
            self._done = True
            return None
        return event

    def done(self) -> None:
        self._q.put(_DONE)

    def is_done(self) -> bool:
        return self._done


class KafkaBus(Bus):
    """kafka-python wrapper. Imported lazily so the dep isn't needed for tests."""

    def __init__(self, bootstrap: str, topic: str, *, group_id: str = "dispatch") -> None:
        from kafka import KafkaConsumer, KafkaProducer  # lazy

        self.topic = topic
        self._producer = KafkaProducer(
            bootstrap_servers=bootstrap,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )
        self._consumer = KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap,
            group_id=group_id,
            auto_offset_reset="earliest",
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            consumer_timeout_ms=1000,
        )

    def publish(self, event: Event) -> None:
        self._producer.send(self.topic, event)

    def poll(self, timeout: float = 1.0) -> Event | None:
        batch = self._consumer.poll(timeout_ms=int(timeout * 1000), max_records=1)
        for _tp, records in batch.items():
            if records:
                return records[0].value
        return None

    def close(self) -> None:
        self._producer.flush()
        self._producer.close()
        self._consumer.close()


def make_bus(cfg: Config | None = None, *, kind: str | None = None) -> Bus:
    """Build the bus named by ``stream.bus`` in config (or the ``kind`` override)."""
    cfg = cfg or load_config()
    scfg = cfg.stream
    kind = kind or scfg.get("bus", "memory")
    if kind == "memory":
        return InMemoryBus()
    if kind == "kafka":
        return KafkaBus(scfg.get("kafka_bootstrap", "localhost:9092"),
                        scfg.get("topic", "pickup_events"))
    raise ValueError(f"unknown bus kind: {kind!r}")
