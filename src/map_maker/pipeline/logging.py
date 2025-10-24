"""Asynchronous logging utilities for pipeline runs."""

from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .models import StageResult, StageStats


class RunLogger:
    """Writes structured events to disk asynchronously."""

    def __init__(self, log_path: Path, summary_path: Optional[Path] = None) -> None:
        self._log_path = log_path
        self._summary_path = summary_path or log_path.with_suffix(".md")
        self._queue: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._stop = threading.Event()
        self._stage_records: list[dict[str, Any]] = []
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._thread.start()

    def _worker(self) -> None:
        with self._log_path.open("a", encoding="utf8") as fh:
            while not self._stop.is_set():
                try:
                    item = self._queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if item is None:
                    break
                json.dump(item, fh, sort_keys=True)
                fh.write("\n")
                fh.flush()

    def log_event(self, event: Dict[str, Any]) -> None:
        payload = {"timestamp": time.time(), **event}
        self._queue.put(payload)

    def log_stage_start(self, stage_name: str, cache_key: str) -> None:
        self.log_event({"type": "stage_start", "stage": stage_name, "cache_key": cache_key})

    def log_stage_end(self, stage_result: StageResult) -> None:
        stats = stage_result.stats
        payload = {
            "type": "stage_end",
            "stage": stage_result.stage_name,
            "cache_key": stage_result.cache_key,
            "artifacts": stage_result.artifact_checksums,
            "metadata": stage_result.metadata,
        }
        if stats:
            payload["stats"] = stats.to_dict()
            self._stage_records.append(
                {
                    "stage": stage_result.stage_name,
                    "duration_ns": stats.duration_ns,
                    "memory_bytes": stats.memory_bytes,
                    "cache_hit": stats.cache_hit,
                }
            )
        self.log_event(payload)

    def close(self) -> None:
        self._stop.set()
        self._queue.put(None)
        self._thread.join(timeout=2)
        self._write_summary()

    def _write_summary(self) -> None:
        if not self._stage_records:
            return
        total_duration = sum(record["duration_ns"] for record in self._stage_records)
        lines = ["# Pipeline Run Summary", "", f"- Total stages: {len(self._stage_records)}"]
        lines.append(f"- Total duration (ms): {total_duration / 1e6:.2f}")
        cache_hits = sum(1 for record in self._stage_records if record["cache_hit"])
        lines.append(f"- Cache hits: {cache_hits}")
        lines.append("")
        lines.append("| Stage | Duration (ms) | Memory (MB) | Cache Hit |")
        lines.append("| --- | ---: | ---: | --- |")
        for record in self._stage_records:
            duration_ms = record["duration_ns"] / 1e6
            memory_mb = record["memory_bytes"] / (1024 * 1024)
            cache = "yes" if record["cache_hit"] else "no"
            lines.append(f"| {record['stage']} | {duration_ms:.2f} | {memory_mb:.2f} | {cache} |")
        self._summary_path.parent.mkdir(parents=True, exist_ok=True)
        self._summary_path.write_text("\n".join(lines), encoding="utf8")


__all__ = ["RunLogger"]
