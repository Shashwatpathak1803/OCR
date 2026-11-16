"""Persistent, disk-backed queue for offline visiting-card processing."""

import hashlib
import json
import os
import platform
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple


def _default_data_dir() -> Path:
    configured = os.environ.get("OCR_DATA_DIR")
    if configured:
        return Path(configured).expanduser()
    if platform.system() == "Windows":
        return Path(os.environ.get("LOCALAPPDATA", Path.home())) / "VisitingCardOCR"
    return Path.home() / ".local" / "share" / "visiting-card-ocr"


class ProcessingQueue:
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else _default_data_dir()
        self.image_dir = self.data_dir / "images"
        self.database_path = self.data_dir / "queue.sqlite3"
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        connection = sqlite3.connect(self.database_path, timeout=30)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA busy_timeout = 30000")
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS cards (
                    id TEXT PRIMARY KEY,
                    image_hash TEXT NOT NULL UNIQUE,
                    filename TEXT NOT NULL,
                    image_path TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'Pending',
                    raw_text TEXT NOT NULL DEFAULT '',
                    record_json TEXT NOT NULL DEFAULT '',
                    timings_json TEXT NOT NULL DEFAULT '{}',
                    confidence REAL NOT NULL DEFAULT 0,
                    error TEXT NOT NULL DEFAULT '',
                    attempts INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS app_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "INSERT OR IGNORE INTO app_state (key, value) VALUES ('paused', '1')"
            )

    def add_image(self, filename: str, image_bytes: bytes) -> Tuple[Dict[str, Any], bool]:
        image_hash = hashlib.sha256(image_bytes).hexdigest()
        extension = Path(filename).suffix.lower()
        if extension not in {".jpg", ".jpeg", ".png", ".webp"}:
            extension = ".img"
        safe_name = Path(filename).name or f"{image_hash}{extension}"

        with self._connect() as connection:
            existing = connection.execute(
                "SELECT * FROM cards WHERE image_hash = ?", (image_hash,)
            ).fetchone()
            if existing:
                return dict(existing), False

            image_path = self.image_dir / f"{image_hash}{extension}"
            if not image_path.exists():
                temporary_path = image_path.with_name(
                    f"{image_path.name}.{uuid.uuid4().hex}.tmp"
                )
                temporary_path.write_bytes(image_bytes)
                temporary_path.replace(image_path)

            card_id = uuid.uuid4().hex
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO cards (id, image_hash, filename, image_path)
                VALUES (?, ?, ?, ?)
                """,
                (card_id, image_hash, safe_name, str(image_path)),
            )
            if cursor.rowcount == 0:
                row = connection.execute(
                    "SELECT * FROM cards WHERE image_hash = ?", (image_hash,)
                ).fetchone()
                if row is None:
                    raise RuntimeError("Image queue insert conflicted without a matching row.")
                return dict(row), False
            row = connection.execute(
                "SELECT * FROM cards WHERE id = ?", (card_id,)
            ).fetchone()
            if row is None:
                raise RuntimeError("Image queue insert did not create a card row.")
            return dict(row), True

    def list_cards(self) -> List[Dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM cards ORDER BY created_at, id"
            ).fetchall()
        return [dict(row) for row in rows]

    def get_card(self, card_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM cards WHERE id = ?", (card_id,)
            ).fetchone()
        return dict(row) if row else None

    def get_image_path(self, card_id: str) -> Optional[Path]:
        row = self.get_card(card_id)
        return Path(row["image_path"]) if row and row["image_path"] else None

    def get_completed_records(self) -> List[Dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, record_json FROM cards
                WHERE record_json != ''
                ORDER BY created_at, id
                """
            ).fetchall()
        records = []
        for row in rows:
            record = json.loads(row["record_json"])
            record["_queue_id"] = row["id"]
            records.append(record)
        return records

    def next_serial_number(self) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM cards WHERE record_json != ''"
            ).fetchone()
        return int(row["count"]) + 1

    def save_record(self, card_id: str, record: Dict[str, Any]) -> None:
        record_data = dict(record)
        record_data.pop("_queue_id", None)
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE cards
                SET record_json = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ? AND record_json != ''
                """,
                (json.dumps(record_data, ensure_ascii=False), card_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"No completed queue record exists for card {card_id}.")

    def claim_next(self) -> Optional[Dict[str, Any]]:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM cards WHERE status = 'Pending' ORDER BY created_at, id LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                """
                UPDATE cards
                SET status = 'Processing', attempts = attempts + 1,
                    error = '', updated_at = CURRENT_TIMESTAMP
                WHERE id = ? AND status = 'Pending'
                """,
                (row["id"],),
            )
            claimed = connection.execute(
                "SELECT * FROM cards WHERE id = ?", (row["id"],)
            ).fetchone()
            return dict(claimed) if claimed else None

    def complete(
        self,
        card_id: str,
        status: str,
        raw_text: str,
        record: Dict[str, Any],
        confidence: float,
        timings: Dict[str, Any],
    ) -> None:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE cards
                SET status = ?, raw_text = ?, record_json = ?, confidence = ?,
                    timings_json = ?, error = '', updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (
                    status,
                    raw_text,
                    json.dumps(record, ensure_ascii=False),
                    confidence,
                    json.dumps(timings, ensure_ascii=False),
                    card_id,
                ),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Queue card {card_id} no longer exists.")

    def update_timings(self, card_id: str, timings: Dict[str, Any]) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE cards SET timings_json = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (json.dumps(timings, ensure_ascii=False), card_id),
            )

    def fail(self, card_id: str, error: str, timings: Dict[str, Any]) -> None:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE cards
                SET status = 'Failed', error = ?, timings_json = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (error, json.dumps(timings, ensure_ascii=False), card_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Queue card {card_id} no longer exists.")

    def summary(self) -> Dict[str, Any]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT status, timings_json FROM cards"
            ).fetchall()
            state = connection.execute(
                "SELECT value FROM app_state WHERE key = 'paused'"
            ).fetchone()
        counts: Dict[str, int] = {}
        stage_values: Dict[str, List[float]] = {}
        for row in rows:
            counts[row["status"]] = counts.get(row["status"], 0) + 1
            timings = json.loads(row["timings_json"] or "{}")
            for name in (
                "total_card_s",
                "preprocessing_s",
                "detection_s",
                "recognition_s",
                "ocr_s",
                "ocr_model_initialization_s",
                "data_extraction_s",
                "database_saving_s",
            ):
                value = timings.get(name)
                if isinstance(value, (int, float)):
                    stage_values.setdefault(name, []).append(float(value))
        total_time = sum(stage_values.get("total_card_s", []))
        completed_count = (
            counts.get("Completed", 0) + counts.get("Awaiting Verification", 0)
        )
        return {
            "counts": counts,
            "total": len(rows),
            "completed": completed_count,
            "total_processing_s": round(total_time, 3),
            "average_card_s": (
                round(total_time / len(stage_values["total_card_s"]), 3)
                if stage_values.get("total_card_s") else 0.0
            ),
            "stage_averages": {
                name: round(sum(values) / len(values), 3)
                for name, values in stage_values.items()
                if values
            },
            "paused": not state or state["value"] != "0",
        }

    def set_paused(self, paused: bool) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO app_state (key, value) VALUES ('paused', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                ("1" if paused else "0",),
            )

    def is_paused(self) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT value FROM app_state WHERE key = 'paused'"
            ).fetchone()
        return not row or row["value"] != "0"

    def retry_failed(self) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE cards SET status = 'Pending', error = '',
                    updated_at = CURRENT_TIMESTAMP
                WHERE status = 'Failed'
                """
            )
        return cursor.rowcount

    def remove_card(self, card_id: str) -> None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT image_path, record_json FROM cards WHERE id = ?", (card_id,)
            ).fetchone()
            if row is None:
                return
            if row["record_json"]:
                connection.execute(
                    """
                    UPDATE cards SET image_path = '', updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (card_id,),
                )
            else:
                connection.execute("DELETE FROM cards WHERE id = ?", (card_id,))
        image_path = Path(row["image_path"]) if row["image_path"] else None
        if image_path and image_path.exists():
            image_path.unlink()

    def clear(self) -> None:
        self.set_paused(True)
        with self._connect() as connection:
            paths = [
                Path(row["image_path"])
                for row in connection.execute("SELECT image_path FROM cards")
                if row["image_path"]
            ]
            connection.execute("DELETE FROM cards")
        for path in paths:
            if path.exists():
                path.unlink()

    def clear_queue(self) -> None:
        self.set_paused(True)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, image_path, record_json FROM cards"
            ).fetchall()
            paths = [Path(row["image_path"]) for row in rows if row["image_path"]]
            connection.execute("DELETE FROM cards WHERE record_json = ''")
            connection.execute(
                """
                UPDATE cards SET image_path = '', updated_at = CURRENT_TIMESTAMP
                WHERE record_json != ''
                """
            )
        for path in paths:
            if path.exists():
                path.unlink()

    def recover_interrupted(self) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE cards SET status = 'Pending', error = '',
                    updated_at = CURRENT_TIMESTAMP
                WHERE status = 'Processing'
                """
            )
        return cursor.rowcount


class QueueWorker:
    def __init__(self, queue: ProcessingQueue):
        self.queue = queue
        self.queue.recover_interrupted()
        self._thread: Optional[threading.Thread] = None
        self._options = {
            "auto_perspective": True,
            "auto_deskew": True,
            "enhance_contrast": True,
        }
        self._lock = threading.Lock()

    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def start(self, options: Optional[Dict[str, Any]] = None) -> None:
        with self._lock:
            if options:
                self._options = dict(options)
            self.queue.set_paused(False)
            if self._thread and self._thread.is_alive():
                return
            self._thread = threading.Thread(
                target=self._run,
                name="visiting-card-ocr-queue",
                daemon=True,
            )
            self._thread.start()

    def pause(self) -> None:
        self.queue.set_paused(True)

    def retry_failed(self, options: Optional[Dict[str, Any]] = None) -> int:
        count = self.queue.retry_failed()
        if count:
            self.start(options)
        return count

    def _run(self) -> None:
        while not self.queue.is_paused():
            card = self.queue.claim_next()
            if card is None:
                self.queue.set_paused(True)
                return
            self._process_card(card)

    def _process_card(self, card: Dict[str, Any]) -> None:
        from image_processing import preprocess_visiting_card
        from ocr_engine import extract_text_from_image, get_ocr_engine, get_engine_metrics
        from rule_extractor import extract_visiting_card_data

        timings: Dict[str, Any] = {}
        try:
            engine = get_ocr_engine(lang="hi")
            engine_metrics = get_engine_metrics()
            timings["ocr_model_initialization_s"] = engine_metrics.get(
                "initialization_s", 0.0
            )

            started = time.perf_counter()
            prep = preprocess_visiting_card(
                card["image_path"],
                **self._options,
            )
            timings.update(prep.get("timings", {}))
            timings["image_loading_s"] = prep.get("timings", {}).get(
                "image_loading_s", 0.0
            )
            timings["preprocessing_s"] = prep.get("timings", {}).get(
                "preprocessing_s", time.perf_counter() - started
            )

            ocr_started = time.perf_counter()
            ocr_result = extract_text_from_image(
                prep["processed_cv"], ocr_engine=engine
            )
            timings.update(ocr_result.get("timings", {}))
            if not timings.get("ocr_s"):
                timings["ocr_s"] = round(time.perf_counter() - ocr_started, 4)
            if not ocr_result["success"] and not ocr_result["raw_text"]:
                raise ValueError(
                    ocr_result.get("error_message")
                    or "OCR detected no text on the card."
                )

            started = time.perf_counter()
            record = extract_visiting_card_data(
                ocr_text=ocr_result["raw_text"],
                lines_info=ocr_result["lines"],
                image_filename=card["filename"],
                serial_number=self.queue.next_serial_number(),
            )
            timings["data_extraction_s"] = round(
                time.perf_counter() - started, 4
            )
            timings["total_card_s"] = round(
                sum(
                    float(timings.get(key) or 0.0)
                    for key in (
                        "image_loading_s",
                        "preprocessing_s",
                        "ocr_s",
                        "data_extraction_s",
                    )
                ),
                4,
            )
            status = (
                "Awaiting Verification"
                if record.verification_needed
                else "Completed"
            )
            save_started = time.perf_counter()
            self.queue.complete(
                card["id"],
                status,
                ocr_result["raw_text"],
                record.to_dict(),
                ocr_result["average_confidence"],
                timings,
            )
            timings["database_saving_s"] = round(
                time.perf_counter() - save_started, 4
            )
            timings["total_card_s"] = round(
                timings["total_card_s"] + timings["database_saving_s"], 4
            )
            self.queue.update_timings(card["id"], timings)
        except Exception as exc:
            self.queue.fail(card["id"], str(exc), timings)
