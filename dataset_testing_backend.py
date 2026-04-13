from __future__ import annotations

import json
import threading
import time
from copy import deepcopy
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


BASE_DIR = Path(__file__).resolve().parent
CATALOG_PATH = BASE_DIR / "dataset_catalog.json"
STATE_PATH = BASE_DIR / "dataset_verification_state.json"
DASHBOARD_PATH = BASE_DIR / "dataset_testing_dashboard.html"
HOST = "127.0.0.1"
PORT = 8011
RUN_INTERVAL_SECONDS = 60


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class DatasetVerifier:
    def __init__(self, catalog_path: Path, state_path: Path) -> None:
        self.catalog_path = catalog_path
        self.state_path = state_path
        self.lock = threading.Lock()
        self._latest_result: dict[str, Any] = {
            "status": "booting",
            "started_at": utc_now_iso(),
            "completed_at": None,
            "summary": {
                "total": 0,
                "ready": 0,
                "attention": 0,
                "multimodal_eligible": 0,
                "multimodal_ready": 0,
                "facial_eligible": 0,
                "facial_ready": 0,
            },
            "results": [],
        }
        self._load_state()

    def _load_state(self) -> None:
        if not self.state_path.exists():
            return

        try:
          state = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return

        with self.lock:
            self._latest_result = state

    def _save_state(self) -> None:
        snapshot = self.snapshot()
        self.state_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")

    def load_resources(self) -> list[dict[str, Any]]:
        return json.loads(self.catalog_path.read_text(encoding="utf-8"))

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return deepcopy(self._latest_result)

    def _evaluate_resource(self, resource: dict[str, Any]) -> dict[str, Any]:
        notes: list[str] = []
        checks: list[dict[str, Any]] = []

        modalities = set(resource.get("modalities", []))
        tasks = set(resource.get("tasks", []))
        suites = set(resource.get("suites", []))
        protocol = resource.get("protocol", {})
        protocol_splits = protocol.get("splits", [])
        emotions = resource.get("emotions", [])
        resource_type = resource.get("type", "dataset")

        core_ok = all(resource.get(field) for field in ("name", "description", "url", "category", "type"))
        checks.append({"name": "core-metadata", "ok": core_ok})
        if not core_ok:
            notes.append("missing core metadata")

        url_ok = str(resource.get("url", "")).startswith(("http://", "https://"))
        checks.append({"name": "source-url", "ok": url_ok})
        if not url_ok:
            notes.append("invalid source URL")

        multimodal_ready = False
        if "multimodal" in suites:
            has_audio = bool({"audio", "speech"} & modalities)
            has_visual = bool({"video", "image", "facial"} & modalities)
            has_target_task = "emotion-recognition" in tasks or "multimodal-fusion" in tasks or resource_type == "model"
            multimodal_ready = has_audio and has_visual and has_target_task
            checks.append({"name": "multimodal-suite", "ok": multimodal_ready})
            if not multimodal_ready:
                notes.append("multimodal suite is missing required modalities or tasks")

        facial_ready = False
        if "facial" in suites:
            has_face_signal = bool({"facial", "image", "video"} & modalities)
            has_face_task = "facial-emotion" in tasks or "expression-classification" in tasks or resource_type != "dataset"
            facial_ready = has_face_signal and has_face_task
            checks.append({"name": "facial-suite", "ok": facial_ready})
            if not facial_ready:
                notes.append("facial suite is missing expression inputs or tasks")

        if resource_type == "dataset":
            protocol_ok = bool(protocol_splits)
            emotion_ok = len(emotions) >= 5
            checks.append({"name": "evaluation-protocol", "ok": protocol_ok})
            checks.append({"name": "emotion-coverage", "ok": emotion_ok})
            if not protocol_ok:
                notes.append("evaluation protocol is not defined")
            if not emotion_ok:
                notes.append("emotion coverage is below the testing threshold")
        else:
            checks.append({"name": "suite-linking", "ok": bool(suites)})
            if not suites:
                notes.append("resource is not linked to a testing suite")

        status = "ready" if not notes else "attention"
        ready_suites = []
        if multimodal_ready:
            ready_suites.append("multimodal")
        if facial_ready:
            ready_suites.append("facial")

        return {
            "name": resource.get("name"),
            "category": resource.get("category"),
            "type": resource_type,
            "status": status,
            "notes": notes,
            "ready_suites": ready_suites,
            "target_suites": sorted(suites),
            "checks": checks,
            "resource": resource,
        }

    def run_once(self, trigger: str = "manual") -> dict[str, Any]:
        resources = self.load_resources()
        started_at = utc_now_iso()
        results = [self._evaluate_resource(resource) for resource in resources]

        summary = {
            "total": len(results),
            "ready": sum(1 for item in results if item["status"] == "ready"),
            "attention": sum(1 for item in results if item["status"] == "attention"),
            "multimodal_eligible": sum(1 for item in results if "multimodal" in item["target_suites"]),
            "multimodal_ready": sum(1 for item in results if "multimodal" in item["ready_suites"]),
            "facial_eligible": sum(1 for item in results if "facial" in item["target_suites"]),
            "facial_ready": sum(1 for item in results if "facial" in item["ready_suites"]),
        }

        payload = {
            "status": "ready" if summary["attention"] == 0 else "attention",
            "trigger": trigger,
            "started_at": started_at,
            "completed_at": utc_now_iso(),
            "summary": summary,
            "results": results,
        }

        with self.lock:
            self._latest_result = payload

        self._save_state()
        return deepcopy(payload)


class VerificationScheduler(threading.Thread):
    def __init__(self, verifier: DatasetVerifier, interval_seconds: int) -> None:
        super().__init__(daemon=True)
        self.verifier = verifier
        self.interval_seconds = interval_seconds
        self.stop_event = threading.Event()

    def run(self) -> None:
        self.verifier.run_once(trigger="startup")
        while not self.stop_event.wait(self.interval_seconds):
            self.verifier.run_once(trigger="scheduler")

    def stop(self) -> None:
        self.stop_event.set()


VERIFIER = DatasetVerifier(CATALOG_PATH, STATE_PATH)
SCHEDULER = VerificationScheduler(VERIFIER, RUN_INTERVAL_SECONDS)


class DatasetTestingHandler(BaseHTTPRequestHandler):
    server_version = "DatasetTestingBackend/1.0"

    def _send_json(self, payload: Any, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str) -> None:
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ("/", "/dataset-testing", "/dataset-testing/"):
            self._send_file(DASHBOARD_PATH, "text/html; charset=utf-8")
            return

        if path == "/api/health":
            latest = VERIFIER.snapshot()
            self._send_json({
                "status": "ok",
                "service": "dataset-testing-backend",
                "time": utc_now_iso(),
                "latest_status": latest.get("status"),
            })
            return

        if path == "/api/resources":
            self._send_json({"resources": VERIFIER.load_resources()})
            return

        if path == "/api/verification/latest":
            latest = VERIFIER.snapshot()
            query = parse_qs(parsed.query)
            include_resources = query.get("include_resources", ["false"])[0].lower() == "true"
            if not include_resources:
                latest = {
                    key: value
                    for key, value in latest.items()
                    if key != "results"
                } | {
                    "results": [
                        {
                            "name": item["name"],
                            "category": item["category"],
                            "type": item["type"],
                            "status": item["status"],
                            "notes": item["notes"],
                            "ready_suites": item["ready_suites"],
                            "target_suites": item["target_suites"],
                        }
                        for item in latest["results"]
                    ]
                }
            self._send_json(latest)
            return

        self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if self.path != "/api/verification/run":
            self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
            return

        payload = VERIFIER.run_once(trigger="manual")
        self._send_json(payload)


def serve() -> None:
    if not SCHEDULER.is_alive():
        SCHEDULER.start()

    httpd = ThreadingHTTPServer((HOST, PORT), DatasetTestingHandler)
    print(f"Dataset testing backend running on http://{HOST}:{PORT}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        SCHEDULER.stop()


if __name__ == "__main__":
    serve()
