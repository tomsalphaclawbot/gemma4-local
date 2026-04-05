#!/usr/bin/env python3
"""
Serializing proxy for local Gemma 4 MLX server.

Sits between clients and the MLX server, ensuring:
1. Only ONE inference request at a time (queued, not rejected)
2. Memory pressure check before forwarding
3. Clean timeout/error responses instead of hanging or OOM

Listens on PROXY_PORT (8891), forwards to MLX_PORT (8890).
OpenClaw and all other clients should point here, never directly at MLX.
"""

import http.server
import json
import threading
import time
import urllib.request
import urllib.error
import subprocess
import re
import os
import sys
import signal

PROXY_HOST = os.environ.get("GEMMA_PROXY_HOST", "127.0.0.1")
PROXY_PORT = int(os.environ.get("GEMMA_PROXY_PORT", "8891"))
MLX_HOST = os.environ.get("GEMMA_MLX_HOST", "127.0.0.1")
MLX_PORT = int(os.environ.get("GEMMA_MLX_PORT", "8890"))
MLX_BASE = f"http://{MLX_HOST}:{MLX_PORT}"

# Max time a request will wait in the queue before getting 503'd
QUEUE_TIMEOUT_S = int(os.environ.get("GEMMA_QUEUE_TIMEOUT", "120"))

# Minimum free RAM (GB) required to accept a new request
MIN_FREE_RAM_GB = float(os.environ.get("GEMMA_MIN_FREE_RAM_GB", "4.0"))
BACKEND_TIMEOUT_S = int(os.environ.get("GEMMA_BACKEND_TIMEOUT", "300"))

# Serialization lock — the whole point of this proxy
_inference_lock = threading.Lock()

# Stats
_stats = {
    "requests_total": 0,
    "requests_completed": 0,
    "requests_queued": 0,
    "requests_rejected_memory": 0,
    "requests_rejected_timeout": 0,
    "requests_failed": 0,
    "current_queue_depth": 0,
    "started_at": time.time(),
}
_stats_lock = threading.Lock()


def get_free_ram_gb():
    """Get available RAM (free + inactive) in GB via vm_stat."""
    try:
        out = subprocess.check_output(["vm_stat"], timeout=5).decode()
        page = 16384
        free = int(re.search(r"Pages free:\s+(\d+)", out).group(1))
        inactive = int(re.search(r"Pages inactive:\s+(\d+)", out).group(1))
        return (free + inactive) * page / 1e9
    except Exception:
        return 999.0  # fail open — don't block on vm_stat failure


def check_mlx_health():
    """Quick check that MLX server is responding."""
    try:
        req = urllib.request.Request(f"{MLX_BASE}/v1/models", method="GET")
        resp = urllib.request.urlopen(req, timeout=5)
        return resp.status == 200
    except Exception:
        return False


class GemmaProxyHandler(http.server.BaseHTTPRequestHandler):
    """HTTP handler that serializes inference requests to the MLX backend."""

    def log_message(self, format, *args):
        # Structured logging
        ts = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        sys.stderr.write(f"[{ts}] {format % args}\n")

    def _normalize_responses_payload(self, body):
        """
        MLX /v1/responses rejects some item types (e.g. `output_text`) that
        OpenClaw can include when replaying prior model outputs.
        Convert those to equivalent input text items before forwarding.
        """
        if not body:
            return body
        try:
            payload = json.loads(body.decode("utf-8"))
        except Exception:
            return body

        changed = False

        def normalize_content_item(item):
            nonlocal changed
            if not isinstance(item, dict):
                return item
            item_type = item.get("type")
            if item_type == "output_text":
                changed = True
                text = item.get("text")
                if text is None:
                    text = item.get("content", "")
                return {"type": "input_text", "text": text}
            return item

        def normalize_input_item(item):
            nonlocal changed
            if not isinstance(item, dict):
                return item
            if item.get("type") == "output_text":
                changed = True
                text = item.get("text")
                if text is None:
                    text = item.get("content", "")
                return {"type": "input_text", "text": text}
            content = item.get("content")
            if isinstance(content, list):
                item = dict(item)
                item["content"] = [normalize_content_item(c) for c in content]
            return item

        def content_text(value):
            if isinstance(value, str):
                return value
            if isinstance(value, list):
                parts = []
                for item in value:
                    if isinstance(item, dict):
                        text = item.get("text")
                        if text is None and isinstance(item.get("content"), str):
                            text = item.get("content")
                        if text is not None:
                            parts.append(str(text))
                    elif item is not None:
                        parts.append(str(item))
                return "\n".join(parts)
            return ""

        def collapse_typed_input_to_messages(items):
            parts = []
            for item in items:
                if not isinstance(item, dict):
                    if item is not None:
                        parts.append(str(item))
                    continue
                text = item.get("text")
                if text is None:
                    text = item.get("content")
                if isinstance(text, list):
                    text = content_text(text)
                if text is not None:
                    parts.append(str(text))
            return [{"role": "user", "content": "\n".join(p for p in parts if p)}]

        input_value = payload.get("input")
        if isinstance(input_value, list):
            normalized_items = [normalize_input_item(i) for i in input_value]
            # MLX responses expects input as either plain string or list of
            # chat messages with role/content; typed item arrays can 422.
            if any(isinstance(i, dict) and "type" in i and "role" not in i for i in normalized_items):
                changed = True
                payload["input"] = collapse_typed_input_to_messages(normalized_items)
            else:
                payload["input"] = normalized_items
        elif isinstance(input_value, dict):
            normalized_item = normalize_input_item(input_value)
            if isinstance(normalized_item, dict) and "type" in normalized_item and "role" not in normalized_item:
                changed = True
                text = normalized_item.get("text")
                if text is None:
                    text = content_text(normalized_item.get("content"))
                payload["input"] = [{"role": "user", "content": str(text or "")}]
            else:
                payload["input"] = normalized_item

        if not changed:
            return body
        return json.dumps(payload).encode("utf-8")

    def _proxy_passthrough(self, method="GET", body=None):
        """Forward non-inference requests directly (no lock needed)."""
        url = f"{MLX_BASE}{self.path}"
        headers = {"Content-Type": self.headers.get("Content-Type", "application/json")}
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            resp = urllib.request.urlopen(req, timeout=BACKEND_TIMEOUT_S)
            data = resp.read()
            self.send_response(resp.status)
            self.send_header("Content-Type", resp.headers.get("Content-Type", "application/json"))
            self.send_header("X-Gemma-Proxy", "passthrough")
            self.end_headers()
            self.wfile.write(data)
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            self.end_headers()
            self.wfile.write(e.read())
        except Exception as e:
            self._send_error(502, f"MLX backend unavailable: {e}")

    def _send_error(self, code, message):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("X-Gemma-Proxy", "error")
        self.end_headers()
        self.wfile.write(json.dumps({
            "error": {"message": message, "type": "proxy_error", "code": code}
        }).encode())

    def _send_json(self, code, data, proxy_header="stats"):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("X-Gemma-Proxy", proxy_header)
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode())

    def do_GET(self):
        # Stats endpoint
        if self.path == "/proxy/stats":
            with _stats_lock:
                s = dict(_stats)
                s["uptime_s"] = round(time.time() - s["started_at"], 1)
                s["free_ram_gb"] = round(get_free_ram_gb(), 1)
                s["mlx_healthy"] = check_mlx_health()
            self._send_json(200, s)
            return

        if self.path == "/proxy/health":
            healthy = check_mlx_health()
            self._send_json(200 if healthy else 503, {
                "status": "ok" if healthy else "mlx_down",
                "free_ram_gb": round(get_free_ram_gb(), 1),
            })
            return

        # Everything else: passthrough to MLX (model listing, etc.)
        self._proxy_passthrough("GET")

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b""

        # Non-inference POST endpoints: passthrough
        is_inference = any(
            segment in self.path for segment in ("/chat/completions", "/completions", "/responses")
        )
        if not is_inference:
            self._proxy_passthrough("POST", body)
            return

        # === INFERENCE REQUEST — serialize through lock ===

        with _stats_lock:
            _stats["requests_total"] += 1
            _stats["current_queue_depth"] += 1
            queue_depth = _stats["current_queue_depth"]

        self.log_message("QUEUE: request accepted (depth=%d)", queue_depth)

        # Memory pressure check
        free_ram = get_free_ram_gb()
        if free_ram < MIN_FREE_RAM_GB:
            with _stats_lock:
                _stats["requests_rejected_memory"] += 1
                _stats["current_queue_depth"] -= 1
            self.log_message("REJECT: low memory (%.1f GB free, need %.1f GB)", free_ram, MIN_FREE_RAM_GB)
            self._send_error(503, f"Insufficient memory: {free_ram:.1f}GB free, need {MIN_FREE_RAM_GB}GB")
            return

        # Try to acquire the inference lock with timeout
        acquired = _inference_lock.acquire(timeout=QUEUE_TIMEOUT_S)
        if not acquired:
            with _stats_lock:
                _stats["requests_rejected_timeout"] += 1
                _stats["current_queue_depth"] -= 1
            self.log_message("REJECT: queue timeout after %ds", QUEUE_TIMEOUT_S)
            self._send_error(503, f"Request queued too long ({QUEUE_TIMEOUT_S}s). Server busy.")
            return

        try:
            # Re-check memory after waiting in queue
            free_ram = get_free_ram_gb()
            if free_ram < MIN_FREE_RAM_GB:
                with _stats_lock:
                    _stats["requests_rejected_memory"] += 1
                    _stats["current_queue_depth"] -= 1
                self.log_message("REJECT: low memory after queue (%.1f GB free)", free_ram)
                self._send_error(503, f"Insufficient memory after queue: {free_ram:.1f}GB free")
                return

            if queue_depth > 1:
                self.log_message("FORWARD: acquired lock after queuing (was depth=%d)", queue_depth)
            else:
                self.log_message("FORWARD: no queue contention")

            if "/responses" in self.path:
                body = self._normalize_responses_payload(body)

            # Forward to MLX
            url = f"{MLX_BASE}{self.path}"
            headers = {
                "Content-Type": self.headers.get("Content-Type", "application/json"),
            }
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            try:
                resp = urllib.request.urlopen(req, timeout=BACKEND_TIMEOUT_S)
                data = resp.read()
                self.send_response(resp.status)
                self.send_header("Content-Type", resp.headers.get("Content-Type", "application/json"))
                self.send_header("X-Gemma-Proxy", "serialized")
                self.send_header("X-Gemma-Queue-Depth", str(queue_depth))
                self.end_headers()
                self.wfile.write(data)

                with _stats_lock:
                    _stats["requests_completed"] += 1

            except urllib.error.HTTPError as e:
                with _stats_lock:
                    _stats["requests_failed"] += 1
                self.send_response(e.code)
                self.end_headers()
                self.wfile.write(e.read())

            except Exception as e:
                with _stats_lock:
                    _stats["requests_failed"] += 1
                self.log_message("ERROR: MLX request failed: %s", str(e))
                self._send_error(502, f"MLX backend error: {e}")

        finally:
            _inference_lock.release()
            with _stats_lock:
                _stats["current_queue_depth"] -= 1


class ThreadedHTTPServer(http.server.HTTPServer):
    """Handle each request in a new thread so queued requests don't block accept()."""
    allow_reuse_address = True

    def process_request(self, request, client_address):
        t = threading.Thread(target=self._handle_request_thread, args=(request, client_address))
        t.daemon = True
        t.start()

    def _handle_request_thread(self, request, client_address):
        try:
            self.finish_request(request, client_address)
        except Exception:
            self.handle_error(request, client_address)
        finally:
            self.shutdown_request(request)


def main():
    server = ThreadedHTTPServer((PROXY_HOST, PROXY_PORT), GemmaProxyHandler)

    def shutdown_handler(signum, frame):
        sys.stderr.write(f"\n[gemma4-proxy] Shutting down (signal {signum})...\n")
        server.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown_handler)
    signal.signal(signal.SIGINT, shutdown_handler)

    print(f"[gemma4-proxy] Serializing proxy started", file=sys.stderr)
    print(f"[gemma4-proxy] Listening: http://{PROXY_HOST}:{PROXY_PORT}", file=sys.stderr)
    print(f"[gemma4-proxy] Backend:   {MLX_BASE}", file=sys.stderr)
    print(f"[gemma4-proxy] Queue timeout: {QUEUE_TIMEOUT_S}s", file=sys.stderr)
    print(f"[gemma4-proxy] Min free RAM: {MIN_FREE_RAM_GB} GB", file=sys.stderr)
    print(f"[gemma4-proxy] Stats: http://{PROXY_HOST}:{PROXY_PORT}/proxy/stats", file=sys.stderr)
    server.serve_forever()


if __name__ == "__main__":
    main()
