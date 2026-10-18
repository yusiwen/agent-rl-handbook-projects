"""The served policy: one HTTP round trip, parsed with the same parser the loop uses."""

from __future__ import annotations

import json
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
for path in (str(REPO), str(HERE.parent / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

from p2_reagent import env  # noqa: E402
from p2_reagent.served_policy import ServedPolicy  # noqa: E402


class _Handler(BaseHTTPRequestHandler):
    reply = "ACTION: ls"
    seen: list[dict] = []

    def do_POST(self) -> None:  # noqa: N802 - the base class names it
        length = int(self.headers.get("Content-Length", "0"))
        _Handler.seen.append(json.loads(self.rfile.read(length).decode("utf-8")))
        body = json.dumps({"choices": [{"message": {"content": _Handler.reply}}]}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:
        return None


class ServedPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = HTTPServer(("127.0.0.1", 0), _Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()

    def setUp(self) -> None:
        _Handler.seen.clear()
        _Handler.reply = "ACTION: ls"

    def test_a_legal_reply_becomes_the_action(self) -> None:
        policy = ServedPolicy(self.base_url, "test-model", temperature=0.0)
        attempt = env.start("files-delete-count")
        self.assertEqual(policy.choose(attempt), "ls")
        self.assertEqual(policy.reasons["ok"], 1)
        self.assertEqual(policy.calls, 1)

    def test_the_request_carries_the_menu_and_no_thinking(self) -> None:
        policy = ServedPolicy(self.base_url, "test-model")
        attempt = env.start("files-delete-count")
        policy.choose(attempt)
        payload = _Handler.seen[-1]
        self.assertEqual(payload["model"], "test-model")
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})
        text = payload["messages"][-1]["content"]
        for action in attempt.menu():
            self.assertIn(f"- {action}", text)

    def test_an_unparsed_reply_passes_through_as_text(self) -> None:
        _Handler.reply = "I would rather not."
        policy = ServedPolicy(self.base_url, "test-model")
        attempt = env.start("files-delete-count")
        action = policy.choose(attempt)
        self.assertEqual(action, "I would rather not.")
        self.assertEqual(policy.reasons["no_action_line"], 1)
        self.assertEqual(attempt.step(action).signal, "error")

    def test_apply_does_nothing_and_says_why(self) -> None:
        policy = ServedPolicy(self.base_url, "test-model")
        self.assertIsNone(policy.apply([]))

    def test_the_repertoire_is_reported_where_entropy_would_be(self) -> None:
        policy = ServedPolicy(self.base_url, "test-model")
        attempt = env.start("files-delete-count")
        policy.choose(attempt)
        policy.choose(attempt)
        self.assertEqual(policy.distinct_action_ratio, 0.5)
        self.assertEqual(policy.summary()["distinct_actions"], 1)


if __name__ == "__main__":
    unittest.main()
