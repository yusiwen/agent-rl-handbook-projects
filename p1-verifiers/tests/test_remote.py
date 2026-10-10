# pyright: reportMissingImports=false

"""Tests for the two HTTP clients in ``remote.py``.

    python3 -m unittest discover -s p1-verifiers/tests -t p1-verifiers/tests

The server below is a real one, on a real socket, speaking the two routes P1 needs. Not a
mock: the failures worth testing here - a 500 that should be retried and a 400 that should
not, a socket that never answers, a body that is not JSON - are exactly the failures a mock
would have been written to produce anyway, and a mock proves only that the test author and
the mock agree.

What is *not* tested here: a real model. That needs a GPU and a served checkpoint, and the
place to prove it is a run record, not a unit test.
"""

from __future__ import annotations

import http.server
import json
import os
import sys
import threading
import time
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "p1-verifiers" / "src"
# The package sits in a directory whose name has a hyphen, so it cannot be imported as a
# package from the repo root. Both roots go on the path, the same way the check script does it.
for _path in (str(REPO), str(SRC)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from p1_verifiers import remote  # noqa: E402
from p1_verifiers.harness import Provenance, Task, run_taskset  # noqa: E402
from p1_verifiers.verifiers import default_registry  # noqa: E402

REPLY = "17 * 23 = 391\n\n391"


class FakeServer:
    """A tiny server shaped like vLLM: a chat route and a tokenize route.

    Every request it receives is recorded, so a test can assert on what was *sent* (the
    model name, the header, the number of attempts) rather than only on what came back.
    """

    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.reply: str | None = REPLY
        self.malformed = False
        self.broken_json = False
        self.fail_remaining = 0
        self.fail_status = 500
        self.delay = 0.0
        self.missing: tuple[str, ...] = ()  # routes that answer 404
        self.models = ["test-model"]  # what /v1/models claims to serve
        self.models_payload: dict | None = None  # set to answer something malformed
        self.replies: list[str] | None = None  # when set, answers cycle through these
        self._reply_index = 0

        state = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 - the name is http.server's
                state.handle(self)

            def do_GET(self) -> None:  # noqa: N802 - the name is http.server's
                state.handle_get(self)

            def log_message(self, format: str, *args) -> None:  # noqa: A002 - the base signature
                """Silence: a test that prints server logs is harder to read than one that does not."""

        self._httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        # The default poll interval is 0.5s and every teardown waits for one tick; with a
        # server per test that is seconds of a unit suite spent doing nothing.
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        self._thread.start()

    @property
    def base_url(self) -> str:
        host, port = self._httpd.server_address[:2]
        return f"http://{host}:{port}"

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()

    def _respond(self, handler: http.server.BaseHTTPRequestHandler, status: int, payload) -> None:
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        try:
            handler.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            # The client hung up on purpose: the timeout test points a 0.4s deadline at a 3s
            # handler. A traceback from the server thread would be noise in the test output.
            pass

    def handle_get(self, handler: http.server.BaseHTTPRequestHandler) -> None:
        """The one GET route P1 uses: /v1/models, for the preflight."""
        self.requests.append({"path": handler.path, "payload": None,
                              "headers": dict(handler.headers)})
        if handler.path.endswith("/v1/models"):
            if self.models_payload is not None:
                self._respond(handler, 200, self.models_payload)
                return
            self._respond(handler, 200, {"data": [{"id": name} for name in self.models]})
            return
        self._respond(handler, 404, {"error": f"no route {handler.path}"})

    def handle(self, handler: http.server.BaseHTTPRequestHandler) -> None:
        length = int(handler.headers.get("Content-Length") or 0)
        raw = handler.rfile.read(length)
        try:
            payload = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            payload = None
        self.requests.append({"path": handler.path, "payload": payload,
                              "headers": dict(handler.headers)})

        if self.delay:
            time.sleep(self.delay)
        if handler.path in self.missing:
            self._respond(handler, 404, {"error": "not found"})
            return
        if self.fail_remaining > 0:
            self.fail_remaining -= 1
            self._respond(handler, self.fail_status, {"error": "the server is having a moment"})
            return
        if self.broken_json:
            self._respond(handler, 200, b"this is not json")
            return
        if handler.path.endswith("/v1/chat/completions"):
            reply = self.reply
            if self.replies:
                reply = self.replies[self._reply_index % len(self.replies)]
                self._reply_index += 1
            message = {} if reply is None else {"role": "assistant", "content": reply}
            self._respond(handler, 200, {
                "choices": [{"message": message, "finish_reason": "stop" if reply else "length"}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15},
            })
            return
        if handler.path.endswith("/tokenize"):
            prompt = (payload or {}).get("prompt", "")
            self._respond(handler, 200, {"count": max(1, len(str(prompt).split())),
                                        "tokens": [1] * max(1, len(str(prompt).split()))})
            return
        self._respond(handler, 404, {"error": f"no route {handler.path}"})


class ServerTestCase(unittest.TestCase):
    """A server per test, torn down even when the test fails."""

    def setUp(self) -> None:
        self.server = FakeServer()
        self.addCleanup(self.server.stop)

    def responder(self, **kwargs):
        defaults = dict(temperature=0.7, top_p=0.95, max_tokens=512, timeout=5.0, retries=0,
                        sleep=lambda _: None)
        defaults.update(kwargs)
        return remote.OpenAIChatResponder(self.server.base_url, "test-model", **defaults)


class RequestShapeTests(ServerTestCase):
    def test_the_request_carries_what_the_model_needs(self) -> None:
        responder = self.responder()
        assert responder.respond(Task(id="t", prompt="What is 17*23?", verifier="exact_match"), 0) == REPLY
        sent = self.server.requests[-1]["payload"]
        self.assertEqual(sent["model"], "test-model")
        self.assertEqual(sent["temperature"], 0.7)
        self.assertEqual(sent["max_tokens"], 512)
        self.assertEqual([m["role"] for m in sent["messages"]], ["system", "user"])
        self.assertIn("17*23", sent["messages"][1]["content"])

    def test_the_path_is_the_openai_one(self) -> None:
        self.responder().respond(Task(id="t", prompt="p", verifier="exact_match"), 0)
        self.assertTrue(self.server.requests[-1]["path"].endswith("/v1/chat/completions"))

    def test_a_seed_is_offset_per_sample_so_samples_are_reproducible_not_identical(self) -> None:
        responder = self.responder(seed=100)
        responder.respond(Task(id="t", prompt="p", verifier="exact_match"), 0)
        responder.respond(Task(id="t", prompt="p", verifier="exact_match"), 1)
        seeds = [r["payload"].get("seed") for r in self.server.requests]
        self.assertEqual(seeds, [100, 101])

    def test_no_seed_is_sent_when_none_was_asked_for(self) -> None:
        self.responder().respond(Task(id="t", prompt="p", verifier="exact_match"), 0)
        self.assertNotIn("seed", self.server.requests[-1]["payload"])

    def test_usage_and_finish_reason_are_kept_for_the_run_record(self) -> None:
        responder = self.responder()
        responder.respond(Task(id="t", prompt="p", verifier="exact_match"), 0)
        self.assertEqual(responder.last_usage["completion_tokens"], 4)
        self.assertEqual(responder.last_finish_reason, "stop")


class FailureTests(ServerTestCase):
    def test_a_4xx_is_not_retried(self) -> None:
        self.server.fail_remaining = 1
        self.server.fail_status = 400
        responder = self.responder(retries=3)
        with self.assertRaises(remote.RemoteError) as ctx:
            responder.respond(Task(id="t", prompt="p", verifier="exact_match"), 0)
        self.assertEqual(ctx.exception.status, 400)
        self.assertEqual(len(self.server.requests), 1, "a bad request was repeated")

    def test_a_5xx_is_retried_and_can_succeed(self) -> None:
        self.server.fail_remaining = 2
        responder = self.responder(retries=2)
        text = responder.respond(Task(id="t", prompt="p", verifier="exact_match"), 0)
        self.assertEqual(text, REPLY)
        self.assertEqual(len(self.server.requests), 3, "two failures then the success")

    def test_a_5xx_that_never_recovers_raises_with_the_status(self) -> None:
        self.server.fail_remaining = 10
        responder = self.responder(retries=1)
        with self.assertRaises(remote.RemoteError) as ctx:
            responder.respond(Task(id="t", prompt="p", verifier="exact_match"), 0)
        self.assertEqual(ctx.exception.status, 500)
        self.assertEqual(len(self.server.requests), 2)

    def test_backoff_is_bounded_and_actually_waited(self) -> None:
        self.server.fail_remaining = 3
        slept: list[float] = []
        responder = self.responder(retries=3, sleep=slept.append)
        responder.respond(Task(id="t", prompt="p", verifier="exact_match"), 0)
        self.assertEqual(len(slept), 3)
        self.assertTrue(all(0 < value <= 8.0 for value in slept), slept)

    def test_an_empty_answer_is_an_error_not_a_wrong_answer(self) -> None:
        self.server.reply = None
        with self.assertRaises(remote.RemoteError) as ctx:
            self.responder().respond(Task(id="t", prompt="p", verifier="exact_match"), 0)
        self.assertIn("no text", str(ctx.exception))

    def test_a_body_that_is_not_json_is_reported_as_such(self) -> None:
        self.server.broken_json = True
        with self.assertRaises(remote.RemoteError) as ctx:
            self.responder().respond(Task(id="t", prompt="p", verifier="exact_match"), 0)
        self.assertIn("did not answer with JSON", str(ctx.exception))

    def test_a_socket_that_never_answers_times_out(self) -> None:
        self.server.delay = 3.0
        responder = self.responder(timeout=0.4)
        started = time.monotonic()
        with self.assertRaises(remote.RemoteError) as ctx:
            responder.respond(Task(id="t", prompt="p", verifier="exact_match"), 0)
        self.assertIn("timed out", str(ctx.exception))
        self.assertLess(time.monotonic() - started, 2.5, "the client waited past its timeout")

    def test_an_unreachable_port_says_so(self) -> None:
        responder = remote.OpenAIChatResponder("http://127.0.0.1:1", "test-model", timeout=2.0)
        with self.assertRaises(remote.RemoteError) as ctx:
            responder.respond(Task(id="t", prompt="p", verifier="exact_match"), 0)
        self.assertIn("unreachable", str(ctx.exception))


class ApiKeyTests(ServerTestCase):
    """The key comes from the environment, and never appears in anything this code prints."""

    ENV = "ARL_TEST_API_KEY"
    VALUE = "unit-test-value"

    def setUp(self) -> None:
        super().setUp()
        self._saved = os.environ.pop(self.ENV, None)
        self.addCleanup(self._restore)

    def _restore(self) -> None:
        if self._saved is None:
            os.environ.pop(self.ENV, None)
        else:
            os.environ[self.ENV] = self._saved

    def test_no_header_when_the_variable_is_unset(self) -> None:
        responder = self.responder(api_key_env=self.ENV)
        responder.respond(Task(id="t", prompt="p", verifier="exact_match"), 0)
        self.assertNotIn("Authorization", self.server.requests[-1]["headers"])

    def test_the_header_is_sent_when_the_variable_is_set(self) -> None:
        os.environ[self.ENV] = self.VALUE
        responder = self.responder(api_key_env=self.ENV)
        responder.respond(Task(id="t", prompt="p", verifier="exact_match"), 0)
        self.assertEqual(self.server.requests[-1]["headers"].get("Authorization"),
                         f"Bearer {self.VALUE}")

    def test_describe_names_the_variable_and_never_the_value(self) -> None:
        os.environ[self.ENV] = self.VALUE
        described = self.responder(api_key_env=self.ENV).describe()
        self.assertIn(self.ENV, described)
        self.assertNotIn(self.VALUE, described)

    def test_describe_is_honest_about_having_no_key(self) -> None:
        self.assertIn("no key", self.responder(api_key_env=self.ENV).describe())
        self.assertIn("no key", self.responder().describe())

    def test_an_error_message_never_contains_the_key(self) -> None:
        os.environ[self.ENV] = self.VALUE
        self.server.fail_remaining = 5
        responder = self.responder(api_key_env=self.ENV, retries=0)
        with self.assertRaises(remote.RemoteError) as ctx:
            responder.respond(Task(id="t", prompt="p", verifier="exact_match"), 0)
        self.assertNotIn(self.VALUE, str(ctx.exception))


class TokeniserTests(ServerTestCase):
    def tokeniser(self, **kwargs):
        return remote.ServerTokeniser(self.server.base_url, "test-model", timeout=5.0, **kwargs)

    def test_it_counts_with_the_servers_own_tokeniser(self) -> None:
        tokeniser = self.tokeniser()
        self.assertEqual(tokeniser.count("one two three four"), 4)
        self.assertTrue(self.server.requests[-1]["path"].endswith("/tokenize"))
        self.assertEqual(self.server.requests[-1]["payload"]["model"], "test-model")

    def test_its_name_says_what_counted_and_where(self) -> None:
        name = self.tokeniser().name
        self.assertIn("test-model", name)
        self.assertIn("127.0.0.1", name)

    def test_the_working_endpoint_is_remembered(self) -> None:
        tokeniser = self.tokeniser()
        tokeniser.count("a b")
        tokeniser.count("a b c")
        self.assertEqual([r["path"] for r in self.server.requests],
                         ["/tokenize", "/tokenize"])

    def test_a_prefixed_route_is_found_when_the_bare_one_is_missing(self) -> None:
        self.server.missing = ("/tokenize",)
        tokeniser = self.tokeniser()
        self.assertEqual(tokeniser.count("a b c"), 3)
        self.assertEqual([r["path"] for r in self.server.requests], ["/tokenize", "/v1/tokenize"])
        tokeniser.count("a b c d")
        self.assertEqual(self.server.requests[-1]["path"], "/v1/tokenize", "the probe was repeated")

    def test_a_server_with_no_tokenize_route_says_what_it_tried(self) -> None:
        self.server.missing = ("/tokenize", "/v1/tokenize")
        with self.assertRaises(remote.RemoteError) as ctx:
            self.tokeniser().count("a b")
        message = str(ctx.exception)
        self.assertIn("/tokenize", message)
        self.assertIn("/v1/tokenize", message)
        self.assertIn("pseudo-words/v1", message)

    def test_a_bare_token_list_is_accepted_too(self) -> None:
        self.server.missing = ()
        original = self.server._respond

        def respond_without_count(handler, status, payload):
            if isinstance(payload, dict) and "count" in payload:
                payload = {"tokens": [0] * payload["count"]}
            original(handler, status, payload)

        self.server._respond = respond_without_count
        self.assertEqual(self.tokeniser().count("a b c d e"), 5)


class LiveWiringTests(ServerTestCase):
    """The point of all of the above: a run whose rows carry real counts and the right marker."""

    def test_a_run_through_a_real_socket_records_real_token_counts(self) -> None:
        responder = self.responder()
        tokeniser = remote.ServerTokeniser(self.server.base_url, "test-model", timeout=5.0)
        task = Task(id="t01", prompt="What is 17 * 23?", verifier="exact_match",
                    fields={"answer": "391"})
        rows = run_taskset(
            [task], responder, default_registry(), samples_per_task=1,
            provenance=Provenance(env_version=tokeniser.name, config_hash="sha256:test",
                                  code_sha="git:test"),
            tokeniser=tokeniser,
        )
        row = rows[0]
        self.assertEqual(row.env_version, tokeniser.name)
        self.assertNotIn("pseudo-words/v1", row.env_version)
        self.assertTrue(row.extras["passed"], row.extras["detail"])
        # Every count in this row came over HTTP, and none of them is the pseudo counter:
        # the server splits its own way, so the numbers are the server's, not a word count
        # that happens to agree. 11 words in the system prompt, 5 in the question, 6 in the
        # reply; only the reply is trainable.
        self.assertEqual(row.total_tokens, 22)
        self.assertEqual(row.model_tokens, 6)
        self.assertEqual(sum(row.mask or []), row.model_tokens)

    def test_a_real_tokeniser_with_the_pseudo_marker_stops_the_run(self) -> None:
        tokeniser = remote.ServerTokeniser(self.server.base_url, "test-model", timeout=5.0)
        task = Task(id="t01", prompt="p", verifier="exact_match", fields={"answer": "1"})
        with self.assertRaises(ValueError) as ctx:
            run_taskset([task], self.responder(), default_registry(), samples_per_task=1,
                        tokeniser=tokeniser)
        self.assertIn("env_version", str(ctx.exception))


class ModelListingTests(ServerTestCase):
    def test_it_lists_what_the_server_serves(self) -> None:
        self.assertEqual(remote.list_models(self.server.base_url), ["test-model"])

    def test_two_models_are_both_listed(self) -> None:
        self.server.models = ["qwen3-8b", "qwen3.6-35b-a3b"]
        self.assertEqual(remote.list_models(self.server.base_url),
                         ["qwen3-8b", "qwen3.6-35b-a3b"])

    def test_a_listing_without_a_data_list_is_an_error(self) -> None:
        self.server.models_payload = {"data": "not a list"}
        with self.assertRaises(remote.RemoteError) as ctx:
            remote.list_models(self.server.base_url)
        self.assertIn("data", str(ctx.exception))

    def test_an_unreachable_server_is_reported(self) -> None:
        with self.assertRaises(remote.RemoteError):
            remote.list_models("http://127.0.0.1:1", timeout=2.0)


def load_runner():
    """The live runner is a script, not part of the package; load it from its path."""
    import importlib.util

    path = REPO / "p1-verifiers" / "checks" / "run_p1_live.py"
    spec = importlib.util.spec_from_file_location("run_p1_live", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LiveRunnerTests(ServerTestCase):
    """The runner end to end over a real socket: the closest thing to a Spark run with no GPU.

    This is where "P1 can run against a served model" stops being a claim: the preflight, the
    tokeniser, the rows, the checks and the run record are all exercised against a real server.
    What it still cannot prove is a real *model* - only a real protocol.
    """

    def setUp(self) -> None:
        super().setUp()
        import tempfile

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.tasks = self.root / "tasks.jsonl"
        self.tasks.write_text(
            '{"id": "t01", "prompt": "What is 17 * 23?", "verifier": "exact_match", "answer": "391"}\n'
            '{"id": "t02", "prompt": "What is 17 * 23 again?", "verifier": "exact_match", "answer": "391"}\n'
            '{"id": "t03", "prompt": "pi to two decimals", "verifier": "numeric", "answer": "3.14"}\n',
            encoding="utf-8")
        self.runs = self.root / "runs"
        self.runner = load_runner()

    def run_main(self, *extra: str):
        import contextlib
        import io

        argv = ["--base-url", self.server.base_url, "--model", "test-model",
                "--tasks", str(self.tasks), "--runs-dir", str(self.runs), *extra]
        buffer = io.StringIO()
        # Both streams: the runner prints its failure summary to stderr, which is the right
        # choice for a script but means the test has to look at both to see the whole story.
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            code = self.runner.main(argv)
        return code, buffer.getvalue()

    #: What a wrong attempt looks like: same shape, wrong final answer, so the verifier can
    #: tell them apart and the group gets a spread.
    WRONG = "17 * 23 = 400\n\n400"

    def test_a_live_run_produces_checked_rows_and_a_record(self) -> None:
        # Alternating answers, so each group holds one pass and one miss. A run where every
        # attempt succeeds has zero advantage in every group, and the checks refuse it - see
        # test_a_run_with_no_contrast_is_rejected_by_the_checks.
        self.server.replies = [REPLY, self.WRONG]
        code, output = self.run_main("--samples", "2")
        self.assertEqual(code, 0, output)
        self.assertIn("all checks passed on this run", output)
        self.assertIn("not scoreable: t03", output)
        self.assertIn("tokeniser: server-tokenize/test-model@127.0.0.1", output)
        self.assertIn("2/4 (50%)", output)

        records = list(self.runs.glob("p1-live-*/"))
        self.assertEqual(len(records), 1)
        for name in ("header.json", "rounds.jsonl", "round-000.jsonl"):
            self.assertTrue((records[0] / name).exists(), f"{name} was not written")

        header = json.loads((records[0] / "header.json").read_text(encoding="utf-8"))
        self.assertEqual(header["env_version"], "server-tokenize/test-model@127.0.0.1")
        self.assertIn("test-model", header["notes"])
        self.assertIn("covered 2/3 tasks", header["notes"])

        rows = [json.loads(line) for line in
                (records[0] / "round-000.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual(len(rows), 4)
        self.assertEqual([row["extras"]["passed"] for row in rows], [True, False, True, False])
        for row in rows:
            self.assertEqual(sum(row["mask"]),
                             sum(seg["tokens"] for seg in row["segments"] if seg["role"] == "model"))

    def test_a_run_with_no_contrast_is_rejected_by_the_checks(self) -> None:
        # Every attempt identical and correct: the rows are valid, the reward is perfect, and
        # the run is worthless, because a group whose samples all score the same produces no
        # gradient. The record is still written - a record of a rejected run is evidence.
        code, output = self.run_main("--samples", "2")
        self.assertEqual(code, 1, output)
        self.assertIn("advantage spread", output)
        self.assertIn("FAILED", output)
        self.assertEqual(len(list(self.runs.glob("p1-live-*/"))), 1)

    def test_a_wrong_model_name_stops_before_any_attempt(self) -> None:
        import contextlib
        import io

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            code = self.runner.main(["--base-url", self.server.base_url, "--model", "not-served",
                                     "--tasks", str(self.tasks), "--runs-dir", str(self.runs)])
        self.assertEqual(code, 2)
        self.assertIn("does not serve", buffer.getvalue())
        self.assertIn("test-model", buffer.getvalue(), "it should say what the server does serve")
        self.assertEqual([r for r in self.server.requests if r["payload"]], [],
                         "an attempt was made despite the preflight failing")

    def test_a_partially_covered_task_set_still_runs_and_says_so(self) -> None:
        self.server.replies = [REPLY, self.WRONG]
        code, output = self.run_main("--limit", "2", "--samples", "2")
        self.assertEqual(code, 0, output)
        self.assertIn("2 covered, x2 samples = 4 attempts", output)

    def test_nothing_scoreable_is_refused_before_the_model_is_touched(self) -> None:
        only_stub = self.root / "stubs.jsonl"
        only_stub.write_text('{"id": "n1", "prompt": "pi", "verifier": "numeric", "answer": "3.14"}\n',
                             encoding="utf-8")
        import contextlib
        import io

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            code = self.runner.main(["--base-url", self.server.base_url, "--model", "test-model",
                                     "--tasks", str(only_stub), "--runs-dir", str(self.runs)])
        self.assertEqual(code, 2)
        self.assertIn("nothing in this task set can be scored", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
