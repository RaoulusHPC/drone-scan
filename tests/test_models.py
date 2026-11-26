# SPDX-License-Identifier: AGPL-3.0-or-later
"""Model registry `models.lock` and `drone-scan models pull`/`list` (KAN-24).

Downloads only go to a local fake HTTP server on 127.0.0.1 serving test bytes; no real
network and no real model weights are used.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from dronescan.cli import app
from dronescan.common.models_lock import (
    ModelEntry,
    ModelNotAvailableError,
    ModelsLock,
    ModelStatus,
    load_models_lock,
    model_path,
    model_status,
    resolve_model,
)

runner = CliRunner()

WEIGHTS = b"fake weights for the test, not a model\n" * 100
OTHER = b"other fake weights\n" * 50


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class FakeServer:
    """Serves ``files`` (path -> bytes) on 127.0.0.1 and records every requested path."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.requests: list[str] = []
        self.redirects: dict[str, str] = {}
        self.truncate = False
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                server.requests.append(self.path)
                if self.path in server.redirects:
                    self.send_response(302)
                    self.send_header("Location", server.redirects[self.path])
                    self.end_headers()
                    return
                body = server.files.get(self.path)
                if body is None:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body[: len(body) // 2] if server.truncate else body)

            def log_message(self, format: str, *args: Any) -> None:
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}{path}"


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeServer]:
    for name in (
        "http_proxy",
        "HTTP_PROXY",
        "https_proxy",
        "HTTPS_PROXY",
        "all_proxy",
        "ALL_PROXY",
    ):
        monkeypatch.delenv(name, raising=False)  # 127.0.0.1 must never go through a proxy
    fake = FakeServer()
    fake.thread.start()
    yield fake
    fake.httpd.shutdown()
    fake.httpd.server_close()


def entry(
    server: FakeServer, name: str = "tiny", data: bytes = WEIGHTS, **extra: Any
) -> dict[str, Any]:
    path = f"/files/{name}.pth"
    server.files[path] = data
    return {
        "name": name,
        "version": "1.0",
        "source": "https://example.org/model-card",
        "url": server.url(path),
        "sha256": sha(data),
        "weights_license": "Apache-2.0",
        "code_license": "MIT",
        **extra,
    }


def write_lock(path: Path, *entries: dict[str, Any]) -> Path:
    path.write_text(json.dumps({"models": list(entries)}), encoding="utf-8")  # JSON is YAML
    return path


def pull(tmp_path: Path, *args: str) -> Any:
    lock, models = tmp_path / "models.lock", tmp_path / "models"
    return runner.invoke(
        app, ["models", "pull", *args, "--lock", str(lock), "--models-dir", str(models)]
    )


def files_below(directory: Path) -> list[str]:
    if not directory.exists():
        return []
    return sorted(p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file())


# ---------- AC 1: models.lock schema ----------


def test_entry_holds_name_version_source_hash_and_both_licenses(tmp_path: Path) -> None:
    """Given a complete models.lock entry,
    When it is loaded,
    Then name, version, source, URL, SHA-256, code and weights license are available.
    """
    lock = write_lock(
        tmp_path / "models.lock",
        {
            "name": "tiny",
            "version": "2024-01",
            "source": "https://example.org/model-card",
            "url": "https://example.org/dl/tiny_v1.pth?download=1",
            "sha256": "b" * 64,
            "weights_license": "CC-BY-4.0",
            "code_license": "Apache-2.0",
        },
    )

    [model] = load_models_lock(lock).models

    assert (model.name, model.version, model.sha256) == ("tiny", "2024-01", "b" * 64)
    assert (model.weights_license, model.code_license) == ("CC-BY-4.0", "Apache-2.0")
    assert model.source == "https://example.org/model-card"
    assert model.file_name == "tiny_v1.pth"
    assert model_path(model, tmp_path) == tmp_path / "tiny" / "2024-01" / "tiny_v1.pth"


def test_shipped_models_lock_is_valid() -> None:
    """Given the repository's models.lock,
    When it is loaded,
    Then it is valid (new entries need a PO license decision, KAN-24).
    """
    load_models_lock(Path(__file__).resolve().parents[1] / "models.lock")


VALID = {
    "name": "tiny",
    "version": "1.0",
    "source": "https://example.org/card",
    "url": "https://example.org/tiny.pth",
    "sha256": "c" * 64,
    "weights_license": "MIT",
    "code_license": "MIT",
}


@pytest.mark.parametrize(
    "change",
    [
        {"code_license": None},
        {"weights_license": None},
        {"url": None},
        {"license": "MIT"},  # old single field is gone: weights and code are explicit
        {"url": "http://example.org/tiny.pth"},  # plain http only for loopback
        {"url": "ftp://example.org/tiny.pth"},
        {"url": "file:///etc/passwd"},
        {"name": "../evil"},
        {"name": "a/b"},
        {"version": ".."},
        {"filename": "../x.pth"},
        {"url": "https://example.org/"},  # no file name derivable
        {"url": "https://example.org/a%00b.pth"},  # NUL from the URL
        {"url": "https://example.org/a%20b.pth"},  # whitespace from the URL
        {"url": "https://example.org/%2e%2e"},
        {"sha256": "C" * 64},
        {"size_bytes": 0},
    ],
)
def test_incomplete_or_unsafe_entries_are_rejected(change: dict[str, Any]) -> None:
    """Given an entry with a missing license, a non-https URL or a path-traversing name,
    When it is validated,
    Then it is rejected.
    """
    data = {k: v for k, v in {**VALID, **change}.items() if v is not None}

    with pytest.raises(ValidationError):
        ModelEntry.model_validate(data)


def test_duplicate_names_are_rejected() -> None:
    """Given two entries with the same name,
    When the lock is validated,
    Then it is rejected (names select the download and the directory).
    """
    with pytest.raises(ValidationError, match="mehrfach"):
        ModelsLock.model_validate({"models": [VALID, {**VALID, "version": "2.0"}]})


# ---------- AC 2: pull only listed models, verify hashes ----------


def test_pull_downloads_and_verifies_the_listed_model(tmp_path: Path, server: FakeServer) -> None:
    """Given a listed model on the (fake) source,
    When `drone-scan models pull` runs,
    Then the file is stored under models/<name>/<version>/ with the verified hash.
    """
    write_lock(tmp_path / "models.lock", entry(server))

    result = pull(tmp_path)

    assert result.exit_code == 0, result.output
    target = tmp_path / "models" / "tiny" / "1.0" / "tiny.pth"
    assert target.read_bytes() == WEIGHTS
    assert files_below(tmp_path / "models") == ["tiny/1.0/tiny.pth"]  # no .part left
    assert "heruntergeladen" in result.output and "SHA-256 geprüft" in result.output


def test_hash_mismatch_aborts_and_leaves_no_file(tmp_path: Path, server: FakeServer) -> None:
    """Given a source whose file differs from the SHA-256 in models.lock,
    When the model is pulled,
    Then the run aborts (exit 1) before the next model, names cause and remedy, keeps no file.
    """
    tampered = entry(server, "tiny")
    server.files["/files/tiny.pth"] = b"tampered"
    write_lock(tmp_path / "models.lock", tampered, entry(server, "other", OTHER))

    result = pull(tmp_path)

    assert result.exit_code == 1
    assert "SHA-256" in result.output and "Abbruch" in result.output
    assert "Abhilfe" in result.output
    assert files_below(tmp_path / "models") == []
    assert server.requests == ["/files/tiny.pth"]  # aborted before "other"


def test_unlisted_model_is_refused_without_any_request(tmp_path: Path, server: FakeServer) -> None:
    """Given a model name that is not in models.lock,
    When it is pulled,
    Then the CLI refuses it (exit 2) and sends no request at all.
    """
    write_lock(tmp_path / "models.lock", entry(server))

    result = pull(tmp_path, "tiny", "unlisted")

    assert result.exit_code == 2
    assert "unlisted" in result.output and "Nicht in models.lock" in result.output
    assert server.requests == []


def test_pull_by_name_fetches_only_that_model(tmp_path: Path, server: FakeServer) -> None:
    """Given two listed models,
    When one is pulled by name and then all,
    Then only the named one is fetched first and the second run fetches only the missing one.
    """
    write_lock(tmp_path / "models.lock", entry(server, "tiny"), entry(server, "other", OTHER))

    assert pull(tmp_path, "other").exit_code == 0
    assert server.requests == ["/files/other.pth"]

    result = pull(tmp_path)

    assert result.exit_code == 0
    assert server.requests == ["/files/other.pth", "/files/tiny.pth"]
    assert "bereits vorhanden" in result.output


def test_corrupt_local_file_is_downloaded_again(tmp_path: Path, server: FakeServer) -> None:
    """Given a local file that no longer matches its hash,
    When the model is pulled,
    Then it is replaced by a verified download; --force reloads even a valid file.
    """
    write_lock(tmp_path / "models.lock", entry(server))
    target = tmp_path / "models" / "tiny" / "1.0" / "tiny.pth"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"broken")

    result = pull(tmp_path)

    assert result.exit_code == 0 and "beschädigt" in result.output
    assert target.read_bytes() == WEIGHTS
    assert pull(tmp_path, "--force").exit_code == 0
    assert len(server.requests) == 2


def test_download_larger_than_size_bytes_is_aborted(tmp_path: Path, server: FakeServer) -> None:
    """Given size_bytes smaller than what the source sends,
    When the model is pulled,
    Then the download is aborted and no file is kept.
    """
    write_lock(tmp_path / "models.lock", entry(server, size_bytes=10))

    result = pull(tmp_path)

    assert result.exit_code == 1 and "size_bytes" in result.output
    assert files_below(tmp_path / "models") == []


def test_http_error_and_unreachable_source_exit_1(tmp_path: Path, server: FakeServer) -> None:
    """Given a 404 resp. an unreachable server,
    When the model is pulled,
    Then it exits 1 with cause and remedy and keeps no partial file.
    """
    missing = entry(server)
    del server.files["/files/tiny.pth"]
    write_lock(tmp_path / "models.lock", missing)

    result = pull(tmp_path)
    assert result.exit_code == 1 and "HTTP 404" in result.output

    server.httpd.shutdown()
    server.httpd.server_close()
    result = pull(tmp_path)
    assert result.exit_code == 1 and "Download fehlgeschlagen" in result.output
    assert files_below(tmp_path / "models") == []


def test_missing_lock_exits_10_and_invalid_lock_exits_1(tmp_path: Path) -> None:
    """Given no resp. an invalid models.lock,
    When `models pull` runs,
    Then it exits 10 (input missing) resp. 1 with the required fields as remedy.
    """
    result = pull(tmp_path)
    assert result.exit_code == 10 and "--lock" in result.output

    (tmp_path / "models.lock").write_text("models:\n  - name: x\n", encoding="utf-8")
    result = pull(tmp_path)
    assert result.exit_code == 1 and "weights_license" in result.output


def test_empty_lock_pulls_nothing(tmp_path: Path) -> None:
    """Given the shipped, empty models.lock,
    When all models are pulled,
    Then nothing is downloaded and the run succeeds.
    """
    write_lock(tmp_path / "models.lock")

    result = pull(tmp_path)

    assert result.exit_code == 0 and "nichts zu laden" in result.output


def test_environment_selects_lock_and_models_dir(
    tmp_path: Path, server: FakeServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given DRONESCAN_MODELS_LOCK and DRONESCAN_MODELS_DIR,
    When `models pull` runs without options,
    Then both are used.
    """
    write_lock(tmp_path / "reg.lock", entry(server))
    monkeypatch.setenv("DRONESCAN_MODELS_LOCK", str(tmp_path / "reg.lock"))
    monkeypatch.setenv("DRONESCAN_MODELS_DIR", str(tmp_path / "weights"))

    result = runner.invoke(app, ["models", "pull"])

    assert result.exit_code == 0, result.output
    assert files_below(tmp_path / "weights") == ["tiny/1.0/tiny.pth"]


# ---------- runtime: offline resolution ----------


def test_resolve_model_returns_only_verified_local_files(tmp_path: Path) -> None:
    """Given a listed model that is missing, corrupt and finally valid,
    When a stage resolves it,
    Then it gets an error with the pull command until the verified file is there.
    """
    lock = ModelsLock.model_validate({"models": [{**VALID, "sha256": sha(WEIGHTS)}]})
    target = model_path(lock.models[0], tmp_path)

    with pytest.raises(ModelNotAvailableError, match="drone-scan models pull tiny"):
        resolve_model("tiny", lock, tmp_path)
    target.parent.mkdir(parents=True)
    target.write_bytes(b"broken")
    assert model_status(lock.models[0], tmp_path) is ModelStatus.CORRUPT
    with pytest.raises(ModelNotAvailableError, match="beschädigt"):
        resolve_model("tiny", lock, tmp_path)
    with pytest.raises(ModelNotAvailableError, match="nicht in models.lock"):
        resolve_model("unlisted", lock, tmp_path)

    target.write_bytes(WEIGHTS)
    assert resolve_model("tiny", lock, tmp_path) == target


def test_models_list_shows_status_and_licenses(tmp_path: Path, server: FakeServer) -> None:
    """Given one missing and one downloaded model,
    When `models list` runs,
    Then it shows status and both licenses without any request.
    """
    write_lock(tmp_path / "models.lock", entry(server, "tiny"), entry(server, "other", OTHER))
    assert pull(tmp_path, "other").exit_code == 0
    server.requests.clear()

    result = runner.invoke(
        app,
        [
            "models",
            "list",
            "--lock",
            str(tmp_path / "models.lock"),
            "--models-dir",
            str(tmp_path / "models"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "tiny 1.0: fehlt, Gewichte Apache-2.0, Code MIT" in result.output
    assert "other 1.0: ok" in result.output
    assert server.requests == []


def test_redirect_to_an_insecure_url_is_refused(tmp_path: Path, server: FakeServer) -> None:
    """Given a source that redirects to plain http on another host,
    When the model is pulled,
    Then the redirect is not followed and the run aborts without a file.
    """
    write_lock(tmp_path / "models.lock", entry(server))
    server.redirects["/files/tiny.pth"] = "http://example.org/tiny.pth"

    result = pull(tmp_path)

    assert result.exit_code == 1 and "Weiterleitung" in result.output
    assert files_below(tmp_path / "models") == []


def test_redirect_to_an_allowed_url_is_followed(tmp_path: Path, server: FakeServer) -> None:
    """Given a source that redirects to an allowed URL (like a CDN),
    When the model is pulled,
    Then the file is downloaded from there and verified.
    """
    write_lock(tmp_path / "models.lock", entry(server))
    server.files["/cdn/blob"] = server.files.pop("/files/tiny.pth")
    server.redirects["/files/tiny.pth"] = server.url("/cdn/blob")

    assert pull(tmp_path).exit_code == 0
    assert server.requests == ["/files/tiny.pth", "/cdn/blob"]


def test_broken_http_response_is_reported(tmp_path: Path, server: FakeServer) -> None:
    """Given a server whose response ends before the announced length,
    When the model is pulled,
    Then it exits 1 with cause and remedy instead of a traceback.
    """
    write_lock(tmp_path / "models.lock", entry(server))
    server.truncate = True

    result = pull(tmp_path)

    assert result.exit_code == 1 and "Download fehlgeschlagen" in result.output
    assert files_below(tmp_path / "models") == []
