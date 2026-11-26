# SPDX-License-Identifier: AGPL-3.0-or-later
"""``drone-scan models pull`` (KAN-24): the ONLY network access of drone-scan (CLAUDE.md rule 3).

Downloads exactly the weight files listed in ``models.lock`` and verifies their SHA-256 while
streaming. A mismatch aborts the whole run and leaves no file behind; only a verified file is
moved into place (atomic rename). Nothing else in ``src/`` may import network modules
(``tests/test_offline.py``).
"""

from __future__ import annotations

import hashlib
import http.client
import os
import ssl
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import IO

from dronescan import __version__
from dronescan.common.models_lock import (
    ModelEntry,
    ModelStatus,
    is_allowed_url,
    model_path,
    model_status,
)

DEFAULT_TIMEOUT_S = 60.0  # per socket operation, not for the whole file
_CHUNK_BYTES = 1024 * 1024


class PullError(RuntimeError):
    """Download failed or the file does not match models.lock; message names cause and remedy."""

    def __init__(self, message: str, code: str = "models.download_failed") -> None:
        super().__init__(message)
        self.code = code  # id from error_catalog.yaml


class PullResult(StrEnum):
    DOWNLOADED = "heruntergeladen"
    PRESENT = "bereits vorhanden"


class _HttpsOnlyRedirects(urllib.request.HTTPRedirectHandler):
    """Follow redirects (e.g. to a CDN) only to URLs models.lock itself would accept."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: http.client.HTTPMessage,
        newurl: str,
    ) -> urllib.request.Request | None:
        if not is_allowed_url(newurl):
            raise PullError(
                f"Weiterleitung auf eine unsichere Adresse ({newurl.split('?')[0]}) abgelehnt. "
                "Abhilfe: Quelle prüfen; nur https-Downloads sind erlaubt."
            )
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _write_error(entry: ModelEntry, exc: OSError) -> PullError:
    return PullError(
        f"{entry.name}: Datei im Modellverzeichnis nicht les- oder schreibbar "
        f"({exc.strerror or exc}). Abhilfe: freien Speicherplatz und Schreibrechte prüfen "
        "(--models-dir bzw. DRONESCAN_MODELS_DIR).",
        code="models.write_failed",
    )


def pull_model(
    entry: ModelEntry,
    models_dir: Path,
    *,
    force: bool = False,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    progress: Callable[[str], None] | None = None,
) -> PullResult:
    """Download and verify one model; a verified existing file is kept unless ``force``."""
    target = model_path(entry, models_dir)
    try:
        status = model_status(entry, models_dir)
        if status is ModelStatus.OK and not force:
            return PullResult.PRESENT
        if status is ModelStatus.CORRUPT and progress:
            progress(f"{entry.name}: vorhandene Datei ist beschädigt und wird neu geladen.")
        target.parent.mkdir(parents=True, exist_ok=True)
        # Unique name: parallel pulls of the same model must not share a partial file.
        handle, name = tempfile.mkstemp(
            dir=target.parent, prefix=f".{target.name}.", suffix=".part"
        )
    except OSError as exc:
        raise _write_error(entry, exc) from exc
    part = Path(name)
    try:
        with os.fdopen(handle, "wb") as out:
            digest = _download(entry, out, timeout_s)
        if digest != entry.sha256:
            raise PullError(
                f"{entry.name}: SHA-256 der heruntergeladenen Datei ({digest}) weicht von "
                f"models.lock ({entry.sha256}) ab. Download verworfen, Abbruch. Abhilfe: später "
                "erneut versuchen; tritt es wieder auf, hat sich die Datei an der Quelle "
                "geändert und der Eintrag muss neu geprüft werden (PO, license-auditor).",
                code="models.hash_mismatch",
            )
        os.replace(part, target)
    except OSError as exc:
        raise _write_error(entry, exc) from exc
    finally:
        part.unlink(missing_ok=True)
    return PullResult.DOWNLOADED


def _download(entry: ModelEntry, out: IO[bytes], timeout_s: float) -> str:
    request = urllib.request.Request(entry.url, headers={"User-Agent": f"drone-scan/{__version__}"})
    opener = urllib.request.build_opener(_HttpsOnlyRedirects)
    digest = hashlib.sha256()
    size = 0
    try:
        with opener.open(request, timeout=timeout_s) as response:
            while chunk := response.read(_CHUNK_BYTES):
                size += len(chunk)
                if entry.size_bytes is not None and size > entry.size_bytes:
                    raise PullError(
                        f"{entry.name}: Download ist größer als size_bytes in models.lock "
                        f"({entry.size_bytes} Bytes). Abbruch. Abhilfe: Quelle prüfen, der "
                        "Eintrag muss neu geprüft werden (PO, license-auditor)."
                    )
                digest.update(chunk)
                out.write(chunk)
            if getattr(response, "length", None):  # connection ended before Content-Length
                raise http.client.IncompleteRead(b"", response.length)
    except urllib.error.HTTPError as exc:
        raise PullError(
            f"{entry.name}: Server antwortet mit HTTP {exc.code}. Abhilfe: URL in models.lock "
            "prüfen; ist die Datei umgezogen, muss der Eintrag neu geprüft werden."
        ) from exc
    except (
        urllib.error.URLError,
        http.client.HTTPException,
        ssl.SSLError,
        TimeoutError,
        ConnectionError,
    ) as exc:
        reason = getattr(exc, "reason", None) or exc
        raise PullError(
            f"{entry.name}: Download fehlgeschlagen ({reason}). Abhilfe: Internetverbindung "
            "und Proxy prüfen, dann erneut 'drone-scan models pull' ausführen."
        ) from exc
    return digest.hexdigest()
