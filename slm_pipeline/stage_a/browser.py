"""Headless-browser rendering fallback for JavaScript-only pages.

Some target sites are single-page apps whose server response contains no
content text.  When a system browser is available, the pipeline renders such
pages with Chrome headless and extracts from the rendered DOM.  The rendered
DOM is stored back into the download cache, so later runs extract without
launching the browser again.

Chrome must be run in its own process group and killed as a group on timeout:
the main process dying does not stop its renderer/sandbox children, and those
orphans hold the output pipes open, which would deadlock the caller.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
from pathlib import Path

# A real browser UA: several SPA hosts (OSF, Kaggle) serve 403 /empty shells
# to unknown user agents.
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0.0.0 Safari/537.36"
)

_CHROMIUM_FAMILY = (
    "google-chrome",
    "google-chrome-stable",
    "google-chrome-beta",
    "chromium",
    "chromium-browser",
    "chrome",
    "msedge",
    "brave-browser",
)


def find_chrome(binary: str | None = None) -> str | None:
    """Return an executable path for a Chromium-family browser, else None."""
    if not binary:
        for name in _CHROMIUM_FAMILY:
            found = shutil.which(name)
            if found:
                return found
        return None
    found = shutil.which(binary)
    if found:
        return found
    # Allow an absolute path that is not on PATH.
    path = Path(binary)
    if path.is_file() and os.access(str(path), os.X_OK):
        return str(path)
    return None


def render_page(
    url: str,
    *,
    chrome_path: str | None,
    timeout_seconds: float = 45.0,
    virtual_time_budget: float = 15.0,
) -> str:
    """Render a page with headless Chrome and return the DOM, or "" on failure."""
    exe = chrome_path or find_chrome()
    if exe is None or not url.lower().startswith(("http://", "https://")):
        return ""
    command = [
        exe,
        "--headless=new",
        "--disable-gpu",
        "--disable-dev-shm-usage",
        "--disable-blink-features=AutomationControlled",
        "--no-first-run",
        "--no-default-browser-check",
        "--mute-audio",
        "--hide-scrollbars",
        f"--user-agent={BROWSER_USER_AGENT}",
        "--window-size=1280,2000",
        f"--virtual-time-budget={int(virtual_time_budget * 1000)}",
        f"--timeout={int(timeout_seconds * 1000)}",
        "--incognito",
        "--noerrdialogs",
        "--dump-dom",
        url,
    ]

    # start_new_session makes the browser a session/group leader, so the whole
    # tree (renderers, crashpad, sandbox helpers) can be killed together.
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
    except OSError:  # missing executable etc.
        return ""
    hard_deadline = timeout_seconds + 20
    try:
        html, _ = process.communicate(timeout=hard_deadline)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            process.kill()
        try:
            html, _ = process.communicate(timeout=15)
        except (subprocess.TimeoutExpired, ValueError) as exc2:  # pragma: no cover
            print(f"  browser render: cleanup failed for {url}: {exc2!r}", flush=True)
            return ""
    except OSError:
        return ""
    html = (html or "").strip()
    if not html or "<html" not in html[:2000].lower():
        return ""
    return html
