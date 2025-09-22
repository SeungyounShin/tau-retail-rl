#!/usr/bin/env python3
import re
from pathlib import Path

LOG_PATH = Path("train.log")
CLEAN_PATH = Path("train.clean.log")
ERROR_PATH = Path("train.error.txt")


ANSI_RE = re.compile(
    r"\x1B\[[0-?]*[ -/]*[@-~]"  # CSI sequences
    r"|\x1B[@-Z\\-_]"           # 2-byte sequences
)


def strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s)


def main():
    if not LOG_PATH.exists():
        raise SystemExit(f"No {LOG_PATH} found in repo root.")

    text = LOG_PATH.read_text(errors="replace")

    # 1) Write cleaned log without ANSI codes
    clean = strip_ansi(text)
    CLEAN_PATH.write_text(clean)

    # 2) Extract last traceback/error block for quick inspection
    # Find common error markers near the end
    markers = [
        "Traceback (most recent call last):",
        "ray.exceptions.RayTaskError",
        "RuntimeError:",
    ]

    last_pos = -1
    for m in markers:
        pos = clean.rfind(m)
        last_pos = max(last_pos, pos)

    if last_pos != -1:
        # Grab from last marker to the end, but cap size to keep it readable
        tail = clean[last_pos:]
        # Also include up to 800 lines of context preceding the marker
        pre_start = clean.rfind("\n", 0, last_pos)
        for _ in range(800):
            if pre_start <= 0:
                break
            pre_start = clean.rfind("\n", 0, pre_start)
        snippet = clean[max(0, pre_start):]
        ERROR_PATH.write_text(snippet)
    else:
        ERROR_PATH.write_text("No error markers found in log.\n")

    print(f"Wrote: {CLEAN_PATH} and {ERROR_PATH}")


if __name__ == "__main__":
    main()

