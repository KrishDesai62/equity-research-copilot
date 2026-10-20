"""Coordinate new public acquisition jobs at at most three request starts/second."""
import fcntl
from pathlib import Path
import time

from src.validation.download import fetch

STATE = Path('data/raw/.public_acquisition_rate')


def coordinated_fetch(url, destination):
    if not Path(destination).exists():
        STATE.parent.mkdir(parents=True, exist_ok=True)
        with STATE.open('a+') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            handle.seek(0)
            value = handle.read().strip()
            now = time.monotonic()
            previous = float(value) if value else 0.
            if previous > now + 1:  # Host reboot: monotonic origin changed.
                previous = 0.
            time.sleep(max(0., .34 - (now - previous)))
            handle.seek(0)
            handle.truncate()
            handle.write(str(time.monotonic()))
            handle.flush()
            fcntl.flock(handle, fcntl.LOCK_UN)
    return fetch(url, destination)
