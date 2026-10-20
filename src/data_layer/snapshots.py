"""Publish validated local research versions through an atomic current pointer."""
import json
import os
from pathlib import Path
from uuid import uuid4


def publish_snapshot(version, current):
    version, current = Path(version).resolve(), Path(current).absolute()
    summary = json.loads((version / "summary.json").read_text())
    checks = json.loads((version / "research_checks.json").read_text())
    qa = checks["qa_checks"]
    if not summary.get("complete") or qa["passed"] != summary["companies"] or not qa["early_date_abstains"] or not qa["unknown_company_abstains"]:
        raise ValueError("Research snapshot failed publication checks")
    if not (version / "recent_update.json").exists() or not (version / "filings.sqlite").exists():
        raise ValueError("Research snapshot is incomplete")
    if version == current.resolve() or current in version.parents:
        raise ValueError("Publish a separate immutable version directory")
    current.parent.mkdir(parents=True, exist_ok=True)
    lock = current.parent / ".research_publish.lock"
    # Never remove another publisher's lock or publish over a concurrent run.
    with lock.open("x") as marker:
        marker.write(str(version))
    pointer = current.parent / f".research_pointer_{uuid4().hex}"
    legacy = None
    try:
        pointer.symlink_to(version, target_is_directory=True)
        if current.exists() and not current.is_symlink():
            # One-time local migration. No deployed service uses this directory.
            legacy = current.parent / f"research_legacy_{uuid4().hex}"
            os.replace(current, legacy)
        try:
            os.replace(pointer, current)
        except Exception:
            if legacy is not None:
                os.replace(legacy, current)
            raise
    finally:
        pointer.unlink(missing_ok=True)
        lock.unlink(missing_ok=True)
    return {"current": str(current), "version": str(version), "legacy_preserved": str(legacy) if legacy else None}
