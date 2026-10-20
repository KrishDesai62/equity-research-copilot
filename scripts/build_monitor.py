"""Build a lean Linux Python 3.12 Lambda ZIP without the ML/RAG dependencies.

Run with your project's Python: python scripts/build_monitor.py
Requires internet access to download runtime wheels. Does not deploy anything.
"""
import argparse
from hashlib import sha256
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import zipfile


ROOT = Path(__file__).resolve().parents[1]
MODULES = (
    "src/__init__.py", "src/aws/__init__.py", "src/aws/s3_cache.py",
    "src/aws/filing_monitor.py", "src/data_layer/__init__.py",
    "src/data_layer/edgar_ingest.py", "src/data_layer/filing_changes.py",
    "src/operations/__init__.py", "src/operations/status.py",
)


def build(output):
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="monitor-build-") as temporary:
        stage = Path(temporary)
        subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
            "--platform", "manylinux2014_x86_64", "--implementation", "cp", "--python-version", "3.12",
            "--abi", "cp312", "--only-binary=:all:", "--no-compile", "--target", str(stage),
            "-r", str(ROOT / "infra/aws/monitor-requirements.txt")], check=True)
        for module in MODULES:
            target = stage / module
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / module, target)
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(stage.rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                    archive.write(path, path.relative_to(stage))
    print(f"Built {output} ({output.stat().st_size:,} bytes)")
    print(f"SHA256: {sha256(output.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT / "build/filing-monitor.zip"))
    build(parser.parse_args().output)
