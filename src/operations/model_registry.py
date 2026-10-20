"""Immutable approved-model versions and atomic local activation/rollback."""
from contextlib import contextmanager
from hashlib import sha256
import os
from pathlib import Path
import re
import shutil
from uuid import uuid4

from src.ml.release import validate_release
from src.ml.artifacts import load_model
from src.ml.risk_contracts import target_definition


class ModelRegistry:
    def __init__(self, root):
        self.root = Path(root).absolute()
        self.versions = self.root / 'versions'
        self.versions.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def lock(self):
        path = self.root / '.registry.lock'
        with path.open('x') as handle:
            handle.write('Registry update in progress')
        try:
            yield
        finally:
            path.unlink(missing_ok=True)

    def register(self, artifact, release):
        payload, approval = Path(artifact).read_bytes(), Path(release).read_bytes()
        version = sha256(payload).hexdigest()[:16] + '-' + sha256(approval).hexdigest()[:8]
        with self.lock():
            destination = self.versions / version
            if destination.exists():
                validate_release(destination / 'distress.release.json', destination / 'distress.joblib')
                return version
            staged = self.versions / ('.staged-' + uuid4().hex)
            staged.mkdir()
            try:
                (staged / 'distress.joblib').write_bytes(payload)
                (staged / 'distress.release.json').write_bytes(approval)
                approval = validate_release(staged / 'distress.release.json', staged / 'distress.joblib')
                model = load_model(staged / 'distress.joblib', expected_sha256=approval['artifact_sha256'])
                target_definition('distress', model['target'])
                if list(model['model'].classes_) != [0, 1]:
                    raise ValueError('Registry requires a binary bankruptcy model')
                staged.rename(destination)
            finally:
                if staged.exists():
                    shutil.rmtree(staged)
        return version

    def activate(self, version):
        if not re.fullmatch(r'[a-f0-9]{16}-[a-f0-9]{8}', version):
            raise ValueError('Use a registered immutable version ID')
        with self.lock():
            target = (self.versions / version).resolve()
            if target.parent != self.versions.resolve():
                raise ValueError('Version escaped the registry')
            validate_release(target / 'distress.release.json', target / 'distress.joblib')
            actual_id = sha256((target / 'distress.joblib').read_bytes()).hexdigest()[:16] + '-' + sha256((target / 'distress.release.json').read_bytes()).hexdigest()[:8]
            if actual_id != version:
                raise ValueError('Immutable registered version changed')
            current = self.root / 'current'
            if current.exists() and not current.is_symlink():
                raise ValueError('Current model pointer must be a managed symlink')
            pointer = self.root / ('.pointer-' + uuid4().hex)
            try:
                pointer.symlink_to(target, target_is_directory=True)
                os.replace(pointer, current)
            finally:
                pointer.unlink(missing_ok=True)
        return {'active_version': version, 'registry_path': str(current)}

    def rollback(self, version):
        return self.activate(version)  # Older releases must still pass today's policy.


if __name__ == '__main__':
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    commands = parser.add_subparsers(dest='action', required=True)
    register = commands.add_parser('register')
    register.add_argument('--artifact', required=True)
    register.add_argument('--release', required=True)
    for action in ['activate', 'rollback']:
        commands.add_parser(action).add_argument('--version', required=True)
    args = parser.parse_args()
    registry = ModelRegistry(args.root)
    result = ({'registered_version': registry.register(args.artifact, args.release)} if args.action == 'register' else
              getattr(registry, args.action)(args.version))
    print(json.dumps(result, indent=2))
