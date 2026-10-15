"""OS isolation boundary for executing generated Over-Editing repairs.

The upstream evaluator runs ``code_utils.untrusted_check`` inside its Docker
service. OpenCompass does not own such a service, so this adapter gives the
byte-identical executor an equivalent local container boundary with
bubblewrap. There is deliberately no unsafe in-process fallback.
"""

import hashlib
import json
import site
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Iterable, List, Tuple

from . import code_utils

_EXECUTOR_DEPENDENCIES = {'matplotlib', 'numpy'}
_NLTK_DATA_DIR = Path(__file__).resolve().parent / 'nltk_data'
_NLTK_RESOURCES = ('corpora/stopwords', )
_NLTK_RESOURCE_FILES = {
    'corpora/stopwords/english': (
        'f6d005956f407dbc6ea32e5ff0c7e8e6f71488d3239b9023efdc7fc139d6375b'),
}
_SANDBOX_LAUNCH_LOCK = Lock()


def _bind_if_present(command, path):
    if Path(path).exists():
        command.extend(['--ro-bind', path, path])


def _validated_nltk_data_dir() -> Path:
    invalid = []
    for relative_path, expected_digest in _NLTK_RESOURCE_FILES.items():
        resource_path = _NLTK_DATA_DIR / relative_path
        if (not resource_path.is_file()
                or hashlib.sha256(resource_path.read_bytes()).hexdigest()
                != expected_digest):
            invalid.append(relative_path)
    if invalid:
        raise RuntimeError(
            'Over-Editing packaged NLTK resources are unavailable: '
            + ', '.join(invalid) + '; reinstall OpenCompass with the '
            'over_editing extra')
    return _NLTK_DATA_DIR


def _sandbox_command(executor_path=None) -> list:
    bwrap = shutil.which('bwrap')
    if not bwrap:
        raise RuntimeError(
            'Over-Editing requires bubblewrap (bwrap) to execute generated '
            'programs in an isolated evaluator boundary')

    package_dir = Path(__file__).resolve().parent
    nltk_data_dir = _validated_nltk_data_dir()
    executor_path = Path(executor_path or package_dir / 'code_utils.py')
    if not executor_path.is_file():
        raise FileNotFoundError(
            f'Over-Editing executor source not found: {executor_path}')
    base_prefix = Path(sys.base_prefix).resolve()
    python = Path(sys.executable).resolve()
    try:
        python_inside = Path('/runtime/base') / python.relative_to(base_prefix)
    except ValueError as error:
        raise RuntimeError(
            'Over-Editing cannot locate the Python executable inside its '
            'base runtime') from error

    site_paths = []
    # ``--system-site-packages`` virtual environments can add dependency
    # trees through .pth files which ``site.getsitepackages()`` does not
    # report. Include only Python package roots from sys.path; this excludes
    # the working checkout even when the launcher prepends it to PYTHONPATH.
    for path in [*site.getsitepackages(), *sys.path]:
        if not path or Path(path).name not in {
                'site-packages', 'dist-packages'}:
            continue
        resolved = Path(path).resolve()
        if resolved.is_dir() and resolved not in site_paths:
            site_paths.append(resolved)
    isolated_site_paths = [
        f'/runtime/site/{index}' for index in range(len(site_paths))
    ]

    # Keep outbound networking as the upstream Docker service does: the
    # frozen BigCodeBench/590 test performs a real HTTPS fetch. Filesystem,
    # environment, IPC, and process namespaces remain isolated.
    command = [
        bwrap,
        '--die-with-parent',
        '--new-session',
        '--unshare-user',
        '--uid', '65534',
        '--gid', '65534',
        '--unshare-pid',
        '--unshare-ipc',
        '--unshare-uts',
        '--clearenv',
        '--setenv', 'HOME', '/tmp',
        '--setenv', 'TMPDIR', '/tmp',
        '--setenv', 'PATH', '/runtime/base/bin:/usr/bin:/bin',
        '--setenv', 'PYTHONPATH', ':'.join(isolated_site_paths),
        '--setenv', 'PYTHONNOUSERSITE', '1',
        '--setenv', 'NLTK_DATA', '/runtime/nltk_data',
        '--setenv', 'MPLCONFIGDIR', '/tmp/matplotlib',
        '--setenv', 'TZ', 'UTC',
        '--dir', '/runtime',
        '--ro-bind', str(base_prefix), '/runtime/base',
        '--ro-bind', str(nltk_data_dir), '/runtime/nltk_data',
        '--dir', '/runtime/site',
        '--dir', '/sandbox',
        '--ro-bind', str(executor_path.resolve()),
        '/sandbox/code_utils.py',
        '--ro-bind', str(package_dir / 'sandbox_worker.py'),
        '/sandbox/sandbox_worker.py',
        '--tmpfs', '/tmp',
        '--tmpfs', '/workspace',
        '--tmpfs', '/home',
        '--tmpfs', '/root',
        '--tmpfs', '/run',
        '--tmpfs', '/var',
        # A private procfs mount is blocked by some container runtimes. An
        # empty mount keeps host process metadata inaccessible and is enough
        # for the pinned executor and selected tests.
        '--tmpfs', '/proc',
        '--tmpfs', '/sys',
        '--dev', '/dev',
        '--tmpfs', '/dev/shm',
        '--dir', '/etc',
    ]
    for index, path in enumerate(site_paths):
        command.extend([
            '--ro-bind', str(path), f'/runtime/site/{index}',
        ])
    # Supply only the interpreter, shared libraries, and installed dependency
    # trees. In particular, never expose the host root or target checkout.
    for path in ('/usr', '/lib', '/lib64'):
        _bind_if_present(command, path)
    for path in ('/etc/ca-certificates.conf', '/etc/hosts',
                 '/etc/nsswitch.conf', '/etc/resolv.conf', '/etc/ssl'):
        _bind_if_present(command, path)
    command.extend([
        '--chdir', '/tmp',
        str(python_inside),
        '/sandbox/sandbox_worker.py',
    ])
    return command


def _run_isolated(request: dict, timeout: float, executor_path=None) -> dict:
    try:
        # The pinned evaluator starts multiprocessing managers and a test
        # process of its own. In constrained containers, concurrent namespace
        # launches have intermittently exited successfully before a protocol
        # result was emitted. Keep that nested lifecycle single-owner; model
        # inference and all non-execution scoring remain concurrent.
        with _SANDBOX_LAUNCH_LOCK:
            completed = subprocess.run(
                _sandbox_command(executor_path),
                input=json.dumps(request),
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
    except subprocess.TimeoutExpired as error:
        raise TimeoutError(
            f'isolated Over-Editing evaluator timed out: {error}') from error
    if completed.returncode:
        cause = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            'isolated Over-Editing evaluator failed: '
            f'{cause or f"exit status {completed.returncode}"}')
    try:
        result = json.loads(completed.stdout)
    except (TypeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            'isolated Over-Editing evaluator returned an invalid result') \
            from error
    if not isinstance(result, dict):
        raise RuntimeError(
            'isolated Over-Editing evaluator returned an invalid result')
    return result


def isolated_untrusted_check(code: str, test_code: str,
                             timeout: float,
                             executor_path=None) -> Tuple[str, dict]:
    # The pinned executor has a 240-second floor and two seconds of process
    # cleanup allowance. The outer boundary is only a final containment cap.
    outer_timeout = max(float(timeout), code_utils.TIMEOUT_LIMIT) + 10
    try:
        result = _run_isolated({
            'mode': 'evaluate',
            'code': code,
            'test_code': test_code,
            'timeout': timeout,
        }, outer_timeout, executor_path=executor_path)
    except TimeoutError as error:
        return 'timeout', {'ALL': str(error)}
    try:
        status = result['status']
        details = result['details']
    except KeyError as error:
        raise RuntimeError(
            'isolated Over-Editing evaluator returned an invalid result') \
            from error
    if status not in {'pass', 'fail', 'timeout'} or not isinstance(
            details, dict):
        raise RuntimeError(
            'isolated Over-Editing evaluator returned an invalid verdict')
    return status, details


def isolated_untrusted_check_many(
        samples: Iterable[Tuple[str, str]], timeout: float,
        executor_path=None) -> List[Tuple[str, dict]]:
    """Run ordered samples through one isolated executor worker.

    Sandbox launches are deliberately serialized because the pinned executor
    creates nested multiprocessing children. Reusing the outer namespace and
    interpreter avoids paying their substantial startup cost once per sample;
    the worker still invokes the pinned ``untrusted_check`` independently for
    every code/test pair, including its original process and timeout handling.
    """
    samples = list(samples)
    if not samples:
        return []
    per_sample_timeout = max(float(timeout), code_utils.TIMEOUT_LIMIT) + 10
    result = _run_isolated({
        'mode': 'evaluate_many',
        'samples': [
            {'code': code, 'test_code': test_code}
            for code, test_code in samples
        ],
        'timeout': timeout,
    }, per_sample_timeout * len(samples), executor_path=executor_path)
    items = result.get('items')
    if not isinstance(items, list) or len(items) != len(samples):
        raise RuntimeError(
            'isolated Over-Editing evaluator returned an invalid batch')
    verdicts = []
    for item in items:
        if not isinstance(item, dict):
            raise RuntimeError(
                'isolated Over-Editing evaluator returned an invalid batch')
        status = item.get('status')
        details = item.get('details')
        if status not in {'pass', 'fail', 'timeout'} or not isinstance(
                details, dict):
            raise RuntimeError(
                'isolated Over-Editing evaluator returned an invalid verdict')
        verdicts.append((status, details))
    return verdicts


@lru_cache(maxsize=8)
def _check_isolated_dependencies(modules: Tuple[str, ...]) -> None:
    # Import each dependency in a fresh child inside the namespace. Keeping
    # tensorflow, scipy, matplotlib, and the web stacks resident together can
    # exceed ordinary evaluator memory limits even though each selected test
    # imports only its own small subset.
    result = _run_isolated({
        'mode': 'dependencies',
        'modules': modules,
        'nltk_resources': _NLTK_RESOURCES,
    }, max(60, len(modules) * 60))
    failures = result.get('failures')
    if not isinstance(failures, dict):
        raise RuntimeError(
            'isolated dependency preflight returned an invalid result')
    if failures:
        reason = '; '.join(
            f'{module}: {cause}' for module, cause in failures.items())
        raise RuntimeError(
            'pinned evaluator dependencies are unavailable inside the '
            f'isolated runtime: {reason}')


def check_isolated_dependencies(modules: Iterable[str]) -> None:
    _check_isolated_dependencies(tuple(sorted(
        set(modules) | _EXECUTOR_DEPENDENCIES)))
