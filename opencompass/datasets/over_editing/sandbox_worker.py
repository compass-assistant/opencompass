"""Private entry point for the isolated Over-Editing executor."""

import contextlib
import json
import os
import subprocess
import sys

import code_utils


def main():
    request = json.load(sys.stdin)
    if request.get('mode') == 'dependencies':
        failures = {}
        for module in request.get('modules', []):
            try:
                completed = subprocess.run(
                    [sys.executable, '-c',
                     'import importlib, sys; '
                     'importlib.import_module(sys.argv[1])', module],
                    text=True,
                    capture_output=True,
                    timeout=60,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                failures[module] = 'import timed out after 60 seconds'
                continue
            if completed.returncode:
                failures[module] = (
                    completed.stderr.strip() or completed.stdout.strip()
                    or f'exit status {completed.returncode}')
        resource_probes = {
            'corpora/stopwords': (
                'import nltk, sys; '
                'nltk.data.find(sys.argv[1]); '
                'from nltk.corpus import stopwords; '
                'words = stopwords.words("english"); '
                'assert "the" in words and "and" in words'),
        }
        for resource in request.get('nltk_resources', []):
            probe = resource_probes.get(resource)
            if probe is None:
                failures[f'nltk:{resource}'] = 'unknown required resource'
                continue
            try:
                completed = subprocess.run(
                    [sys.executable, '-c', probe, resource],
                    text=True,
                    capture_output=True,
                    timeout=60,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                failures[f'nltk:{resource}'] = (
                    'resource probe timed out after 60 seconds')
                continue
            if completed.returncode:
                failures[f'nltk:{resource}'] = (
                    completed.stderr.strip() or completed.stdout.strip()
                    or f'exit status {completed.returncode}')
        json.dump({'failures': failures}, sys.stdout)
        return
    if request.get('mode') not in {'evaluate', 'evaluate_many'}:
        raise ValueError('unknown isolated evaluator mode')
    # Keep the protocol stream separate from incidental library output. A
    # Python-level redirect is insufficient because generated programs may
    # launch children which inherit and write directly to file descriptor 1.
    # Preserve that descriptor solely for the final protocol record and point
    # stdout at stderr throughout untrusted execution. The pinned executor
    # continues to supply all verdict and timeout logic.
    protocol_fd = os.dup(sys.stdout.fileno())
    try:
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        with contextlib.redirect_stdout(sys.stderr):
            if request['mode'] == 'evaluate':
                status, details = code_utils.untrusted_check(
                    request['code'],
                    request['test_code'],
                    min_time_limit=request['timeout'],
                    gt_time_limit=request['timeout'],
                )
                result = {'status': status, 'details': details}
            else:
                items = []
                for sample in request.get('samples', []):
                    status, details = code_utils.untrusted_check(
                        sample['code'],
                        sample['test_code'],
                        min_time_limit=request['timeout'],
                        gt_time_limit=request['timeout'],
                    )
                    items.append({'status': status, 'details': details})
                result = {'items': items}
        payload = memoryview(json.dumps(result).encode('utf-8'))
        while payload:
            payload = payload[os.write(protocol_fd, payload):]
    finally:
        os.close(protocol_fd)


if __name__ == '__main__':
    main()
