#!/usr/bin/env python3
"""Run source-only HRM checks, with no datasets, GPU allocation or training."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
TASK_REL = Path('rsi-tasks/signature-tasks/hrm_text_pretraining')
TASK = ROOT / TASK_REL
HRM_REVISION = 'aaa948ea674fd84b7bc455c9cfb455ecfefdf914'
UNITTEST = '''
import json, sys, unittest
suite = unittest.defaultTestLoader.discover(sys.argv[1], pattern=sys.argv[2])
result = unittest.TextTestRunner(verbosity=2).run(suite)
print('HRM_TEST_RESULT=' + json.dumps({'tests': result.testsRun, 'skipped': len(result.skipped)}))
sys.exit(not result.wasSuccessful() or bool(result.skipped) or not result.testsRun)
'''


def pinned_checkout(path, revision):
    path = path.resolve()
    git = ['git', '-c', f'safe.directory={path}', '-C', str(path)]
    head = subprocess.check_output([*git, 'rev-parse', 'HEAD'], text=True).strip()
    if head != revision:
        raise ValueError(f'{path}: expected upstream revision {revision}, found {head}')
    subprocess.run([*git, 'diff', '--exit-code', 'HEAD', '--'], check=True)
    return path


def run_check(name, command, cwd, env):
    start = time.monotonic()
    try:
        result = subprocess.run(command, cwd=cwd, env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=600)
        record = {'name': name, 'passed': result.returncode == 0,
                  'seconds': round(time.monotonic() - start, 3), 'output': result.stdout}
        match = re.search(r'^HRM_TEST_RESULT=(.+)$', result.stdout, re.MULTILINE)
        if match:
            record.update(json.loads(match[1]))
    except subprocess.TimeoutExpired:
        record = {'name': name, 'passed': False, 'seconds': 600, 'output': 'Check timed out.'}
    print(f"{'PASS' if record['passed'] else 'FAIL'} {name} ({record['seconds']:.1f}s)", flush=True)
    if not record['passed']:
        print(record['output'], flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--static-only', action='store_true', help='Skip CPU tests and upstream checkouts')
    parser.add_argument('--hrm-source', type=Path, help='Clean pinned HRM-Text checkout')
    parser.add_argument('--data-io-source', type=Path, help='Clean pinned data_io checkout')
    parser.add_argument('--report', type=Path, help='Write results and tested source hashes as JSON')
    args = parser.parse_args()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
               PYTHONDONTWRITEBYTECODE='1')
    # sudo may reset PATH; launch.sh and static checks must use this interpreter.
    env['PATH'] = str(Path(sys.executable).parent) + os.pathsep + env.get('PATH', '')
    env.pop('HRM_ENFORCE_PROFILE', None)
    if not args.static_only:
        if sys.platform != 'linux' or os.geteuid() != 0:
            parser.error('Full CPU checks require Linux root to exercise the actual OS sandbox.')
        if not args.hrm_source or not args.data_io_source:
            parser.error('Full checks require --hrm-source and --data-io-source; see the task README.')
        pins = json.loads((TASK / 'environment/setup/data_sources.json').read_text())
        try:
            hrm = pinned_checkout(args.hrm_source, HRM_REVISION)
            data_io = pinned_checkout(args.data_io_source, pins['data_io_revision'])
        except (ValueError, subprocess.CalledProcessError) as exc:
            parser.error(str(exc))
        env['HRM_INFERENCE_TEST_SOURCE'] = str(hrm / 'simple_inference_engine.py')
        env['HRM_DATA_IO_TEST_ROOT'] = str(data_io)
    records = []
    # Use the existing static-check inventory, including future additions.
    workflow = (ROOT / '.github/workflows/static-checks.yml').read_text()
    scripts = re.findall(r'"[^"\n]+\|(check-[a-z-]+\.sh)"', workflow)
    if not scripts or len(scripts) != len(set(scripts)):
        raise ValueError('Cannot identify the static-check workflow inventory')
    with tempfile.TemporaryDirectory(prefix='hrm-source-checks-') as tmp:
        snapshot = Path(tmp)
        shutil.copytree(TASK, snapshot / TASK_REL,
                        ignore=shutil.ignore_patterns('staged', '__pycache__', '*.pyc'))
        with ThreadPoolExecutor(max_workers=4) as pool:
            records.extend(pool.map(
                lambda script: run_check(script, ['bash', str(ROOT / 'checks' / script), str(TASK_REL)],
                                         snapshot, env), scripts))
    if not args.static_only:
        suites = [
            ('evaluator and profiles', TASK / 'tests', 'test_*.py'),
            ('training tools', TASK / 'environment/task-tools', 'test_*.py'),
            ('data staging', TASK / 'environment/setup', 'test_*.py'),
            ('timeout policy', ROOT / 'checks', 'test_task_timeout.py'),
        ]
        for name, directory, pattern in suites:
            records.append(run_check(name, [sys.executable, '-c', UNITTEST, str(directory), pattern], ROOT, env))
    if args.report:
        sources = [p for p in TASK.rglob('*') if p.is_file()
                   and not {'staged', '__pycache__', 'validation'}.intersection(p.relative_to(TASK).parts)
                   and p.suffix != '.pyc']
        sources += [ROOT / 'checks/check_hrm_text.py', ROOT / 'checks/check-task-timeout.sh',
                    ROOT / 'checks/test_task_timeout.py', ROOT / '.github/workflows/hrm-text-checks.yml']
        report = {'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'python': sys.version,
                  'command': [sys.executable, *sys.argv],
                  'packages': {} if args.static_only else {
                      name: importlib.metadata.version(name) for name in
                      ('torch', 'numpy', 'safetensors', 'omegaconf', 'pydantic', 'PyYAML', 'tqdm', 'huggingface_hub')},
                  'gpu_training': False, 'static_only': args.static_only, 'checks': records,
                  'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in sorted(sources)}}
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + '\n')
    failed = sum(not record['passed'] for record in records)
    tests = sum(record.get('tests', 0) for record in records)
    print(f'{len(records) - failed}/{len(records)} checks passed; {tests} CPU tests, no skips permitted.')
    return int(bool(failed))


if __name__ == '__main__':
    raise SystemExit(main())
