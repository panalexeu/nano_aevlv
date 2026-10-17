import sys 
import json 
import resource 
import subprocess
from pathlib import Path

def sandboxed(cmd: list[str]) -> list[str]:
    '''run cmd with a read-only filesystem, hidden home dir, no network, and a throwaway /tmp.'''
    return [
        'bwrap',
        '--ro-bind', '/', '/',                          # whole filesystem read-only
        '--tmpfs', str(Path.home()),                    # hide your home dir (ssh keys, .env, projects)
        '--ro-bind', sys.prefix, sys.prefix,            # ...but keep the venv (numpy etc.) visible
        '--ro-bind', sys.base_prefix, sys.base_prefix,  # ...and the python it's built on
        '--dev', '/dev',                                # minimal /dev without host disks or gpu
        '--proc', '/proc',                              # own /proc, hides host processes and their env vars
        '--tmpfs', '/tmp',                              # writable scratch, discarded afterwards
        '--unshare-all',                                # no network, own pids/ipc/users
        '--die-with-parent',                            # killed if your search crashes
        '--new-session',
        *cmd,
    ]

def _limits():
    resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))      # 2 GB memory cap

def call_sandboxed(code: str, fn: str, runner: str, timeout: float = 10.0, **kwargs) -> dict:
    '''run `runner` in a sandbox with {code, func, kwargs} on stdin; return its json output.'''
    payload = json.dumps({'code': code, 'func': fn, 'kwargs': kwargs})
    try:
        proc = subprocess.run(
            sandboxed([sys.executable, '-I', '-c', runner]),
            input=payload, capture_output=True, text=True,
            timeout=timeout, preexec_fn=_limits,
        )
    except subprocess.TimeoutExpired:
        raise ValueError(f'timeout after {timeout}s')               # short message for the fail map
    lines = proc.stdout.strip().splitlines()
    if not lines:                                                    # bwrap failed, oom kill, interpreter crash
        raise ValueError(f'sandbox produced no output: {proc.stderr.strip()[-300:]}')
    res = json.loads(lines[-1])
    if 'error' in res:
        raise ValueError(res['error'])
    return res