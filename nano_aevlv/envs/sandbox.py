import sys 
import json 
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

# rather biased runner for bin packing problem: todo rework this somehow 
_RUNNER = r'''
import json, math, sys
import numpy as np

req = json.load(sys.stdin)
try:
    ns = {"np": np, "math": math}
    exec(req["code"], ns)
    args = [np.asarray(a) if isinstance(a, list) else a for a in req["args"]]
    out = ns[req["func"]](*args)
    out = out.tolist() if isinstance(out, np.ndarray) else out
    print(json.dumps({"result": out}))
except Exception as e:
    print(json.dumps({"error": f"{type(e).__name__}: {e}"}))
'''

def call_sandboxed(code: str, fn: str, args: list, timeout: float = 10.0):
    '''run `func(*args)` from `code` in a sandbox; lists are passed as numpy arrays.'''
    payload = json.dumps({'code': code, 'func': fn, 'args': args})
    proc = subprocess.run(
        sandboxed([sys.executable, '-I', '-c', _RUNNER]),   # sandboxed() from the previous message
        input=payload, capture_output=True, text=True, timeout=timeout,
    )
    res = json.loads(proc.stdout.strip().splitlines()[-1])
    if 'error' in res:
        raise ValueError(res['error'])
    return res['result']

if __name__ == '__main__': 
    code = '''
def priority(item, bins):
    return -(bins - item)
    '''.strip()
    print(call_sandboxed(code, 'priority', [30.0, [50.0, 35.0, 90.0]]))