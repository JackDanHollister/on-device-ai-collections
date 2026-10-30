#!/usr/bin/env python3
"""Thor owner of the fixed GB10 country run; also provides safe lifecycle commands."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

RUN = '20260913_esites_country_24h_v2'
CLIENT = 'tdwg-esites-country-24h-v2.service'
GUARD = 'tdwg-esites-country-24h-v2-guard.service'
LOCK = Path('/path/to/model-service/.nemotron-accuracy-service.lock')
READY = Path('/run/user/' + str(os.getuid()) + '/tdwg-esites-country-24h-v2.ready')
SSH = ['ssh', '-i', '/path/to/ssh-key', '-o', 'IdentitiesOnly=yes',
       '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '-o', 'ServerAliveInterval=5',
       '-o', 'ServerAliveCountMax=2', 'user@gb10-host']
PROBE = r'''
import fcntl,json,math,re,subprocess,urllib.request
from pathlib import Path
root=Path('/path/to/gb10-workspace/outputs/20260913_esites_country_24h_v2')
state=dict(line.split('=',1) for line in subprocess.run(['systemctl','--user','show',
    'tdwg-esites-country-24h-v2.service','-p','ActiveState','-p','SubState','-p','MainPID',
    '-p','Result'],capture_output=True,text=True,timeout=10).stdout.splitlines() if '=' in line)
idle=False
try:
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open('http://127.0.0.1:8000/metrics',timeout=10) as response:
        raw=response.read(4*1024*1024+1)
    seen=[]
    for line in raw.decode().splitlines():
        match=re.fullmatch(r'vllm:num_requests_(running|waiting)(?:\{[^}]*\})?\s+(\S+)',line)
        if match: seen.append((match[1],float(match[2])))
    idle=(len(raw)<=4*1024*1024 and {k for k,v in seen}=={'running','waiting'}
          and all(math.isfinite(v) and v==0 for k,v in seen))
except Exception: pass
free=False
lock=Path('/path/to/gb10-workspace/work/.nemotron-client.lock')
if not any(p.is_symlink() for p in (lock,*lock.parents)):
    with lock.open('r+') as stream:
        try:
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
            free=True
        except BlockingIOError: pass
progress=None
if (root/'progress.json').is_file():
    progress=json.loads((root/'progress.json').read_text())
    if progress.get('contains_record_values') is not False or progress.get('contract')!='esites-country-24h-thinking-v2':
        raise RuntimeError('UNSAFE_PROGRESS')
print(json.dumps({'service':state,'queue_idle':idle,'client_lock_free':free,
    'stopped_and_drained':state.get('ActiveState') in {'inactive','failed'} and state.get('MainPID')=='0' and idle and free,
    'progress':progress}))
'''


def remote(command=None, code=None):
    return subprocess.run(SSH + ([command] if command else ['/usr/bin/python3 -']), input=code,
        capture_output=True, text=True, timeout=50, check=True)


def guard():
    os.umask(0o077)
    if os.getuid() != 1003 or any(p.is_symlink() for p in (LOCK, *LOCK.parents, READY)):
        raise RuntimeError('GUARD_PATH_INVALID')
    with LOCK.open('r+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # An old ready file is recoverable only after the actual flock is acquired.
        READY.write_text(json.dumps({'pid': os.getpid(), 'run': RUN}) + '\n')
        try:
            # systemctl start is idempotent for an already active/restarting client.
            remote('systemctl --user reset-failed ' + CLIENT + '; systemctl --user start ' + CLIENT)
            while True:
                try:
                    value = json.loads(remote(code=PROBE).stdout)
                    if value['stopped_and_drained']:
                        print('CLIENT_STOPPED_AND_DRAINED', flush=True)
                        return
                except (subprocess.SubprocessError, OSError, ValueError):
                    pass
                time.sleep(10)
        finally:
            READY.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('start', 'resume', 'status', 'stop', 'guard'))
    action = parser.parse_args().action
    if action == 'guard':
        guard()
    elif action == 'status':
        print(remote(code=PROBE).stdout.strip())
        subprocess.run(['systemctl', '--user', 'show', GUARD, '-p', 'ActiveState', '-p', 'MainPID'], check=True)
    elif action == 'stop':
        remote('systemctl --user stop ' + CLIENT)
        print('Client stop requested; guard will wait for queue drain.')
    else:
        state = subprocess.run(['systemctl', '--user', 'show', GUARD, '-p', 'LoadState', '--value'],
                               capture_output=True, text=True, check=True).stdout.strip()
        if state == 'loaded':
            subprocess.run(['systemctl', '--user', 'start', GUARD], check=True)
        else:
            subprocess.run(['systemd-run', '--user', '--unit=' + GUARD.removesuffix('.service'),
                '--property=Restart=on-failure', '--property=RestartSec=15',
                '/usr/bin/python3', '-B', str(Path(__file__).resolve()), 'guard'], check=True)


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, lambda *args: (_ for _ in ()).throw(SystemExit(1)))
    main()
