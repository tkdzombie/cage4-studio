#!/usr/bin/env python3
"""macOS per-user dashboard launchd lifecycle. Never controls training or TB."""
import argparse
import json
import os
from pathlib import Path
import plistlib
import shutil
import subprocess
import sys
import time
import urllib.request
LABEL='com.cage4studio.dashboard'
ROOT=Path(__file__).resolve().parents[1]

def command(*args,check=True):
    return subprocess.run(list(args),check=check,capture_output=True,text=True)
def domain():
    for name in ('gui','user'):
        d=f'{name}/{os.getuid()}'
        if command('launchctl','print',d,check=False).returncode==0:return d
    raise RuntimeError('No user launchd domain; log in to macOS first')
def loaded(d):return command('launchctl','print',d+'/'+LABEL,check=False).returncode==0
def stop(d):
    if loaded(d):
        command('launchctl','bootout',d+'/'+LABEL)
        # bootout may return before the old job disappears from the domain.
        for _ in range(60):
            if not loaded(d):return
            time.sleep(.5)
        raise RuntimeError('launchd job did not unload; inspect status before retrying')
def start(d,plist):
    if not loaded(d):command('launchctl','bootstrap',d,str(plist))
def health(port):
    for _ in range(30):
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/_stcore/health',timeout=1) as response:
                if response.status==200:return
        except OSError:pass
        time.sleep(.5)
    raise RuntimeError('Dashboard failed health check')
def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['install','start','stop','status','rollback'])
    p.add_argument('--python',type=Path,default=Path.home()/'cage4-studio/.conda-env/bin/python')
    p.add_argument('--runs',type=Path,default=Path.home()/'cage4-experiments')
    p.add_argument('--legacy',type=Path,default=Path.home()/'cage4-studio')
    p.add_argument('--port',type=int,default=8501)
    p.add_argument('--backup',type=Path)
    a=p.parse_args()
    if sys.platform!='darwin':p.error('macOS only')
    if not 1<=a.port<=65535:p.error('Invalid port')
    d=domain();agents=Path.home()/'Library/LaunchAgents';plist=agents/(LABEL+'.plist')
    if a.action=='status':
        result=command('launchctl','print',d+'/'+LABEL,check=False)
        print(result.stdout or result.stderr);return
    if a.action=='stop':stop(d);return
    if a.action=='start':start(d,plist);return
    if a.action=='rollback':
        if not a.backup or not (a.backup/'dashboard.plist').is_file():p.error('--backup with dashboard.plist required')
        stop(d);shutil.copy2(a.backup/'dashboard.plist',plist)
        info=json.loads((a.backup/'state.json').read_text())
        if info['was_loaded']:start(d,plist)
        print('Restored:',a.backup);return
    # Preflight before any service mutation. Reuse existing runtime; no downloads.
    py=a.python.expanduser().absolute()
    if not py.is_file() or not os.access(py,os.X_OK):p.error('Existing Python not executable')
    env=os.environ.copy()
    paths=[ROOT/'protocol',ROOT/'server/src']
    env['PYTHONPATH']=os.pathsep.join(map(str,paths))
    subprocess.run([str(py),'-c','import streamlit,pandas,cage4_server,cage4_protocol; print("runtime OK")'],env=env,check=True)
    # If another manually started server occupies the port, never kill it.
    if not loaded(d):
        import socket
        with socket.socket() as s:
            if s.connect_ex(('127.0.0.1',a.port))==0:
                raise RuntimeError('Port already occupied by unmanaged service; inspect and stop it explicitly')
    backup=Path.home()/'cage4-studio-backups'/time.strftime('%Y%m%d_%H%M%S')
    backup.mkdir(parents=True,exist_ok=False)
    was_loaded=loaded(d)
    if plist.exists():shutil.copy2(plist,backup/'dashboard.plist')
    (backup/'state.json').write_text(json.dumps({'was_loaded':was_loaded,'release':str(ROOT)}))
    logs=Path.home()/'cage4-studio-logs';logs.mkdir(parents=True,exist_ok=True)
    roots=[a.runs.expanduser().absolute(),a.legacy.expanduser().absolute()/'runs']
    roots[0].mkdir(parents=True,exist_ok=True)
    data={'Label':LABEL,'ProgramArguments':[str(py),'-m','streamlit','run',str(ROOT/'server/src/cage4_server/app.py'),
          '--server.address','127.0.0.1','--server.port',str(a.port),'--server.headless','true',
          '--server.fileWatcherType','none','--browser.gatherUsageStats','false'],
          'WorkingDirectory':str(ROOT),'RunAtLoad':True,'KeepAlive':{'SuccessfulExit':False},'ThrottleInterval':15,
          'EnvironmentVariables':{'PYTHONPATH':env['PYTHONPATH'],'PYTHONUNBUFFERED':'1',
                                 'CAGE4_RUN_ROOTS':os.pathsep.join(map(str,roots))},
          'StandardOutPath':str(logs/'dashboard.out.log'),'StandardErrorPath':str(logs/'dashboard.err.log')}
    agents.mkdir(parents=True,exist_ok=True)
    temp=plist.with_suffix('.new')
    try:
        with temp.open('wb') as f:plistlib.dump(data,f)
        temp.chmod(0o600)
        stop(d);os.replace(temp,plist);start(d,plist)
        health(a.port)
    except BaseException:
        stop(d)
        if (backup/'dashboard.plist').exists():
            shutil.copy2(backup/'dashboard.plist',plist)
            if was_loaded:start(d,plist)
        elif plist.exists():plist.unlink()
        raise
    finally:
        if temp.exists():temp.unlink()
    print('Installed:',ROOT,'\nBackup:',backup,'\nDashboard:',f'http://127.0.0.1:{a.port}')
    print('LaunchAgent starts at user login, not before login. Existing TensorBoard and training are untouched.')
if __name__=='__main__':main()
