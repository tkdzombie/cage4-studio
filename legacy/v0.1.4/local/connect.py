#!/usr/bin/env python3
"""Run on YOUR LAPTOP from a LOCAL Python interpreter, not SSH interpreter.

Starts dashboard on the Mac (idempotent), creates an SSH tunnel, then opens
local browser. Requires ssh command and an existing working SSH login.
"""
import argparse
import shutil
import socket
import subprocess
import sys
import time
import webbrowser


def is_ready(host, port):
    try:
        with socket.create_connection((host, port), timeout=0.4):
            return True
    except OSError:
        return False


def main():
    p = argparse.ArgumentParser(description='Open the CAGE4 Studio dashboard via SSH')
    p.add_argument('--host', required=True, help='Mac Studio hostname or IP')
    p.add_argument('--user', required=True, help='Mac Studio ssh user')
    p.add_argument('--port', type=int, default=22, help='SSH port')
    p.add_argument('--identity-file', default=None)
    p.add_argument('--remote-dir', default='cage4-studio',
                   help='Directory directly under remote user home')
    p.add_argument('--local-port', type=int, default=8501)
    p.add_argument('--tensorboard-port', type=int, default=6006)
    args = p.parse_args()
    if not shutil.which('ssh'):
        sys.exit('OpenSSH client not found on your laptop.')
    if not all(1 <= x <= 65535 for x in (args.port, args.local_port, args.tensorboard_port)):
        sys.exit('Invalid port value.')
    if any(x in args.remote_dir for x in ('/', '\\', '..', ' ', '\n', ';', '&', '$', '`')):
        sys.exit('remote-dir must be a simple directory name under remote home')
    dest = f'{args.user}@{args.host}'
    base = ['ssh', '-p', str(args.port), '-o', 'ConnectTimeout=10']
    if args.identity_file:
        base += ['-i', args.identity_file]
    print('Checking dashboard on Mac Studio ...')
    # Remote command is fixed except for strictly checked remote directory.
    startup = base + ['-o', 'BatchMode=yes', dest,
                      f'cd ~/{args.remote_dir} && ./start.sh']
    result = subprocess.run(startup, capture_output=True, text=True, timeout=40)
    if result.returncode:
        sys.exit('Remote launch failed. Check SSH key and path.\n' +
                 result.stdout + '\n' + result.stderr)
    print(result.stdout.strip())
    if is_ready('127.0.0.1', args.local_port):
        sys.exit(f'Local port {args.local_port} is already in use; choose another --local-port')
    if is_ready('127.0.0.1', args.tensorboard_port):
        sys.exit(f'Local port {args.tensorboard_port} is already in use; choose another --tensorboard-port')
    command = base + [
        '-N', '-o', 'BatchMode=yes', '-o', 'ExitOnForwardFailure=yes',
        '-L', f'{args.local_port}:127.0.0.1:8501',
        '-L', f'{args.tensorboard_port}:127.0.0.1:6006', dest,
    ]
    print('Creating SSH tunnel (Ctrl+C closes this tunnel) ...')
    proc = subprocess.Popen(command)
    try:
        for _ in range(45):
            if proc.poll() is not None:
                sys.exit(f'SSH tunnel exited ({proc.returncode})')
            if is_ready('127.0.0.1', args.local_port):
                break
            time.sleep(0.2)
        else:
            sys.exit('SSH tunnel not ready; see SSH authentication/server logs')
        url = f'http://127.0.0.1:{args.local_port}'
        print('Opening', url)
        webbrowser.open_new(url)
        print('TensorBoard:', f'http://127.0.0.1:{args.tensorboard_port}')
        proc.wait()
    except KeyboardInterrupt:
        print('\nClosing SSH tunnel ...')
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == '__main__':
    main()
