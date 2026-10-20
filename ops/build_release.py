#!/usr/bin/env python3
"""Create UTF-8 source ZIP without runtime/data/metadata or symlink escapes."""
import argparse
import hashlib
from pathlib import Path
import zipfile
VERSION='0.4.0'
SKIP={'settings.local','.git','__pycache__','.DS_Store','__MACOSX','runs','test-runs','acceptance','logs','.conda-env','.venv'}
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 root=Path(__file__).resolve().parents[1];a.output.mkdir(parents=True,exist_ok=True)
 target=a.output/f'CAGE4_Studio_v{VERSION}_Source.zip'
 with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
  for file in sorted(root.rglob('*')):
   rel=file.relative_to(root)
   if any(v in SKIP or v.endswith('.egg-info') for v in rel.parts):continue
   if file.is_symlink():raise ValueError('Symlink in source: '+str(file))
   if not file.is_file() or file.suffix in ('.pyc','.pt','.log','.zip','.tgz','.pem','.key'):continue
   if not str(rel).isascii():raise ValueError('Non-ASCII archive filename: '+str(rel))
   if file.name.startswith('._'):continue
   z.write(file,f'CAGE4_Studio_v{VERSION}/{rel}')
 with zipfile.ZipFile(target) as z:
  assert z.testzip() is None
 print(target)
 print(hashlib.sha256(target.read_bytes()).hexdigest())
if __name__=='__main__':main()
