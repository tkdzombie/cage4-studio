import json
import os
from pathlib import Path
import tempfile
import unittest
from cage4_protocol.store import atomic_json,read_json,RunWriter,run_status
from cage4_server import reader

class StoreTests(unittest.TestCase):
 def test_atomic_and_reader(self):
  with tempfile.TemporaryDirectory() as t:
   w=RunWriter(t,{'seed':1},{'algorithm':'sleep'})
   w.record('metrics',1,10,{'team_return':0,'agents':{}})
   self.assertEqual(len(reader.runs([t])),1)
   self.assertEqual(reader.metrics(w.path)[0]['episode'],1)
   self.assertEqual(run_status(w.path)['state'],'running')
   self.assertFalse(list(w.path.glob('.*')))
   with self.assertRaises(FileExistsError):w.record('metrics',1,10,{})
   w.finish();self.assertEqual(run_status(w.path)['state'],'finished')
 def test_latest_run_uses_time_not_directory_prefix(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t)
   old=root/'acceptance_old';new=root/'20261020_new'
   for path,stamp in ((old,100),(new,200)):
    path.mkdir()
    atomic_json(path/'manifest.json',{'schema':'cage4.run/1'})
    atomic_json(path/'status.json',{'state':'finished','started_at':stamp})
   self.assertEqual(reader.runs([root]),[new.resolve(),old.resolve()])
 def test_reused_pid(self):
  with tempfile.TemporaryDirectory() as t:
   atomic_json(Path(t)/'status.json',{'state':'running','pid':os.getpid(),'identity':'another process'})
   self.assertEqual(run_status(t)['state'],'interrupted')
 def test_symlink_escape(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);(p/'evil.json').symlink_to('/etc/passwd')
   self.assertIsNone(reader.read(p,p/'evil.json'))
 def test_partial_legacy(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);(p/'metrics.jsonl').write_text('{"episode":1}\n{"episode":')
   self.assertEqual(reader.metrics(p),[{'episode':1}])
 def test_failed_write_leaves_old_record(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'a.json';atomic_json(p,{'ok':1})
   with self.assertRaises(ValueError):atomic_json(p,{'bad':float('nan')})
   self.assertEqual(read_json(p),{'ok':1})
if __name__=='__main__':unittest.main()
