import json
import os
from pathlib import Path
import tempfile
import unittest
from cage4_protocol.store import atomic_json,read_json,RunWriter,run_status,prune_runs
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
class RetentionTests(unittest.TestCase):
 def test_only_latest_five_survive_with_all_old_files_deleted(self):
  with tempfile.TemporaryDirectory() as t:
   paths=[]
   for i in range(8):
    w=RunWriter(t,{},{});(w.path/'weights.pt').write_bytes(b'weights');w.finish();paths.append(w.path)
   self.assertFalse(any(p.exists() for p in paths[:3]))
   self.assertTrue(all((p/'weights.pt').exists() for p in paths[3:]))
 def test_live_writer_and_unknown_files_and_symlinks_survive(self):
  with tempfile.TemporaryDirectory() as t, tempfile.TemporaryDirectory() as outside:
   active=RunWriter(t,{},{});active.status(started_at=1)
   root=Path(t);(root/'notes').mkdir();(root/'escape').symlink_to(outside,target_is_directory=True)
   for _ in range(6):w=RunWriter(t,{},{});w.finish()
   self.assertTrue(active.path.exists())
   self.assertTrue((root/'notes').exists());self.assertTrue((root/'escape').is_symlink())
   active.finish();self.assertFalse(active.path.exists())
 def test_unverifiable_process_is_protected(self):
  from unittest.mock import patch
  with tempfile.TemporaryDirectory() as t:
   w=RunWriter(t,{},{});w.status(started_at=1)
   for _ in range(5):new=RunWriter(t,{},{});new.finish()
   with patch('cage4_protocol.store.process_identity',return_value=None):
    prune_runs([t])
   self.assertTrue(w.path.exists())
 def test_global_retention_across_roots_and_stale_pid(self):
  with tempfile.TemporaryDirectory() as t:
   roots=[Path(t)/'a',Path(t)/'b'];paths=[]
   for i in range(8):
    w=RunWriter(roots[i%2],{},{});w.finish();paths.append(w.path)
   atomic_json(paths[0]/'status.json',{'state':'running','started_at':1,'pid':os.getpid(),'identity':'stale'})
   removed=prune_runs(roots)
   self.assertEqual(len(removed),3);self.assertFalse(paths[0].exists())
   self.assertTrue(all(p.exists() for p in paths[3:]))

if __name__=='__main__':unittest.main()
