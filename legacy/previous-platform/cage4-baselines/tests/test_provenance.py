from pathlib import Path
import tempfile
import unittest
from cage4_baselines.paper.provenance import source_digest, IMPLEMENTATION_EXCLUDES

class ProvenanceTests(unittest.TestCase):
 def test_launcher_and_unused_template_do_not_invalidate_other_models(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t)
   for path in IMPLEMENTATION_EXCLUDES:
    p=root/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('original')
   original=source_digest(root,IMPLEMENTATION_EXCLUDES)
   for path in IMPLEMENTATION_EXCLUDES:(root/path).write_text('changed')
   self.assertEqual(source_digest(root,IMPLEMENTATION_EXCLUDES),original)
 def test_runtime_observation_source_changes_invalidate_checkpoint(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);p=root/'envs/observations.py';p.parent.mkdir();p.write_text('v1')
   original=source_digest(root,IMPLEMENTATION_EXCLUDES);p.write_text('v2')
   self.assertNotEqual(source_digest(root,IMPLEMENTATION_EXCLUDES),original)
