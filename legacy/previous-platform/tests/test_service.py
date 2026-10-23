import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
spec=importlib.util.spec_from_file_location('service',Path(__file__).resolve().parents[1]/'ops/service.py')
service=importlib.util.module_from_spec(spec);spec.loader.exec_module(service)
class ServiceTests(unittest.TestCase):
 def test_waits_until_launchd_job_disappears(self):
  with patch.object(service,'loaded',side_effect=[True,True,False]),patch.object(service,'command') as command,patch.object(service.time,'sleep') as sleep:
   service.stop('gui/505')
   command.assert_called_once_with('launchctl','bootout','gui/505/'+service.LABEL)
   sleep.assert_called_once_with(.5)
 def test_stop_missing_job_is_noop(self):
  with patch.object(service,'loaded',return_value=False),patch.object(service,'command') as command:
   service.stop('gui/505');command.assert_not_called()
 def test_start_loaded_job_is_noop(self):
  with patch.object(service,'loaded',return_value=True),patch.object(service,'command') as command:
   service.start('gui/505',Path('/unused'));command.assert_not_called()
if __name__=='__main__':unittest.main()
