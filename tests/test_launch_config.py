"""Ensure the PyCharm launcher preserves the comparison protocol and initialization."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('launch_config',Path(__file__).resolve().parents[1]/'experiments/run_baseline.py')
launch=importlib.util.module_from_spec(spec);spec.loader.exec_module(launch)

class LaunchTests(unittest.TestCase):
    def test_author_initialization_has_single_critic_unit_conversion(self):
        with patch.object(launch,'INITIALIZATION','author'):
            args=launch.command()
        self.assertEqual(args[args.index('--warm-start-value-scale')+1],'.01')
        self.assertEqual(args[args.index('--protocol')+1],'flat-full-actions-history-correct')
        self.assertEqual(args[args.index('--level')+1],'flat')
        self.assertEqual(args[args.index('--epochs')+1],'30')

    def test_scratch_never_inherits_author_weights(self):
        with patch.object(launch,'INITIALIZATION','scratch'):
            args=launch.command()
        self.assertNotIn('--warm-start',args)
        self.assertEqual(args[args.index('--lr')+1],'.0003')
        self.assertEqual(args[args.index('--epochs')+1],'10')

    def test_unknown_initialization_rejected(self):
        with patch.object(launch,'INITIALIZATION','typo'):
            with self.assertRaises(ValueError):launch.command()

if __name__=='__main__':unittest.main()
