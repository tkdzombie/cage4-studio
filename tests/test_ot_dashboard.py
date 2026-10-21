"""The read-only dashboard consumes live OT records without loading models."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from cage4_server import reader


class OTDashboardTests(unittest.TestCase):
    def test_group_discovery_live_tail_and_verified_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();run=root/'ot-continuity-v1'/'macpo-test';run.mkdir(parents=True)
            def save(name,data): (run/name).write_text(json.dumps(data))
            save('manifest.json',dict(schema=reader.OT_SCHEMA,event_budget=.05,config=dict(episodes=1792)))
            save('status.json',dict(state='running',pid=12,identity='expected'))
            save('progress.json',dict(completed_training=1,training=True,step=200,heartbeat=100))
            row=dict(steps=499,return_=-123,event_cost=1)
            save('training.json',[row])
            (run/'training.jsonl').write_text(json.dumps(row)+'\n'+json.dumps({**row,'return_':-99})+'\n'+ '{"unfinished":')
            save('updates.json',[dict(episode=2,metrics={'blue_agent_0':dict(actual_policy_kl=.001)})])
            self.assertEqual(reader.runs([root,run.parent]),[run])
            rows=reader.metrics(run)
            self.assertEqual([x['official_team_return'] for x in rows],[-123,-99])
            self.assertEqual(rows[1]['event_budget'],.05)
            self.assertEqual(rows[1]['losses']['blue_agent_0']['actual_policy_kl'],.001)
            with patch('cage4_protocol.store.process_identity',return_value='expected'):
                state=reader.status(run)
            self.assertEqual(state['state'],'running')
            self.assertEqual(state['completed_episodes'],2)
            self.assertEqual(state['step'],998)
            with patch('cage4_protocol.store.process_identity',return_value='different'):
                self.assertEqual(reader.status(run)['state'],'interrupted')

    def test_snapshot_fallback_and_symlink_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();run=root/'run';run.mkdir()
            (run/'manifest.json').write_text(json.dumps(dict(schema=reader.OT_SCHEMA,event_budget=.05)))
            (run/'training.json').write_text(json.dumps([dict(steps=499,return_=-12,event_cost=0)]))
            self.assertEqual(reader.metrics(run)[0]['ot_event'],0)
            (root/'alias').symlink_to(run,target_is_directory=True)
            self.assertEqual(reader.runs([root]),[run])


if __name__=='__main__': unittest.main()
