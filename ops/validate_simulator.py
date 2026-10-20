"""Opt-in real M3/CybORG integration acceptance. Writes only to --output."""
import argparse
import json
from pathlib import Path
import hashlib
import yaml
import torch
from cage4_baselines.config import DEFAULTS
from cage4_baselines.runner import run
from cage4_protocol.store import read_json,atomic_json
from cage4_server.reader import metrics,records

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);a=p.parse_args()
 root=a.output;root.mkdir(parents=True,exist_ok=True)
 c={**DEFAULTS,'steps':20,'episodes':2,'checkpoint_every':1,'snapshot_every':5}
 trained=run(c,root/'runs')
 checkpoints=sorted((trained/'checkpoints').glob('*.pt'))
 assert len(checkpoints)==2
 first=torch.load(checkpoints[0],weights_only=True);last=torch.load(checkpoints[1],weights_only=True)
 changes={agent:any(not torch.equal(v,last['policy']['models'][agent][k]) for k,v in model.items())
          for agent,model in first['policy']['models'].items()}
 assert all(changes.values()),changes
 rows=metrics(trained);assert len(rows)==2 and len(rows[-1]['losses'])==5
 frames=records(trained,'frames');assert len(frames)>20
 for f in frames:
  assert all(v['legal_mask'] for v in f.get('blue',{}).values())
  assert 'red' in f
 snapshots=[f['topology'] for f in frames if 'topology' in f]
 assert all(len(x['hosts'])>20 and len(x['links'])>20 for x in snapshots)
 h=digest(checkpoints[-1])
 evaluated=run({**c,'episodes':1},root/'runs','evaluate',checkpoints[-1],seeds=[1001,1002,1003])
 assert digest(checkpoints[-1])==h
 evaluation=read_json(evaluated/'evaluation.json');assert evaluation['n_seeds']==3
 baseline_dirs={}
 for algo in ('random','sleep'):
  baseline_dirs[algo]=str(run({**c,'algorithm':algo,'episodes':1},root/'runs','evaluate',seeds=[1001,1002,1003]))
 # Streamlit executes the real app with these records, no browser required.
 import os
 os.environ['CAGE4_RUN_ROOTS']=str(root/'runs')
 import jsonschema
 schema=json.loads((Path(__file__).resolve().parents[1]/'protocol/schema.json').read_text())
 for record in [read_json(trained/'manifest.json'),read_json(trained/'status.json'),*rows,*frames]:
  jsonschema.validate(record,schema)
 from streamlit.testing.v1 import AppTest
 app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'server/src/cage4_server/app.py')).run(timeout=30)
 assert not app.exception,[str(x) for x in app.exception]
 report={'train_run':str(trained),'eval_run':str(evaluated),'baseline_runs':baseline_dirs,
         'weights_changed_all_five_agents':changes,'checkpoint_unchanged_by_evaluation':True,
         'evaluation':evaluation,'first_topology':{'hosts':len(snapshots[0]['hosts']),'links':len(snapshots[0]['links'])},
         'all_selected_blue_actions_mask_legal':True,'streamlit_apptest':'PASS','json_schema_validation':'PASS'}
 atomic_json(root/'validation.json',report)
 print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
