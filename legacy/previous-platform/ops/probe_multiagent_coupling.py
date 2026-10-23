"""Zero-transition factorial fixture for a shared official Green transaction.

Two different defenders can deny the same permitted business flow. This tests
structural reward coupling, not learned cooperation or communication benefits.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    from cage4_baselines.envs.adapter import Cage4Adapter
    from cage4_baselines.paper.provenance import runtime_identity
    from CybORG.Simulator.Actions.ConcreteActions.ControlTraffic import BlockTrafficZone
    from CybORG.Simulator.Actions.GreenActions import GreenAccessService
    from CybORG.Shared.BlueRewardMachine import BlueRewardMachine
    if args.output.exists():raise FileExistsError(args.output)
    env=Cage4Adapter(54001,500,include_graph=False);rows=[]
    for block0,block4 in ((0,0),(1,0),(0,1),(1,1)):
        inputs=env.reset(54001);state=env.base.environment_controller.state
        source=next(h for h in sorted(state.hosts) if h.startswith('restricted_zone_a_subnet_user_host_'))
        target=next(h for h in sorted(state.hosts) if h.startswith('admin_network_subnet_server_host_'))
        specs=[('blue_agent_0','admin_network_subnet','restricted_zone_a_subnet',block0),
               ('blue_agent_4','restricted_zone_a_subnet','admin_network_subnet',block4)]
        applied=[]
        for a,src,dst,on in specs:
            label=next((s for i,s in enumerate(inputs[a].labels) if inputs[a].mask[i] and s.startswith('BlockTrafficZone '+dst+' ') and ('<- '+src+' (') in s),None)
            assert label is not None, 'Intervention must be a legal public action'
            if on:
                outcome=BlockTrafficZone(0,a,src,dst).execute(state)
                assert outcome.success==True
                applied.append(dict(agent=a,label=label))
        green=next(a for a,sessions in state.sessions.items() if a.startswith('green') and any(s.hostname==source for s in sessions.values()))
        session=next(k for k,s in state.sessions[green].items() if s.hostname==source)
        flow=GreenAccessService(green,session,state.hostname_ip_map[source],[],0.)
        # Same selected allowed pair in all four cases; destination selection is
        # held fixed to identify the two legal firewall interventions' effect.
        flow.random_reachable_ip=lambda _,ip=state.hostname_ip_map[target]:ip
        success=flow.execute(state).success==True
        weight=BlueRewardMachine('Blue').get_phase_rewards(1)['restricted_zone_a_subnet']['ASF']
        rows.append(dict(block_blue0=block0,block_blue4=block4,success=success,
                         mission_a_green_reward=0 if success else weight,interventions=applied))
    assert [r['success'] for r in rows]==[True,False,False,False]
    y=[r['mission_a_green_reward'] for r in rows]
    interaction=y[3]-y[1]-y[2]+y[0]
    assert interaction!=0
    result=dict(schema='cage4.multiagent-coupling-fixture/1',environment_transitions=0,training_updates=0,
                reset_seed=54001,source_zone='restricted_zone_a_subnet',destination_zone='admin_network_subnet',
                runtime=runtime_identity(),rows=rows,reward_interaction=interaction,
                official_sources={c.__name__:hashlib.sha256(Path(inspect.getfile(c)).read_bytes()).hexdigest() for c in (BlockTrafficZone,GreenAccessService)},
                scope='Constructed fixed allowed endpoint pair, direct official executes; non-additive business-flow coupling only. No population effect size, learned safety cooperation or timing claim.')
    with args.output.open('x') as file:file.write(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__=='__main__':main()
