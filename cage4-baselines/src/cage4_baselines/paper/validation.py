"""Analytical contract checks against the *actual* pinned upstream buffer/API.

Run by ops/validate_paper_baselines.py on the research runtime. This is not a
performance benchmark: synthetic two-episode signals isolate bridge semantics.
"""
import numpy as np
import torch


def check_bridge(policy, inputs):
    rewards=[1.,2.,100.,200.];costs=[.1,.2,.7,.8];done=[False,True,False,True]
    policy.begin_episode()
    for reward,cost,terminal in zip(rewards,costs,done):
        policy.act(inputs,training=True)
        policy.observe({a:reward for a in inputs},cost,terminal)
    originals={a:t.train for a,t in policy.trainers.items()}
    checked=[]
    for agent,trainer in policy.trainers.items():
        def wrapped(buffer, a=agent):
            expected_masks=np.array([1,1,0,1,0],dtype=np.float32)
            np.testing.assert_array_equal(buffer.masks[:,0,0],expected_masks)
            # Old log probability must equal evaluation before any agent update.
            np.testing.assert_allclose(policy.logp(a,buffer).cpu().numpy(),buffer.action_log_probs[:,0],atol=1e-5)
            def gae(signal,values):
                out=np.zeros(4,dtype=np.float32);acc=0.
                for t in reversed(range(4)):
                    nxt=values[t+1] if t<3 else 0.
                    delta=signal[t]+policy.args.gamma*nxt*expected_masks[t+1]-values[t]
                    acc=delta+policy.args.gamma*policy.args.gae_lambda*expected_masks[t+1]*acc
                    out[t]=acc+values[t]
                return out
            np.testing.assert_allclose(buffer.returns[:-1,0,0],gae(np.array(rewards)*.01,buffer.value_preds[:-1,0,0]),atol=1e-5)
            if policy.safe:
                np.testing.assert_allclose(buffer.cost_returns[:-1,0,0],gae(costs,buffer.cost_preds[:-1,0,0]),atol=1e-5)
                np.testing.assert_allclose(buffer.aver_episode_costs,.9,atol=1e-6)
                np.testing.assert_allclose(policy.trainers[a].safety_bound,policy.rate*2,atol=1e-6)
            checked.append(a)
            return originals[a](buffer)
        trainer.train=wrapped
    try:
        metrics=policy.update(.01)
    finally:
        for a,trainer in policy.trainers.items():trainer.train=originals[a]
    if sorted(checked)!=sorted(inputs):raise AssertionError('Not all five upstream trainers checked')
    return dict(agents=checked,checks=['old logp parity','two episode terminal masks','analytical reward GAE']+
                (['analytical cost GAE','batch mean episode cost','episode budget scale'] if policy.safe else []))
