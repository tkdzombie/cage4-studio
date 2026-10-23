import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

try:
    import torch
except ImportError:
    torch=None

spec=importlib.util.spec_from_file_location('value_probe',Path(__file__).resolve().parents[2]/'ops/probe_macpo_value_fit.py')
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)


@unittest.skipIf(torch is None,'torch required')
class MacpoValueFitTests(unittest.TestCase):
    def test_copies_do_not_share_parameters_or_adam_moments(self):
        model=torch.nn.Linear(2,1);optimizer=torch.optim.Adam(model.parameters(),lr=.01)
        model(torch.ones((3,2))).sum().backward();optimizer.step();optimizer.zero_grad()
        initial=[p.detach().clone() for p in model.parameters()]
        source_state=optimizer.state_dict()
        a,oa=probe.fork_critic(model,optimizer,source_state)
        b,ob=probe.fork_critic(model,optimizer,source_state)
        b_state=[s['exp_avg'].clone() for s in ob.state.values()]
        original_state=[s['exp_avg'].clone() for s in optimizer.state.values()]
        a(torch.ones((3,2))).sum().backward();oa.step()
        for p,old,other in zip(model.parameters(),initial,b.parameters()):
            self.assertTrue(torch.equal(p,old));self.assertTrue(torch.equal(other,old))
        for actual,old in zip(optimizer.state.values(),original_state):self.assertTrue(torch.equal(actual['exp_avg'],old))
        for actual,old in zip(ob.state.values(),b_state):self.assertTrue(torch.equal(actual['exp_avg'],old))
        self.assertTrue(any(not torch.equal(p,old) for p,old in zip(a.parameters(),initial)))

    def test_pinned_author_loss_clipping_can_zero_a_needed_value_gradient(self):
        upstream=Path.home()/'cage4-upstreams'
        if not upstream.exists():self.skipTest('Pinned external upstream required')
        from cage4_baselines.paper.upstreams import activate
        activate(upstream,'macpo')
        from macpo.algorithms.r_mappo.r_macpo import R_MACTRPO_CPO
        trainer=SimpleNamespace(_use_popart=False,clip_param=.2,_use_huber_loss=True,huber_delta=10.,_use_clipped_value_loss=True,_use_value_active_masks=True)
        old=torch.tensor([[0.]]);target=torch.tensor([[-5.]]);mask=torch.ones((1,1))
        clipped=torch.tensor([[-1.]],requires_grad=True)
        loss=R_MACTRPO_CPO.cal_value_loss(trainer,clipped,old,target,mask)
        self.assertEqual(float(torch.autograd.grad(loss,clipped)[0]),0.)
        trainer._use_clipped_value_loss=False
        unclipped=torch.tensor([[-1.]],requires_grad=True)
        loss=R_MACTRPO_CPO.cal_value_loss(trainer,unclipped,old,target,mask)
        self.assertGreater(float(torch.autograd.grad(loss,unclipped)[0]),0.)


if __name__=='__main__':unittest.main()
