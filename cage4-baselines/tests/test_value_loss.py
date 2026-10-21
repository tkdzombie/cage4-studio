from types import SimpleNamespace as NS
import unittest
try:
    import torch
except ImportError:
    torch=None


@unittest.skipIf(torch is None,'torch required')
class ValueLossTests(unittest.TestCase):
    def trainer(self,**overrides):
        values=dict(_use_popart=False,_use_huber_loss=True,huber_delta=10.,_use_clipped_value_loss=False,clip_param=.2,_use_value_active_masks=False)
        values.update(overrides);return NS(**values)

    def test_matches_torch_huber_on_both_tails_and_boundary_gradients(self):
        from cage4_baselines.paper.value_loss import symmetric_value_loss
        pred=torch.tensor([[-25.],[-10.],[-5.],[0.],[5.],[10.],[25.]],requires_grad=True)
        target=torch.zeros_like(pred)
        actual=symmetric_value_loss(self.trainer(),pred,pred.detach(),target,torch.ones_like(pred))
        expected=torch.nn.functional.huber_loss(pred,target,delta=10.)
        self.assertTrue(torch.allclose(actual,expected))
        self.assertTrue(torch.allclose(torch.autograd.grad(actual,pred,retain_graph=True)[0],torch.autograd.grad(expected,pred)[0]))

    def test_negative_return_tail_has_nonzero_loss_and_gradient(self):
        from cage4_baselines.paper.value_loss import symmetric_value_loss
        pred=torch.tensor([[0.]],requires_grad=True)
        loss=symmetric_value_loss(self.trainer(),pred,pred.detach(),torch.tensor([[-20.]]),torch.ones_like(pred))
        self.assertEqual(float(loss),150.)
        self.assertEqual(float(torch.autograd.grad(loss,pred)[0]),10.)

    def test_actual_macpo_trainer_uses_corrected_loss(self):
        from pathlib import Path
        upstream=Path.home()/'cage4-upstreams'
        if not upstream.exists():self.skipTest('Pinned upstream required')
        from cage4_baselines.paper.upstreams import activate
        activate(upstream,'macpo')
        from cage4_baselines.paper.discrete_macpo import classes
        _,trainer_class=classes()
        pred=torch.zeros((1,1),requires_grad=True)
        loss=trainer_class.cal_value_loss(self.trainer(),pred,pred.detach(),torch.tensor([[-20.]]),torch.ones_like(pred))
        self.assertEqual(float(loss),150.)
        self.assertEqual(float(torch.autograd.grad(loss,pred)[0]),10.)

    def test_inactive_rows_have_no_loss_or_gradient(self):
        from cage4_baselines.paper.value_loss import symmetric_value_loss
        pred=torch.zeros((2,1),requires_grad=True);target=torch.tensor([[-20.],[1000.]])
        loss=symmetric_value_loss(self.trainer(_use_value_active_masks=True),pred,pred.detach(),target,torch.tensor([[1.],[0.]]))
        self.assertEqual(float(loss),150.)
        self.assertTrue(torch.equal(torch.autograd.grad(loss,pred)[0],torch.tensor([[10.],[0.]])))

    def test_clipping_still_blocks_an_overlarge_value_change(self):
        from cage4_baselines.paper.value_loss import symmetric_value_loss
        pred=torch.tensor([[-1.]],requires_grad=True)
        loss=symmetric_value_loss(self.trainer(_use_clipped_value_loss=True),pred,torch.zeros_like(pred),torch.tensor([[-5.]]),torch.ones_like(pred))
        self.assertEqual(float(torch.autograd.grad(loss,pred)[0]),0.)

    def test_non_huber_option_remains_half_squared_error(self):
        from cage4_baselines.paper.value_loss import symmetric_value_loss
        pred=torch.tensor([[2.]])
        self.assertEqual(float(symmetric_value_loss(self.trainer(_use_huber_loss=False),pred,pred,torch.tensor([[-2.]]),torch.ones_like(pred))),8.)


if __name__=='__main__':unittest.main()
