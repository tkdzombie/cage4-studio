import unittest
from types import SimpleNamespace as NS
try:
    import torch
except ImportError:
    torch=None


@unittest.skipIf(torch is None,'torch is required for optimizer adaptation checks')
class FiniteHorizonTests(unittest.TestCase):
    def test_narrow_cpo_rescale_restores_even_on_error(self):
        from cage4_baselines.paper.finite_horizon import finite_cpo_rescale
        trainer=NS(gamma=.99)
        def original():
            self.assertEqual(trainer.gamma,0.)
            raise RuntimeError('test')
        with self.assertRaises(RuntimeError):finite_cpo_rescale(original)(trainer)
        self.assertEqual(trainer.gamma,.99)

    def test_dual_is_fixed_during_epochs(self):
        from cage4_baselines.paper.finite_horizon import preserve_dual
        trainer=NS(lamda_lagr=torch.tensor(.78))
        before=trainer.lamda_lagr
        def original():trainer.lamda_lagr=torch.tensor(100.);return 7
        self.assertEqual(preserve_dual(original)(trainer),7)
        self.assertIs(trainer.lamda_lagr,before)

    def test_episode_scaling_matches_exact_first_hit_policy_gradient(self):
        # Enumerate a two-step Bernoulli process: violation is either action=1.
        # Emitting once means the second action has no remaining event cost if
        # the first action already triggered it. The analytic derivative of
        # Pr(any hit)=1-(1-p)^2 is 2*p*(1-p)^2 for sigmoid(logit).
        theta=torch.tensor(.3,requires_grad=True);p=theta.sigmoid();p0=p.detach()
        loss=torch.tensor(0.)
        for a in (0,1):
            for b in (0,1):
                probability=(p0 if a else 1-p0)*(p0 if b else 1-p0)
                actions=torch.tensor([a,b],dtype=torch.float32)
                ratios=torch.exp(torch.distributions.Bernoulli(probs=p).log_prob(actions)-torch.distributions.Bernoulli(probs=p0).log_prob(actions))
                to_go=torch.tensor([float(a or b),float((not a) and b)])
                loss=loss+probability*(ratios*to_go*2).mean()
        grad=torch.autograd.grad(loss,theta)[0]
        self.assertAlmostEqual(float(grad),float(2*p0*(1-p0)**2),places=6)


if __name__=='__main__':unittest.main()
