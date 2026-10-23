import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('constrained_campaign',Path(__file__).resolve().parents[1]/'ops/constrained_campaign.py')
campaign=importlib.util.module_from_spec(spec);spec.loader.exec_module(campaign)


class CampaignTests(unittest.TestCase):
    def test_plan_only_allows_constrained_algorithms(self):
        plan=campaign.build_plan(['macpo','mappo_lagrangian'],[101,102],2000)
        self.assertEqual(len(plan),4)
        self.assertEqual(plan[0],dict(algorithm='macpo',seed=101,episodes=2000))
        for names in (['mappo'],['ippo'],['macpo','macpo'],[]):
            with self.assertRaises(ValueError):campaign.build_plan(names,[101],2000)
        for seeds in ([101,101],[0],[],[True]):
            with self.assertRaises(ValueError):campaign.build_plan(['macpo'],seeds,2000)
        self.assertTrue(set(campaign.DEVELOPMENT_SEEDS).isdisjoint(range(53001,53101)))

    def test_summary_uses_shared_reward_once(self):
        with tempfile.TemporaryDirectory() as t:
            folder=Path(t)/'metrics/episode_000001';folder.mkdir(parents=True)
            (folder/'step_000499.json').write_text(json.dumps(dict(official_team_return=-123.,team_return=-615.,
                safety=dict(mean_cost=.04),budget_violated=False,optimizer_update=True,losses={'a':{'actual_policy_kl':.001}})))
            summary=campaign.training_summary(t)
            self.assertEqual(summary['last100']['mean_return'],-123.)
            self.assertEqual(summary['last100']['mean_cost'],.04)
            self.assertEqual(summary['optimizer_updates'],1)

    def test_duplicate_supervisor_is_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            with campaign.campaign_lock(t):
                with self.assertRaises(RuntimeError):
                    with campaign.campaign_lock(t):pass

    def test_overrides_allow_one_factor_and_cannot_change_protocol(self):
        self.assertEqual(campaign.load_overrides(None,['macpo']),{})
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'overrides.json'
            def check(values):
                p.write_text(json.dumps(dict(schema='cage4.campaign-overrides/1',algorithms=values)))
                return campaign.load_overrides(p,['macpo','mappo_lagrangian'])
            good=dict(macpo=dict(fraction_coef=.5),mappo_lagrangian=dict(lr=.0001))
            self.assertEqual(check(good),good)
            for bad in ({},dict(mappo=dict(lr=.0001)),dict(macpo=dict(budget_rate=.1)),
                        dict(macpo=dict(fraction_coef=.5,kl_threshold=.02)),
                        dict(macpo=dict(fraction_coef=True)),dict(macpo=dict(fraction_coef=float('nan'))),
                        dict(macpo=dict(fraction_coef=1.1)),dict(macpo=dict(ls_step=2.5)),
                        dict(mappo_lagrangian=dict(lr=0))):
                with self.subTest(bad=bad),self.assertRaises(ValueError):check(bad)
            p.write_text(json.dumps(dict(schema='wrong',algorithms=good)))
            with self.assertRaises(ValueError):campaign.load_overrides(p,['macpo'])
