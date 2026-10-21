"""Characterize installed CAGE4 business semantics without changing its source.

Run with the pinned simulator's Python interpreter. These constructed fixtures
describe behavior, not frequencies, and do not execute an environment episode.
"""
import argparse
import hashlib
import inspect
import json
from ipaddress import IPv4Address
from pathlib import Path
from types import SimpleNamespace as NS


class FixedRng:
    def random(self):
        return 1.0

    def integers(self, *args, **kwargs):
        return 0

    def choice(self, items):
        return items[0]


def characterize():
    from CybORG.Simulator.Actions.GreenActions import GreenAccessService, GreenLocalWork
    from CybORG.Shared.Enums import ProcessName

    src, dst = IPv4Address('10.0.0.1'), IPv4Address('10.0.1.1')
    destination = NS(services={}, get_ephemeral_port=lambda: 12345,
                     events=NS(network_connections=[]))
    state = NS(ip_addresses={src: 'src', dst: 'dst'},
               hostname_ip_map={'src': src, 'dst': dst},
               hostname_subnet_map={'src': NS(value='src_zone'), 'dst': NS(value='dst_zone')},
               hosts={'dst': destination}, blocks={}, np_random=FixedRng())
    action = GreenAccessService('green_probe', 0, src, [], 0.0)
    # Hold destination selection fixed and make the health predicate explicitly
    # reject it. This isolates whether execute actually invokes that predicate.
    action.random_reachable_ip = lambda _: dst
    calls = []

    def reject_service(_):
        calls.append(1)
        return False

    action.available_dest_service = reject_service
    unblocked = action.execute(state)
    predicate_calls = len(calls)
    state.blocks = {'dst_zone': ['src_zone']}
    blocked = action.execute(state)

    ssh = NS(active=True, get_service_reliability=lambda: 100)
    ot = NS(active=False, get_service_reliability=lambda: 0)
    local_state = NS(sessions={'green_probe': {0: NS(hostname='local')}},
                     hosts={'local': NS(services={ProcessName.SSHD: ssh, ProcessName.OTSERVICE: ot})},
                     np_random=FixedRng())
    local = GreenLocalWork('green_probe', 0, src, 0.0, 0.0).execute(local_state)
    classes = (GreenAccessService, GreenLocalWork)
    return {
        'schema': 'cage4.business-semantics-probe/1',
        'scope': 'Constructed characterization fixtures; zero environment transitions; no policy or simulator file changes',
        'source_sha256': {cls.__name__: hashlib.sha256(Path(inspect.getfile(cls)).read_bytes()).hexdigest()
                          for cls in classes},
        'access_service': {
            'destination_service_count': 0,
            'rejecting_health_predicate_calls_unblocked': predicate_calls,
            'unblocked_success': unblocked.success == True,
            'blocked_success': blocked.success == True,
        },
        'local_work': {'ot_active': False, 'other_service_active': True,
                       'local_work_success': local.success == True},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Optional new private JSON file; refuses overwrites')
    args = parser.parse_args()
    result = characterize()
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        with args.output.open('x') as file:
            file.write(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
