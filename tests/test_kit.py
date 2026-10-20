from types import SimpleNamespace

from cage4studio.topology import snapshot, dot_graph, filtered_view
from cage4studio.train import required_config


def fake_env():
    def make_host(links, sessions=None):
        return SimpleNamespace(interfaces=[SimpleNamespace(data_links=links)], sessions=sessions or {})
    hosts = {
        'a_subnet_router': make_host(['b_subnet_router', 'a_subnet_user_host_0']),
        'b_subnet_router': make_host(['a_subnet_router']),
        'a_subnet_user_host_0': make_host(['a_subnet_router'], {'red_agent_0': [1]}),
    }
    state = SimpleNamespace(hosts=hosts, hostname_subnet_map={
        'a_subnet_router': 'a_subnet', 'b_subnet_router': 'b_subnet',
        'a_subnet_user_host_0': 'a_subnet'}, mission_phase=1)
    return SimpleNamespace(
        environment_controller=SimpleNamespace(state=state),
        get_ip_map=lambda: {k: '10.0.0.1' for k in hosts},
    )


def test_true_edges_and_compromise():
    s = snapshot(fake_env(), episode=2, step=3)
    assert len(s['hosts']) == 3
    assert len(s['links']) == 2
    assert sum(h['red_sessions'] > 0 for h in s['hosts']) == 1
    nodes, edges = filtered_view(s, limit_hosts=True)
    assert len(nodes) == 2 and len(edges) == 1
    assert 'RED' in dot_graph(s, overview=False)


def test_config_validation():
    c = required_config({'steps': 30, 'episodes': 2, 'device': 'cpu'})
    assert c['episodes'] == 2 and c['steps'] == 30
    try:
        required_config({'steps': -1})
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid steps not rejected')
