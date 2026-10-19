"""Ground-truth snapshots for researcher visualization ONLY.

Never feed this data to a learning policy. The learning policy must use the
BlueFlatWrapper observations and permitted action masks exclusively.
"""
from __future__ import annotations


def subnet_of(hostname: str) -> str:
    if hostname == 'root_internet_host_0':
        return 'internet_subnet'
    if '_subnet_' in hostname:
        return hostname.split('_subnet_', 1)[0] + '_subnet'
    return 'unknown'


def _value(x):
    return str(getattr(x, 'value', x))


def snapshot(env, *, episode: int, step: int):
    state = env.environment_controller.state
    ip_map = env.get_ip_map()
    hosts = []
    undirected_links = set()
    for name, h in sorted(state.hosts.items()):
        sessions = {}
        for agent, active in getattr(h, 'sessions', {}).items():
            if active:
                sessions[str(agent)] = len(active)
        iface = getattr(h, 'interfaces', [])
        links = set()
        for i in iface:
            for other in getattr(i, 'data_links', []) or []:
                other = str(other)
                if other in state.hosts and other != name:
                    links.add(other)
                    undirected_links.add(tuple(sorted((name, other))))
        subnet_obj = state.hostname_subnet_map.get(name)
        subnet = _value(subnet_obj).lower() if subnet_obj is not None else subnet_of(name)
        hosts.append({
            'name': name, 'ip': str(ip_map.get(name, '')), 'subnet': subnet,
            'role': 'router' if '_router' in name else ('internet' if name == 'root_internet_host_0' else 'host'),
            'red_sessions': sum(n for a, n in sessions.items() if a.startswith('red')),
            'blue_sessions': sum(n for a, n in sessions.items() if a.startswith('blue')),
            'sessions': sessions,
            'links': sorted(links),
        })
    return {
        'episode': int(episode), 'step': int(step),
        'mission_phase': int(getattr(state, 'mission_phase', -1)),
        'hosts': hosts,
        'links': [{'source': a, 'target': b} for a, b in sorted(undirected_links)],
        'provenance': 'CybORG state.hosts[*].interfaces[*].data_links (ground truth, researcher only)',
    }


def filtered_view(snap: dict, selected_subnet: str = '全部', limit_hosts: bool = False):
    """Return actual host graph, optionally restricted to a subnet.

    A 'subnet overview' plots routers, the internet host, and their actual
    data_links; it NEVER fabricates subnet connectivity.
    """
    hosts = snap.get('hosts', [])
    if limit_hosts:
        nodes = {h['name'] for h in hosts if h['role'] in ('router', 'internet')}
    else:
        nodes = {h['name'] for h in hosts if selected_subnet == '全部' or h['subnet'] == selected_subnet}
    return ([h for h in hosts if h['name'] in nodes],
            [e for e in snap.get('links', []) if e['source'] in nodes and e['target'] in nodes])


def dot_graph(snap: dict, selected_subnet: str = '全部', overview: bool = True) -> str:
    """DOT graph with distinct colors for red-session hosts; safe ID quoting."""
    import json
    hosts, edges = filtered_view(snap, selected_subnet, overview)
    out = ['graph cyborg {', 'graph [layout=neato, overlap=false, splines=true, outputorder=edgesfirst];',
           'node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=9];',
           'edge [color="#9CA3AF", penwidth=1.3];']
    for h in hosts:
        if h['red_sessions'] > 0:
            fill, border = '#FECACA', '#DC2626'
        elif h['role'] == 'router':
            fill, border = '#BFDBFE', '#2563EB'
        elif h['role'] == 'internet':
            fill, border = '#E9D5FF', '#7C3AED'
        else:
            fill, border = '#D1FAE5', '#059669'
        title = h['name']
        label = f"{h['name']}\\n{h['ip']}"
        if h['red_sessions']:
            label += f"\\nRED ×{h['red_sessions']}"
        out.append(f'{json.dumps(title)} [label={json.dumps(label)}, fillcolor={json.dumps(fill)}, color={json.dumps(border)}];')
    for e in edges:
        out.append(f'{json.dumps(e["source"])} -- {json.dumps(e["target"])};')
    out.append('}')
    return '\n'.join(out)
