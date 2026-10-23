"""Render recorded graphs only; no simulator dependency."""
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
