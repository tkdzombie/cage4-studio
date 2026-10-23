"""Researcher-only telemetry. Never imported by an algorithm or observation encoder."""
import json

def safe(value):
    return json.loads(json.dumps(value,default=str))

def snapshot(base, episode, step):
    state = base.environment_controller.state
    hosts,edges = [],set()
    ips = base.get_ip_map()
    for name,h in sorted(state.hosts.items()):
        sessions = {str(a):len(v) for a,v in h.sessions.items() if v}
        links = set()
        for interface in h.interfaces:
            for other in getattr(interface,'data_links',[]) or []:
                other = str(other)
                if other in state.hosts and other != name:
                    edges.add(tuple(sorted((name,other))))
                    links.add(other)
        subnet = state.hostname_subnet_map.get(name,'unknown')
        hosts.append({'name':name,'ip':str(ips.get(name,'')),
                      'subnet':str(getattr(subnet,'value',subnet)).lower(),
                      'role':'router' if '_router' in name else ('internet' if name=='root_internet_host_0' else 'host'),
                      'sessions':sessions,'red_sessions':sum(v for a,v in sessions.items() if a.startswith('red')),
                      'blue_sessions':sum(v for a,v in sessions.items() if a.startswith('blue')),
                      'links':sorted(links)})
    return {'episode':episode,'step':step,'hosts':hosts,
            'links':[{'source':a,'target':b} for a,b in sorted(edges)],
            'mission_phase':int(state.mission_phase),
            'provenance':'CybORG state.hosts[*].interfaces[*].data_links; researcher only'}

def red_telemetry(base):
    red = {}
    for a in base.agents:
        if not a.startswith('red'):
            continue
        actions = base.get_last_action(a)
        seq = actions if isinstance(actions,(list,tuple)) else [actions]
        invalid = any(type(x).__name__=='InvalidAction' for x in seq if x is not None)
        red[a] = {'executed_actions':safe(actions),
                  'validity':'invalid' if invalid else ('no_action' if not actions else 'not_rejected'),
                  'observation':safe(base.get_observation(a)),
                  'sessions':[safe({key:getattr(session,key,None) for key in
                      ('ident','hostname','username','agent','pid','parent','session_type','active')})
                      for session in base.environment_controller.state.sessions.get(a,{}).values()]}
    return red

def frame(adapter, episode, step, blue=None):
    return {'topology':snapshot(adapter.base,episode,step),'red':red_telemetry(adapter.base),'blue':blue or {},
            'blue_observations':safe(adapter.raw)}
