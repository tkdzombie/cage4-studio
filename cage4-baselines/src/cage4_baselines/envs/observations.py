"""Versioned host-slot features from Blue observations and public action metadata.

No simulator/state reference is accepted. Histories are episode-local observations,
not estimates of hidden compromise. See docs/OBSERVATIONS_zh.md for the layout.
"""
from collections.abc import Mapping
import numpy as np
from .belief import CHANNELS, observation_config, ObservationTransport, update_event_belief, update_confidence

ENCODER = 'blue-belief-slots/1'
HOST_FIELDS = ('known', 'observed_now', 'own_session', 'process_event',
               'connection_event', 'process_history', 'connection_history',
               'analysed', 'file_count', 'alert_age', 'analysis_age', 'last_target')
BASE_FIELDS = HOST_FIELDS
SUCCESS = ('UNKNOWN', 'TRUE', 'FALSE', 'IN_PROGRESS')


def name(value):
    return getattr(value, 'name', str(value)).upper()


class StructuredObservation:
    def __init__(self, hosts, commands, max_hosts, horizon, agent_index, config=None):
        self.hosts = tuple(hosts)
        self.commands = tuple(commands)
        self.max_hosts = max_hosts
        self.horizon = horizon
        self.agent_index = agent_index
        self.config = observation_config(config)
        self.fields = BASE_FIELDS + ('server','user','other')
        self.fields += tuple('observed/'+c for c in CHANNELS)
        self.fields += tuple('age_linear/'+c for c in CHANNELS)
        self.fields += tuple('age_log/'+c for c in CHANNELS)
        self.fields += tuple('confidence/'+c for c in CHANNELS)
        self.fields += tuple('last_host_action/'+c for c in self.commands)
        self.fields += ('action_age','action_success','action_result_known','ever_analyse','ever_restore','ever_remove')
        self.reset()

    def reset(self, seed=None):
        transport_config = dict(self.config)
        if seed is not None: transport_config['seed'] = int(seed)
        self.transport = ObservationTransport(transport_config, self.hosts)
        self.history = {h: dict(known=False, session=False, process=False,
                               connection=False, analysed=False, files=0,
                               alert=None, analysis=None, last={k:None for k in CHANNELS},
                               action=None, action_step=None, action_success=False,
                               ever_analyse=False, ever_restore=False, ever_remove=False) for h in self.hosts}

    def schema(self):
        head = ['step_fraction'] + [f'agent_{i}' for i in range(5)]
        head += [f'success/{s}' for s in SUCCESS]
        head += [f'last_action/{s}' for s in self.commands]
        head += [f'message/{i}/{j}' for i in range(4) for j in range(8)]
        slots = [dict(slot=i, hostname=h, fields=list(self.fields)) for i,h in enumerate(self.hosts)]
        return dict(encoder=ENCODER, head=head, hosts=slots, max_hosts=self.max_hosts,
                    padding='zero host rows after own fixed host slots',
                    dimension=len(head)+self.max_hosts*len(self.fields),
                    sources='own Blue observation; own public fixed action slots; episode clock',
                    history='reset every episode; Remove does not certify recovery; Restore clears current alert belief',
                    options=self.config, channel_masks='event presence only; no event does not certify absence',
                    graph='optional membership graph; not a trained GNN')

    def encode(self, observation, step):
        if not isinstance(observation, Mapping): raise ValueError('Blue observation must be a mapping')
        observation, source_step, dropped = self.transport.push(observation, step)
        stamp = step if source_step is None else source_step
        mode = self.config['mode']
        if mode == 'current':
            for h in self.history.values():
                h.update(process=False,connection=False,analysed=False,files=0,alert=None,analysis=None)
                h['last'] = {k:None for k in CHANNELS}
        success = name(observation.get('success', 'UNKNOWN'))
        action = observation.get('action')
        action_name = type(action).__name__ if action is not None and not isinstance(action,str) else str(action or '').split(' ')[0]
        target = getattr(action, 'hostname', None)
        if target is None and isinstance(action,str):
            parts=action.split(); target=parts[1] if len(parts)>1 else None
        # A recovery result changes remembered observations, not hidden truth.
        if success == 'TRUE' and target in self.history:
            h = self.history[target]
            if action_name == 'Restore' and target not in dropped:
                h.update(process=False, connection=False, alert=None, analysed=False, files=0, analysis=None)
                # A successful restore is intervention evidence, not a new sensor sample.
                h['last'] = {k:None for k in CHANNELS}
        if source_step is not None and target in self.history and target not in dropped:
            h=self.history[target]
            if success != 'IN_PROGRESS':
                h.update(action=action_name, action_step=stamp, action_success=success=='TRUE')
                if success=='TRUE' and action_name in ('Analyse','Restore','Remove'):
                    h['ever_'+action_name.lower()]=True
        rows = np.zeros((self.max_hosts,len(self.fields)), dtype=np.float32)
        age_stats=[];confidence_stats=[]
        for i,host in enumerate(self.hosts):
            h = self.history[host]
            value = observation.get(host, {})
            seen = isinstance(value, Mapping) and bool(value)
            processes = value.get('Processes', []) if seen else []
            connections = source_step is not None and source_step>0 and any(bool(p.get('Connections')) for p in processes)
            # Initial inventory includes benign processes; it is not an alert.
            process = source_step is not None and source_step > 0 and any(not p.get('Connections') for p in processes)
            process = self.transport.noisy(process, source_step is not None and source_step>0 and bool(processes))
            connections = self.transport.noisy(connections, source_step is not None and source_step>0 and bool(processes))
            if seen:
                update_event_belief(h['last'], 'host', stamp)
                h['known'] = True
                if 'Sessions' in value: h['session'] = bool(value['Sessions'])
            if process: update_event_belief(h['last'], 'process', stamp)
            if connections: update_event_belief(h['last'], 'connection', stamp)
            if process or connections: h['alert'] = stamp
            h['process'] |= process; h['connection'] |= connections
            if success == 'TRUE' and action_name == 'Analyse' and target == host and host not in dropped:
                h.update(analysed=True, analysis=stamp, files=len(value.get('Files', [])))
                update_event_belief(h['last'], 'file', stamp)
            age = lambda t: 1. if t is None else min(1., (step-t)/self.horizon)
            base = (h['known'], seen, h['session'], process, connections,
                       h['process'], h['connection'], h['analysed'],
                       min(h['files'],10)/10., age(h['alert']), age(h['analysis']), target==host)
            observed,linear,logarithmic,confidence = update_confidence(h['last'],step,self.config['age_max'],self.config['decay'])
            age_stats.extend(linear);confidence_stats.extend(confidence)
            if mode in ('current','persistent'):
                linear=[0.]*4;logarithmic=[0.]*4
                base=base[:9]+(0.,0.)+base[11:]
            if mode != 'belief': confidence=[0.]*4
            types = ('_server_host_' in host, '_user_host_' in host, '_server_host_' not in host and '_user_host_' not in host)
            history = [h['action']==c for c in self.commands]
            history += [0. if mode in ('current','persistent') else age(h['action_step']), h['action_success'], h['action_step'] is not None,
                        h['ever_analyse'],h['ever_restore'],h['ever_remove']]
            rows[i] = (*base,*types,*observed,*linear,*logarithmic,*confidence,*history)
        self.host_features=rows.copy()
        self.last_step=step
        self.diagnostics = dict(mean_confidence=float(np.mean(confidence_stats)),
                                mean_age=float(np.mean(age_stats)),
                                age_p50=float(np.quantile(age_stats,.5)),age_p95=float(np.quantile(age_stats,.95)))
        messages = np.asarray(observation.get('message', np.zeros((4,8))),dtype=np.float32)
        if messages.shape != (4,8) or not np.isfinite(messages).all():
            raise ValueError('Expected four finite 8-bit Blue messages')
        head = [min(step/self.horizon,1.)] + [float(i==self.agent_index) for i in range(5)]
        head += [float(success==s) for s in SUCCESS]
        head += [float(action_name==s) for s in self.commands]
        out = np.concatenate((np.asarray(head,dtype=np.float32), messages.reshape(-1), rows.reshape(-1)))
        if not np.isfinite(out).all(): raise ValueError('Nonfinite structured observation')
        self.global_features=out[:-self.max_hosts*len(self.fields)].copy()
        return out


    def graph(self):
        """Pure NumPy graph. Membership only; unknown physical links are omitted.

        Host/subnet indexing is deterministic across episodes. There is no
        trainable graph model here; existing paper baselines consume the vector.
        """
        def subnet(host):
            for marker in ('_user_host_', '_server_host_', '_router'):
                if marker in host: return host.split(marker)[0]
            return 'unknown'
        subnets=sorted({subnet(h) for h in self.hosts})
        if len(subnets)>3: raise ValueError('CAGE4 supports at most three owned subnets')
        H=self.max_hosts;F=len(self.fields)
        nodes=np.zeros((H+3,F),dtype=np.float32)
        nodes[:H]=self.host_features
        edges=[]
        for i,h in enumerate(self.hosts):
            j=H+subnets.index(subnet(h));edges.extend(((i,j),(j,i)))
        for j,s in enumerate(subnets):
            indices=[i for i,h in enumerate(self.hosts) if subnet(h)==s]
            # Subnet first fields: observed occupancy, current alerts, remembered
            # alerts, mean age, mean confidence; remaining fields zero.
            x=self.host_features[indices]
            nodes[H+j,:5]=(x[:,0].mean(),np.any(x[:,3:5]>0,axis=1).mean(),
                            np.any(x[:,5:7]>0,axis=1).mean(),
                            x[:,[self.fields.index('age_linear/'+k) for k in CHANNELS]].mean(),
                            x[:,[self.fields.index('confidence/'+k) for k in CHANNELS]].mean())
        return dict(node_features=nodes, edge_index=np.asarray(edges,dtype=np.int64).T,
                    edge_attr=np.ones((len(edges),1),dtype=np.float32),
                    global_features=self.global_features.copy(),
                    node_type=np.asarray([0]*H+[1]*3,dtype=np.int64),
                    node_mask=np.asarray([i<len(self.hosts) for i in range(H)]+[i<len(subnets) for i in range(3)]),
                    metadata=dict(host_indices={h:i for i,h in enumerate(self.hosts)},
                                  subnet_indices={s:H+i for i,s in enumerate(subnets)},
                                  edge_type='belongs_to / contains; possible inventory, not confirmed physical topology',
                                  subnet_fields=['observed_occupancy','current_alert_fraction','remembered_alert_fraction','mean_age','mean_confidence']))


    def decision_diagnostics(self, label, threshold=.5):
        """Research logging only; does not change available actions or rewards."""
        parts=label.split();target=parts[1] if len(parts)>1 else None
        risky=bool(parts and parts[0] in ('Remove','Restore') and target in self.history)
        stale=False
        if risky:
            *_,confidence=update_confidence(self.history[target]['last'],self.last_step,
                                            self.config['age_max'],self.config['decay'])
            stale=min(confidence[:3])<threshold
        return dict(high_risk_host_action=int(risky),stale_high_risk_host_action=int(risky and stale))
