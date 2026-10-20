"""Feature hashing of ONLY the agent's own official observation.

No environment object, topology, global host inventory or red telemetry enters
this encoder. Stable BLAKE2b avoids Python hash randomization across processes.
"""
import hashlib
import math
from collections.abc import Mapping
import numpy as np
ENCODER = 'blue-observation-hash/1'

def encode(observation, dim=512):
    out = np.zeros(dim, dtype=np.float32)
    def add(token):
        digest = hashlib.blake2b(token.encode('utf-8'), digest_size=8).digest()
        out[int.from_bytes(digest,'big') % dim] += 1
    def visit(value, path):
        if isinstance(value, Mapping):
            for key, child in sorted(value.items(), key=lambda item:str(item[0])):
                visit(child, path+'/'+str(key))
        elif isinstance(value, (list,tuple,np.ndarray)):
            for index, child in enumerate(value):
                visit(child, path+'/'+str(index))
        else:
            # Categories, enums, IP addresses and numeric values use stable strings.
            if isinstance(value,(float,np.floating)) and not math.isfinite(float(value)):
                raise ValueError('Non-finite observation')
            add(path+'='+str(value))
    visit(observation,'obs')
    return np.tanh(out)
