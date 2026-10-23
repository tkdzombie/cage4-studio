"""Extensible registry. IPPO and MAPPO are deliberately unimplemented."""
import importlib
REGISTRY = {}
def register(name):
    def decorate(cls):
        if name in REGISTRY:
            raise ValueError('Duplicate algorithm: '+name)
        REGISTRY[name] = cls
        return cls
    return decorate

def create(name, inputs, config):
    for module in config.get('plugins',[]):
        importlib.import_module(module)  # explicit trusted research configuration only
    if name not in REGISTRY:
        raise ValueError(f'Unknown/unimplemented algorithm {name}; available: {sorted(REGISTRY)}')
    return REGISTRY[name](inputs,config)
