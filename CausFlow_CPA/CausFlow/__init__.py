"""CausFlow package with backwards-compatible lazy original API imports."""
from .cpa_adapter import CausFlowCPADose


def __getattr__(name):
    if name == 'CausFlow':
        from .CausFlow import CausFlow
        return CausFlow
    if name == 'Dataset':
        from .Dataset import Dataset
        return Dataset
    raise AttributeError(name)
