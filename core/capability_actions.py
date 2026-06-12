from dataclasses import dataclass
@dataclass
class CapabilityAction:
    action_type:str
    priority:str
    source:str
    reason:str
