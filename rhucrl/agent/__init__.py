"""Python Script Template."""
from .action_robust_agent import ActionRobustAgent
from .action_robust_hucrl_agent import ActionRobustHUCRLAgent
from .adversarial_mpc_agent import AdversarialMPCAgent
from .dr_agent import DomainRandomizationAgent
from .ep_opt import EPOPTAgent
from .maximin_agent import MaxiMinAgent
from .rarl_agent import HRARLAgent, RAPAgent, RARLAgent
from .rhucrl_agent import BestResponseAgent, RHUCRLAgent

AR_AGENTS = [
    "ActionRobust",
    "ActionRobustHUCRL",
]
DR_AGENTS = [
    "DomainRandomization",
    "EPOPT",
]
ADVERSARIAL_AGENTS = [
    "RARL",
    "RAP",
    "HRARL",
    "AdversarialMPC",
    "RHUCRL",
    "MaxiMin",
    "BestResponse",
]
