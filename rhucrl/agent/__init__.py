"""Python Script Template."""
from .action_robust_agent import NoisyActionRobustAgent, ProbabilisticActionRobustAgent
from .adversarial_mpc_agent import AdversarialMPCAgent
from .maximin_agent import MaxiMinAgent
from .ep_opt import EPOPTAgent
from .rarl_agent import HRARLAgent, RAPAgent, RARLAgent
from .rhucrl_agent import BestResponseAgent, RHUCRLAgent
from .arhucrl_agent import NoisyActionRHUCRLAgent, ProbabilisticActionRHUCRLAgent

AGENTS = [
    "RARL",
    "RAP",
    "HRARL",
    "AdversarialMPC",
    "EPOPT",
    "RHUCRL",
    "MaxiMin",
    "BestResponse",
    "NoisyActionRobust",
    "ProbabilisticActionRobust",
    "NoisyActionRHUCRL",
    "ProbabilisticActionRHUCRL",
]
