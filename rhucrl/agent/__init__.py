"""Python Script Template."""
from .adversarial_mpc_agent import AdversarialMPCAgent
from .maximin_agent import MaxiMinAgent
from .rarl_agent import HRARLAgent, RAPAgent, RARLAgent
from .rhucrl_agent import BestResponseAgent, RHUCRLAgent

AGENTS = ["RARL", "RAP", "HRARL", "AdversarialMPC", "RHUCRL", "MaxiMin", "BestResponse"]
