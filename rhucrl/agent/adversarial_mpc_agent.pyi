"""Python Script Template."""
from typing import Any

from rllib.algorithms.mpc.abstract_solver import MPCSolver
from rllib.policy import MPCPolicy

from .adversarial_agent import AdversarialAgent

class AdversarialMPCAgent(AdversarialAgent):
    policy: MPCPolicy
    def __init__(self, mpc_solver: MPCSolver, *args: Any, **kwargs: Any) -> None: ...
