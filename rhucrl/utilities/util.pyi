"""Utility functions for RHUCRL project."""
from typing import Any, List, Optional, Tuple

from rllib.model import AbstractModel, TransformedModel

from rhucrl.agent.adversarial_agent import AdversarialAgent
from rhucrl.environment import AdversarialEnv

def wrap_adversarial_environment(
    environment: AdversarialEnv,
    wrapper_name: str,
    alpha: float,
    force_body_names: Optional[List[str]] = ...,
) -> AdversarialEnv: ...
def get_default_models(
    environment: AdversarialEnv,
    known_model: Optional[AbstractModel] = ...,
    hallucinate: bool = ...,
    strong_antagonist: bool = ...,
) -> Tuple[TransformedModel, TransformedModel]: ...
def get_agent(
    agent: str, environment: AdversarialEnv, **kwargs: Any
) -> AdversarialAgent: ...
