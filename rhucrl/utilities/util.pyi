"""Utility functions for RHUCRL project."""
from typing import Any, List, Optional

from rllib.dataset.transforms import AbstractTransform
from rllib.model import AbstractModel, TransformedModel

from rhucrl.agent.adversarial_agent import AdversarialAgent
from rhucrl.environment import AdversarialEnv

def wrap_adversarial_environment(
    environment: AdversarialEnv,
    wrapper_name: str,
    alpha: float,
    force_body_names: Optional[List[str]] = ...,
) -> AdversarialEnv: ...
def get_default_model(
    environment: AdversarialEnv,
    known_model: Optional[AbstractModel] = ...,
    known_transforms: List[AbstractTransform] = ...,
    hallucinate: bool = ...,
    protagonist: bool = ...,
    weak_antagonist: bool = ...,
    strong_antagonist: bool = ...,
) -> TransformedModel: ...
def get_agent(
    agent: str, environment: AdversarialEnv, **kwargs: Any
) -> AdversarialAgent: ...
