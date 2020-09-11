"""Utility functions for RHUCRL project."""
import importlib

from rllib.model import TransformedModel

from rhucrl.environment import AdversarialEnv
from rhucrl.environment.wrappers import (
    HallucinationWrapper,
    MujocoAdversarialWrapper,
    NoisyActionRobustWrapper,
    ProbabilisticActionRobustWrapper,
)
from rhucrl.model import HallucinatedModel


def wrap_adversarial_environment(
    environment: AdversarialEnv, wrapper_name: str, alpha: float, force_body_names=None
):
    """Wrap environment with an adversarial wrapper."""
    if wrapper_name == "noisy_action":
        environment.add_wrapper(NoisyActionRobustWrapper, alpha=alpha)
    elif wrapper_name == "probabilistic_action":
        environment.add_wrapper(ProbabilisticActionRobustWrapper, alpha=alpha)
    elif wrapper_name == "external_force":
        environment.add_wrapper(
            MujocoAdversarialWrapper, alpha=alpha, force_body_names=force_body_names
        )
    else:
        raise NotImplementedError(f"{wrapper_name} not implemented.")
    return environment


def get_default_models(
    environment, known_model=None, hallucinate=False, strong_antagonist=True
):
    """Get default protagonist/antagonist models.

    Notes
    -----
    This has a side effect on the environment.
    """
    if known_model is not None:
        expected_model = TransformedModel(known_model, [])
        return expected_model, expected_model

    if hallucinate:
        dynamical_model = HallucinatedModel.default(environment)
        environment.add_wrapper(HallucinationWrapper)
    else:
        dynamical_model = TransformedModel.default(environment)

    expected_model = TransformedModel(
        dynamical_model.base_model, dynamical_model.forward_transformations
    )
    protagonist_dynamical_model = dynamical_model
    if strong_antagonist:
        antagonist_dynamical_model = dynamical_model
    else:
        antagonist_dynamical_model = expected_model

    return protagonist_dynamical_model, antagonist_dynamical_model


def get_agent(agent, environment, **kwargs):
    """Get default agent."""
    agent_module = importlib.import_module("rhucrl.agent")
    agent = getattr(agent_module, f"{agent}Agent").default(environment, **kwargs)
    return agent
