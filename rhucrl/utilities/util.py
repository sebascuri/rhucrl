"""Utility functions for RHUCRL project."""
import importlib

from rllib.model import TransformedModel

from rhucrl.environment import AdversarialEnv
from rhucrl.environment.wrappers import (
    HallucinationWrapper,
    NoisyActionRobustWrapper,
    ProbabilisticActionRobustWrapper,
)
from rhucrl.model import HallucinatedModel


def get_environment(environment, adversarial_wrapper=None, alpha=0.1, seed=0, **kwargs):
    """Get environment."""
    if adversarial_wrapper == "noisy_action":
        environment = AdversarialEnv(environment, seed=seed, **kwargs)
        environment.add_wrapper(NoisyActionRobustWrapper, alpha=alpha)
    elif adversarial_wrapper == "probabilistic_action":
        environment = AdversarialEnv(environment, seed=seed, **kwargs)
        environment.add_wrapper(ProbabilisticActionRobustWrapper, alpha=alpha)
    else:
        environment = AdversarialEnv(environment, seed=seed, alpha=alpha, **kwargs)
    return environment


def get_default_models(environment, hallucinate=False, strong_antagonist=True):
    """Get default protagonist/antagonist models.

    Notes
    -----
    This has a side effect on the environment.
    """
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
