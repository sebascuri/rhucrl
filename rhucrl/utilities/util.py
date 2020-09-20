"""Utility functions for RHUCRL project."""
import importlib

from rllib.model import TransformedModel

from rhucrl.environment.wrappers import (
    AdversarialPendulumWrapper,
    MujocoAdversarialWrapper,
    NoisyActionRobustWrapper,
    ProbabilisticActionRobustWrapper,
)
from rhucrl.model import HallucinatedModel


def wrap_adversarial_environment(
    environment, wrapper_name, alpha, force_body_names=None
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
    elif wrapper_name == "adversarial_pendulum":
        environment.add_wrapper(
            AdversarialPendulumWrapper, alpha=alpha, force_body_names=force_body_names
        )
    else:
        raise NotImplementedError(f"{wrapper_name} not implemented.")
    return environment


def _get_default_dynamical_model(environment, known_dynamical_model, hallucinate):
    if known_dynamical_model is not None:
        if not isinstance(known_dynamical_model, TransformedModel):
            dynamical_model = TransformedModel(known_dynamical_model, [])
        else:
            dynamical_model = known_dynamical_model
    else:
        if hallucinate:
            dynamical_model = HallucinatedModel.default(environment)
        else:
            dynamical_model = TransformedModel.default(environment)
    return dynamical_model


def _get_default_reward_model(
    environment, known_reward_model, hallucinate, dynamical_model
):
    if known_reward_model is not None:
        if not isinstance(known_reward_model, TransformedModel):
            reward_model = TransformedModel(known_reward_model, [])
        else:
            reward_model = known_reward_model
    else:
        try:
            reward_model = TransformedModel(environment.env.reward_model(), [])
        except AttributeError:
            if hallucinate:
                reward_model = HallucinatedModel.default(
                    environment,
                    model_kind="rewards",
                    transformations=dynamical_model.forward_transformations,
                )
            else:
                reward_model = TransformedModel.default(
                    environment,
                    model_kind="rewards",
                    transformations=dynamical_model.forward_transformations,
                )
    return reward_model


def _get_default_termination_model(environment, known_termination_model):
    termination_model = known_termination_model
    if termination_model is None:
        try:
            termination_model = environment.env.termination_model()
        except AttributeError:
            pass
    return termination_model


def get_default_models(
    environment,
    known_dynamical_model=None,
    known_reward_model=None,
    known_termination_model=None,
    hallucinate=False,
    protagonist=True,
    weak_antagonist=False,
    strong_antagonist=False,
):
    """Get default protagonist/antagonist models."""
    dynamical_model = _get_default_dynamical_model(
        environment, known_dynamical_model, hallucinate
    )
    reward_model = _get_default_reward_model(
        environment, known_reward_model, hallucinate, dynamical_model
    )
    termination_model = _get_default_termination_model(
        environment, known_termination_model
    )

    if protagonist:
        return dynamical_model, reward_model, termination_model
    elif weak_antagonist:
        return (
            TransformedModel(
                dynamical_model.base_model, dynamical_model.forward_transformations
            ),
            TransformedModel(
                reward_model.base_model, reward_model.base_transformations
            ),
            termination_model,
        )
    elif strong_antagonist:
        return dynamical_model, reward_model, termination_model
    else:
        raise NotImplementedError


def get_agent(agent, environment, **kwargs):
    """Get default agent."""
    agent_module = importlib.import_module("rhucrl.agent")
    agent = getattr(agent_module, f"{agent}Agent").default(environment, **kwargs)
    return agent
