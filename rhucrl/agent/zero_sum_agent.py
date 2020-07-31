"""Python Script Template."""
from rllib.dataset.datatypes import Observation, RawObservation

from .adversarial_agent import AdversarialAgent


class ZeroSumAgent(AdversarialAgent):
    """Zero-Sum Agent.

    Zero-Sum has two dependent agents.
    The protagonist receives (s, a, r, s') and the adversary (s, a, -r, s').

    """

    def __init__(
        self,
        protagonist_agent,
        adversarial_agent,
        train_frequency=1,
        num_rollouts=0,
        exploration_steps=0,
        exploration_episodes=0,
        gamma=0.99,
        comment="",
    ):
        super().__init__(
            protagonist_agent=protagonist_agent,
            adversarial_agent=adversarial_agent,
            train_frequency=train_frequency,
            num_rollouts=num_rollouts,
            exploration_steps=exploration_steps,
            exploration_episodes=exploration_episodes,
            gamma=gamma,
            comment=comment,
        )
        self.policy = protagonist_agent.policy
        assert (
            self.protagonist_agent.policy.base_policy
            is self.adversarial_agent.policy.base_policy
        ), "Adversarial and Protagonist agent must share the base policy."

    def observe(self, observation: Observation) -> None:
        """Send observations to both players.

        This is the crucial method as it needs to separate the actions.
        """
        super().observe(observation)
        state, action, reward, next_state, done, *r = observation
        protagonist_observation = RawObservation(
            state, action, reward, next_state, done
        ).to_torch()
        adversarial_observation = RawObservation(
            state, action, -reward, next_state, done
        ).to_torch()

        self.send_observations(protagonist_observation, adversarial_observation)
