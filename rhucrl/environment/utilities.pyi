from typing import Any

from .adversarial_environment import AdversarialEnv

def adversarial_to_protagonist_environment(
    environment: AdversarialEnv,
) -> AdversarialEnv: ...
def adversarial_to_antagonist_environment(
    environment: AdversarialEnv,
) -> AdversarialEnv: ...

class Hallucinate(object):
    environment: AdversarialEnv
    hallucinate_rewards: bool
    def __init__(
        self, environment: AdversarialEnv, hallucinate_rewards: bool = ...
    ) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(self, *args: Any) -> None: ...
