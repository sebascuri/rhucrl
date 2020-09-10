"""Python Script Template."""
from .adversarial_environment import AdversarialEnv

def adversarial_to_protagonist_environment(
    environment: AdversarialEnv,
) -> AdversarialEnv: ...
def adversarial_to_antagonist_environment(
    environment: AdversarialEnv,
) -> AdversarialEnv: ...
