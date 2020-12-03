"""Python Script Template."""
from typing import Any

from rhucrl.policy.adversarial_policy import AdversarialPolicy

class ProtagonistMode(object):
    policy: AdversarialPolicy
    old: bool
    protagonist: bool
    def __init__(self, policy: AdversarialPolicy, protagonist: bool = ...) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(self, *args: Any, **kwargs: Any) -> None: ...

class AntagonistMode(ProtagonistMode): ...
