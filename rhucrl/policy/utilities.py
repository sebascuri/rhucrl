"""Python Script Template."""


class ProtagonistMode(object):
    """Context Manager to set the policy to be protagonist."""

    def __init__(self, policy, protagonist=True):
        self.policy = policy
        self.old = self.policy.protagonist
        self.protagonist = protagonist

    def __enter__(self):
        """Enter into a Hallucination Context."""
        self.policy.protagonist = self.protagonist

    def __exit__(self, *args):
        """Exit the Hallucination Context."""
        self.policy.protagonist = self.old


class AntagonistMode(ProtagonistMode):
    """Context Manager to set the policy to be protagonist."""

    def __init__(self, policy, protagonist=False):
        super().__init__(policy, protagonist=protagonist)
