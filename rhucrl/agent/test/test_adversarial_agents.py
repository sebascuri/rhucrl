import pytest

from rhucrl.utilities.util import get_agent, get_default_models, get_environment


@pytest.fixture(params=[True, False])
def hallucinate(request):
    return request.param


@pytest.fixture(params=[True, False])
def strong_antagonist(request):
    return request.param


@pytest.fixture(params=["TD3", "SAC", "MPO"])
def base_agent(request):
    return request.param


@pytest.fixture(params=["SVG", "BPTT", "MVE", "STEVE", "Dyna"])
def model_based_agent(request):
    return request.param


@pytest.fixture(params=["PendulumAdvEnv-v0"])
def environment_name(request):
    return request.param


def delete_logs(robust_agent):
    for agent in robust_agent.agents:
        agent.logger.delete_directory()
    robust_agent.logger.delete_directory()


class TestAdversarialMPC(object):
    def test_model_free(self, environment_name):
        environment = get_environment(environment=environment_name)
        agent = get_agent("AdversarialMPC", environment)
        assert agent.policy.dim_state == environment.dim_state
        assert agent.policy.dim_action == environment.dim_action
        assert agent.weak_antagonist_agent is None
        delete_logs(agent)

    def test_model_model_based(self, hallucinate, strong_antagonist, environment_name):
        environment = get_environment(environment=environment_name)
        protagonist_model, antagonist_model = get_default_models(
            environment, strong_antagonist=strong_antagonist, hallucinate=hallucinate
        )
        agent = get_agent(
            "AdversarialMPC",
            environment,
            protagonist_dynamical_model=protagonist_model,
            antagonist_dynamical_model=antagonist_model,
        )
        assert agent.policy.dim_state[0] == environment.dim_state[0]

        raw_env = get_environment(environment=environment_name)
        if hallucinate:
            assert (
                agent.policy.dim_action[0]
                == raw_env.dim_action[0] + raw_env.dim_state[0]
            )
        else:
            assert agent.policy.dim_action[0] == raw_env.dim_action[0]

        assert agent.protagonist_agent.dynamical_model.dim_state == raw_env.dim_state
        assert agent.protagonist_agent.dynamical_model.dim_action == raw_env.dim_action
        assert agent.weak_antagonist_agent is None
        delete_logs(agent)


class TestRARL(object):
    def test_model_free(self, base_agent, environment_name):
        environment = get_environment(environment=environment_name)
        agent = get_agent(
            "RARL", environment, protagonist_name=base_agent, antagonist_name=base_agent
        )
        # Test Names.
        for agent_ in agent.agents:
            assert agent_.name[: len(base_agent)] == base_agent

        # Test Protagonist
        protagonist = agent.protagonist_agent
        assert protagonist.policy.dim_action == environment.protagonist_dim_action
        assert protagonist.policy.dim_state == environment.dim_state

        # Test Antagonist
        antagonist = agent.antagonist_agent
        assert antagonist.policy.dim_action == environment.antagonist_dim_action
        assert antagonist.policy.dim_state == environment.dim_state

        # Test Weak Antagonist
        assert agent.weak_antagonist_agent is None

        delete_logs(agent)

    def test_model_model_based(
        self,
        hallucinate,
        strong_antagonist,
        model_based_agent,
        base_agent,
        environment_name,
    ):
        environment = get_environment(environment=environment_name)

        protagonist_model, antagonist_model = get_default_models(
            environment, strong_antagonist=strong_antagonist, hallucinate=hallucinate
        )

        agent = get_agent(
            "RARL",
            environment,
            protagonist_name=model_based_agent,
            antagonist_name=model_based_agent,
            base_agent=base_agent,
            protagonist_dynamical_model=protagonist_model,
            antagonist_dynamical_model=antagonist_model,
        )
        protagonist_dim_action = environment.protagonist_dim_action
        antagonist_dim_action = environment.antagonist_dim_action

        dim_state = environment.dim_state
        # Test Names.
        for agent_ in agent.agents:
            assert agent_.name[: len(model_based_agent)] == model_based_agent
            if base_agent in ["MVE", "STEVE", "Dyna"]:
                assert agent.algorithm.base_algorithm == base_agent

        # Test protagonist
        protagonist = agent.protagonist_agent

        assert protagonist.policy.dim_state == dim_state
        assert protagonist.dynamical_model.dim_state == dim_state
        assert protagonist.dynamical_model.dim_action == protagonist_dim_action
        if hallucinate:
            assert protagonist.policy.dim_action == (
                dim_state[0] + protagonist_dim_action[0],
            )
        else:
            assert protagonist.policy.dim_action == protagonist_dim_action

        # Test antagonist
        antagonist = agent.antagonist_agent

        assert antagonist.policy.dim_state == dim_state
        assert antagonist.dynamical_model.dim_state == dim_state
        assert antagonist.dynamical_model.dim_action == antagonist_dim_action
        if hallucinate and strong_antagonist:
            assert antagonist.policy.dim_action == (
                dim_state[0] + antagonist_dim_action[0],
            )
        else:
            assert antagonist.policy.dim_action == antagonist_dim_action

        # Test weak antagonist
        assert agent.weak_antagonist_agent is None
        delete_logs(agent)


class TestZeroSum(object):
    def test_model_free(self, base_agent, environment_name):

        environment = get_environment(environment=environment_name)

        agent = get_agent(
            "ZeroSum",
            environment,
            protagonist_name=base_agent,
            antagonist_name=base_agent,
        )
        # Test Names.
        for agent_ in agent.agents:
            assert agent_.name[: len(base_agent)] == base_agent

        # Test Protagonist
        protagonist = agent.protagonist_agent
        assert protagonist.policy.dim_action == environment.dim_action
        assert protagonist.policy.dim_state == environment.dim_state

        # Test Antagonist
        antagonist = agent.antagonist_agent
        assert antagonist.policy.dim_action == environment.dim_action
        assert antagonist.policy.dim_state == environment.dim_state

        delete_logs(agent)

    def test_model_model_based(
        self,
        hallucinate,
        strong_antagonist,
        model_based_agent,
        base_agent,
        environment_name,
    ):
        environment = get_environment(environment=environment_name)

        protagonist_model, antagonist_model = get_default_models(
            environment, strong_antagonist=strong_antagonist, hallucinate=hallucinate
        )

        agent = get_agent(
            "ZeroSum",
            environment,
            protagonist_name=model_based_agent,
            antagonist_name=model_based_agent,
            base_agent=base_agent,
            protagonist_dynamical_model=protagonist_model,
            antagonist_dynamical_model=antagonist_model,
        )
        raw_env = get_environment(environment.env_name)
        dim_action = raw_env.dim_action
        dim_h_action = environment.dim_action
        dim_state = environment.dim_state
        if hallucinate:
            assert dim_h_action[0] == dim_action[0] + dim_state[0]
        else:
            assert dim_h_action[0] == dim_action[0]
        # Test Names.
        for agent_ in agent.agents:
            assert agent_.name[: len(model_based_agent)] == model_based_agent
            if base_agent in ["MVE", "STEVE", "Dyna"]:
                assert agent.algorithm.base_algorithm == base_agent

        # Test protagonist
        protagonist = agent.protagonist_agent

        assert protagonist.policy.dim_state == dim_state
        assert protagonist.dynamical_model.dim_state == dim_state
        assert protagonist.dynamical_model.dim_action == dim_action
        assert protagonist.policy.dim_action == dim_h_action

        # Test antagonist
        antagonist = agent.antagonist_agent

        assert antagonist.policy.dim_state == dim_state
        assert antagonist.dynamical_model.dim_state == dim_state
        assert antagonist.dynamical_model.dim_action == dim_action
        assert antagonist.policy.dim_action == dim_h_action

        # Test weak antagonist
        weak_antagonist = agent.weak_antagonist_agent
        if strong_antagonist and hallucinate:
            assert weak_antagonist.policy.dim_state == dim_state
            assert weak_antagonist.dynamical_model.dim_state == dim_state
            assert weak_antagonist.dynamical_model.dim_action == dim_action
            assert weak_antagonist.policy.dim_action == dim_h_action
        else:
            assert weak_antagonist is None
        delete_logs(agent)
