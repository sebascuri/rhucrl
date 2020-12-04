"""Python Script Template."""
from hucrl.environment.hallucination_wrapper import HallucinationWrapper
from hucrl.model.hallucinated_model import HallucinatedModel
from rllib.model.transformed_model import TransformedModel
from rllib.util.training.agent_training import train_agent
from rllib.util.utilities import set_random_seed

from applications.util import get_agent
from rhucrl.agent import AGENTS as ADVERSARIAL_AGENTS
from rhucrl.agent.antagonist_agent import AntagonistAgent
from rhucrl.environment import AdversarialEnv
from rhucrl.environment.wrappers import MujocoAdversarialWrapper

alpha = 10.0
hallucinate = False
name = "MBHalfCheetah-v0"
agent_name = "RARL"
base_agent_name = "SAC"
beta = 1.0
seed = 0
num_episodes = 30
max_steps = 1000
force_body_names = ["torso", "bfoot", "ffoot"]

set_random_seed(seed=seed)
# Define environment
environment = AdversarialEnv(name, seed=seed)
if agent_name in ADVERSARIAL_AGENTS:
    environment.add_wrapper(
        MujocoAdversarialWrapper, alpha=alpha, force_body_names=force_body_names
    )
if agent_name in ["RHUCRL", "BestResponse", "HRARL"] or hallucinate:
    dynamical_model = HallucinatedModel.default(environment, beta=beta)
    environment.add_wrapper(HallucinationWrapper)
else:
    dynamical_model = TransformedModel.default(environment)

# Define agent
agent = get_agent(
    environment,
    agent_name=agent_name,
    dynamical_model=dynamical_model,
    base_agent_name=base_agent_name,
)

train_agent(
    agent=agent,
    environment=environment,
    num_episodes=num_episodes,
    max_steps=max_steps,
    print_frequency=1,
)

if agent_name not in ADVERSARIAL_AGENTS:
    environment.add_wrapper(
        MujocoAdversarialWrapper, alpha=alpha, force_body_names=force_body_names
    )
robust_antagonist = AntagonistAgent.default(
    environment=environment, protagonist_agent=agent, base_agent_name="SAC"
)
train_agent(
    robust_antagonist,
    environment=environment,
    num_episodes=num_episodes,
    max_steps=max_steps,
    print_frequency=1,
)
