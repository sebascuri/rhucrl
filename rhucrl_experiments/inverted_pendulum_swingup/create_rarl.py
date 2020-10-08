"""Python Script Template."""

from rllib.dataset.datatypes import Observation

from rhucrl_experiments.run import init_experiment
from rhucrl_experiments.utilities import get_command_line_parser

for wrapper in ["adversarial_pendulum"]:
    for alpha in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]:
        parser = get_command_line_parser()
        parser.set_defaults(
            environment="PendulumSwingUp-v0",
            agent="RARL",
            adversarial_wrapper=wrapper,
            alpha=alpha,
            force_body_names=["gravity", "mass"],
            horizon=40,
            max_steps=500,
            train_episodes=10,
            train_antagonist_episodes=0,
            exploration_episodes=0,
            eval_episodes=5,
            nominal_model=False,
        )
        args = parser.parse_args()
        agent, environment = init_experiment(args)
        agent.eval()
        for _ in range(5):
            state = environment.reset()
            agent.start_episode()
            for i in range(args.max_steps):
                action = environment.action_space.sample()
                next_state, reward, done, info = environment.step(action)
                agent.observe(Observation(state, action, reward, next_state).to_torch())
                state = next_state
            agent.end_episode()
