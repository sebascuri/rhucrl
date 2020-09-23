"""Python Script Template."""
import os

import pandas as pd

H_PARAMS = "hparams.json"
STATISTICS = "statistics.json"


def get_name(h_params):
    """Get experiment name from hyper parameter json file."""
    protagonist_name = h_params.protagonist_name[0]
    if protagonist_name in ["MVE", "Dyna", "STEVE"]:
        protagonist_name += f"-{h_params.base_agent[0]}"
    if protagonist_name in ["MVE", "Dyna", "STEVE", "BPTT"]:
        protagonist_name += f"-{h_params.num_steps[0]}"

    label = protagonist_name
    if h_params.hallucinate[0]:
        hallucination = True
        if h_params.strong_antagonist[0]:
            label, strong = "H-" + protagonist_name + "-strong", True
        else:
            label, strong = "H-" + protagonist_name + "-weak", False
    else:
        hallucination, strong = False, False

    wrapper = h_params.adversarial_wrapper[0]
    alpha = h_params.alpha[0]
    return dict(
        protagonist_name=protagonist_name,
        wrapper=wrapper,
        alpha=alpha,
        label=label,
        hallucination=hallucination,
        strong=strong,
    )


def get_player_data_frame(run_dir, player="Protagonist"):
    """Get Player dataframe."""
    agent_listdir = os.listdir(run_dir)
    agents = [*filter(lambda x: player in x, agent_listdir)]
    if len(agents) == 0:
        return pd.DataFrame()
    name = agents[0]
    player_dir = f"{run_dir}/{name}"
    player_dir = player_dir + "/" + os.listdir(player_dir)[0]
    if len(os.listdir(player_dir)) == 0:
        return pd.DataFrame()
    return pd.read_json(f"{player_dir}/statistics.json")


def extend_data_frame(df, name_dict):
    """Extend data frame with a name dictionary."""
    df["counter"] = range(len(df))
    for key, value in name_dict.items():
        df[key] = value


def get_all_data_frames(base_dir="runs/RARLAgent/"):
    """Get experiment data frames."""
    if base_dir[-1] != "/":
        base_dir = base_dir + "/"

    joint_df = pd.DataFrame()
    protagonist_df = pd.DataFrame()
    antagonist_df = pd.DataFrame()
    weak_antagonist_df = pd.DataFrame()

    for run in os.listdir(base_dir):
        environment = run.split(" ")[0]
        run_dir = f"{base_dir}/{run}"
        agent_listdir = os.listdir(run_dir)
        if H_PARAMS not in agent_listdir:
            continue
        hyper_params = pd.read_json(f"{run_dir}/{H_PARAMS}")
        name_dict = get_name(hyper_params)
        name_dict.update(environment=environment)

        joint_df_ = pd.read_json(f"{run_dir}/{STATISTICS}")
        extend_data_frame(joint_df_, name_dict)

        protagonist_df_ = get_player_data_frame(run_dir, player="Protagonist")
        extend_data_frame(protagonist_df_, name_dict)

        antagonist_df_ = get_player_data_frame(run_dir, player="Antagonist")
        extend_data_frame(antagonist_df_, name_dict)

        weak_antagonist_df_ = get_player_data_frame(run_dir, player="WeakAntagonist")
        extend_data_frame(weak_antagonist_df_, name_dict)

        joint_df = pd.concat([joint_df, joint_df_])
        protagonist_df = pd.concat([protagonist_df, protagonist_df_])
        antagonist_df = pd.concat([antagonist_df, antagonist_df_])
        weak_antagonist_df = pd.concat([weak_antagonist_df, weak_antagonist_df_])

    return joint_df, protagonist_df, antagonist_df, weak_antagonist_df


if __name__ == "__main__":

    base_dir = "runs/RARLAgent/"
    # base_dir = "runs/ZeroSumAgent"
    joint, protagonist, antagonist, weak_antagonist = get_all_data_frames(base_dir)
    # protagonist.to_pickle("Nominal.pkl")
