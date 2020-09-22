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
    alpha = h_params.alpha[0]
    hallucinate = h_params.hallucinate[0]
    strong = h_params.strong_antagonist[0]
    wrapper = h_params.adversarial_wrapper[0]
    return f"{protagonist_name}_{wrapper}_{alpha}_{hallucinate}_{strong}"


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
        name = f"{environment}_{get_name(hyper_params)}"

        joint_df_ = pd.read_json(f"{run_dir}/{STATISTICS}")
        joint_df_["counter"] = range(len(joint_df_))
        joint_df_["name"] = name

        protagonist_df_ = get_player_data_frame(run_dir, player="Protagonist")
        protagonist_df_["counter"] = range(len(protagonist_df_))
        protagonist_df_["name"] = name

        antagonist_df_ = get_player_data_frame(run_dir, player="Antagonist")
        antagonist_df_["counter"] = range(len(antagonist_df_))
        antagonist_df_["name"] = name

        weak_antagonist_df_ = get_player_data_frame(run_dir, player="WeakAntagonist")
        weak_antagonist_df_["counter"] = range(len(weak_antagonist_df_))
        weak_antagonist_df_["name"] = name

        joint_df = pd.concat([joint_df, joint_df_])
        protagonist_df = pd.concat([protagonist_df, protagonist_df_])
        antagonist_df = pd.concat([antagonist_df, antagonist_df_])
        weak_antagonist_df = pd.concat([weak_antagonist_df, weak_antagonist_df_])

    return joint_df, protagonist_df, antagonist_df, weak_antagonist_df
