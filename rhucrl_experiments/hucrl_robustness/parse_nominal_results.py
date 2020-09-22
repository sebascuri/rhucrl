"""Python Script Template."""
from rhucrl_experiments.parse_results import get_all_data_frames

joint_df, protagonist_df, _, _ = get_all_data_frames(base_dir="runs/RARLAgent/")
protagonist_df.to_pickle("Nominal.pkl")
