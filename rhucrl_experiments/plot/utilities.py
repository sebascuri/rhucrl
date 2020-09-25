"""Python Script Template."""
from collections import OrderedDict

from matplotlib import rcParams

LABELS = OrderedDict(
    robust=r"\textbf{RH-UCRL}",
    weak=r"RH-UCRL-weak",
    expected="H-UCRL",
    bptt="BPTT",
    td3="TD3",
    sac="SAC",
    mpo="MPO",
    ppo="PPO",
    vmpo="VMPO",
)

COLORS = OrderedDict(
    robust="C0",
    weak="C1",
    expected="C2",
    bptt="C3",
    td3="C4",
    sac="C5",
    mpo="C6",
    ppo="C7",
    vmpo="C8",
)


def set_figure_params(serif=False, fontsize=9):
    """Define default values for font, fontsize and use latex.

    Parameters
    ----------
    serif: bool, optional
        Whether to use a serif or sans-serif font.
    fontsize: int, optional
        Size to use for the font size

    """
    latex_preamble = (
        r"\DeclareMathAlphabet{\mathcal}{OMS}{cmsy}{m}{n} \usepackage{newtxmath}"
    )

    params = {
        "font.serif": [
            "Times",
            "Palatino",
            "New Century Schoolbook",
            "Bookman",
            "Computer Modern Roman",
        ]
        + rcParams["font.serif"],
        "font.sans-serif": [
            "Times",
            "Helvetica",
            "Avant Garde",
            "Computer Modern Sans serif",
        ]
        + rcParams["font.sans-serif"],
        "font.family": "sans-serif",
        "text.usetex": True,
        # Make sure mathcal doesn't use the Times style
        "text.latex.preamble": latex_preamble,
        "axes.labelsize": fontsize,
        "axes.linewidth": 0.75,
        "font.size": fontsize,
        "legend.fontsize": fontsize * 0.7,
        "xtick.labelsize": fontsize * 8 / 9,
        "ytick.labelsize": fontsize * 8 / 9,
        "figure.dpi": 100,
        "savefig.dpi": 600,
        "legend.numpoints": 1,
    }

    if serif:
        params["font.family"] = "serif"

    rcParams.update(params)
