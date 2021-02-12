"""Project Plotter utilities."""
from collections import OrderedDict

from matplotlib import rcParams

AXES = {
    "HalfCheetah": (0, 0),
    "Hopper": (0, 1),
    "InvertedPendulum": (0, 2),
    "Reacher2d": (1, 0),
    "Swimmer": (1, 1),
    "Walker2d": (1, 2),
}
LABELS = OrderedDict(
    RHUCRL=r"\textbf{RH-UCRL}",
    BestResponse=r"BestResponse",
    MaxiMin="MaxiMin",
    MaxiMinMF="MaxiMinMF",
    HUCRL="H-UCRL",
    BPTT="H-UCRL",
    SAC="SAC",
    PPO="PPO",
    RARL="RARL",
    RAP="RAP",
    DomainRandomization="DomainRandomization",
    EPOPT="EPOPT",
    ActionRobust="ActionRobust",
    baseline="Baseline",
)

COLORS = OrderedDict(
    RHUCRL="C0",
    BestResponse="C1",
    MaxiMin="C2",
    MaxiMinMF="C2",
    HUCRL="C3",
    BPTT="C3",
    SAC="C3",
    PPO="C3",
    RARL="C4",
    RAP="C4",
    DomainRandomization="C4",
    EPOPT="C4",
    baseline="C4",
    ActionRobust="C4",
)

LINESTYLE = OrderedDict(
    RHUCRL="solid",
    BestResponse="dashed",
    MaxiMin="dotted",
    MaxiMinMF="dashdot",
    HUCRL=(0, (1, 5)),
    BPTT=(0, (1, 5)),
    SAC=(0, (5, 10)),
    PPO=(0, (3, 10, 1, 10)),
    RARL=(0, (1, 1)),
    RAP=(0, (5, 1)),
    DomainRandomization=(0, (1, 1)),
    EPOPT=(0, (5, 1)),
    baseline=(0, (1, 1)),
    ActionRobust=(0, (1, 1)),
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
