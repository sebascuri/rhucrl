"""Project Plotter utilities."""
from collections import OrderedDict

from matplotlib import rcParams

LABELS = OrderedDict(
    RHUCRL=r"\textbf{RH-UCRL}",
    BestResponse=r"BestResponse",
    MaxiMin="MaxiMin",
    HUCRL="H-UCRL",
    sac="SAC",
    ppo="PPO",
    RARL="RARL",
    RAP="RAP",
    DomainRandomization="DomainRandomization",
    EPOPT="EPOPT",
    baseline="baseline",
)

COLORS = OrderedDict(
    RHUCRL="C0",
    BestResponse="C1",
    MaxiMin="C2",
    HUCRL="C3",
    sac="C3",
    ppo="C3",
    RARL="C4",
    RAP="C4",
    DomainRandomization="C4",
    EPOPT="C4",
    baseline="C4",
)

LINESTYLE = OrderedDict(
    RHUCRL="solid",
    BestResponse="dashed",
    MaxiMin="dotted",
    HUCRL="dashed",
    sac="solid",
    ppo="dashed",
    RARL="dotted",
    RAP="dashed",
    DomainRandomization="dashed",
    EPOPT="dotted",
    baseline="dotted",
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
