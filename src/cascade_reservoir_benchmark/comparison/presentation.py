"""Paper display names and typography only; no data or metric transformations."""
import re

from matplotlib.text import Text

FONT_FAMILY = "Times New Roman"
LABEL_FT = 13
TICK_FT = 11
LEGEND_FT = 10
PANEL_FT = 12
FIGURE_DPI = 600
ACTION_FIGSIZE = (13.0, 8.6)
RESERVOIR_LABELS = {"XW": "Reservoir X", "NZD": "Reservoir N"}
FONT_RC = {
    "font.family": FONT_FAMILY,
    "font.serif": [FONT_FAMILY],
    "font.size": LABEL_FT,
    "axes.labelsize": LABEL_FT,
    "axes.titlesize": PANEL_FT,
    "xtick.labelsize": TICK_FT,
    "ytick.labelsize": TICK_FT,
    "legend.fontsize": LEGEND_FT,
    "mathtext.fontset": "custom",
    "mathtext.rm": FONT_FAMILY,
    "mathtext.it": f"{FONT_FAMILY}:italic",
    "mathtext.bf": f"{FONT_FAMILY}:bold",
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "path",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
}


def display_text(text: str) -> str:
    text = re.sub(r"xiaowan", "Reservoir X", text, flags=re.IGNORECASE)
    text = re.sub(r"nuozhadu", "Reservoir N", text, flags=re.IGNORECASE)
    return re.sub(r"\bISO\b(?!-RF)", "ISO-RF", text, flags=re.IGNORECASE)


def finalize_figure(fig) -> None:
    """Change visible artists only; internal station keys and result tables stay intact."""
    for item in fig.findobj(match=Text):
        item.set_text(display_text(item.get_text()))
        item.set_fontfamily(FONT_FAMILY)
