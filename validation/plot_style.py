"""Shared scientific plot settings; missing LaTeX/fonts print installation guidance."""

def plot_style():
    import shutil
    import matplotlib.pyplot as plt
    from matplotlib import rcParams, font_manager
    from cycler import cycler
    colors = ['navy', 'coral', 'c', 'crimson', 'springgreen', 'gold',
              'dodgerblue', 'hotpink', 'forestgreen', 'moccasin', 'darkred', 'teal']
    rcParams.update({'axes.labelsize': 14, 'axes.titlesize': 14, 'legend.fontsize': 14,
        'xtick.labelsize': 12, 'ytick.labelsize': 12, 'text.usetex': True,
        'text.latex.preamble': r'\usepackage{txfonts}', 'font.size': 14,
        'font.family': 'serif', 'font.serif': 'times', 'mathtext.default': 'sf',
        'lines.markeredgewidth': 1, 'lines.linewidth': 2,
        'axes.prop_cycle': cycler(color=colors)})
    missing = [name for name in ('latex', 'dvipng') if shutil.which(name) is None]
    if missing:
        print(f"Install {', '.join(missing)} separately for LaTeX plot text.", flush=True)
        rcParams['text.usetex'] = False
    try:
        font_manager.findfont('Times', fallback_to_default=False)
    except ValueError:
        print('Install a Times font separately for the requested plot font.', flush=True)
        rcParams['font.serif'] = ['DejaVu Serif']
    return plt

