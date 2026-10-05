# Language
I am a scientist, not a software developer. I prefer simple words like "jobs", "configurations", "settings", "parameters", "splitting", "up to date" over terms like "manifests", "shards", "being current", etc.

# Programming
Use serena index

Write simple, easy to understand, well documented code. Favour performance and simplicity over defensive programming.

Be generous with comments and documentation.

If implementing an equation, or using a specific algorithm, write it in the docs. Add link to the original if implementing something specific based on some resource (e.g. you can give a link to the Whittle paper in a function that codes the Gaussian likelihood in gravitational wave data analysis).

There is no need to restrict the code to use PEP standard number of characters per line. Use a limit of 100 instead.

Good example:
```python
import numpy as np
import numpy.typing as npt

def normalize(a: npt.ArrayLike) -> np.ndarray:
    """
    Return a normalized copy of the input array.
    $x \rightarrow \dfrac{x - \langle x \rangle}{\mathrm{std}(x)}$
    """
    a = np.asarray(a)
    return (a - a.mean()) / a.std()
```

Bad example:
```python
from typing import Any
import numpy as np
def normalize(a : Any) -> Any:
    a = np.asarray(a)
    if np.any(np.isnan(a) | np.isinf(a)):
        raise ValueError('Array must not contain infs or nans')
    if len(a) == 0:
        raise ValueError('Array must have more than 1 values')
    mean_a = np.mean(a)
    std_a = np.std(a)
    return (a - mean_a) / std_a
```

Unless requested explicitly, programs should by default run with a single thread. Use os.environment to set this.

# Testing and Debugging
Use the `gw` micromamba environment. Ask if you need any mandatory extra packages not already present.

# Plots
Use the following style
```
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from cycler import cycler
my_colors = ['navy', 'coral', 'c', 'crimson', 'springgreen', 'gold',
             'dodgerblue', 'hotpink', 'forestgreen', 'moccasin', 'darkred', 'teal']
rc_params = {'axes.labelsize': 14,
             'axes.titlesize': 14,
             'legend.fontsize': 14,
             'xtick.labelsize': 12,
             'ytick.labelsize': 12,
             'text.usetex' : True,
             'text.latex.preamble' : '\\usepackage{txfonts}',
             'font.size': 14,
             'font.family': 'serif',
             'font.serif': 'times',
             'mathtext.default': 'sf',
             'lines.markeredgewidth' : 1,
             'lines.linewidth' : 2,
             'axes.prop_cycle' : cycler(color=my_colors),
            }
rcParams.update(rc_params)

def sci_fmt(x: float) -> str:
    # e.g. 10299.546 -> 1.0 x 10^4
    m, e = (f'{x:.1e}').split('e')
    m = float(m)
    e = int(e)
    if e in [-1,0,1]: return r'%.1f' % x
    else: return r'%.1f \times 10^{%d}' % (float(m), int(e))
```

If latex or something is unavailable for the above, print an instruction for me to install it separately.

Use standard guidelines for presenting graphics. All scientifically relevant details should be visible and legible.

If saving, save them as jpg with dpi=200.

# Writing
Reports should describe the science, algorithms, or documentation in a stand-alone manner. They should not sound like they are your response to given prompts. They should not discuss progress and recent developements.

Unless explicitly requested otherwise, reports should be in the markdown format.
