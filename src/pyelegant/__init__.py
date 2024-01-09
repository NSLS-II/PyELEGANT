import importlib.metadata
import json
import os
from pathlib import Path
import re
import shlex
from subprocess import PIPE, Popen

import matplotlib.pylab as plt

plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["mathtext.rm"] = "serif"

__version__ = {"PyELEGANT": importlib.metadata.version(__name__)}

this_folder = os.path.dirname(os.path.abspath(__file__))

_default_rpn_defns_path = str(Path(this_folder).joinpath(".defns.rpn").resolve())
if "RPN_DEFNS" in os.environ:
    if not Path(os.environ["RPN_DEFNS"]).exists():
        print(
            f'Environment variable $RPN_DEFNS="{Path(os.environ["RPN_DEFNS"])}" does NOT exist.'
        )
        print(f'So, "{_default_rpn_defns_path}" will be used instead.')
        os.environ["RPN_DEFNS"] = _default_rpn_defns_path
else:
    print('Environment variable "RPN_DEFNS" is NOT defined.')
    print(f'So, "$RPN_DEFNS" is defined to be "{_default_rpn_defns_path}".')
    os.environ["RPN_DEFNS"] = _default_rpn_defns_path

del this_folder


std_print_enabled = dict(out=True, err=True)
sbatch_std_print_enabled = dict(out=True, err=True)

from . import correct, elebuilder, eleutil, errors, geneopt, linopt_correct, local

try:
    from . import latex
except:
    print("\n## pyelegant:WARNING ##")
    print(
        'Failed to load "latex" module. All relevant functionalities will result in errors.'
    )
from . import ltemanager, nonlin, notation, orbit, respmat, sdds, sigproc, twiss, util
from .local import (
    disable_sbatch_stderr,
    disable_sbatch_stdout,
    disable_stderr,
    disable_stdout,
    enable_sbatch_stderr,
    enable_sbatch_stdout,
    enable_stderr,
    enable_stdout,
    run,
)
from .remote import remote
from .twiss import calc_line_twiss, calc_ring_twiss, plot_twiss

if remote:
    __version__["ELEGANT"] = remote.__elegant_version__
else:
    p = Popen(shlex.split("which elegant"), stdout=PIPE, stderr=PIPE, encoding="utf-8")
    out, err = p.communicate()
    if out.strip():
        p = Popen(shlex.split("elegant"), stdout=PIPE, stderr=PIPE, encoding="utf-8")
        out, err = p.communicate()

        m = re.search("This is elegant ([^,]+),", out)
        if m is None:
            __version__["ELEGANT"] = "unknown"
        else:
            __version__["ELEGANT"] = m.groups()[0].strip()

        del m
    else:
        print("\n*** pyelegant:WARNING: ELEGANT not available.")
        __version__["ELEGANT"] = None

    del p, out, err
