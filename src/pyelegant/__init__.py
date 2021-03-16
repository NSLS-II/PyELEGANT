import os
import json
from subprocess import Popen, PIPE
import shlex
from pathlib import Path

import matplotlib.pylab as plt
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'

this_folder = os.path.dirname(os.path.abspath(__file__))
__version__ = {}
with open(os.path.join(this_folder, 'version.json'), 'r') as f:
    __version__['PyELEGANT'] = json.load(f)

with open(os.path.join(this_folder, 'facility.json'), 'r') as f:
    facility_name = json.load(f)['name']

_default_rpn_defns_path = str(Path(this_folder).joinpath('.defns.rpn').resolve())
if 'RPN_DEFNS' in os.environ:
    if not Path(os.environ['RPN_DEFNS']).exists():
        print(f'Environment variable $RPN_DEFNS="{Path(os.environ["RPN_DEFNS"])}" does NOT exist.')
        print(f'So, "{_default_rpn_defns_path}" will be used instead.')
        os.environ['RPN_DEFNS'] = _default_rpn_defns_path
else:
    print('Environment variable "RPN_DEFNS" is NOT defined.')
    print(f'So, "$RPN_DEFNS" is defined to be "{_default_rpn_defns_path}".')
    os.environ['RPN_DEFNS'] = _default_rpn_defns_path

del this_folder


std_print_enabled = dict(out=True, err=True)
sbatch_std_print_enabled = dict(out=True, err=True)

from . import local
#from . import remote
from . import correct
from . import elebuilder
from . import eleutil
from . import geneopt
try:
    from . import latex
except:
    print('\n## pyelegant:WARNING ##')
    print('Failed to load "latex" module for "{}"'.format(facility_name))
    print('All the functionality that requires this module will result in errors.')
from . import ltemanager
from . import nonlin
from . import notation
from . import orbit
from . import respmat
from . import sdds
from . import sigproc
from . import twiss
from . import util

from .local import (
    run,
    enable_stdout,
    enable_stderr,
    disable_stdout,
    disable_stderr,
    enable_sbatch_stdout,
    enable_sbatch_stderr,
    disable_sbatch_stdout,
    disable_sbatch_stderr,
)
from .remote import remote
from .twiss import calc_line_twiss, calc_ring_twiss, plot_twiss

if remote:
    __version__['ELEGANT'] = remote.__elegant_version__
else:
    p = Popen(shlex.split('which elegant'), stdout=PIPE, stderr=PIPE, encoding='utf-8')
    out, err = p.communicate()
    if out.strip():
        p = Popen(shlex.split('elegant'), stdout=PIPE, stderr=PIPE, encoding='utf-8')
        out, err = p.communicate()
        temp = out.split(',')[0]
        if temp.startswith('This is elegant '):
            __version__['ELEGANT'] = temp[len('This is elegant '):]
        else:
            __version__['ELEGANT'] = 'unknown'
        del temp
    else:
        print('\n*** pyelegant:WARNING: ELEGANT not available.')
        __version__['ELEGANT'] = None
    del p, out, err
