import matplotlib.pylab as plt
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'
import os
import json

this_folder = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(this_folder, 'version.json'), 'r') as f:
    __version__ = json.load(f)

del this_folder

std_print_enabled = dict(out=True, err=True)

from . import local
#from . import remote
from . import correct
from . import elebuilder
from . import eleutil
from . import geneopt
from . import ltemanager
from . import nonlin
from . import notation
from . import orbit
from . import sdds
from . import sigproc
from . import twiss
from . import util

from .local import run, enable_stdout, enable_stderr, disable_stdout, disable_stderr
from .remote import remote
from .twiss import calc_line_twiss, calc_ring_twiss, plot_twiss
