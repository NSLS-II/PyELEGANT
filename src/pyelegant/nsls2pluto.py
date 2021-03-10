import sys
import os
import time
import datetime
from copy import deepcopy
from subprocess import Popen, PIPE
import shlex
import tempfile
import glob
from pathlib import Path
import json
import re

import numpy as np
import dill
from ruamel import yaml

from . import util

_IMPORT_TIMESTAMP = time.time()

_SLURM_CONFIG_FILEPATH = Path.home().joinpath('.pyelegant', 'slurm_config.yaml')
_SLURM_CONFIG_FILEPATH.parent.mkdir(parents=True, exist_ok=True)
if not _SLURM_CONFIG_FILEPATH.exists():
    _SLURM_CONFIG_FILEPATH.write_text('''\
exclude: []
abs_time_limit: {}
''')
#
SLURM_PARTITIONS = {}
SLURM_EXCL_NODES = None
SLURM_ABS_TIME_LIMIT = {}

p = Popen(shlex.split('which elegant'), stdout=PIPE, stderr=PIPE, encoding='utf-8')
out, err = p.communicate()
if out.strip():
    path_tokens = out.split('/')
    if 'elegant' in path_tokens:
        __elegant_version__ = path_tokens[path_tokens.index('elegant')+1]
    else:
        __elegant_version__ = 'unknown'
    del path_tokens
else:
    print('\n*** pyelegant:WARNING: ELEGANT not available.')
    __elegant_version__ = None
del p, out, err
