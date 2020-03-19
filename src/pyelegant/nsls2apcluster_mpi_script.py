import os
import sys
import importlib
import pickle
import gzip
import tempfile
from pathlib import Path
from functools import partial
import traceback

from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor
import dill
# To allow class methods usable in mpi4py
MPI.pickle.__init__(dill.dumps, dill.loads) # <= This works for mpi4py v.3.0

#----------------------------------------------------------------------
def _mpi_starmap(func_or_classmethod, param_list, *args):
    """"""

    executor = MPIPoolExecutor()

    #print('param_list:', param_list)
    #print('args:', args)
    #print('actual:', [tuple([param] + list(args)) for param in param_list])

    futures = executor.starmap(
        func_or_classmethod,
        [tuple([param] + list(args)) for param in param_list])

    results = list(futures)

    executor.shutdown(wait=True)

    return results

def _wrapper(func, param_list, *args):
    """"""

    try:
        # Put your body of code you are trying to debug here
        return func(param_list, *args)

    except Exception as err:

        tmp = tempfile.NamedTemporaryFile(
            dir=Path.cwd(), delete=False, prefix='error.', suffix='.txt')
        Path(tmp.name).write_text(traceback.format_exc())
        tmp.close()

        raise

if __name__ == '__main__':

    # ## CRITICAL ## You must have this section below

    if (len(sys.argv) == 3) and (sys.argv[1] == '_mpi_starmap'):
        input_filepath = sys.argv[2]
        with open(input_filepath, 'rb') as f:

            paths_to_prepend = dill.load(f)
            for _path in paths_to_prepend:
                sys.path.insert(0, _path)

            d = dill.load(f)

        #mod = importlib.import_module('pyelegant.nonlin')
        mod = importlib.import_module(d['module_name'])
        #func = getattr(mod, '_calc_chrom_track_get_tbt')
        func = getattr(mod, d['func_name'])

        #results = _mpi_starmap(func, d['param_list'], *d['args'])
        results = _mpi_starmap(partial(_wrapper, func), d['param_list'], *d['args'])

        if d['output_filepath']:
            with gzip.GzipFile(d['output_filepath'], 'w') as f:
                pickle.dump(results, f, protocol=2)

        sys.exit(0)