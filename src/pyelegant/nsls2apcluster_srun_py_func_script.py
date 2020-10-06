import sys
import importlib
import pickle
import gzip

import dill

if __name__ == '__main__':

    input_filepath = sys.argv[1]
    with open(input_filepath, 'rb') as f:

        paths_to_prepend = dill.load(f)
        for _path in paths_to_prepend:
            sys.path.insert(0, str(_path))

        d = dill.load(f)

    #mod = importlib.import_module('pyelegant.nonlin')
    mod = importlib.import_module(d['module_name'])
    #func = getattr(mod, '_calc_chrom_track_get_tbt')
    func = getattr(mod, d['func_name'])

    result = func(*d['func_args'], **d['func_kwargs'])

    if d['output_filepath']:
        with gzip.GzipFile(d['output_filepath'], 'w') as f:
            pickle.dump(result, f, protocol=2)

    sys.exit(0)