from __future__ import print_function, division, absolute_import
from __future__ import unicode_literals

import os
import time
import gzip
try:
    from six.moves import cPickle as pickle
except:
    print('Package "six" could not be found.')
    import cPickle as pickle

def is_file_updated(filepath, timestamp_to_compare):
    """
    "timestamp_to_compare" is a float (time elapsed from the Unit epoch time),
    which could be an output of time.time() or os.stat(filepath).st_mtime.
    """

    try:
        file_timestamp = os.stat(filepath).st_mtime
        if file_timestamp > timestamp_to_compare:
            return True
        else:
            return False
    except FileNotFoundError:
        return False

def robust_text_file_write(output_filepath, contents, nMaxTry=10, sleep=10.0):
    """"""

    success = False

    for iTry in range(nMaxTry):
        try:
            with open(output_filepath, 'w') as f:
                f.write(contents)
            success = True
            break
        except:
            if iTry != nMaxTry - 1:
                time.sleep(sleep)

    return success

def robust_pgz_file_write(output_filepath, contents, nMaxTry=10, sleep=10.0):
    """"""

    success = False

    for iTry in range(nMaxTry):
        try:
            with gzip.GzipFile(output_filepath, 'w') as f:
                pickle.dump(contents, f)
            success = True
            break
        except:
            if iTry != nMaxTry - 1:
                time.sleep(sleep)

    return success

def get_abspath(filepath_in_ele_file, ele_filepath, rootname=None):
    """"""

    if rootname is None:
        rootname = '.'.join(os.path.basename(ele_filepath).split('.')[:-1])

    if '%s' in filepath_in_ele_file:
        subs_path = filepath_in_ele_file.replace('%s', rootname)
    else:
        subs_path = filepath_in_ele_file

    return os.path.abspath(subs_path)

def get_run_setup_output_abspaths(
    ele_filepath, rootname=None,
    output=None, centroid=None, sigma=None, final=None, acceptance=None,
    losses=None, magnets=None, semaphore_file=None, parameters=None):
    """"""

    filepath_dict = {}

    if output is not None:
        filepath_dict['output'] = get_abspath(output, ele_filepath, rootname=rootname)
    if centroid is not None:
        filepath_dict['centroid'] = get_abspath(centroid, ele_filepath, rootname=rootname)
    if sigma is not None:
        filepath_dict['sigma'] = get_abspath(sigma, ele_filepath, rootname=rootname)
    if final is not None:
        filepath_dict['final'] = get_abspath(final, ele_filepath, rootname=rootname)
    if acceptance is not None:
        filepath_dict['acceptance'] = get_abspath(acceptance, ele_filepath, rootname=rootname)
    if losses is not None:
        filepath_dict['losses'] = get_abspath(losses, ele_filepath, rootname=rootname)
    if magnets is not None:
        filepath_dict['magnets'] = get_abspath(magnets, ele_filepath, rootname=rootname)
    if semaphore_file is not None:
        filepath_dict['semaphore_file'] = get_abspath(semaphore_file, ele_filepath, rootname=rootname)
    if parameters is not None:
        filepath_dict['parameters'] = get_abspath(parameters, ele_filepath, rootname=rootname)

    return filepath_dict