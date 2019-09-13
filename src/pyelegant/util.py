import os
import time
import gzip
import pickle
import h5py
import numpy as np

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

def load_pgz_file(pgz_filepath):
    """"""

    with gzip.GzipFile(pgz_filepath, 'r') as f:
        out = pickle.load(f)

    return out

def robust_sdds_hdf5_write(output_filepath, sdds_dict_list, nMaxTry=10, sleep=10.0):
    """"""

    success = False

    output, meta = sdds_dict_list

    for iTry in range(nMaxTry):
        try:
            f = h5py.File(output_filepath, 'w')
        except:
            if iTry != nMaxTry - 1:
                time.sleep(sleep)
                continue
        try:
            for sdds_file_type, v1 in output.items():
                g1 = f.create_group(sdds_file_type)

                if ('params' in v1) and (v1['params'] != {}):
                    g2 = g1.create_group('scalars')
                    for k2, v2 in v1['params'].items():
                        g2[k2] = v2

                        m_d = meta[sdds_file_type]['params'][k2]
                        for mk, mv in m_d.items():
                            g2[k2].attrs[mk] = mv

                if ('columns' in v1) and (v1['columns'] != {}):
                    g2 = g1.create_group('arrays')
                    for k2, v2 in v1['columns'].items():
                        if isinstance(v2[0], str):
                            g2.create_dataset(
                                k2, data=[u.encode('utf-8') for u in v2],
                                compression='gzip')
                        else:
                            try:
                                g2.create_dataset(k2, data=v2, compression='gzip')
                            except:
                                g2[k2] = v2

                        m_d = meta[sdds_file_type]['columns'][k2]
                        for mk, mv in m_d.items():
                            g2[k2].attrs[mk] = mv


            f.close()

            success = True
            break

        except:
            try:
                f.close()
            except:
                pass

            if iTry != nMaxTry - 1:
                time.sleep(sleep)

    return success

def load_sdds_hdf5_file(hdf5_filepath):
    """"""

    d = {}
    meta = {}

    f = h5py.File(hdf5_filepath, 'r')
    for sdds_file_type in f.keys():
        g1 = f[sdds_file_type]

        d2 = d[sdds_file_type] = {}
        m2 = meta[sdds_file_type] = {}

        if 'scalars' in g1:
            g2 = g1['scalars']
            d3 = d2['scalars'] = {}
            m3 = m2['scalars'] = {}

            for k2 in g2.keys():
                if isinstance(g2[k2], h5py.Dataset):
                    d3[k2] = g2[k2][()]
                    if isinstance(d3[k2], bytes):
                        d3[k2] = d3[k2].decode('utf-8')

                    m3[k2] = {}
                    for mk, mv in g2[k2].attrs.items():
                        m3[k2][mk] = mv

                else:
                    for k3 in g2[k2].keys():
                        d3[f'{k2}/{k3}'] = g2[k2][k3][()]
                        if isinstance(d3[f'{k2}/{k3}'], bytes):
                            d3[f'{k2}/{k3}'] = d3[f'{k2}/{k3}'].decode('utf-8')

                        m3[f'{k2}/{k3}'] = {}
                        for mk, mv in g2[k2][k3].attrs.items():
                            m3[f'{k2}/{k3}'][mk] = mv

        if 'arrays' in g1:
            g2 = g1['arrays']
            d3 = d2['arrays'] = {}
            m3 = m2['arrays'] = {}

            for k2 in g2.keys():
                d3[k2] = g2[k2][()]

                if isinstance(d3[k2][0], bytes):
                    d3[k2] = np.array([b.decode('utf-8') for b in d3[k2]])

                m3[k2] = {}
                for mk, mv in g2[k2].attrs.items():
                    m3[k2][mk] = mv

    f.close()

    return d, meta

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