import os
import time
import gzip
import pickle
import h5py
import numpy as np
import itertools
from pathlib import Path

def get_current_local_time_str():
    """"""

    DEF_FILENAME_TIMESTAMP_STR_FORMAT = '%Y-%m-%dT%H-%M-%S'

    return time.strftime(DEF_FILENAME_TIMESTAMP_STR_FORMAT, time.localtime())

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

def robust_sdds_hdf5_write(
    output_filepath, sdds_dict_list, nMaxTry=10, sleep=10.0, mode='w'):
    """"""

    success = False

    output, meta = sdds_dict_list

    for iTry in range(nMaxTry):
        try:
            f = h5py.File(output_filepath, mode)
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
                        if v2.size == 0:
                            g2[k2] = []
                        elif isinstance(v2[0], str):
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

            raise

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
    for sdds_file_type in list(f):
        g1 = f[sdds_file_type]

        try:
            'scalars' in g1
        except TypeError:
            # Definitely not an SDDS file group
            continue

        d2 = d[sdds_file_type] = {}
        m2 = meta[sdds_file_type] = {}

        if 'scalars' in g1:
            g2 = g1['scalars']
            d3 = d2['scalars'] = {}
            m3 = m2['scalars'] = {}

            for k2 in list(g2):
                if isinstance(g2[k2], h5py.Dataset):
                    d3[k2] = g2[k2][()]
                    if isinstance(d3[k2], bytes):
                        d3[k2] = d3[k2].decode('utf-8')

                    m3[k2] = {}
                    for mk, mv in g2[k2].attrs.items():
                        m3[k2][mk] = mv

                else:
                    for k3 in list(g2[k2]):
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

            for k2 in list(g2):
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

def delete_temp_files(filepath_list):
    """"""

    for fp in np.unique(filepath_list):
        if fp.startswith('/dev'):
            continue
        else:
            try:
                fp_pathobj = Path(fp)
                if fp_pathobj.exists():
                    fp_pathobj.unlink()
            except:
                print(f'Failed to delete "{fp}"')

def chunk_list(full_flat_list, ncores):
    """"""

    n = len(full_flat_list)

    full_inds = np.array(range(n))

    chunked_list = []
    reverse_mapping = np.full((n, 2), np.nan)
    # ^ 1st column will contain the core index on which each task is performed.
    #   2nd column will contain the task index within each core task list
    for iCore in range(ncores):

        chunked_list.append(full_flat_list[iCore::ncores])

        assigned_flat_inds = full_inds[iCore::ncores]
        reverse_mapping[assigned_flat_inds, 0] = iCore
        reverse_mapping[assigned_flat_inds, 1] = np.array(range(
            len(assigned_flat_inds)))

    assert np.all(~np.isnan(reverse_mapping))

    return chunked_list, reverse_mapping.astype(int)

def unchunk_list_of_lists(chunked_list, reverse_mapping):
    """"""

    full_flat_list = []
    for iCore, sub_task_index in reverse_mapping:
        full_flat_list.append(chunked_list[iCore][sub_task_index])

    return full_flat_list

########################################################################
class ResonanceDiagram():
    """
    """

    #----------------------------------------------------------------------
    def __init__(self):
        """Constructor"""

    #----------------------------------------------------------------------
    def getResonanceCoeffs(self, norder):
        """"""

        assert norder >= 1

        n = norder

        xy_coeffs = list(set(filter(
            lambda nxy: np.abs(nxy[0]) + np.abs(nxy[1]) == n,
            itertools.product(range(-n,n+1), range(-n,n+1)))))

        return xy_coeffs

    #----------------------------------------------------------------------
    def getLineSegment(self, nx, ny, r, nuxlim, nuylim):
        """
        nx * nux + ny * nuy = r
        """

        if nx == 0:
            # Horizontal Line
            nuy_const = r / ny
            return [[nuxlim[0], nuy_const],
                    [nuxlim[1], nuy_const]]

        elif ny == 0:
            # Vertical Line
            nux_const = r / nx
            return [[nux_const, nuylim[0]],
                    [nux_const, nuylim[1]]]

        else:

            line_seg = []

            # Intersecting Coords. @ Left Boundary
            nux = nuxlim[0]
            nuy = (r - nx * nux) / ny
            if nuylim[0] <= nuy <= nuylim[1]:
                line_seg.append([nux, nuy])

            # Intersecting Coords. @ Right Boundary
            nux = nuxlim[1]
            nuy = (r - nx * nux) / ny
            if nuylim[0] <= nuy <= nuylim[1]:
                line_seg.append([nux, nuy])

            if len(line_seg) == 2:
                return line_seg

            # Intersecting Coords. @ Bottom Boundary
            nuy = nuylim[0]
            nux = (r - ny * nuy) / nx
            if nuxlim[0] <= nux <= nuxlim[1]:
                line_seg.append([nux, nuy])

            if len(line_seg) == 2:
                return line_seg

            # Intersecting Coords. @ Top Boundary
            nuy = nuylim[1]
            nux = (r - ny * nuy) / nx
            if nuxlim[0] <= nux <= nuxlim[1]:
                line_seg.append([nux, nuy])

            return line_seg

    #----------------------------------------------------------------------
    def getResonanceLines(self, nuxlim, nuylim, resonance_coeffs):
        """"""

        nux_min, nux_max = nuxlim
        nuy_min, nuy_max = nuylim

        line_list = []
        intersecting_resonance_coeffs_list = []

        for nx, ny in resonance_coeffs:
            # nx * nux + ny * nuy = r
            r_list = [
                nx * nux_min + ny * nuy_min, nx * nux_min + ny * nuy_max,
                nx * nux_max + ny * nuy_min, nx * nux_max + ny * nuy_max,
            ]

            rmin = np.min(r_list)
            rmax = np.max(r_list)

            gcd_nx_ny = np.gcd(nx, ny)

            for r_int in range(int(np.floor(rmin)), int(np.ceil(rmax))):

                if r_int < rmin:
                    continue

                if np.gcd(gcd_nx_ny, r_int) != 1:
                    continue

                line = self.getLineSegment(nx, ny, r_int, nuxlim, nuylim)

                nVertexes = len(line)
                if nVertexes == 2:
                    # Found intersecting line segment
                    line_list.append(line)
                    intersecting_resonance_coeffs_list.append((nx, ny, r_int))

        return line_list, intersecting_resonance_coeffs_list

    #----------------------------------------------------------------------
    def getResonanceCoeffsAndLines(self, norder, nuxlim, nuylim):
        """"""

        all_resonance_coeffs = self.getResonanceCoeffs(norder)

        lines, matching_resonance_coeffs = self.getResonanceLines(
            nuxlim, nuylim, all_resonance_coeffs)

        return dict(
            coeffs=matching_resonance_coeffs, lines=lines)

    #----------------------------------------------------------------------
    def getResonanceCoeffLabelString(self, nx, ny):
        """"""

        if nx == 0:
            label = ''
        elif nx == 1:
            label = r'\nu_x '
        elif nx == -1:
            label = r'-\nu_x '
        else:
            label = fr'{nx:d} \nu_x '

        if ny == 0:
            pass
        elif ny == 1:
            if label == '':
                label += r'\nu_y '
            else:
                label += r'+ \nu_y '
        elif ny == -1:
            label += r'- \nu_y '
        else:
            if label == '':
                label += fr'{ny:d} \nu_y '
            else:
                label += fr'{ny:+d} \nu_y '

        return f'${label}$'





