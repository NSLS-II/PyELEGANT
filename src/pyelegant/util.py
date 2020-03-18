from typing import Union
import os
import sys
import time
import gzip
import pickle
import h5py
import numpy as np
import itertools
from pathlib import Path

from . import __version__

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
            if '_version_PyELEGANT' not in f:
                f['_version_PyELEGANT'] = __version__['PyELEGANT']
        except:
            pass
        try:
            if '_version_ELEGANT' not in f:
                f['_version_ELEGANT'] = __version__['ELEGANT']
        except:
            pass

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
    version = {}

    f = h5py.File(hdf5_filepath, 'r')
    for k in list(f):

        if k.startswith('_version_'):
            version[k[len('_version_'):]] = f[k][()]
            continue
        else:
            sdds_file_type = k

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

    return d, meta, version

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

def pprint_sci_notation(float_val, format):
    """"""

    raw_str = f'{float_val:{format}}'

    if 'e' in raw_str:
        s1, s2 = raw_str.split('e')
        pretty_str = fr'{s1} \times 10^{{{int(s2):d}}} '
    else:
        pretty_str = raw_str

    return pretty_str

def pprint_optim_term_log(tlog_sdds_dict):
    """"""

    tlog = tlog_sdds_dict

    total_term_val = np.sum(tlog['columns']['Contribution'])
    max_term_val = np.max(tlog['columns']['Contribution'])
    full_bar_char_width = 6
    for val, expr in zip(tlog['columns']['Contribution'],
                         tlog['columns']['Term']):
        bar_width = int(np.ceil( (val / max_term_val) / (1 / full_bar_char_width) ))
        print(f'{"*" * bar_width:>{full_bar_char_width:d}} {val:9.4g}  ::  "{expr}"')
    print(f'## Sum of All Terms = {total_term_val:9.4g}')

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

def auto_check_output_file_type(
    output_filepath: str, output_file_type: Union[None, str]) -> str:
    """"""

    if output_file_type is None:
        # Auto-detect file type from "output_filepath"
        if output_filepath.endswith(('.hdf5', '.h5')):
            output_file_type = 'hdf5'
        elif output_filepath.endswith('.pgz'):
            output_file_type = 'pgz'
        else:
            raise ValueError(
                ('"output_file_type" could NOT be automatically deduced from '
                 '"output_filepath". Please specify "output_file_type".'))

    if output_file_type.lower() not in ('hdf5', 'h5', 'pgz'):
        raise ValueError('Invalid output file type: {}'.format(output_file_type))

    return output_file_type.lower()

def save_input_to_hdf5(output_filepath: Union[str, Path], input_dict: dict) -> None:
    """"""

    f = h5py.File(output_filepath, 'w')
    g = f.create_group('input')
    for k, v in input_dict.items():
        if v is None:
            continue
        elif isinstance(v, dict):
            g2 = g.create_group(k)
            for k2, v2 in v.items():
                try:
                    g2[k2] = v2
                except:
                    g2[k2] = np.array(v2, dtype='S')
        else:
            try:
                g[k] = v
            except:
                g[k] = np.array(v, dtype='S')
    f.close()

def run_cmd_w_realtime_print(cmd_list, return_stdout=False, return_stderr=False):
    """
    Based on an answer posted at

    https://stackoverflow.com/questions/31926470/run-command-and-get-its-stdout-stderr-separately-in-near-real-time-like-in-a-te
    """

    import pty
    from subprocess import Popen
    from select import select
    import errno

    masters, slaves = zip(pty.openpty(), pty.openpty())
    p = Popen(cmd_list, stdin=slaves[0], stdout=slaves[0], stderr=slaves[1])
    for fd in slaves:
        os.close(fd)

    if return_stdout:
        stdout = '' # to store entire stdout history
    if return_stderr:
        stderr = '' # to store entire stderr history

    readable = { masters[0]: sys.stdout, masters[1]: sys.stderr}
    try:
        while readable:
            for fd in select(readable, [], [])[0]:
                try:
                    data = os.read(fd, 1024)
                except OSError as e:
                    if e.errno != errno.EIO:
                        raise
                    del readable[fd]
                finally:
                    if not data:
                        del readable[fd]
                    else:
                        if return_stdout or return_stderr:
                            if fd == masters[0]:
                                stdout += data.decode('utf-8')
                            else:
                                stderr += data.decode('utf-8')

                        readable[fd].write(data.decode('utf-8'))
                        readable[fd].flush()
    except:
        pass

    finally:
        p.wait()
        for fd in masters:
            os.close(fd)

    output = {}
    if return_stdout:
        output['out'] = stdout
    if return_stderr:
        output['err'] = stderr
    if output == {}:
        output = None

    return output

def deepcopy_dict(d):
    """"""

    return pickle.loads(pickle.dumps(d))
