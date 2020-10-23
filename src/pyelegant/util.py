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
from subprocess import Popen, PIPE
import shlex
import matplotlib.patches as patches

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
        elif k == 'input':
            d2 = d[k] = {}
            for k2, v2 in f[k].items():
                d2[k2] = v2[()]
            continue
        else:
            sdds_file_type = k

        g1 = f[sdds_file_type]

        try:
            'scalars' in g1
        except TypeError:
            # not an HDF5 group, but an HDF5 dataset
            d[k] = g1[()]
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

def chained_Popen(cmd_list):
    """"""

    if len(cmd_list) == 1:
        cmd = cmd_list[0]
        p = Popen(shlex.split(cmd), stdout=PIPE, stderr=PIPE,
                  encoding='utf-8')

    else:
        cmd = cmd_list[0]
        p = Popen(shlex.split(cmd), stdout=PIPE, stderr=PIPE)
        for cmd in cmd_list[1:-1]:
            p = Popen(shlex.split(cmd), stdin=p.stdout, stdout=PIPE, stderr=PIPE)
        cmd = cmd_list[-1]
        p = Popen(shlex.split(cmd), stdin=p.stdout, stdout=PIPE, stderr=PIPE,
                  encoding='utf-8')

    out, err = p.communicate()

    return out, err, p.returncode

def get_visible_spos_inds(all_s_array, slim, s_margin_m=0.1):
    """
    s_margin_m [m]
    """

    shifted_slim = slim - slim[0]

    _visible = np.logical_and(
        all_s_array - slim[0] >= shifted_slim[0] - s_margin_m,
        all_s_array - slim[0] <= shifted_slim[1] + s_margin_m
    )

    return _visible

def add_magnet_profiles(
    ax, twi_arrays, parameters_arrays, slim, s_margin_m=0.1, s0_m=0.0):
    """"""

    prof_center_y = 0.0
    quad_height = 0.5
    sext_height = quad_height * 1.5
    oct_height = quad_height * 1.75
    bend_half_height = quad_height/3.0

    ax.set_yticks([])

    ax.set_xlim(slim)
    max_height = max([quad_height, sext_height, oct_height, bend_half_height])
    ax.set_ylim(np.array([-max_height, +max_height]))

    twi_ar = twi_arrays
    parameters = parameters_arrays

    prev_s = 0.0 - s0_m
    assert len(twi_ar['s']) == len(twi_ar['ElementType']) == \
           len(twi_ar['ElementName']) == len(twi_ar['ElementOccurence'])
    for ei, (s, elem_type, elem_name, elem_occur) in enumerate(zip(
        twi_ar['s'], twi_ar['ElementType'], twi_ar['ElementName'],
        twi_ar['ElementOccurence'])):

        cur_s = s - s0_m

        if (s < slim[0] - s_margin_m) or (s > slim[1] + s_margin_m):
            prev_s = cur_s
            continue

        elem_type = elem_type.upper()

        if elem_type in ('QUAD', 'KQUAD'):

            K1 = _get_param_val('K1', parameters, elem_name, elem_occur)
            c = 'r'
            if K1 >= 0.0: # Focusing Quad
                bottom, top = 0.0, quad_height
            else: # Defocusing Quad
                bottom, top = -quad_height, 0.0

            # Shift vertically
            bottom += prof_center_y
            top += prof_center_y

            width = cur_s - prev_s
            height = top - bottom

            p = patches.Rectangle((prev_s, bottom), width, height, fill=True, color=c)
            ax.add_patch(p)

        elif elem_type in ('SEXT', 'KSEXT'):

            K2 = _get_param_val('K2', parameters, elem_name, elem_occur)
            c = 'b'
            if K2 >= 0.0: # Focusing Sext
                bottom, mid_h, top = 0.0, sext_height / 2, sext_height
            else: # Defocusing Sext
                bottom, mid_h, top = -sext_height, -sext_height / 2, 0.0

            # Shift vertically
            bottom += prof_center_y
            mid_h += prof_center_y
            top += prof_center_y

            mid_s = (prev_s + cur_s) / 2

            if K2 >= 0.0: # Focusing Sext
                xy = np.array([
                    [prev_s, bottom], [prev_s, mid_h], [mid_s, top],
                    [cur_s, mid_h], [cur_s, bottom]
                ])
            else:
                xy = np.array([
                    [prev_s, top], [prev_s, mid_h], [mid_s, bottom],
                    [cur_s, mid_h], [cur_s, top]
                ])
            p = patches.Polygon(xy, closed=True, fill=True, color=c)
            ax.add_patch(p)

        elif elem_type in ('OCTU', 'KOCT'):

            K3 = _get_param_val('K3', parameters, elem_name, elem_occur)
            c = 'g'
            if K3 >= 0.0: # Focusing Octupole
                bottom, mid_h, top = 0.0, oct_height / 2, oct_height
            else: # Defocusing Octupole
                bottom, mid_h, top = -oct_height, -oct_height / 2, 0.0

            # Shift vertically
            bottom += prof_center_y
            mid_h += prof_center_y
            top += prof_center_y

            mid_s = (prev_s + cur_s) / 2

            if K3 >= 0.0: # Focusing Octupole
                xy = np.array([
                    [prev_s, bottom], [prev_s, mid_h], [mid_s, top],
                    [cur_s, mid_h], [cur_s, bottom]
                ])
            else:
                xy = np.array([
                    [prev_s, top], [prev_s, mid_h], [mid_s, bottom],
                    [cur_s, mid_h], [cur_s, top]
                ])
            p = patches.Polygon(xy, closed=True, fill=True, color=c)
            ax.add_patch(p)

        elif elem_type in ('RBEND', 'SBEND', 'SBEN', 'CSBEND'):
            bottom, top = -bend_half_height, bend_half_height

            # Shift vertically
            bottom += prof_center_y
            top += prof_center_y

            width = cur_s - prev_s
            height = top - bottom

            p = patches.Rectangle((prev_s, bottom), width, height, fill=True, color='k')
            ax.add_patch(p)
        else:
            ax.plot([prev_s, cur_s], np.array([0.0, 0.0]) + prof_center_y, 'k-')

        prev_s = cur_s

def _get_param_val(param_name, parameters_dict, elem_name, elem_occur):
    """
    Used only by add_magnet_profiles()
    """

    parameters = parameters_dict

    matched_elem_names = (parameters['ElementName'] == elem_name)
    matched_elem_occurs = (parameters['ElementOccurence'] == elem_occur)
    m = np.logical_and(matched_elem_names, matched_elem_occurs)
    if np.sum(m) == 0:
        m = np.where(matched_elem_names)[0]
        u_elem_occurs_int = np.unique(parameters['ElementOccurence'][m])
        if np.all(u_elem_occurs_int > elem_occur):
            elem_occur = np.min(u_elem_occurs_int)
        elif np.all(u_elem_occurs_int < elem_occur):
            elem_occur = np.max(u_elem_occurs_int)
        else:
            elem_occur = np.min(
                u_elem_occurs_int[u_elem_occurs_int >= elem_occur])
        matched_elem_occurs = (parameters['ElementOccurence'] == elem_occur)
        m = np.logical_and(matched_elem_names, matched_elem_occurs)
    m = np.logical_and(m, parameters['ElementParameter'] == param_name)
    assert np.sum(m) == 1

    return parameters['ParameterValue'][m][0]
