import sys
import os
from pathlib import Path
import numpy as np
import scipy.constants as PHYSCONST
import matplotlib.pylab as plt
import tempfile
import h5py
import shlex
from subprocess import Popen, PIPE
import time

from .local import run
from .remote import remote
from . import std_print_enabled
from . import elebuilder
from . import util
from . import sdds
from . import twiss
from . import sigproc

def _auto_check_output_file_type(output_filepath, output_file_type):
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

def _save_input_to_hdf5(output_filepath, input_dict):
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
            g[k] = v
    f.close()

def _add_transmute_blocks(ed, transmute_elements):
    """
    ed := EleDesigner object
    """

    if transmute_elements is None:

        actual_transmute_elems = dict(
            SBEN='CSBEN', RBEN='CSBEN', QUAD='KQUAD', SEXT='KSEXT',
            RFCA='MARK', SREFFECTS='MARK')
    else:

        actual_transmute_elems = {}
        for old_type, new_type in transmute_elements.items():
            actual_transmute_elems[old_type] = new_type

    for old_type, new_type in actual_transmute_elems.items():
        ed.add_block('transmute_elements',
                     name='*', type=old_type, new_type=new_type)

def _add_N_KICKS_alter_elements_blocks(ed, N_KICKS):
    """
    ed := EleDesigner object
    """

    if N_KICKS is None:
        N_KICKS = dict(KQUAD=20, KSEXT=20, CSBEND=20)

    for k, v in N_KICKS.items():
        if k.upper() not in ('KQUAD', 'KSEXT', 'CSBEND'):
            raise ValueError(f'The key "{k}" in N_KICKS dict is invalid. '
                             f'Must be one of KQUAD, KSEXT, or CSBEND')
        ed.add_block('alter_elements',
                     name='*', type=k.upper(), item='N_KICKS', value=v)

def calc_fma_xy(
    output_filepath, LTE_filepath, E_MeV, xmin, xmax, ymin, ymax, nx, ny,
    n_turns=1024, delta_offset=0.0, quadratic_spacing=False, full_grid_output=False,
    use_beamline=None, N_KICKS=None, transmute_elements=None, ele_filepath=None,
    output_file_type=None, del_tmp_files=True,
    run_local=False, remote_opts=None):
    """"""

    return _calc_fma(
        output_filepath, LTE_filepath, E_MeV, 'xy',
        xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
        nx=nx, ny=ny, n_turns=n_turns, delta_offset=delta_offset,
        quadratic_spacing=quadratic_spacing, full_grid_output=full_grid_output,
        use_beamline=use_beamline, N_KICKS=N_KICKS,
        transmute_elements=transmute_elements, ele_filepath=ele_filepath,
        output_file_type=output_file_type, del_tmp_files=del_tmp_files,
        run_local=run_local, remote_opts=remote_opts)

def calc_fma_px(
    output_filepath, LTE_filepath, E_MeV, delta_min, delta_max, xmin, xmax, ndelta, nx,
    n_turns=1024, y_offset=0.0, quadratic_spacing=False, full_grid_output=False,
    use_beamline=None, N_KICKS=None, transmute_elements=None, ele_filepath=None,
    output_file_type=None, del_tmp_files=True,
    run_local=False, remote_opts=None):
    """"""

    return _calc_fma(
        output_filepath, LTE_filepath, E_MeV, 'px',
        xmin=xmin, xmax=xmax, delta_min=delta_min, delta_max=delta_max,
        nx=nx, ndelta=ndelta, n_turns=n_turns, y_offset=y_offset,
        quadratic_spacing=quadratic_spacing, full_grid_output=full_grid_output,
        use_beamline=use_beamline, N_KICKS=N_KICKS,
        transmute_elements=transmute_elements, ele_filepath=ele_filepath,
        output_file_type=output_file_type, del_tmp_files=del_tmp_files,
        run_local=run_local, remote_opts=remote_opts)

def _calc_fma(
    output_filepath, LTE_filepath, E_MeV, plane, xmin=-0.1, xmax=0.1, ymin=1e-6, ymax=0.1,
    delta_min=0.0, delta_max=0.0, nx=21, ny=21, ndelta=1, n_turns=1024,
    delta_offset=0.0, y_offset=0.0, quadratic_spacing=False, full_grid_output=False,
    use_beamline=None, N_KICKS=None, transmute_elements=None, ele_filepath=None,
    output_file_type=None, del_tmp_files=True,
    run_local=False, remote_opts=None):
    """"""

    if plane == 'xy':
        pass
    elif plane == 'px':
        pass
    else:
        raise ValueError('"plane" must be either "xy" or "px".')

    with open(LTE_filepath, 'r') as f:
        file_contents = f.read()

    input_dict = dict(
        LTE_filepath=os.path.abspath(LTE_filepath), E_MeV=E_MeV, n_turns=n_turns,
        quadratic_spacing=quadratic_spacing, use_beamline=use_beamline,
        N_KICKS=N_KICKS, transmute_elements=transmute_elements,
        ele_filepath=ele_filepath, del_tmp_files=del_tmp_files, run_local=run_local,
        remote_opts=remote_opts,
        lattice_file_contents=file_contents,
        timestamp_ini=util.get_current_local_time_str(),
    )
    input_dict['fma_plane'] = plane
    if plane == 'xy':
        input_dict['xmin'] = xmin
        input_dict['xmax'] = xmax
        input_dict['ymin'] = ymin
        input_dict['ymax'] = ymax
        input_dict['nx'] = nx
        input_dict['ny'] = ny
        input_dict['delta_offset'] = delta_offset

        plane_specific_freq_map_block_opts = dict(
            xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
            delta_min=delta_offset, delta_max=delta_offset, nx=nx, ny=ny, ndelta=1)
    else:
        input_dict['delta_min'] = delta_min
        input_dict['delta_max'] = delta_max
        input_dict['xmin'] = xmin
        input_dict['xmax'] = xmax
        input_dict['ndelta'] = ndelta
        input_dict['nx'] = nx
        input_dict['y_offset'] = y_offset

        plane_specific_freq_map_block_opts = dict(
            xmin=xmin, xmax=xmax, ymin=y_offset, ymax=y_offset,
            delta_min=delta_min, delta_max=delta_max, nx=nx, ny=1, ndelta=ndelta)

    output_file_type = _auto_check_output_file_type(output_filepath, output_file_type)
    input_dict['output_file_type'] = output_file_type

    if output_file_type in ('hdf5', 'h5'):
        _save_input_to_hdf5(output_filepath, input_dict)

    if ele_filepath is None:
        tmp = tempfile.NamedTemporaryFile(
            dir=os.getcwd(), delete=False, prefix=f'tmpFMA{plane}_', suffix='.ele')
        ele_filepath = os.path.abspath(tmp.name)
        tmp.close()

    ed = elebuilder.EleDesigner(double_format='.12g')

    _add_transmute_blocks(ed, transmute_elements)

    ed.add_newline()

    ed.add_block('run_setup',
        lattice=LTE_filepath, p_central_mev=E_MeV, use_beamline=use_beamline,
        semaphore_file='%s.done')

    ed.add_newline()

    ed.add_block('run_control', n_passes=n_turns)

    ed.add_newline()

    _add_N_KICKS_alter_elements_blocks(ed, N_KICKS)

    ed.add_block('bunched_beam', n_particles_per_bunch=1)

    ed.add_newline()

    ed.add_block('frequency_map',
        output='%s.fma', include_changes=True, quadratic_spacing=quadratic_spacing,
        full_grid_output=full_grid_output, **plane_specific_freq_map_block_opts
    )

    ed.write(ele_filepath)

    ed.update_output_filepaths(ele_filepath[:-4]) # Remove ".ele"
    #print(ed.actual_output_filepath_list)

    for fp in ed.actual_output_filepath_list:
        if fp.endswith('.fma'):
            fma_output_filepath = fp
        elif fp.endswith('.done'):
            done_filepath = fp
        else:
            raise ValueError('This line should not be reached.')

    # Run Elegant
    if run_local:
        run(ele_filepath, print_cmd=False,
            print_stdout=std_print_enabled['out'],
            print_stderr=std_print_enabled['err'])
    else:
        if remote_opts is None:
            remote_opts = dict(
                use_sbatch=False, pelegant=True, job_name='fma',
                output='fma.%J.out', error='fma.%J.err',
                partition='normal', ntasks=50)

        remote.run(remote_opts, ele_filepath, print_cmd=True,
                   print_stdout=std_print_enabled['out'],
                   print_stderr=std_print_enabled['err'],
                   output_filepaths=None)

    tmp_filepaths = dict(fma=fma_output_filepath)
    output, meta = {}, {}
    for k, v in tmp_filepaths.items():
        try:
            output[k], meta[k] = sdds.sdds2dicts(v)
        except:
            continue

    timestamp_fin = util.get_current_local_time_str()

    if output_file_type in ('hdf5', 'h5'):
        util.robust_sdds_hdf5_write(
            output_filepath, [output, meta], nMaxTry=10, sleep=10.0, mode='a')
        f = h5py.File(output_filepath)
        f['timestamp_fin'] = timestamp_fin
        f.close()

    elif output_file_type == 'pgz':
        mod_output = {}
        for k, v in output.items():
            mod_output[k] = {}
            if 'params' in v:
                mod_output[k]['scalars'] = v['params']
            if 'columns' in v:
                mod_output[k]['arrays'] = v['columns']
        mod_meta = {}
        for k, v in meta.items():
            mod_meta[k] = {}
            if 'params' in v:
                mod_meta[k]['scalars'] = v['params']
            if 'columns' in v:
                mod_meta[k]['arrays'] = v['columns']
        util.robust_pgz_file_write(
            output_filepath, dict(data=mod_output, meta=mod_meta,
                                  input=input_dict, timestamp_fin=timestamp_fin),
            nMaxTry=10, sleep=10.0)
    else:
        raise ValueError()

    if del_tmp_files:
        for fp in ed.actual_output_filepath_list + [ele_filepath]:
            if fp.startswith('/dev'):
                continue
            else:
                try:
                    os.remove(fp)
                except:
                    print(f'Failed to delete "{fp}"')

    return output_filepath

def plot_fma_xy(
    output_filepath, title='', xlim=None, ylim=None, scatter=True,
    is_diffusion=True, cmin=-10, cmax=-2):
    """"""

    _plot_fma(output_filepath, title=title, xlim=xlim, ylim=ylim,
              scatter=scatter, is_diffusion=is_diffusion, cmin=cmin, cmax=cmax)

def plot_fma_px(
    output_filepath, title='', deltalim=None, xlim=None, scatter=True,
    is_diffusion=True, cmin=-10, cmax=-2):
    """"""

    _plot_fma(output_filepath, title=title, deltalim=deltalim, xlim=xlim,
              scatter=scatter, is_diffusion=is_diffusion, cmin=cmin, cmax=cmax)

def _plot_fma(
    output_filepath, title='', xlim=None, ylim=None, deltalim=None,
    scatter=True, is_diffusion=True, cmin=-10, cmax=-2):
    """"""

    try:
        d = util.load_pgz_file(output_filepath)
        plane = d['input']['fma_plane']
        g = d['data']['fma']['arrays']
        if plane == 'xy':
            v1 = g['x']
            v2 = g['y']
        elif plane == 'px':
            v1 = g['delta']
            v2 = g['x']
        else:
            raise ValueError(f'Unexpected "fma_plane" value: {plane}')

        if is_diffusion:
            diffusion = g['diffusion']
        else:
            diffusionRate = g['diffusionRate']

        if not scatter:
            g = d['input']
            quadratic_spacing = g['quadratic_spacing']
            if plane == 'xy':
                v1max = g['xmax']
                v1min = g['xmin']
                v2max = g['ymax']
                v2min = g['ymin']
                n1 = g['nx']
                n2 = g['ny']
            else:
                v1max = g['delta_max']
                v1min = g['delta_min']
                v2max = g['xmax']
                v2min = g['xmin']
                n1 = g['ndelta']
                n2 = g['nx']

    except:
        f = h5py.File(output_filepath, 'r')
        plane = f['input']['fma_plane'][()]
        g = f['fma']['arrays']
        if plane == 'xy':
            v1 = g['x'][()]
            v2 = g['y'][()]
        elif plane == 'px':
            v1 = g['delta'][()]
            v2 = g['x'][()]
        else:
            raise ValueError(f'Unexpected "fma_plane" value: {plane}')

        if is_diffusion:
            diffusion = g['diffusion'][()]
        else:
            diffusionRate = g['diffusionRate'][()]

        if not scatter:
            g = f['input']
            quadratic_spacing = g['quadratic_spacing'][()]
            if plane == 'xy':
                v1max = g['xmax'][()]
                v1min = g['xmin'][()]
                v2max = g['ymax'][()]
                v2min = g['ymin'][()]
                n1 = g['nx'][()]
                n2 = g['ny'][()]
            else:
                v1max = g['delta_max'][()]
                v1min = g['delta_min'][()]
                v2max = g['xmax'][()]
                v2min = g['xmin'][()]
                n1 = g['ndelta'][()]
                n2 = g['nx'][()]

        f.close()

    if plane == 'xy':
        v1name, v2name = 'x', 'y'
        v1unitsymb, v2unitsymb = r'\mathrm{mm}', r'\mathrm{mm}'
        v1unitconv, v2unitconv = 1e3, 1e3
        v1lim, v2lim = xlim, ylim
    else:
        v1name, v2name = '\delta', 'x'
        v1unitsymb, v2unitsymb = r'\%', r'\mathrm{mm}'
        v1unitconv, v2unitconv = 1e2, 1e3
        v1lim, v2lim = deltalim, xlim

    if is_diffusion:
        DIFFUSION_EQ_STR = r'$\rm{log}_{10}(\Delta{\nu_x}^2+\Delta{\nu_y}^2)$'
        values = diffusion
    else:
        DIFFUSION_EQ_STR = r'$\rm{log}_{10}(\frac{\sqrt{\Delta{\nu_x}^2+\Delta{\nu_y}^2}}{N})$'
        values = diffusionRate

    LB = cmin
    UB = cmax

    if scatter:

        font_sz = 18

        plt.figure()
        plt.scatter(v1 * v1unitconv, v2 * v2unitconv, s=14, c=values, cmap='jet',
                    vmin=LB, vmax=UB)
        plt.xlabel(fr'${v1name}\, [{v1unitsymb}]$', size=font_sz)
        plt.ylabel(fr'${v2name}\, [{v2unitsymb}]$', size=font_sz)
        if v1lim is not None:
            plt.xlim([v * 1e3 for v in v1lim])
        if v2lim is not None:
            plt.ylim([v * 1e3 for v in v2lim])
        if title != '':
            plt.title(title, size=font_sz)
        cb = plt.colorbar()
        cb.set_ticks(range(LB, UB+1))
        cb.set_ticklabels([str(i) for i in range(LB, UB+1)])
        cb.ax.set_title(DIFFUSION_EQ_STR)
        cb.ax.title.set_position((0.5, 1.02))
        plt.tight_layout()

    else:

        font_sz = 18

        if not quadratic_spacing:
            v1array = np.linspace(v1min, v1max, n1)
            v2array = np.linspace(v2min, v2max, n2)
        else:

            dv1 = v1max - max([0.0, v1min])
            v1array = np.sqrt(np.linspace((dv1**2) / n1, dv1**2, n1))
            #v1array - np.unique(v1)
            #plt.figure()
            #plt.plot(np.unique(v1), 'b-', v1array, 'r-')

            dv2 = v2max - max([0.0, v2min])
            v2array = v2min + np.sqrt(np.linspace((dv2**2) / n2, dv2**2, n2))
            #v2array - np.unique(v2)
            #plt.figure()
            #plt.plot(np.unique(v2), 'b-', v2array, 'r-')

        V1, V2 = np.meshgrid(v1array, v2array)
        D = V1 * np.nan

        v1inds = np.argmin(np.abs(
            v1array.reshape((-1,1)) @ np.ones((1, v1.size)) - v1), axis=0)
        v2inds = np.argmin(np.abs(
            v2array.reshape((-1,1)) @ np.ones((1, v2.size)) - v2), axis=0)
        flatinds = np.ravel_multi_index((v1inds, v2inds), V1.shape, order='F')
        D_flat = D.flatten()
        D_flat[flatinds] = values
        D = D_flat.reshape(D.shape)

        D = np.ma.masked_array(D, np.isnan(D))

        plt.figure()
        ax = plt.subplot(111)
        plt.pcolor(V1*v1unitconv, V2*v2unitconv, D, cmap='jet', vmin=LB, vmax=UB)
        plt.xlabel(fr'${v1name}\, [{v1unitsymb}]$', size=font_sz)
        plt.ylabel(fr'${v2name}\, [{v2unitsymb}]$', size=font_sz)
        if v1lim is not None:
            plt.xlim([v * 1e3 for v in v1lim])
        if v2lim is not None:
            plt.ylim([v * 1e3 for v in v2lim])
        if title != '':
            plt.title(title, size=font_sz)
        cb = plt.colorbar()
        cb.set_ticks(range(LB, UB+1))
        cb.set_ticklabels([str(i) for i in range(LB, UB+1)])
        cb.ax.set_title(DIFFUSION_EQ_STR)
        cb.ax.title.set_position((0.5, 1.02))
        plt.tight_layout()

def calc_find_aper_nlines(
    output_filepath, LTE_filepath, E_MeV, xmax=0.1, ymax=0.1, ini_ndiv=21,
    n_lines=11, neg_y_search=False,
    n_turns=1024, use_beamline=None, N_KICKS=None, transmute_elements=None,
    ele_filepath=None, output_file_type=None, del_tmp_files=True,
    run_local=False, remote_opts=None):
    """"""

    assert n_lines >= 3

    with open(LTE_filepath, 'r') as f:
        file_contents = f.read()

    input_dict = dict(
        LTE_filepath=os.path.abspath(LTE_filepath), E_MeV=E_MeV, n_turns=n_turns,
        use_beamline=use_beamline,
        N_KICKS=N_KICKS, transmute_elements=transmute_elements,
        ele_filepath=ele_filepath,
        del_tmp_files=del_tmp_files, run_local=run_local,
        remote_opts=remote_opts,
        lattice_file_contents=file_contents,
        timestamp_ini=util.get_current_local_time_str(),
    )
    input_dict['xmax'] = xmax
    input_dict['ymax'] = ymax
    input_dict['ini_ndiv'] = ini_ndiv
    input_dict['n_lines'] = n_lines
    input_dict['neg_y_search'] = neg_y_search

    output_file_type = _auto_check_output_file_type(output_filepath, output_file_type)
    input_dict['output_file_type'] = output_file_type

    if output_file_type in ('hdf5', 'h5'):
        _save_input_to_hdf5(output_filepath, input_dict)

    if ele_filepath is None:
        tmp = tempfile.NamedTemporaryFile(
            dir=os.getcwd(), delete=False, prefix=f'tmpFindAper_', suffix='.ele')
        ele_filepath = os.path.abspath(tmp.name)
        tmp.close()

    ed = elebuilder.EleDesigner(double_format='.12g')

    _add_transmute_blocks(ed, transmute_elements)

    ed.add_newline()

    ed.add_block('run_setup',
        lattice=LTE_filepath, p_central_mev=E_MeV, use_beamline=use_beamline,
        semaphore_file='%s.done')

    ed.add_newline()

    ed.add_block('run_control', n_passes=n_turns)

    ed.add_newline()

    _add_N_KICKS_alter_elements_blocks(ed, N_KICKS)

    ed.add_newline()

    ed.add_block('find_aperture',
        output='%s.aper', mode='n-lines', xmax=xmax, ymax=ymax, nx=ini_ndiv,
        n_lines=n_lines, full_plane=neg_y_search,
        offset_by_orbit=True, # recommended according to the manual
    )

    ed.write(ele_filepath)

    ed.update_output_filepaths(ele_filepath[:-4]) # Remove ".ele"
    #print(ed.actual_output_filepath_list)

    for fp in ed.actual_output_filepath_list:
        if fp.endswith('.aper'):
            aper_output_filepath = fp
        elif fp.endswith('.done'):
            done_filepath = fp
        else:
            raise ValueError('This line should not be reached.')

    # Run Elegant
    if run_local:
        run(ele_filepath, print_cmd=False,
            print_stdout=std_print_enabled['out'],
            print_stderr=std_print_enabled['err'])
    else:
        if remote_opts is None:
            remote_opts = dict(
                use_sbatch=False, pelegant=True, job_name='findaper',
                output='findaper.%J.out', error='findaper.%J.err',
                partition='normal', ntasks=np.min([50, n_lines]))

        remote.run(remote_opts, ele_filepath, print_cmd=True,
                   print_stdout=std_print_enabled['out'],
                   print_stderr=std_print_enabled['err'],
                   output_filepaths=None)

    tmp_filepaths = dict(aper=aper_output_filepath)
    output, meta = {}, {}
    for k, v in tmp_filepaths.items():
        try:
            output[k], meta[k] = sdds.sdds2dicts(v)
        except:
            continue

    timestamp_fin = util.get_current_local_time_str()

    if output_file_type in ('hdf5', 'h5'):
        util.robust_sdds_hdf5_write(
            output_filepath, [output, meta], nMaxTry=10, sleep=10.0, mode='a')
        f = h5py.File(output_filepath)
        f['timestamp_fin'] = timestamp_fin
        f.close()

    elif output_file_type == 'pgz':
        mod_output = {}
        for k, v in output.items():
            mod_output[k] = {}
            if 'params' in v:
                mod_output[k]['scalars'] = v['params']
            if 'columns' in v:
                mod_output[k]['arrays'] = v['columns']
        mod_meta = {}
        for k, v in meta.items():
            mod_meta[k] = {}
            if 'params' in v:
                mod_meta[k]['scalars'] = v['params']
            if 'columns' in v:
                mod_meta[k]['arrays'] = v['columns']
        util.robust_pgz_file_write(
            output_filepath, dict(data=mod_output, meta=mod_meta,
                                  input=input_dict, timestamp_fin=timestamp_fin),
            nMaxTry=10, sleep=10.0)
    else:
        raise ValueError()

    if del_tmp_files:
        for fp in ed.actual_output_filepath_list + [ele_filepath]:
            if fp.startswith('/dev'):
                continue
            else:
                try:
                    os.remove(fp)
                except:
                    print(f'Failed to delete "{fp}"')

    return output_filepath


def plot_find_aper_nlines(output_filepath, title='', xlim=None, ylim=None):
    """"""

    try:
        d = util.load_pgz_file(output_filepath)
        g = d['data']['aper']['arrays']
        x = g['x']
        y = g['y']
        g = d['data']['aper']['scalars']
        area = g['Area']
        neg_y_search = d['data']['input']['neg_y_search']

    except:
        f = h5py.File(output_filepath, 'r')
        g = f['aper']['arrays']
        x = g['x'][()]
        y = g['y'][()]
        g = f['aper']['scalars']
        area = g['Area'][()]
        neg_y_search = f['input']['neg_y_search'][()]
        f.close()

    font_sz = 18

    plt.figure()
    plt.plot(x * 1e3, y * 1e3, 'b.-')
    plt.xlabel(r'$x\, [\mathrm{mm}]$', size=font_sz)
    plt.ylabel(r'$y\, [\mathrm{mm}]$', size=font_sz)
    if xlim is not None:
        plt.xlim([v * 1e3 for v in xlim])
    if ylim is not None:
        plt.ylim([v * 1e3 for v in ylim])
    area_info_title = (fr'$(x_{{\mathrm{{min}}}}={np.min(x)*1e3:.1f}, '
                       fr'x_{{\mathrm{{max}}}}={np.max(x)*1e3:.1f}, ')
    if neg_y_search:
        area_info_title += fr'y_{{\mathrm{{min}}}}={np.min(y)*1e3:.1f}, '
    area_info_title += fr'y_{{\mathrm{{max}}}}={np.max(y)*1e3:.1f})\, [\mathrm{{mm}}], '
    area_info_title += fr'\mathrm{{Area}}={area*1e6:.1f}\, [\mathrm{{mm}}^2]$'
    if title != '':
        plt.title('\n'.join([title, area_info_title]), size=font_sz)
    else:
        plt.title(area_info_title, size=font_sz)
    plt.tight_layout()

def calc_mom_aper(
    output_filepath, LTE_filepath, E_MeV, x_initial=1e-5, y_initial=1e-5,
    delta_negative_start=-1e-3, delta_negative_limit=-5e-2,
    delta_positive_start=+1e-3, delta_positive_limit=+5e-2,
    init_delta_step_size=5e-3, s_start=0.0, s_end=None, include_name_pattern=None,
    n_turns=1024, use_beamline=None, N_KICKS=None, transmute_elements=None,
    ele_filepath=None, output_file_type=None, del_tmp_files=True,
    run_local=False, remote_opts=None):
    """"""

    with open(LTE_filepath, 'r') as f:
        file_contents = f.read()

    input_dict = dict(
        LTE_filepath=os.path.abspath(LTE_filepath), E_MeV=E_MeV, n_turns=n_turns,
        use_beamline=use_beamline,
        N_KICKS=N_KICKS, transmute_elements=transmute_elements,
        ele_filepath=ele_filepath,
        del_tmp_files=del_tmp_files, run_local=run_local,
        remote_opts=remote_opts,
        lattice_file_contents=file_contents,
        timestamp_ini=util.get_current_local_time_str(),
    )
    input_dict['s_start'] = s_start
    input_dict['s_end'] = s_end

    output_file_type = _auto_check_output_file_type(output_filepath, output_file_type)
    input_dict['output_file_type'] = output_file_type

    if output_file_type in ('hdf5', 'h5'):
        _save_input_to_hdf5(output_filepath, input_dict)

    if ele_filepath is None:
        tmp = tempfile.NamedTemporaryFile(
            dir=os.getcwd(), delete=False, prefix=f'tmpMomAper_', suffix='.ele')
        ele_filepath = os.path.abspath(tmp.name)
        tmp.close()

    ed = elebuilder.EleDesigner(double_format='.12g')

    _add_transmute_blocks(ed, transmute_elements)

    ed.add_newline()

    ed.add_block('run_setup',
        lattice=LTE_filepath, p_central_mev=E_MeV, use_beamline=use_beamline,
        semaphore_file='%s.done')

    ed.add_newline()

    ed.add_block('run_control', n_passes=n_turns)

    ed.add_newline()

    _add_N_KICKS_alter_elements_blocks(ed, N_KICKS)

    ed.add_newline()

    _block_opts = dict(
        output='%s.mmap', x_initial=x_initial, y_initial=y_initial,
        delta_negative_start=delta_negative_start, delta_negative_limit=delta_negative_limit,
        delta_positive_start=delta_positive_start, delta_positive_limit=delta_positive_limit,
        delta_step_size=init_delta_step_size, include_name_pattern=include_name_pattern,
        s_start=s_start, s_end=(s_end if s_end is not None else sys.float_info.max),
        fiducialize=True, verbosity=True,
    )
    if include_name_pattern is not None:
        _block_opts['include_name_pattern'] = include_name_pattern
    ed.add_block('momentum_aperture', **_block_opts)

    ed.write(ele_filepath)

    ed.update_output_filepaths(ele_filepath[:-4]) # Remove ".ele"
    #print(ed.actual_output_filepath_list)

    for fp in ed.actual_output_filepath_list:
        if fp.endswith('.mmap'):
            mmap_output_filepath = fp
        elif fp.endswith('.done'):
            done_filepath = fp
        else:
            raise ValueError('This line should not be reached.')

    # Run Elegant
    if run_local:
        run(ele_filepath, print_cmd=False,
            print_stdout=std_print_enabled['out'],
            print_stderr=std_print_enabled['err'])
    else:
        if remote_opts is None:
            remote_opts = dict(
                use_sbatch=False, pelegant=True, job_name='momaper',
                output='momaper.%J.out', error='momaper.%J.err',
                partition='normal', ntasks=50)

        remote.run(remote_opts, ele_filepath, print_cmd=True,
                   print_stdout=std_print_enabled['out'],
                   print_stderr=std_print_enabled['err'],
                   output_filepaths=None)

    tmp_filepaths = dict(mmap=mmap_output_filepath)
    output, meta = {}, {}
    for k, v in tmp_filepaths.items():
        try:
            output[k], meta[k] = sdds.sdds2dicts(v)
        except:
            continue

    timestamp_fin = util.get_current_local_time_str()

    if output_file_type in ('hdf5', 'h5'):
        util.robust_sdds_hdf5_write(
            output_filepath, [output, meta], nMaxTry=10, sleep=10.0, mode='a')
        f = h5py.File(output_filepath)
        f['timestamp_fin'] = timestamp_fin
        f.close()

    elif output_file_type == 'pgz':
        mod_output = {}
        for k, v in output.items():
            mod_output[k] = {}
            if 'params' in v:
                mod_output[k]['scalars'] = v['params']
            if 'columns' in v:
                mod_output[k]['arrays'] = v['columns']
        mod_meta = {}
        for k, v in meta.items():
            mod_meta[k] = {}
            if 'params' in v:
                mod_meta[k]['scalars'] = v['params']
            if 'columns' in v:
                mod_meta[k]['arrays'] = v['columns']
        util.robust_pgz_file_write(
            output_filepath, dict(data=mod_output, meta=mod_meta,
                                  input=input_dict, timestamp_fin=timestamp_fin),
            nMaxTry=10, sleep=10.0)
    else:
        raise ValueError()

    if del_tmp_files:
        for fp in ed.actual_output_filepath_list + [ele_filepath]:
            if fp.startswith('/dev'):
                continue
            else:
                try:
                    os.remove(fp)
                except:
                    print(f'Failed to delete "{fp}"')

    return output_filepath

def plot_mom_aper(output_filepath, title='', slim=None, deltalim=None):
    """"""

    try:
        d = util.load_pgz_file(output_filepath)
        g = d['data']['mmap']['arrays']
        deltaNegative = g['deltaNegative']
        deltaPositive = g['deltaPositive']
        s = g['s']

    except:
        f = h5py.File(output_filepath, 'r')
        g = f['mmap']['arrays']
        deltaNegative = g['deltaNegative'][()]
        deltaPositive = g['deltaPositive'][()]
        s = g['s'][()]
        f.close()

    font_sz = 18

    sort_inds = np.argsort(s)
    s = s[sort_inds]
    deltaNegative = deltaNegative[sort_inds]
    deltaPositive = deltaPositive[sort_inds]

    plt.figure()
    plt.plot(s, deltaNegative * 1e2, 'b-')
    plt.plot(s, deltaPositive * 1e2, 'r-')
    plt.axhline(0, color='k')
    plt.xlabel(r'$s\, [\mathrm{m}]$', size=font_sz)
    plt.ylabel(r'$\delta_{+}, \delta_{-}\, [\%]$', size=font_sz)
    if slim is not None:
        plt.xlim([v for v in slim])
    if deltalim is not None:
        plt.ylim([v * 1e2 for v in deltalim])
    mmap_info_title = r'${},\, {}\, [\%]$'.format(
        fr'{np.min(deltaPositive)*1e2:.2f} < \delta_{{+}} < {np.max(deltaPositive)*1e2:.2f}',
        fr'{np.min(deltaNegative)*1e2:.2f} < \delta_{{-}} < {np.max(deltaNegative)*1e2:.2f}',
    )
    if title != '':
        plt.title('\n'.join([title, mmap_info_title]), size=font_sz)
    else:
        plt.title(mmap_info_title, size=font_sz)
    plt.tight_layout()

def calc_Touschek_lifetime(
    output_filepath, LTE_filepath, E_MeV, mmap_filepath, charge_C, emit_ratio,
    RFvolt, RFharm, max_mom_aper_percent=None, ignoreMismatch=True,
    use_beamline=None, output_file_type=None, del_tmp_files=True, print_cmd=False):
    """"""

    with open(LTE_filepath, 'r') as f:
        file_contents = f.read()

    nElectrons = int(np.round(charge_C / PHYSCONST.e))

    input_dict = dict(
        LTE_filepath=os.path.abspath(LTE_filepath), E_MeV=E_MeV,
        mmap_filepath=mmap_filepath, charge_C=charge_C, nElectrons=nElectrons,
        emit_ratio=emit_ratio, RFvolt=RFvolt, RFharm=RFharm,
        max_mom_aper_percent=max_mom_aper_percent, ignoreMismatch=ignoreMismatch,
        use_beamline=use_beamline, del_tmp_files=del_tmp_files,
        lattice_file_contents=file_contents,
        timestamp_ini=util.get_current_local_time_str(),
    )

    output_file_type = _auto_check_output_file_type(output_filepath, output_file_type)
    input_dict['output_file_type'] = output_file_type

    if output_file_type in ('hdf5', 'h5'):
        _save_input_to_hdf5(output_filepath, input_dict)

    tmp = tempfile.NamedTemporaryFile(
        dir=os.getcwd(), delete=False, prefix=f'tmpTau_', suffix='.pgz')
    twi_pgz_filepath = os.path.abspath(tmp.name)
    life_filepath = '.'.join(twi_pgz_filepath.split('.')[:-1] + ['life'])
    tmp.close()

    tmp_filepaths = twiss.calc_ring_twiss(
        twi_pgz_filepath, LTE_filepath, E_MeV, use_beamline=use_beamline,
        parameters=None, radiation_integrals=True, run_local=True,
        del_tmp_files=False)
    tmp_filepaths['twi_pgz'] = twi_pgz_filepath
    #print(tmp_filepaths)

    #twi = util.load_pgz_file(twi_pgz_filepath)
    #try:
        #os.remove(twi_pgz_filepath)
    #except:
        #pass

    try:
        sdds.sdds2dicts(mmap_filepath)
        # "mmap_filepath" is a valid SDDS file
        mmap_sdds_filepath = mmap_filepath
    except:
        # "mmap_filepath" is NOT a valid SDDS file, and most likely an HDF5
        # file generated from an SDDS file. Try to convert it back to a valid
        # SDDS file.
        d = util.load_sdds_hdf5_file(mmap_filepath)
        mmap_d = d[0]['mmap']
        mmap_sdds_filepath = '.'.join(twi_pgz_filepath.split('.')[:-1] + ['mmap'])
        sdds.dicts2sdds(
            mmap_sdds_filepath, params=mmap_d['scalars'],
            columns=mmap_d['arrays'], outputMode='binary')
        tmp_filepaths['mmap'] = mmap_sdds_filepath

    cmd_str = (
        f'touschekLifetime {life_filepath} -twiss={tmp_filepaths["twi"]} '
        f'-aperture={mmap_sdds_filepath} -particles={nElectrons:d} '
        f'-coupling={emit_ratio:.9g} '
        f'-RF=Voltage={RFvolt/1e6:.9g},harmonic={RFharm:d},limit ')

    if max_mom_aper_percent is not None:
        cmd_str += f'-deltaLimit={max_mom_aper_percent:.9g} '

    if ignoreMismatch:
        cmd_str += '-ignoreMismatch'

    cmd_list = shlex.split(cmd_str)
    if print_cmd:
        print('\n$ ' + ' '.join(cmd_list) + '\n')

    p = Popen(cmd_list, stdout=PIPE, stderr=PIPE, encoding='utf-8')
    out, err = p.communicate()
    if std_print_enabled['out']:
        print(out)
    if std_print_enabled['err'] and err:
        print('ERROR:')
        print(err)

    output_tmp_filepaths = dict(life=life_filepath)
    output, meta = {}, {}
    for k, v in output_tmp_filepaths.items():
        try:
            output[k], meta[k] = sdds.sdds2dicts(v)
        except:
            continue

    timestamp_fin = util.get_current_local_time_str()

    if output_file_type in ('hdf5', 'h5'):
        util.robust_sdds_hdf5_write(
            output_filepath, [output, meta], nMaxTry=10, sleep=10.0, mode='a')
        f = h5py.File(output_filepath)
        f['timestamp_fin'] = timestamp_fin
        f.close()

    elif output_file_type == 'pgz':
        mod_output = {}
        for k, v in output.items():
            mod_output[k] = {}
            if 'params' in v:
                mod_output[k]['scalars'] = v['params']
            if 'columns' in v:
                mod_output[k]['arrays'] = v['columns']
        mod_meta = {}
        for k, v in meta.items():
            mod_meta[k] = {}
            if 'params' in v:
                mod_meta[k]['scalars'] = v['params']
            if 'columns' in v:
                mod_meta[k]['arrays'] = v['columns']
        util.robust_pgz_file_write(
            output_filepath, dict(data=mod_output, meta=mod_meta,
                                  input=input_dict, timestamp_fin=timestamp_fin),
            nMaxTry=10, sleep=10.0)
    else:
        raise ValueError()

    tmp_filepaths.update(output_tmp_filepaths)
    if del_tmp_files:
        for fp in tmp_filepaths.values():
            if fp.startswith('/dev'):
                continue
            else:
                try:
                    os.remove(fp)
                except:
                    print(f'Failed to delete "{fp}"')

    return output_filepath

def plot_Touschek_lifetime(output_filepath, title='', slim=None):
    """"""

    try:
        d = util.load_pgz_file(output_filepath)
        tau_hr = d['data']['life']['scalars']['tLifetime']
        g = d['data']['life']['arrays']
        FN = g['FN']
        FP = g['FP']
        s = g['s']

    except:
        f = h5py.File(output_filepath, 'r')
        tau_hr = f['life']['scalars']['tLifetime'][()]
        g = f['life']['arrays']
        FN = g['FN'][()]
        FP = g['FP'][()]
        s = g['s'][()]
        f.close()

    font_sz = 18

    plt.figure()
    plt.plot(s, FN, 'b-', label=r'$\delta < 0$')
    plt.plot(s, FP, 'r-', label=r'$\delta > 0$')
    plt.axhline(0, color='k')
    plt.xlabel(r'$s\, [\mathrm{m}]$', size=font_sz)
    plt.ylabel(r'$f_{+}, f_{-}\, [\mathrm{s}^{-1}]$', size=font_sz)
    if slim is not None:
        plt.xlim([v for v in slim])
    tau_info_title = fr'$\tau_{{\mathrm{{Touschek}}}} = {tau_hr:.6g} [\mathrm{{hr}}]$'
    if title != '':
        plt.title('\n'.join([title, tau_info_title]), size=font_sz)
    else:
        plt.title(tau_info_title, size=font_sz)
    plt.legend(loc='best')
    plt.tight_layout()

def calc_chrom_twiss(
    output_filepath, LTE_filepath, E_MeV, delta_min, delta_max, ndelta,
    use_beamline=None, transmute_elements=None, ele_filepath=None,
    output_file_type=None, del_tmp_files=True, print_cmd=False,
    run_local=True, remote_opts=None):
    """"""

    with open(LTE_filepath, 'r') as f:
        file_contents = f.read()

    input_dict = dict(
        LTE_filepath=os.path.abspath(LTE_filepath), E_MeV=E_MeV,
        delta_min=delta_min, delta_max=delta_max, ndelta=ndelta,
        use_beamline=use_beamline, transmute_elements=transmute_elements,
        ele_filepath=ele_filepath, del_tmp_files=del_tmp_files,
        run_local=run_local, remote_opts=remote_opts,
        lattice_file_contents=file_contents,
        timestamp_ini=util.get_current_local_time_str(),
    )

    output_file_type = _auto_check_output_file_type(output_filepath, output_file_type)
    input_dict['output_file_type'] = output_file_type

    if output_file_type in ('hdf5', 'h5'):
        _save_input_to_hdf5(output_filepath, input_dict)

    if ele_filepath is None:
        tmp = tempfile.NamedTemporaryFile(
            dir=os.getcwd(), delete=False, prefix=f'tmpChromTwiss_', suffix='.ele')
        ele_filepath = os.path.abspath(tmp.name)
        tmp.close()

    ed = elebuilder.EleDesigner(double_format='.12g')

    if transmute_elements is None:
        transmute_elements = dict(RFCA='MARK', SREFFECTS='MARK')

    _add_transmute_blocks(ed, transmute_elements)

    ed.add_newline()

    ed.add_block('run_setup',
        lattice=LTE_filepath, p_central_mev=E_MeV, use_beamline=use_beamline,
    )

    ed.add_newline()

    ed.add_block('run_control')

    ed.add_newline()

    temp_malign_elem_name = 'ELEGANT_CHROM_TWISS_MAL'
    temp_malign_elem_def = f'{temp_malign_elem_name}: MALIGN'

    ed.add_block('insert_elements',
        name='*', exclude='*', add_at_start=True, element_def=temp_malign_elem_def
    )

    ed.add_newline()

    ed.add_block('alter_elements',
        name=temp_malign_elem_name, type='MALIGN', item='DP', value='<delta>')

    ed.add_newline()

    ed.add_block('twiss_output', filename='%s.twi')

    ed.write(ele_filepath)

    ed.update_output_filepaths(ele_filepath[:-4]) # Remove ".ele"
    #print(ed.actual_output_filepath_list)

    for fp in ed.actual_output_filepath_list:
        if fp.endswith('.twi'):
            twi_filepath = fp
        else:
            raise ValueError('This line should not be reached.')

    delta_array = np.linspace(delta_min, delta_max, ndelta)

    # Run Elegant
    if run_local:
        nuxs = np.full(ndelta, np.nan)
        nuys = np.full(ndelta, np.nan)
        for i, delta in enumerate(delta_array):
            run(ele_filepath, print_cmd=print_cmd,
                macros=dict(delta=f'{delta:.12g}'),
                print_stdout=std_print_enabled['out'],
                print_stderr=std_print_enabled['err'])

            output, _ = sdds.sdds2dicts(twi_filepath)

            nuxs[i] = output['params']['nux']
            nuys[i] = output['params']['nuy']
    else:
        raise NotImplementedError

    timestamp_fin = util.get_current_local_time_str()

    _save_chrom_data(
        output_filepath, output_file_type, delta_array, nuxs, nuys,
        timestamp_fin, input_dict)

    if del_tmp_files:
        for fp in ed.actual_output_filepath_list + [ele_filepath]:
            if fp.startswith('/dev'):
                continue
            else:
                try:
                    os.remove(fp)
                except:
                    print(f'Failed to delete "{fp}"')

    return output_filepath

def calc_chrom_track(
    output_filepath, LTE_filepath, E_MeV, delta_min, delta_max, ndelta,
    n_turns=256, x0_offset=1e-5, y0_offset=1e-5, use_beamline=None, N_KICKS=None,
    transmute_elements=None, ele_filepath=None, output_file_type=None,
    del_tmp_files=True, print_cmd=False,
    run_local=True, remote_opts=None):
    """"""

    LTE_file_pathobj = Path(LTE_filepath)

    file_contents = LTE_file_pathobj.read_text()

    input_dict = dict(
        LTE_filepath=str(LTE_file_pathobj.resolve()), E_MeV=E_MeV,
        delta_min=delta_min, delta_max=delta_max, ndelta=ndelta,
        n_turns=n_turns, x0_offset=x0_offset, y0_offset=y0_offset,
        use_beamline=use_beamline, N_KICKS=N_KICKS, transmute_elements=transmute_elements,
        ele_filepath=ele_filepath, del_tmp_files=del_tmp_files,
        run_local=run_local, remote_opts=remote_opts,
        lattice_file_contents=file_contents,
        timestamp_ini=util.get_current_local_time_str(),
    )

    output_file_type = _auto_check_output_file_type(output_filepath, output_file_type)
    input_dict['output_file_type'] = output_file_type

    if output_file_type in ('hdf5', 'h5'):
        _save_input_to_hdf5(output_filepath, input_dict)

    if ele_filepath is None:
        tmp = tempfile.NamedTemporaryFile(
            dir=Path.cwd(), delete=False, prefix=f'tmpChromTrack_', suffix='.ele')
        ele_pathobj = Path(tmp.name)
        ele_filepath = str(ele_pathobj.resolve())
        tmp.close()

    watch_pathobj = ele_pathobj.with_suffix('.wc')
    twi_pgz_pathobj = ele_pathobj.with_suffix('.twi.pgz')

    ed = elebuilder.EleDesigner(double_format='.12g')

    _add_transmute_blocks(ed, transmute_elements)

    ed.add_newline()

    ed.add_block('run_setup',
        lattice=LTE_filepath, p_central_mev=E_MeV, use_beamline=use_beamline,
    )

    ed.add_newline()

    temp_watch_elem_name = 'ELEGANT_CHROM_TRACK_WATCH'
    temp_watch_elem_def = (
        f'{temp_watch_elem_name}: WATCH, FILENAME="{watch_pathobj.resolve()}", '
        'MODE=coordinate')

    ed.add_block('insert_elements',
        name='*', exclude='*', add_at_start=True, element_def=temp_watch_elem_def
    )

    ed.add_newline()

    _add_N_KICKS_alter_elements_blocks(ed, N_KICKS)

    ed.add_newline()

    ed.add_block('run_control', n_passes=n_turns)

    ed.add_newline()

    centroid = {}
    centroid[0] = x0_offset
    centroid[2] = y0_offset
    centroid[5] = '<delta>'
    #
    ed.add_block(
        'bunched_beam', n_particles_per_bunch=1, centroid=centroid)

    ed.add_newline()

    ed.add_block('track')

    ed.write(ele_filepath)

    ed.update_output_filepaths(ele_filepath[:-4]) # Remove ".ele"
    #print(ed.actual_output_filepath_list)

    twiss.calc_ring_twiss(
        str(twi_pgz_pathobj), LTE_filepath, E_MeV, use_beamline=use_beamline,
        parameters=None, run_local=True, del_tmp_files=True)
    _d = util.load_pgz_file(str(twi_pgz_pathobj))
    nux0 = _d['data']['twi']['scalars']['nux']
    nuy0 = _d['data']['twi']['scalars']['nuy']
    twi_pgz_pathobj.unlink()

    nux0_frac = nux0 - np.floor(nux0)
    nuy0_frac = nuy0 - np.floor(nuy0)

    delta_array = np.linspace(delta_min, delta_max, ndelta)

    # Run Elegant
    if run_local:
        tbt = dict(x = np.full((n_turns, ndelta), np.nan),
                   y = np.full((n_turns, ndelta), np.nan),
                   xp = np.full((n_turns, ndelta), np.nan),
                   yp = np.full((n_turns, ndelta), np.nan))

        #tElapsed = dict(run_ele=0.0, sdds2dicts=0.0, tbt_population=0.0)

        for i, delta in enumerate(delta_array):
            #t0 = time.time()
            run(ele_filepath, print_cmd=print_cmd,
                macros=dict(delta=f'{delta:.12g}'),
                print_stdout=std_print_enabled['out'],
                print_stderr=std_print_enabled['err'])
            #tElapsed['run_ele'] += time.time() - t0

            #t0 = time.time()
            output, _ = sdds.sdds2dicts(watch_pathobj)
            #tElapsed['sdds2dicts'] += time.time() - t0

            #t0 = time.time()
            cols = output['columns']
            for k in list(tbt):
                tbt[k][:len(cols[k]), i] = cols[k]
            #tElapsed['tbt_population'] += time.time() - t0
    else:
        raise NotImplementedError

    #print(tElapsed)

    #t0 = time.time()
    # Estimate tunes from TbT data
    neg_delta_array = delta_array[delta_array < 0.0]
    neg_sort_inds = np.argsort(np.abs(neg_delta_array))
    sorted_neg_delta_inds = np.where(delta_array < 0.0)[0][neg_sort_inds]
    pos_delta_array = delta_array[delta_array >= 0.0]
    pos_sort_inds = np.argsort(pos_delta_array)
    sorted_pos_delta_inds = np.where(delta_array >= 0.0)[0][pos_sort_inds]
    sorted_neg_delta_inds, sorted_pos_delta_inds
    #
    nus = dict(x=np.full(delta_array.shape, np.nan),
               y=np.full(delta_array.shape, np.nan))
    #
    opts = dict(window='sine', resolution=1e-8)
    for sorted_delta_inds in [sorted_neg_delta_inds, sorted_pos_delta_inds]:
        init_nux = nux0_frac
        init_nuy = nuy0_frac
        for i in sorted_delta_inds:
            xarray = tbt['x'][:, i]
            yarray = tbt['y'][:, i]

            if np.any(np.isnan(xarray)) or np.any(np.isnan(yarray)):
                # Particle lost at some point.
                continue

            out = sigproc.getDftPeak(xarray, init_nux, **opts)
            nus['x'][i] = out['nu']
            init_nux = out['nu']

            out = sigproc.getDftPeak(yarray, init_nuy, **opts)
            nus['y'][i] = out['nu']
            init_nuy = out['nu']
    #
    nuxs = nus['x']
    nuys = nus['y']
    #print('* Time elapsed for tune estimation: {:.3f}'.format(time.time() - t0))

    timestamp_fin = util.get_current_local_time_str()

    _save_chrom_data(
        output_filepath, output_file_type, delta_array, nuxs, nuys,
        timestamp_fin, input_dict)

    if del_tmp_files:
        util.delete_temp_files(
            ed.actual_output_filepath_list + [ele_filepath, str(watch_pathobj)])

    return output_filepath

def _save_chrom_data(
    output_filepath, output_file_type, delta_array, nuxs, nuys, timestamp_fin,
    input_dict):
    """"""

    if output_file_type in ('hdf5', 'h5'):
        _kwargs = dict(compression='gzip')
        f = h5py.File(output_filepath)
        f.create_dataset('deltas', data=delta_array, **_kwargs)
        f.create_dataset('nuxs', data=nuxs, **_kwargs)
        f.create_dataset('nuys', data=nuys, **_kwargs)
        f['timestamp_fin'] = timestamp_fin
        f.close()

    elif output_file_type == 'pgz':
        util.robust_pgz_file_write(
            output_filepath, dict(deltas=delta_array, nuxs=nuxs, nuys=nuys,
                                  input=input_dict, timestamp_fin=timestamp_fin),
            nMaxTry=10, sleep=10.0)
    else:
        raise ValueError()

def plot_chrom(
    output_filepath, max_chrom_order=3, title='', deltalim=None,
    nuxlim=None, nuylim=None, max_resonance_line_order=5):
    """"""

    assert max_resonance_line_order <= 5

    try:
        d = util.load_pgz_file(output_filepath)
        deltas = d['deltas']
        nuxs = d['nuxs']
        nuys = d['nuys']
    except:
        f = h5py.File(output_filepath, 'r')
        deltas = f['deltas'][()]
        nuxs = f['nuxs'][()]
        nuys = f['nuys'][()]
        f.close()

    coeffs = dict(x=np.polyfit(deltas, nuxs, max_chrom_order),
                  y=np.polyfit(deltas, nuys, max_chrom_order))

    fit_label = {}
    for plane in ['x', 'y']:
        fit_label[plane] = fr'\nu_{plane} = '
        for i, c in zip(range(max_chrom_order + 1)[::-1], coeffs[plane]):
            coeff_str = f'{c:+.3g} '
            if 'e' in coeff_str:
                s1, s2 = coeff_str.split('e')
                coeff_str = fr'{s1} \times 10^{int(s2):d} '
            fit_label[plane] += coeff_str
            if i == 1:
                fit_label[plane] += r'\delta '
            elif i >= 2:
                fit_label[plane] += fr'\delta^{i:d} '

        fit_label[plane] = '${}$'.format(fit_label[plane].strip())

    is_nuxlim_frac = False
    if nuxlim is not None:
        if (0.0 <= nuxlim[0] <= 1.0) and (0.0 <= nuxlim[1] <= 1.0):
            is_nuxlim_frac = True
    is_nuylim_frac = False
    if nuylim is not None:
        if (0.0 <= nuylim[0] <= 1.0) and (0.0 <= nuylim[1] <= 1.0):
            is_nuylim_frac = True

    font_sz = 22

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    if is_nuxlim_frac:
        offset = np.floor(nuxs)
    else:
        offset = np.zeros(nuxs.shape)
    lines1 = ax1.plot(deltas * 1e2, nuxs - offset, 'b.', label=r'$\nu_x$')
    fit_lines1 = ax1.plot(
        deltas * 1e2, np.poly1d(coeffs['x'])(deltas) - offset, 'b-',
        label=fit_label['x'])
    ax2 = ax1.twinx()
    if is_nuxlim_frac:
        offset = np.floor(nuys)
    else:
        offset = np.zeros(nuys.shape)
    lines2 = ax2.plot(deltas * 1e2, nuys - offset, 'r.', label=r'$\nu_y$')
    fit_lines2 = ax2.plot(
        deltas * 1e2, np.poly1d(coeffs['y'])(deltas) - offset, 'r-',
        label=fit_label['y'])
    ax1.set_xlabel(r'$\delta\, [\%]$', size=font_sz)
    ax1.set_ylabel(r'$\nu_x$', size=font_sz, color='b')
    ax2.set_ylabel(r'$\nu_y$', size=font_sz, color='r')
    if deltalim is not None:
        ax1.set_xlim([v * 1e2 for v in deltalim])
    if nuxlim is not None:
        ax1.set_ylim(nuxlim)
    else:
        nuxlim = ax1.get_ylim()
    if nuylim is not None:
        ax2.set_ylim(nuylim)
    else:
        nuylim = ax2.get_ylim()
    if title != '':
        plt.title(title, size=font_sz, pad=60)
    combined_lines = fit_lines1 + fit_lines2
    leg = ax2.legend(combined_lines, [L.get_label() for L in combined_lines],
                     loc='upper center', bbox_to_anchor=(0.5, 1.3),
                     fancybox=True, shadow=True, prop=dict(size=12))
    plt.tight_layout()

    frac_nuxs = nuxs - np.floor(nuxs)
    frac_nuys = nuys - np.floor(nuys)

    frac_nuxlim = [v - np.floor(v) for v in nuxlim]
    frac_nuylim = [v - np.floor(v) for v in nuylim]

    fig = plt.figure()
    fig.add_subplot(111)
    plt.scatter(frac_nuxs, frac_nuys, s=10, c=deltas * 1e2, marker='o', cmap='jet')
    plt.xlim(frac_nuxlim)
    plt.ylim(frac_nuylim)
    plt.xlabel(r'$\nu_x$', size=font_sz)
    plt.ylabel(r'$\nu_y$', size=font_sz)
    #
    rd = util.ResonanceDiagram()
    lineprops = dict(
        color    =['k',  'k', 'g', 'm', 'm'],
        linestyle=['-', '--', '-', '-', ':'],
        linewidth=[  2,    2, 0.5, 0.5, 0.5],
    )
    for n in range(1, max_resonance_line_order):
        d = rd.getResonanceCoeffsAndLines(n, frac_nuxlim, frac_nuylim)
        prop = {k: lineprops[k][n-1] for k in ['color', 'linestyle', 'linewidth']}
        assert len(d['lines']) == len(d['coeffs'])
        for ((nux1, nuy1), (nux2, nuy2)), (nx, ny, _) in zip(d['lines'], d['coeffs']):
            _x = np.array([nux1, nux2])
            _x -= np.floor(_x)
            _y = np.array([nuy1, nuy2])
            _y -= np.floor(_y)
            plt.plot(_x, _y, label=rd.getResonanceCoeffLabelString(nx, ny), **prop)
    #leg = plt.legend(loc='best')
    #leg.set_draggable(True, use_blit=True)
    #
    cb = plt.colorbar()
    cb.ax.set_title(r'$\delta\, [\%]$', size=16)
    if title != '':
        plt.title(title, size=font_sz)
    plt.tight_layout()


