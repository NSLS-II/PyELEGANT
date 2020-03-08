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

def calc_cmap_xy(
    output_filepath, LTE_filepath, E_MeV, xmin, xmax, ymin, ymax, nx, ny,
    n_turns=1, delta_offset=0.0, forward_backward=1,
    use_beamline=None, N_KICKS=None, transmute_elements=None, ele_filepath=None,
    output_file_type=None, del_tmp_files=True,
    run_local=False, remote_opts=None):
    """"""

    return _calc_cmap(
        output_filepath, LTE_filepath, E_MeV, 'xy',
        xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
        nx=nx, ny=ny, n_turns=n_turns, delta_offset=delta_offset,
        forward_backward=forward_backward,
        use_beamline=use_beamline, N_KICKS=N_KICKS,
        transmute_elements=transmute_elements, ele_filepath=ele_filepath,
        output_file_type=output_file_type, del_tmp_files=del_tmp_files,
        run_local=run_local, remote_opts=remote_opts)

def calc_cmap_px(
    output_filepath, LTE_filepath, E_MeV, delta_min, delta_max, xmin, xmax, ndelta, nx,
    n_turns=1, y_offset=0.0, forward_backward=1,
    use_beamline=None, N_KICKS=None, transmute_elements=None, ele_filepath=None,
    output_file_type=None, del_tmp_files=True,
    run_local=False, remote_opts=None):
    """"""

    return _calc_cmap(
        output_filepath, LTE_filepath, E_MeV, 'px',
        xmin=xmin, xmax=xmax, delta_min=delta_min, delta_max=delta_max,
        nx=nx, ndelta=ndelta, n_turns=n_turns, y_offset=y_offset,
        forward_backward=forward_backward,
        use_beamline=use_beamline, N_KICKS=N_KICKS,
        transmute_elements=transmute_elements, ele_filepath=ele_filepath,
        output_file_type=output_file_type, del_tmp_files=del_tmp_files,
        run_local=run_local, remote_opts=remote_opts)

def _calc_cmap(
    output_filepath, LTE_filepath, E_MeV, plane, xmin=-0.1, xmax=0.1, ymin=1e-6, ymax=0.1,
    delta_min=0.0, delta_max=0.0, nx=20, ny=21, ndelta=1, n_turns=1,
    delta_offset=0.0, y_offset=0.0, forward_backward=1,
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

    if forward_backward < 1:
        raise ValueError('"forward_backward" must be an integer >= 1.')

    with open(LTE_filepath, 'r') as f:
        file_contents = f.read()

    input_dict = dict(
        LTE_filepath=os.path.abspath(LTE_filepath), E_MeV=E_MeV, n_turns=n_turns,
        forward_backward=forward_backward, use_beamline=use_beamline,
        N_KICKS=N_KICKS, transmute_elements=transmute_elements,
        ele_filepath=ele_filepath, del_tmp_files=del_tmp_files, run_local=run_local,
        remote_opts=remote_opts,
        lattice_file_contents=file_contents,
        timestamp_ini=util.get_current_local_time_str(),
    )
    input_dict['cmap_plane'] = plane
    if plane == 'xy':
        input_dict['xmin'] = xmin
        input_dict['xmax'] = xmax
        input_dict['ymin'] = ymin
        input_dict['ymax'] = ymax
        input_dict['nx'] = nx
        input_dict['ny'] = ny
        input_dict['delta_offset'] = delta_offset

        plane_specific_chaos_map_block_opts = dict(
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

        plane_specific_chaos_map_block_opts = dict(
            xmin=xmin, xmax=xmax, ymin=y_offset, ymax=y_offset,
            delta_min=delta_min, delta_max=delta_max, nx=nx, ny=1, ndelta=ndelta)

    output_file_type = util.auto_check_output_file_type(output_filepath, output_file_type)
    input_dict['output_file_type'] = output_file_type

    if output_file_type in ('hdf5', 'h5'):
        util.save_input_to_hdf5(output_filepath, input_dict)

    if ele_filepath is None:
        tmp = tempfile.NamedTemporaryFile(
            dir=os.getcwd(), delete=False, prefix=f'tmpCMAP{plane}_', suffix='.ele')
        ele_filepath = os.path.abspath(tmp.name)
        tmp.close()

    ed = elebuilder.EleDesigner(ele_filepath, double_format='.12g')

    elebuilder.add_transmute_blocks(ed, transmute_elements)

    ed.add_newline()

    ed.add_block('run_setup',
        lattice=LTE_filepath, p_central_mev=E_MeV, use_beamline=use_beamline,
        semaphore_file='%s.done')

    ed.add_newline()

    ed.add_block('run_control', n_passes=n_turns)

    ed.add_newline()

    elebuilder.add_N_KICKS_alter_elements_blocks(ed, N_KICKS)

    ed.add_block('bunched_beam', n_particles_per_bunch=1)

    ed.add_newline()

    ed.add_block('twiss_output', filename='%s.twi')

    ed.add_newline()

    ed.add_block('chaos_map',
        output='%s.cmap', forward_backward=forward_backward, verbosity=False,
        **plane_specific_chaos_map_block_opts
    )

    ed.write()
    #print(ed.actual_output_filepath_list)

    for fp in ed.actual_output_filepath_list:
        if fp.endswith('.cmap'):
            cmap_output_filepath = fp
        elif fp.endswith('.twi'):
            twi_filepath = fp
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
                use_sbatch=False, pelegant=True, job_name='cmap',
                output='cmap.%J.out', error='cmap.%J.err',
                partition='normal', ntasks=50)

        remote.run(remote_opts, ele_filepath, print_cmd=True,
                   print_stdout=std_print_enabled['out'],
                   print_stderr=std_print_enabled['err'],
                   output_filepaths=None)

    tmp_filepaths = dict(cmap=cmap_output_filepath)
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

def plot_cmap_xy(
    output_filepath, title='', xlim=None, ylim=None, scatter=True,
    is_log10=True, cmin=-24, cmax=-10):
    """"""

    _plot_cmap(output_filepath, title=title, xlim=xlim, ylim=ylim,
               scatter=scatter, is_log10=is_log10, cmin=cmin, cmax=cmax)

def plot_cmap_px(
    output_filepath, title='', deltalim=None, xlim=None, scatter=True,
    is_log10=True, cmin=-24, cmax=-10):
    """"""

    _plot_cmap(output_filepath, title=title, deltalim=deltalim, xlim=xlim,
               scatter=scatter, is_log10=is_log10, cmin=cmin, cmax=cmax)

def _plot_cmap(
    output_filepath, title='', xlim=None, ylim=None, deltalim=None,
    scatter=True, is_log10=True, cmin=-24, cmax=-10):
    """"""

    try:
        d = util.load_pgz_file(output_filepath)
        plane = d['input']['cmap_plane']
        g = d['data']['cmap']['arrays']
        if plane == 'xy':
            v1 = g['x']
            v2 = g['y']
        elif plane == 'px':
            v1 = g['delta']
            v2 = g['x']
        else:
            raise ValueError(f'Unexpected "cmap_plane" value: {plane}')

        survived = g['Survived'].astype(bool)

        if is_log10:
            chaos = g['Log10dF']
        else:
            chaos = g['dF']

        if not scatter:
            g = d['input']
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
        plane = f['input']['cmap_plane'][()]
        g = f['cmap']['arrays']
        if plane == 'xy':
            v1 = g['x'][()]
            v2 = g['y'][()]
        elif plane == 'px':
            v1 = g['delta'][()]
            v2 = g['x'][()]
        else:
            raise ValueError(f'Unexpected "cmap_plane" value: {plane}')

        survived = g['Survived'][()].astype(bool)

        if is_log10:
            chaos = g['Log10dF'][()]
        else:
            chaos = g['dF'][()]

        if not scatter:
            g = f['input']
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

    chaos[~survived] = np.nan

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

    if is_log10:
        EQ_STR = r'$\rm{log}_{10}(\Delta)$'
        values = chaos
    else:
        EQ_STR = r'$\Delta$'
        values = chaos

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
            plt.xlim([v * v1unitconv for v in v1lim])
        if v2lim is not None:
            plt.ylim([v * v2unitconv for v in v2lim])
        if title != '':
            plt.title(title, size=font_sz)
        cb = plt.colorbar()
        try:
            cb.set_ticks(range(LB, UB+1))
            cb.set_ticklabels([str(i) for i in range(LB, UB+1)])
        except:
            pass
        cb.ax.set_title(EQ_STR)
        cb.ax.title.set_position((0.5, 1.02))
        plt.tight_layout()

    else:

        font_sz = 18

        v1array = np.linspace(v1min, v1max, n1)
        v2array = np.linspace(v2min, v2max, n2)

        V1, V2 = np.meshgrid(v1array, v2array)
        D = V1 * np.nan

        v1inds = np.argmin(np.abs(
            v1array.reshape((-1,1)) @ np.ones((1, v1.size)) - v1), axis=0)
        v2inds = np.argmin(np.abs(
            v2array.reshape((-1,1)) @ np.ones((1, v2.size)) - v2), axis=0)
        flatinds = np.ravel_multi_index((v1inds, v2inds), V1.T.shape, order='F')
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
            plt.xlim([v * v1unitconv for v in v1lim])
        if v2lim is not None:
            plt.ylim([v * v2unitconv for v in v2lim])
        if title != '':
            plt.title(title, size=font_sz)
        cb = plt.colorbar()
        try:
            cb.set_ticks(range(LB, UB+1))
            cb.set_ticklabels([str(i) for i in range(LB, UB+1)])
        except:
            pass
        cb.ax.set_title(EQ_STR)
        cb.ax.title.set_position((0.5, 1.02))
        plt.tight_layout()


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

    output_file_type = util.auto_check_output_file_type(output_filepath, output_file_type)
    input_dict['output_file_type'] = output_file_type

    if output_file_type in ('hdf5', 'h5'):
        util.save_input_to_hdf5(output_filepath, input_dict)

    if ele_filepath is None:
        tmp = tempfile.NamedTemporaryFile(
            dir=os.getcwd(), delete=False, prefix=f'tmpFMA{plane}_', suffix='.ele')
        ele_filepath = os.path.abspath(tmp.name)
        tmp.close()

    ed = elebuilder.EleDesigner(ele_filepath, double_format='.12g')

    elebuilder.add_transmute_blocks(ed, transmute_elements)

    ed.add_newline()

    ed.add_block('run_setup',
        lattice=LTE_filepath, p_central_mev=E_MeV, use_beamline=use_beamline,
        semaphore_file='%s.done')

    ed.add_newline()

    ed.add_block('run_control', n_passes=n_turns)

    ed.add_newline()

    elebuilder.add_N_KICKS_alter_elements_blocks(ed, N_KICKS)

    ed.add_block('bunched_beam', n_particles_per_bunch=1)

    ed.add_newline()

    ed.add_block('frequency_map',
        output='%s.fma', include_changes=True, quadratic_spacing=quadratic_spacing,
        full_grid_output=full_grid_output, **plane_specific_freq_map_block_opts
    )

    ed.write()
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
            plt.xlim([v * v1unitconv for v in v1lim])
        if v2lim is not None:
            plt.ylim([v * v2unitconv for v in v2lim])
        if title != '':
            plt.title(title, size=font_sz)
        cb = plt.colorbar()
        try:
            cb.set_ticks(range(LB, UB+1))
            cb.set_ticklabels([str(i) for i in range(LB, UB+1)])
        except:
            pass
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
        flatinds = np.ravel_multi_index((v1inds, v2inds), V1.T.shape, order='F')
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
            plt.xlim([v * v1unitconv for v in v1lim])
        if v2lim is not None:
            plt.ylim([v * v2unitconv for v in v2lim])
        if title != '':
            plt.title(title, size=font_sz)
        cb = plt.colorbar()
        try:
            cb.set_ticks(range(LB, UB+1))
            cb.set_ticklabels([str(i) for i in range(LB, UB+1)])
        except:
            pass
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

    output_file_type = util.auto_check_output_file_type(output_filepath, output_file_type)
    input_dict['output_file_type'] = output_file_type

    if output_file_type in ('hdf5', 'h5'):
        util.save_input_to_hdf5(output_filepath, input_dict)

    if ele_filepath is None:
        tmp = tempfile.NamedTemporaryFile(
            dir=os.getcwd(), delete=False, prefix=f'tmpFindAper_', suffix='.ele')
        ele_filepath = os.path.abspath(tmp.name)
        tmp.close()

    ed = elebuilder.EleDesigner(ele_filepath, double_format='.12g')

    elebuilder.add_transmute_blocks(ed, transmute_elements)

    ed.add_newline()

    ed.add_block('run_setup',
        lattice=LTE_filepath, p_central_mev=E_MeV, use_beamline=use_beamline,
        semaphore_file='%s.done')

    ed.add_newline()

    ed.add_block('run_control', n_passes=n_turns)

    ed.add_newline()

    elebuilder.add_N_KICKS_alter_elements_blocks(ed, N_KICKS)

    ed.add_newline()

    ed.add_block('find_aperture',
        output='%s.aper', mode='n-lines', xmax=xmax, ymax=ymax, nx=ini_ndiv,
        n_lines=n_lines, full_plane=neg_y_search,
        offset_by_orbit=True, # recommended according to the manual
    )

    ed.write()
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

    output_file_type = util.auto_check_output_file_type(output_filepath, output_file_type)
    input_dict['output_file_type'] = output_file_type

    if output_file_type in ('hdf5', 'h5'):
        util.save_input_to_hdf5(output_filepath, input_dict)

    if ele_filepath is None:
        tmp = tempfile.NamedTemporaryFile(
            dir=os.getcwd(), delete=False, prefix=f'tmpMomAper_', suffix='.ele')
        ele_filepath = os.path.abspath(tmp.name)
        tmp.close()

    ed = elebuilder.EleDesigner(ele_filepath, double_format='.12g')

    elebuilder.add_transmute_blocks(ed, transmute_elements)

    ed.add_newline()

    ed.add_block('run_setup',
        lattice=LTE_filepath, p_central_mev=E_MeV, use_beamline=use_beamline,
        semaphore_file='%s.done')

    ed.add_newline()

    ed.add_block('run_control', n_passes=n_turns)

    ed.add_newline()

    elebuilder.add_N_KICKS_alter_elements_blocks(ed, N_KICKS)

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

    ed.write()
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

    output_file_type = util.auto_check_output_file_type(output_filepath, output_file_type)
    input_dict['output_file_type'] = output_file_type

    if output_file_type in ('hdf5', 'h5'):
        util.save_input_to_hdf5(output_filepath, input_dict)

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

    output_file_type = util.auto_check_output_file_type(output_filepath, output_file_type)
    input_dict['output_file_type'] = output_file_type

    if output_file_type in ('hdf5', 'h5'):
        util.save_input_to_hdf5(output_filepath, input_dict)

    if ele_filepath is None:
        tmp = tempfile.NamedTemporaryFile(
            dir=os.getcwd(), delete=False, prefix=f'tmpChromTwiss_', suffix='.ele')
        ele_filepath = os.path.abspath(tmp.name)
        tmp.close()

    ed = elebuilder.EleDesigner(ele_filepath, double_format='.12g')

    if transmute_elements is None:
        transmute_elements = dict(RFCA='MARK', SREFFECTS='MARK')

    elebuilder.add_transmute_blocks(ed, transmute_elements)

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

    ed.write()
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
    courant_snyder=True,
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
        courant_snyder=courant_snyder,
        n_turns=n_turns, x0_offset=x0_offset, y0_offset=y0_offset,
        use_beamline=use_beamline, N_KICKS=N_KICKS, transmute_elements=transmute_elements,
        ele_filepath=ele_filepath, del_tmp_files=del_tmp_files,
        run_local=run_local, remote_opts=remote_opts,
        lattice_file_contents=file_contents,
        timestamp_ini=util.get_current_local_time_str(),
    )

    output_file_type = util.auto_check_output_file_type(output_filepath, output_file_type)
    input_dict['output_file_type'] = output_file_type

    if output_file_type in ('hdf5', 'h5'):
        util.save_input_to_hdf5(output_filepath, input_dict)

    if ele_filepath is None:
        tmp = tempfile.NamedTemporaryFile(
            dir=Path.cwd(), delete=False, prefix=f'tmpChromTrack_', suffix='.ele')
        ele_pathobj = Path(tmp.name)
        ele_filepath = str(ele_pathobj.resolve())
        tmp.close()

    watch_pathobj = ele_pathobj.with_suffix('.wc')
    twi_pgz_pathobj = ele_pathobj.with_suffix('.twi.pgz')

    ed = elebuilder.EleDesigner(ele_filepath, double_format='.12g')

    elebuilder.add_transmute_blocks(ed, transmute_elements)

    ed.add_newline()

    ed.add_block('run_setup',
        lattice=LTE_filepath, p_central_mev=E_MeV, use_beamline=use_beamline,
    )

    ed.add_newline()

    temp_watch_elem_name = 'ELEGANT_CHROM_TRACK_WATCH'
    if run_local:
        watch_filepath = str(watch_pathobj.resolve())
    else:
        watch_filepath = watch_pathobj.name
    temp_watch_elem_def = (
        f'{temp_watch_elem_name}: WATCH, FILENAME="{watch_filepath}", '
        'MODE=coordinate')

    ed.add_block('insert_elements',
        name='*', exclude='*', add_at_start=True, element_def=temp_watch_elem_def
    )

    ed.add_newline()

    elebuilder.add_N_KICKS_alter_elements_blocks(ed, N_KICKS)

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

    ed.write()
    #print(ed.actual_output_filepath_list)

    twiss.calc_ring_twiss(
        str(twi_pgz_pathobj), LTE_filepath, E_MeV, use_beamline=use_beamline,
        parameters=None, run_local=True, del_tmp_files=True)
    _d = util.load_pgz_file(str(twi_pgz_pathobj))
    nux0 = _d['data']['twi']['scalars']['nux']
    nuy0 = _d['data']['twi']['scalars']['nuy']
    # alpha & beta at watch element (at the start of the lattice)
    betax = _d['data']['twi']['arrays']['betax'][0]
    betay = _d['data']['twi']['arrays']['betay'][0]
    alphax = _d['data']['twi']['arrays']['alphax'][0]
    alphay = _d['data']['twi']['arrays']['alphay'][0]
    twi_pgz_pathobj.unlink()

    delta_array = np.linspace(delta_min, delta_max, ndelta)

    # Run Elegant
    if run_local:
        tbt = dict(x = np.full((n_turns, ndelta), np.nan),
                   y = np.full((n_turns, ndelta), np.nan),
                   )
        if courant_snyder:
            tbt['xp'] = np.full((n_turns, ndelta), np.nan)
            tbt['yp'] = np.full((n_turns, ndelta), np.nan)

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

        if remote_opts is None:
            remote_opts = dict(ntasks=20)
            #remote_opts = dict(
                #use_sbatch=True, pelegant=False, job_name='chrom',
                #output='chrom.%J.out', error='chrom.%J.err',
                #partition='short', ntasks=50)

        delta_sub_array_list, reverse_mapping = util.chunk_list(
            delta_array, remote_opts['ntasks'])

        coords_list = ['x', 'y']
        if courant_snyder:
            coords_list += ['xp', 'yp']

        module_name = 'pyelegant.nonlin'
        func_name = '_calc_chrom_track_get_tbt'
        chunked_results = remote.run_mpi_python(
            remote_opts, module_name, func_name, delta_sub_array_list,
            (ele_pathobj.read_text(), ele_pathobj.name, watch_pathobj.name,
             print_cmd, std_print_enabled['out'], std_print_enabled['err'],
             coords_list),
        )

        tbt_chunked_list = dict()
        tbt_flat_list = dict()
        for plane in coords_list:
            tbt_chunked_list[plane] = [_d[plane] for _d in chunked_results]
            tbt_flat_list[plane] = util.unchunk_list_of_lists(
                tbt_chunked_list[plane], reverse_mapping)

        tbt = dict(x = np.full((n_turns, ndelta), np.nan),
                   y = np.full((n_turns, ndelta), np.nan))
        if courant_snyder:
            tbt['xp'] = np.full((n_turns, ndelta), np.nan)
            tbt['yp'] = np.full((n_turns, ndelta), np.nan)
        for plane in coords_list:
            for iDelta, array in enumerate(tbt_flat_list[plane]):
                tbt[plane][:len(array), iDelta] = array

    #print(tElapsed)

    #t0 = time.time()
    # Estimate tunes from TbT data
    if courant_snyder:
        nus = calc_chrom_from_tbt_cs(
            delta_array, tbt['x'], tbt['y'], nux0, nuy0,
            tbt['xp'], tbt['yp'], betax, alphax, betay, alphay,
            init_guess_from_prev_step=True)
        extra_save_kwargs = dict(
            xptbt=tbt['xp'], yptbt=tbt['yp'],
            betax=betax, alphax=alphax, betay=betay, alphay=alphay)
    else:
        nus = calc_chrom_from_tbt_ps(delta_array, tbt['x'], tbt['y'], nux0, nuy0)
        extra_save_kwargs = {}
    nuxs = nus['x']
    nuys = nus['y']
    #print('* Time elapsed for tune estimation: {:.3f}'.format(time.time() - t0))

    timestamp_fin = util.get_current_local_time_str()

    _save_chrom_data(
        output_filepath, output_file_type, delta_array, nuxs, nuys,
        timestamp_fin, input_dict, xtbt=tbt['x'], ytbt=tbt['y'],
        nux0=nux0, nuy0=nuy0, **extra_save_kwargs)

    if del_tmp_files:
        util.delete_temp_files(
            ed.actual_output_filepath_list + [ele_filepath, str(watch_pathobj)])

    return output_filepath

def calc_chrom_from_tbt_ps(delta_array, xtbt, ytbt, nux0, nuy0):
    """
    Using phase-space (ps) variables "x" and "y", which can only
    determine tunes within the range of [0, 0.5].
    """

    frac_nux0 = nux0 - np.floor(nux0)
    frac_nuy0 = nuy0 - np.floor(nuy0)

    if frac_nux0 > 0.5:
        frac_nux0 = 1 - frac_nux0
        nux_above_half = True
    else:
        nux_above_half = False

    if frac_nuy0 > 0.5:
        frac_nuy0 = 1 - frac_nuy0
        nuy_above_half = True
    else:
        nuy_above_half = False

    neg_delta_array = delta_array[delta_array < 0.0]
    neg_sort_inds = np.argsort(np.abs(neg_delta_array))
    sorted_neg_delta_inds = np.where(delta_array < 0.0)[0][neg_sort_inds]
    pos_delta_array = delta_array[delta_array >= 0.0]
    pos_sort_inds = np.argsort(pos_delta_array)
    sorted_pos_delta_inds = np.where(delta_array >= 0.0)[0][pos_sort_inds]
    sorted_neg_delta_inds, sorted_pos_delta_inds

    nus = dict(x=np.full(delta_array.shape, np.nan),
               y=np.full(delta_array.shape, np.nan))

    n_turns = xtbt.shape[0]
    nu_vec = np.fft.fftfreq(n_turns)

    opts = dict(window='sine', resolution=1e-8)
    for sorted_delta_inds in [sorted_neg_delta_inds, sorted_pos_delta_inds]:
        init_nux = frac_nux0
        init_nuy = frac_nuy0
        for i in sorted_delta_inds:
            xarray = xtbt[:, i]
            yarray = ytbt[:, i]

            if np.any(np.isnan(xarray)) or np.any(np.isnan(yarray)):
                # Particle lost at some point.
                continue

            if False:
                # This algorithm does NOT work too well if tune change
                # between neighboring delta points are too large.
                out = sigproc.getDftPeak(xarray, init_nux, **opts)
                nus['x'][i] = out['nu']
                init_nux = out['nu']

                out = sigproc.getDftPeak(yarray, init_nuy, **opts)
                nus['y'][i] = out['nu']
                init_nuy = out['nu']
            else:
                # Find the rough peak first
                ff_rect = np.fft.fft(xarray - np.mean(xarray))
                A_arb = np.abs(ff_rect)
                init_nux = nu_vec[np.argmax(A_arb[:(n_turns//2)])]
                # Then fine-tune
                out = sigproc.getDftPeak(xarray, init_nux, **opts)
                nus['x'][i] = out['nu']

                # Find the rough peak first
                ff_rect = np.fft.fft(yarray - np.mean(yarray))
                A_arb = np.abs(ff_rect)
                init_nuy = nu_vec[np.argmax(A_arb[:(n_turns//2)])]
                # Then fine-tune
                out = sigproc.getDftPeak(yarray, init_nuy, **opts)
                nus['y'][i] = out['nu']

    if nux_above_half:
        nus['x'] = 1 - nus['x']
    if nuy_above_half:
        nus['y'] = 1 - nus['y']

    return nus

def angle2p(xp, yp, delta):
    """
    Convert angles (xp, yp) into canonical momenta (px, py)
    """

    fac = (1 + delta) / np.sqrt(1 + xp**2 + yp**2)

    px = xp * fac
    py = yp * fac

    return px, py

def p2angle(px, py, delta):
    """
    Convert canonical momenta (px, py) into angles (xp, yp)
    """

    fac = np.sqrt((1 + delta)**2 - (px**2 + py**2))

    xp = px / fac
    yp = py / fac

    return xp, yp

def approx_angle2p(xp, yp, delta):
    """
    Convert angles (xp, yp) into canonical momenta (px, py)
    **approximately** with small-angle assumption.
    """

    fac = 1 + delta

    px = xp * fac
    py = yp * fac

    return px, py

def approx_p2angle(px, py, delta):
    """
    Convert canonical momenta (px, py) into angles (xp, yp)
    **approximately** with small-angle assumption.
    """

    fac = 1 + delta

    xp = px / fac
    yp = py / fac

    return xp, yp

def calc_chrom_from_tbt_cs(
    delta_array, xtbt, ytbt, nux0, nuy0,
    xptbt, yptbt, betax, alphax, betay, alphay, init_guess_from_prev_step=True):
    """
    Using Courant-Snyder (CS) coordinates "xhat" and "yhat", which enables
    determination of tunes within the range of [0, 1.0].

    If `init_guess_from_prev_step` is True (recommended), then the tune peak
    fine-tuning starts from the tune peak found from the previous momentum offset
    to avoid any sudden tune jump along an increasing momentum offset vector.
    If False, then a rough tune peak is found from a simple FFT spectrum, which
    can potentially lead the initial search point to another strong resonance
    peak, away from the fundamental tune peak.
    """

    frac_nux0 = nux0 - np.floor(nux0)
    frac_nuy0 = nuy0 - np.floor(nuy0)

    if frac_nux0 > 0.5:
        frac_nux0 = frac_nux0 - 1
    if frac_nuy0 > 0.5:
        frac_nuy0 = frac_nuy0 - 1

    neg_delta_array = delta_array[delta_array < 0.0]
    neg_sort_inds = np.argsort(np.abs(neg_delta_array))
    sorted_neg_delta_inds = np.where(delta_array < 0.0)[0][neg_sort_inds]
    pos_delta_array = delta_array[delta_array >= 0.0]
    pos_sort_inds = np.argsort(pos_delta_array)
    sorted_pos_delta_inds = np.where(delta_array >= 0.0)[0][pos_sort_inds]
    sorted_neg_delta_inds, sorted_pos_delta_inds

    nus = dict(x=np.full(delta_array.shape, np.nan),
               y=np.full(delta_array.shape, np.nan))

    n_turns = xtbt.shape[0]
    nu_vec = np.fft.fftfreq(n_turns)

    opts = dict(window='sine', resolution=1e-8)
    for sorted_delta_inds in [sorted_neg_delta_inds, sorted_pos_delta_inds]:
        init_nux = frac_nux0
        init_nuy = frac_nuy0
        for i in sorted_delta_inds:
            xarray = xtbt[:, i]
            yarray = ytbt[:, i]

            if np.any(np.isnan(xarray)) or np.any(np.isnan(yarray)):
                # Particle lost at some point.
                continue

            xparray = xptbt[:, i]
            yparray = yptbt[:, i]
            delta = delta_array[i]
            pxarray, pyarray = angle2p(xparray, yparray, delta)

            xarray -= np.mean(xarray)
            yarray -= np.mean(yarray)
            pxarray -= np.mean(pxarray)
            pyarray -= np.mean(pyarray)

            xhat = xarray / np.sqrt(betax)
            pxhat = alphax / np.sqrt(betax) * xarray + np.sqrt(betax) * pxarray
            yhat = yarray / np.sqrt(betay)
            pyhat = alphay / np.sqrt(betay) * yarray + np.sqrt(betay) * pyarray

            hx = xhat - 1j * pxhat
            hy = yhat - 1j * pyhat

            if init_guess_from_prev_step:
                out = sigproc.getDftPeak(hx, init_nux, **opts)
                nus['x'][i] = out['nu']
                init_nux = out['nu']
            else:
                # Find the rough peak first
                ff_rect = np.fft.fft(hx)
                A_arb = np.abs(ff_rect)
                init_nux = nu_vec[np.argmax(A_arb)]
                # Then fine-tune
                out = sigproc.getDftPeak(hx, init_nux, **opts)
                nus['x'][i] = out['nu']

            if init_guess_from_prev_step:
                out = sigproc.getDftPeak(hy, init_nuy, **opts)
                nus['y'][i] = out['nu']
                init_nuy = out['nu']
            else:
                # Find the rough peak first
                ff_rect = np.fft.fft(hy)
                A_arb = np.abs(ff_rect)
                init_nuy = nu_vec[np.argmax(A_arb)]
                # Then fine-tune
                out = sigproc.getDftPeak(hy, init_nuy, **opts)
                nus['y'][i] = out['nu']

    nonnan_inds = np.where(~np.isnan(nus['x']))[0]
    neg_inds = nonnan_inds[nus['x'][nonnan_inds] < 0]
    nus['x'][neg_inds] += 1
    nonnan_inds = np.where(~np.isnan(nus['y']))[0]
    neg_inds = nonnan_inds[nus['y'][nonnan_inds] < 0]
    nus['y'][neg_inds] += 1

    return nus

def _calc_chrom_track_get_tbt(
    delta_sub_array, ele_contents, ele_filename, watch_filename,
    print_cmd, print_stdout, print_stderr, coords_list, tempdir_path='/tmp'):
    """"""

    if not Path(tempdir_path).exists():
        tempdir_path = Path.cwd()

    sub_tbt = dict()
    for k in coords_list:
        sub_tbt[k] = []

    with tempfile.TemporaryDirectory(
        prefix='tmpCalcChrom_', dir=tempdir_path) as tmpdirname:

        ele_pathobj = Path(tmpdirname).joinpath(ele_filename)
        watch_pathobj = Path(tmpdirname).joinpath(watch_filename)

        ele_contents = ele_contents.replace(
            watch_filename, str(watch_pathobj.resolve()))

        ele_pathobj.write_text(ele_contents)

        ele_filepath = str(ele_pathobj.resolve())

        for delta in delta_sub_array:

            run(ele_filepath, print_cmd=print_cmd,
                macros=dict(delta=f'{delta:.12g}'),
                print_stdout=print_stdout, print_stderr=print_stderr)

            output, _ = sdds.sdds2dicts(watch_pathobj)

            cols = output['columns']
            for k in list(sub_tbt):
                sub_tbt[k].append(cols[k])

    return sub_tbt

def _save_chrom_data(
    output_filepath, output_file_type, delta_array, nuxs, nuys, timestamp_fin,
    input_dict, xtbt=None, ytbt=None, nux0=None, nuy0=None,
    xptbt=None, yptbt=None, betax=None, alphax=None, betay=None, alphay=None):
    """
    nux0, nuy0: on-momentum tunes
    """

    if output_file_type in ('hdf5', 'h5'):
        _kwargs = dict(compression='gzip')
        f = h5py.File(output_filepath)
        f.create_dataset('deltas', data=delta_array, **_kwargs)
        f.create_dataset('nuxs', data=nuxs, **_kwargs)
        f.create_dataset('nuys', data=nuys, **_kwargs)
        if xtbt is not None:
            f.create_dataset('xtbt', data=xtbt, **_kwargs)
        if ytbt is not None:
            f.create_dataset('ytbt', data=ytbt, **_kwargs)
        if nux0:
            f['nux0'] = nux0
        if nuy0:
            f['nuy0'] = nuy0
        if xptbt is not None:
            f.create_dataset('xptbt', data=xptbt, **_kwargs)
        if yptbt is not None:
            f.create_dataset('yptbt', data=yptbt, **_kwargs)
        if betax:
            f['betax'] = betax
        if alphax:
            f['alphax'] = alphax
        if betay:
            f['betay'] = betay
        if alphay:
            f['alphay'] = alphay
        f['timestamp_fin'] = timestamp_fin
        f.close()

    elif output_file_type == 'pgz':
        d = dict(deltas=delta_array, nuxs=nuxs, nuys=nuys,
                 input=input_dict, timestamp_fin=timestamp_fin)
        if xtbt is not None:
            d['xtbt'] = xtbt
        if ytbt is not None:
            d['ytbt'] = ytbt
        if nux0:
            d['nux0'] = nux0
        if nuy0:
            d['nuy0'] = nuy0
        if xptbt is not None:
            d['xptbt'] = xptbt
        if yptbt is not None:
            d['yptbt'] = yptbt
        if betax:
            d['betax'] = betax
        if alphax:
            d['alphax'] = alphax
        if betay:
            d['betay'] = betay
        if alphay:
            d['alphay'] = alphay
        util.robust_pgz_file_write(output_filepath, d, nMaxTry=10, sleep=10.0)
    else:
        raise ValueError()

def plot_chrom(
    output_filepath, max_chrom_order=3, title='', deltalim=None, fit_deltalim=None,
    nuxlim=None, nuylim=None, max_resonance_line_order=5, fit_label_format='+.3g',
    ax_nu_vs_delta=None, ax_nuy_vs_nux=None):
    """"""

    assert max_resonance_line_order <= 5

    is_nuxlim_frac = False
    if nuxlim:
        if (0.0 <= nuxlim[0] <= 1.0) and (0.0 <= nuxlim[1] <= 1.0):
            is_nuxlim_frac = True

    is_nuylim_frac = False
    if nuylim:
        if (0.0 <= nuylim[0] <= 1.0) and (0.0 <= nuylim[1] <= 1.0):
            is_nuylim_frac = True

    if is_nuxlim_frac and is_nuylim_frac:
        _plot_nu_frac = True
    elif (not is_nuxlim_frac) and (not is_nuylim_frac):
        _plot_nu_frac = False
    else:
        raise ValueError('"nuxlim" and "nuylim" must be either both fractional or both non-fractional')

    try:
        d = util.load_pgz_file(output_filepath)
        deltas = d['deltas']
        nuxs = d['nuxs']
        nuys = d['nuys']
        if 'nux0' in d:
            nux0 = d['nux0']
            nuy0 = d['nuy0']

            nux0_int = np.floor(nux0)
            nuy0_int = np.floor(nuy0)

            nuxs += nux0_int
            nuys += nuy0_int
        else:
            nux0 = np.nanmedian(nuxs)
            nuy0 = np.nanmedian(nuys)

            nux0_int = np.floor(nux0)
            nuy0_int = np.floor(nuy0)
    except:
        f = h5py.File(output_filepath, 'r')
        deltas = f['deltas'][()]
        nuxs = f['nuxs'][()]
        nuys = f['nuys'][()]
        if 'nux0' in f:
            nux0 = f['nux0'][()]
            nuy0 = f['nuy0'][()]

            nux0_int = np.floor(nux0)
            nuy0_int = np.floor(nuy0)

            nuxs += nux0_int
            nuys += nuy0_int
        else:
            nux0 = np.nanmedian(nuxs)
            nuy0 = np.nanmedian(nuys)

            nux0_int = np.floor(nux0)
            nuy0_int = np.floor(nuy0)
        f.close()

    if deltalim:
        delta_incl = np.logical_and(deltas >= np.min(deltalim),
                                    deltas <= np.max(deltalim))
    else:
        delta_incl = np.ones(deltas.shape).astype(bool)
    deltas = deltas[delta_incl]
    nuxs = nuxs[delta_incl]
    nuys = nuys[delta_incl]

    if fit_deltalim:
        fit_delta_incl = np.logical_and(deltas >= np.min(fit_deltalim),
                                        deltas <= np.max(fit_deltalim))
        fit_deltas = deltas[fit_delta_incl]
        fit_nuxs = nuxs[fit_delta_incl]
        fit_nuys = nuys[fit_delta_incl]
    else:
        fit_delta_incl = np.ones(deltas.shape).astype(bool)
        fit_deltas = deltas
        fit_nuxs = nuxs
        fit_nuys = nuys
    nux_nan_inds = np.isnan(fit_nuxs)
    nuy_nan_inds = np.isnan(fit_nuys)
    coeffs = dict(
        x=np.polyfit(fit_deltas[~nux_nan_inds], fit_nuxs[~nux_nan_inds], max_chrom_order),
        y=np.polyfit(fit_deltas[~nuy_nan_inds], fit_nuys[~nuy_nan_inds], max_chrom_order))

    fit_label = {}
    for plane in ['x', 'y']:
        fit_label[plane] = fr'\nu_{plane} = '
        for i, c in zip(range(max_chrom_order + 1)[::-1], coeffs[plane]):

            if i != 0:
                fit_label[plane] += util.pprint_sci_notation(c, fit_label_format)
            else:
                fit_label[plane] += util.pprint_sci_notation(c, '+.3f')

            if i == 1:
                fit_label[plane] += r'\delta '
            elif i >= 2:
                fit_label[plane] += fr'\delta^{i:d} '

        fit_label[plane] = '${}$'.format(fit_label[plane].strip())

    font_sz = 22

    if ax_nu_vs_delta:
        ax1 = ax_nu_vs_delta
    else:
        fig, ax1 = plt.subplots()
    if _plot_nu_frac:
        offset = np.floor(nuxs)
    else:
        offset = np.zeros(nuxs.shape)
    lines1 = ax1.plot(deltas * 1e2, nuxs - offset, 'b.', label=r'$\nu_x$')
    fit_lines1 = ax1.plot(
        fit_deltas * 1e2, np.poly1d(coeffs['x'])(fit_deltas) - offset[fit_delta_incl], 'b-',
        label=fit_label['x'])
    ax2 = ax1.twinx()
    if _plot_nu_frac:
        offset = np.floor(nuys)
    else:
        offset = np.zeros(nuys.shape)
    lines2 = ax2.plot(deltas * 1e2, nuys - offset, 'r.', label=r'$\nu_y$')
    fit_lines2 = ax2.plot(
        fit_deltas * 1e2, np.poly1d(coeffs['y'])(fit_deltas) - offset[fit_delta_incl], 'r-',
        label=fit_label['y'])
    ax1.set_xlabel(r'$\delta\, [\%]$', size=font_sz)
    ax1.set_ylabel(r'$\nu_x$', size=font_sz, color='b')
    ax2.set_ylabel(r'$\nu_y$', size=font_sz, color='r')
    if deltalim is not None:
        ax1.set_xlim([v * 1e2 for v in deltalim])
    if nuxlim:
        ax1.set_ylim(nuxlim)
    else:
        nuxlim = ax1.get_ylim()
    if nuylim:
        ax2.set_ylim(nuylim)
    else:
        nuylim = ax2.get_ylim()
    if title != '':
        ax1.set_title(title, size=font_sz, pad=60)
    combined_lines = fit_lines1 + fit_lines2
    leg = ax2.legend(combined_lines, [L.get_label() for L in combined_lines],
                     loc='upper center', bbox_to_anchor=(0.5, 1.3),
                     fancybox=True, shadow=True, prop=dict(size=12))
    plt.sca(ax1)
    plt.tight_layout()



    if _plot_nu_frac:
        _nuxs = nuxs - nux0_int
        _nuys = nuys - nuy0_int

        frac_nuxlim = nuxlim
        frac_nuylim = nuylim
    else:
        _nuxs = nuxs
        _nuys = nuys

        frac_nuxlim = nuxlim - nux0_int
        frac_nuylim = nuylim - nuy0_int

    if ax_nuy_vs_nux:
        ax = ax_nuy_vs_nux
    else:
        _, ax = plt.subplots()
    sc_obj = ax.scatter(_nuxs, _nuys, s=10, c=deltas * 1e2, marker='o', cmap='jet')
    if nuxlim:
        ax.set_xlim(nuxlim)
    if nuylim:
        ax.set_ylim(nuylim)
    ax.set_xlabel(r'$\nu_x$', size=font_sz)
    ax.set_ylabel(r'$\nu_y$', size=font_sz)
    #
    rd = util.ResonanceDiagram()
    lineprops = dict(
        color    =['k',  'k', 'g', 'm', 'm'],
        linestyle=['-', '--', '-', '-', ':'],
        linewidth=[  2,    2, 0.5, 0.5, 0.5],
    )
    for n in range(1, max_resonance_line_order):
        d = rd.getResonanceCoeffsAndLines(
            n, np.array(frac_nuxlim) + nux0_int, frac_nuylim + nuy0_int) # <= CRITICAL: Must pass tunes w/ integer parts
        prop = {k: lineprops[k][n-1] for k in ['color', 'linestyle', 'linewidth']}
        assert len(d['lines']) == len(d['coeffs'])
        for ((nux1, nuy1), (nux2, nuy2)), (nx, ny, _) in zip(d['lines'], d['coeffs']):
            if _plot_nu_frac:
                _x = np.array([nux1 - nux0_int, nux2 - nux0_int])
                _y = np.array([nuy1 - nuy0_int, nuy2 - nuy0_int])
            else:
                _x = np.array([nux1, nux2])
                _y = np.array([nuy1, nuy2])
            ax.plot(_x, _y, label=rd.getResonanceCoeffLabelString(nx, ny), **prop)
            #print(n, nx, ny, _x, _y, prop)
    #leg = plt.legend(loc='best')
    #leg.set_draggable(True, use_blit=True)
    #
    cb = plt.colorbar(sc_obj, ax=ax)
    cb.ax.set_title(r'$\delta\, [\%]$', size=16)
    if title != '':
        ax.set_title(title, size=font_sz)
    plt.sca(ax)
    plt.tight_layout()



def calc_tswa_x(
    output_filepath, LTE_filepath, E_MeV, abs_xmax, nx, xsign='+',
    courant_snyder=True, return_fft_spec=True, n_turns=256, y0_offset=1e-5,
    use_beamline=None, N_KICKS=None, transmute_elements=None, ele_filepath=None,
    output_file_type=None, del_tmp_files=True, print_cmd=False,
    run_local=True, remote_opts=None):
    """"""

    if xsign == '+':
        x0_array = np.linspace(0.0, abs_xmax, nx)[1:] # exclude x == 0.0
    elif xsign == '-':
        x0_array = np.linspace(0.0, abs_xmax * (-1), nx)[1:] # exclude x == 0.0
    else:
        raise ValueError('`xsign` must be either "+" or "-".')
    y0_array = np.full(x0_array.shape, y0_offset)

    plane_specific_input = dict(abs_xmax=abs_xmax, nx=nx, xsign=xsign, y0_offset=y0_offset)

    return _calc_tswa(
        'x', plane_specific_input, output_filepath, LTE_filepath, E_MeV,
        x0_array, y0_array, courant_snyder=courant_snyder,
        return_fft_spec=return_fft_spec, n_turns=n_turns, use_beamline=use_beamline,
        N_KICKS=N_KICKS, transmute_elements=transmute_elements,
        ele_filepath=ele_filepath, output_file_type=output_file_type,
        del_tmp_files=del_tmp_files, print_cmd=print_cmd,
        run_local=run_local, remote_opts=remote_opts)

def calc_tswa_y(
    output_filepath, LTE_filepath, E_MeV, abs_ymax, ny, ysign='+',
    courant_snyder=True, return_fft_spec=True, n_turns=256, x0_offset=1e-5,
    use_beamline=None, N_KICKS=None, transmute_elements=None, ele_filepath=None,
    output_file_type=None, del_tmp_files=True, print_cmd=False,
    run_local=True, remote_opts=None):
    """"""

    if ysign == '+':
        y0_array = np.linspace(0.0, abs_ymax, ny)[1:] # exclude y == 0.0
    elif ysign == '-':
        y0_array = np.linspace(0.0, abs_ymax * (-1), ny)[1:] # exclude y == 0.0
    else:
        raise ValueError('`ysign` must be either "+" or "-".')
    x0_array = np.full(y0_array.shape, x0_offset)

    plane_specific_input = dict(abs_ymax=abs_ymax, ny=ny, ysign=ysign, x0_offset=x0_offset)

    return _calc_tswa(
        'y', plane_specific_input, output_filepath, LTE_filepath, E_MeV,
        x0_array, y0_array, courant_snyder=courant_snyder,
        return_fft_spec=return_fft_spec, n_turns=n_turns, use_beamline=use_beamline,
        N_KICKS=N_KICKS, transmute_elements=transmute_elements,
        ele_filepath=ele_filepath, output_file_type=output_file_type,
        del_tmp_files=del_tmp_files, print_cmd=print_cmd,
        run_local=run_local, remote_opts=remote_opts)

def _calc_tswa(
    scan_plane, plane_specific_input, output_filepath, LTE_filepath, E_MeV,
    x0_array, y0_array, courant_snyder=True, return_fft_spec=True,
    n_turns=256, use_beamline=None, N_KICKS=None,
    transmute_elements=None, ele_filepath=None, output_file_type=None,
    del_tmp_files=True, print_cmd=False,
    run_local=True, remote_opts=None):
    """"""

    assert x0_array.size == y0_array.size
    nscan = x0_array.size

    LTE_file_pathobj = Path(LTE_filepath)

    file_contents = LTE_file_pathobj.read_text()

    input_dict = dict(
        scan_plane=scan_plane, plane_specific_input=plane_specific_input,
        LTE_filepath=str(LTE_file_pathobj.resolve()), E_MeV=E_MeV,
        x0_array=x0_array, y0_array=y0_array, courant_snyder=courant_snyder,
        return_fft_spec=return_fft_spec, n_turns=n_turns,
        use_beamline=use_beamline, N_KICKS=N_KICKS, transmute_elements=transmute_elements,
        ele_filepath=ele_filepath, del_tmp_files=del_tmp_files,
        run_local=run_local, remote_opts=remote_opts,
        lattice_file_contents=file_contents,
        timestamp_ini=util.get_current_local_time_str(),
    )

    output_file_type = util.auto_check_output_file_type(output_filepath, output_file_type)
    input_dict['output_file_type'] = output_file_type

    if output_file_type in ('hdf5', 'h5'):
        util.save_input_to_hdf5(output_filepath, input_dict)

    if ele_filepath is None:
        tmp = tempfile.NamedTemporaryFile(
            dir=Path.cwd(), delete=False, prefix=f'tmpTSwA_', suffix='.ele')
        ele_pathobj = Path(tmp.name)
        ele_filepath = str(ele_pathobj.resolve())
        tmp.close()

    watch_pathobj = ele_pathobj.with_suffix('.wc')
    twi_pgz_pathobj = ele_pathobj.with_suffix('.twi.pgz')

    ed = elebuilder.EleDesigner(ele_filepath, double_format='.12g')

    elebuilder.add_transmute_blocks(ed, transmute_elements)

    ed.add_newline()

    ed.add_block('run_setup',
        lattice=LTE_filepath, p_central_mev=E_MeV, use_beamline=use_beamline,
    )

    ed.add_newline()

    temp_watch_elem_name = 'ELEGANT_TSWA_WATCH'
    if run_local:
        watch_filepath = str(watch_pathobj.resolve())
    else:
        watch_filepath = watch_pathobj.name
    temp_watch_elem_def = (
        f'{temp_watch_elem_name}: WATCH, FILENAME="{watch_filepath}", '
        'MODE=coordinate')

    ed.add_block('insert_elements',
        name='*', exclude='*', add_at_start=True, element_def=temp_watch_elem_def
    )

    ed.add_newline()

    elebuilder.add_N_KICKS_alter_elements_blocks(ed, N_KICKS)

    ed.add_newline()

    ed.add_block('run_control', n_passes=n_turns)

    ed.add_newline()

    centroid = {}
    centroid[0] = '<x0>'
    centroid[2] = '<y0>'
    centroid[5] = 0.0
    #
    ed.add_block(
        'bunched_beam', n_particles_per_bunch=1, centroid=centroid)

    ed.add_newline()

    ed.add_block('track')

    ed.write()
    #print(ed.actual_output_filepath_list)

    twiss.calc_ring_twiss(
        str(twi_pgz_pathobj), LTE_filepath, E_MeV, use_beamline=use_beamline,
        parameters=None, run_local=True, del_tmp_files=True)
    _d = util.load_pgz_file(str(twi_pgz_pathobj))
    nux0 = _d['data']['twi']['scalars']['nux']
    nuy0 = _d['data']['twi']['scalars']['nuy']
    # alpha & beta at watch element (at the start of the lattice)
    betax = _d['data']['twi']['arrays']['betax'][0]
    betay = _d['data']['twi']['arrays']['betay'][0]
    alphax = _d['data']['twi']['arrays']['alphax'][0]
    alphay = _d['data']['twi']['arrays']['alphay'][0]
    twi_pgz_pathobj.unlink()

    # Run Elegant
    if run_local:
        tbt = dict(x = np.full((n_turns, nscan), np.nan),
                   y = np.full((n_turns, nscan), np.nan))
        if courant_snyder:
            tbt['xp'] = np.full((n_turns, nscan), np.nan)
            tbt['yp'] = np.full((n_turns, nscan), np.nan)

        #tElapsed = dict(run_ele=0.0, sdds2dicts=0.0, tbt_population=0.0)

        for i, (x0, y0) in enumerate(zip(x0_array, y0_array)):
            #t0 = time.time()
            run(ele_filepath, print_cmd=print_cmd,
                macros=dict(x0=f'{x0:.12g}', y0=f'{y0:.12g}'),
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

        if remote_opts is None:
            remote_opts = dict(ntasks=20)
            #remote_opts = dict(
                #use_sbatch=True, pelegant=False, job_name='chrom',
                #output='chrom.%J.out', error='chrom.%J.err',
                #partition='short', ntasks=50)

        xy0_array = np.vstack((x0_array, y0_array)).T

        xy0_sub_array_list, reverse_mapping = util.chunk_list(
            xy0_array, remote_opts['ntasks'])

        coords_list = ['x', 'y']
        if courant_snyder:
            coords_list += ['xp', 'yp']

        module_name = 'pyelegant.nonlin'
        func_name = '_calc_tswa_get_tbt'
        chunked_results = remote.run_mpi_python(
            remote_opts, module_name, func_name, xy0_sub_array_list,
            (ele_pathobj.read_text(), ele_pathobj.name, watch_pathobj.name,
             print_cmd, std_print_enabled['out'], std_print_enabled['err'],
             coords_list),
        )

        tbt_chunked_list = dict()
        tbt_flat_list = dict()
        for plane in coords_list:
            tbt_chunked_list[plane] = [_d[plane] for _d in chunked_results]
            tbt_flat_list[plane] = util.unchunk_list_of_lists(
                tbt_chunked_list[plane], reverse_mapping)

        tbt = dict(x = np.full((n_turns, nscan), np.nan),
                   y = np.full((n_turns, nscan), np.nan))
        if courant_snyder:
            tbt['xp'] = np.full((n_turns, nscan), np.nan)
            tbt['yp'] = np.full((n_turns, nscan), np.nan)
        for plane in coords_list:
            for iXY, array in enumerate(tbt_flat_list[plane]):
                tbt[plane][:len(array), iXY] = array

    #print(tElapsed)

    #t0 = time.time()
    # Estimate tunes and amplitudes from TbT data
    if courant_snyder:
        if return_fft_spec:
            nus, As, fft_nus, fft_hAxs, fft_hAys = calc_tswa_from_tbt_cs(
                scan_plane, x0_array, y0_array, tbt['x'], tbt['y'], nux0, nuy0,
                tbt['xp'], tbt['yp'], betax, alphax, betay, alphay,
                init_guess_from_prev_step=True, return_fft_spec=True)
            extra_save_kwargs = dict(
                xptbt=tbt['xp'], yptbt=tbt['yp'],
                fft_nus=fft_nus, fft_hAxs=fft_hAxs, fft_hAys=fft_hAys,
            )
        else:
            nus, As = calc_tswa_from_tbt_cs(
                scan_plane, x0_array, y0_array, tbt['x'], tbt['y'], nux0, nuy0,
                tbt['xp'], tbt['yp'], betax, alphax, betay, alphay,
                init_guess_from_prev_step=True, return_fft_spec=False)
            extra_save_kwargs = dict(xptbt=tbt['xp'], yptbt=tbt['yp'])
    else:
        nus, As = calc_tswa_from_tbt_ps(
            scan_plane, x0_array, y0_array, tbt['x'], tbt['y'], nux0, nuy0)
        extra_save_kwargs = {}
    nuxs, nuys = nus['x'], nus['y']
    Axs, Ays = As['x'], As['y']
    time_domain_Axs = np.std(tbt['x'], axis=0, ddof=1) * np.sqrt(2)
    time_domain_Ays = np.std(tbt['y'], axis=0, ddof=1) * np.sqrt(2)
    #print('* Time elapsed for tune/amplitude estimation: {:.3f}'.format(time.time() - t0))

    timestamp_fin = util.get_current_local_time_str()

    _save_tswa_data(
        output_filepath, output_file_type, x0_array, y0_array, tbt['x'], tbt['y'],
        betax, alphax, betay, alphay, nux0, nuy0, nuxs, nuys, Axs, Ays,
        time_domain_Axs, time_domain_Ays, timestamp_fin, input_dict,
        **extra_save_kwargs)

    if del_tmp_files:
        util.delete_temp_files(
            ed.actual_output_filepath_list + [ele_filepath, str(watch_pathobj)])

    return output_filepath

def _calc_tswa_get_tbt(
    xy0_sub_array_list, ele_contents, ele_filename, watch_filename,
    print_cmd, print_stdout, print_stderr, coords_list, tempdir_path='/tmp'):
    """"""

    if not Path(tempdir_path).exists():
        tempdir_path = Path.cwd()

    sub_tbt = dict()
    for k in coords_list:
        sub_tbt[k] = []

    with tempfile.TemporaryDirectory(
        prefix='tmpCalcTSwA_', dir=tempdir_path) as tmpdirname:

        ele_pathobj = Path(tmpdirname).joinpath(ele_filename)
        watch_pathobj = Path(tmpdirname).joinpath(watch_filename)

        ele_contents = ele_contents.replace(
            watch_filename, str(watch_pathobj.resolve()))

        ele_pathobj.write_text(ele_contents)

        ele_filepath = str(ele_pathobj.resolve())

        for x0, y0 in xy0_sub_array_list:

            run(ele_filepath, print_cmd=print_cmd,
                macros=dict(x0=f'{x0:.12g}', y0=f'{y0:.12g}'),
                print_stdout=print_stdout, print_stderr=print_stderr)

            output, _ = sdds.sdds2dicts(watch_pathobj)

            cols = output['columns']
            for k in list(sub_tbt):
                sub_tbt[k].append(cols[k])

    return sub_tbt

def calc_tswa_from_tbt_ps(scan_plane, x0_array, y0_array, xtbt, ytbt, nux0, nuy0):
    """
    Using phase-space (ps) variables "x" and "y", which can only
    determine tunes within the range of [0, 0.5].
    """

    assert x0_array.shape == y0_array.shape

    nus = dict(x=np.full(x0_array.shape, np.nan),
               y=np.full(x0_array.shape, np.nan))
    As = dict(x=np.full(x0_array.shape, np.nan),
              y=np.full(x0_array.shape, np.nan))

    frac_nux0 = nux0 - np.floor(nux0)
    frac_nuy0 = nuy0 - np.floor(nuy0)

    n_turns, nscans = xtbt.shape
    nu_vec = np.fft.fftfreq(n_turns)

    opts = dict(window='sine', resolution=1e-8)
    init_nux = frac_nux0
    init_nuy = frac_nuy0
    for i in range(nscans):
        xarray = xtbt[:, i]
        yarray = ytbt[:, i]

        if np.any(np.isnan(xarray)) or np.any(np.isnan(yarray)):
            # Particle lost at some point.
            continue

        if False:
            # This algorithm does NOT work too well if tune change
            # between neighboring delta points are too large.
            out = sigproc.getDftPeak(xarray, init_nux, **opts)
            nus['x'][i] = out['nu']
            As['x'][i] = out['A']
            init_nux = out['nu']

            out = sigproc.getDftPeak(yarray, init_nuy, **opts)
            nus['y'][i] = out['nu']
            As['y'][i] = out['A']
            init_nuy = out['nu']
        else:
            # Find the rough peak first
            ff_rect = np.fft.fft(xarray - np.mean(xarray))
            A_arb = np.abs(ff_rect)
            init_nux = nu_vec[np.argmax(A_arb[:(n_turns//2)])]
            # Then fine-tune
            out = sigproc.getDftPeak(xarray, init_nux, **opts)
            nus['x'][i] = out['nu']
            As['x'][i] = out['A']

            # Find the rough peak first
            ff_rect = np.fft.fft(yarray - np.mean(yarray))
            A_arb = np.abs(ff_rect)
            init_nuy = nu_vec[np.argmax(A_arb[:(n_turns//2)])]
            # Then fine-tune
            out = sigproc.getDftPeak(yarray, init_nuy, **opts)
            nus['y'][i] = out['nu']
            As['y'][i] = out['A']

    return nus, As

def calc_tswa_from_tbt_cs(
    scan_plane, x0_array, y0_array, xtbt, ytbt, nux0, nuy0,
    xptbt, yptbt, betax, alphax, betay, alphay, init_guess_from_prev_step=True,
    return_fft_spec=True):
    """
    Using Courant-Snyder (CS) coordinates "xhat" and "yhat", which enables
    determination of tunes within the range of [0, 1.0].

    If `init_guess_from_prev_step` is True (recommended), then the tune peak
    fine-tuning starts from the tune peak found from the previous amplitude to
    avoid any sudden tune jump along an increasing amplitude vector. If False,
    then a rough tune peak is found from a simple FFT spectrum, which can
    potentially lead the initial search point to another strong resonance peak,
    away from the fundamental tune peak.
    """

    assert x0_array.shape == y0_array.shape

    nus = dict(x=np.full(x0_array.shape, np.nan),
               y=np.full(x0_array.shape, np.nan))
    As = dict(x=np.full(x0_array.shape, np.nan),
              y=np.full(x0_array.shape, np.nan))

    frac_nux0 = nux0 - np.floor(nux0)
    frac_nuy0 = nuy0 - np.floor(nuy0)

    if frac_nux0 > 0.5:
        frac_nux0 = frac_nux0 - 1
    if frac_nuy0 > 0.5:
        frac_nuy0 = frac_nuy0 - 1

    n_turns, nscans = xtbt.shape
    nu_vec = np.fft.fftfreq(n_turns)

    opts = dict(window='sine', resolution=1e-8, return_fft_spec=return_fft_spec)
    init_nux = frac_nux0
    init_nuy = frac_nuy0
    if return_fft_spec:
        fft_nus = None
        fft_hAxs = []
        fft_hAys = []
    for i in range(nscans):
        xarray = xtbt[:, i]
        yarray = ytbt[:, i]

        if np.any(np.isnan(xarray)) or np.any(np.isnan(yarray)):
            # Particle lost at some point.
            continue

        xparray = xptbt[:, i]
        yparray = yptbt[:, i]
        delta = 0.0 # Only on-momentum case is assumed
        pxarray, pyarray = angle2p(xparray, yparray, delta)

        xarray -= np.mean(xarray)
        yarray -= np.mean(yarray)
        pxarray -= np.mean(pxarray)
        pyarray -= np.mean(pyarray)

        xhat = xarray / np.sqrt(betax)
        pxhat = alphax / np.sqrt(betax) * xarray + np.sqrt(betax) * pxarray
        yhat = yarray / np.sqrt(betay)
        pyhat = alphay / np.sqrt(betay) * yarray + np.sqrt(betay) * pyarray

        hx = xhat - 1j * pxhat
        hy = yhat - 1j * pyhat

        if init_guess_from_prev_step:
            out = sigproc.getDftPeak(hx, init_nux, **opts)
            nus['x'][i] = out['nu']
            init_nux = out['nu']
        else:
            # Find the rough peak first
            ff_rect = np.fft.fft(hx)
            A_arb = np.abs(ff_rect)
            init_nux = nu_vec[np.argmax(A_arb)]
            # Then fine-tune
            out = sigproc.getDftPeak(hx, init_nux, **opts)
            nus['x'][i] = out['nu']
        sqrt_twoJx = out['A']
        As['x'][i] = sqrt_twoJx * np.sqrt(betax) # Convert CS amplitude to phase-space amplitude
        if return_fft_spec:
            if fft_nus is None:
                fft_nus = out['fft_nus']
            fft_hAxs.append(out['fft_As'])

        if init_guess_from_prev_step:
            out = sigproc.getDftPeak(hy, init_nuy, **opts)
            nus['y'][i] = out['nu']
            init_nuy = out['nu']
        else:
            # Find the rough peak first
            ff_rect = np.fft.fft(hy)
            A_arb = np.abs(ff_rect)
            init_nuy = nu_vec[np.argmax(A_arb)]
            # Then fine-tune
            out = sigproc.getDftPeak(hy, init_nuy, **opts)
            nus['y'][i] = out['nu']
        sqrt_twoJy = out['A']
        As['y'][i] = sqrt_twoJy * np.sqrt(betay) # Convert CS amplitude to phase-space amplitude
        if return_fft_spec:
            fft_hAys.append(out['fft_As'])

    nonnan_inds = np.where(~np.isnan(nus['x']))[0]
    neg_inds = nonnan_inds[nus['x'][nonnan_inds] < 0]
    nus['x'][neg_inds] += 1
    nonnan_inds = np.where(~np.isnan(nus['y']))[0]
    neg_inds = nonnan_inds[nus['y'][nonnan_inds] < 0]
    nus['y'][neg_inds] += 1

    if not return_fft_spec:
        return nus, As
    else:
        return nus, As, fft_nus, np.array(fft_hAxs).T, np.array(fft_hAys).T

def _save_tswa_data(
    output_filepath, output_file_type, x0_array, y0_array, xtbt, ytbt,
    betax, alphax, betay, alphay, nux0, nuy0, nuxs, nuys, Axs, Ays,
    time_domain_Axs, time_domain_Ays, timestamp_fin, input_dict,
    xptbt=None, yptbt=None, fft_nus=None, fft_hAxs=None, fft_hAys=None):
    """
    """

    if output_file_type in ('hdf5', 'h5'):
        _kwargs = dict(compression='gzip')
        f = h5py.File(output_filepath)
        f.create_dataset('x0s', data=x0_array, **_kwargs)
        f.create_dataset('y0s', data=y0_array, **_kwargs)
        f.create_dataset('xtbt', data=xtbt, **_kwargs)
        f.create_dataset('ytbt', data=ytbt, **_kwargs)
        f['nux0'] = nux0
        f['nuy0'] = nuy0
        if xptbt is not None:
            f.create_dataset('xptbt', data=xptbt, **_kwargs)
        if yptbt is not None:
            f.create_dataset('yptbt', data=xptbt, **_kwargs)
        if fft_nus is not None:
            f.create_dataset('fft_nus', data=fft_nus, **_kwargs)
            f.create_dataset('fft_hAxs', data=fft_hAxs, **_kwargs)
            f.create_dataset('fft_hAys', data=fft_hAys, **_kwargs)
        f['betax'] = betax
        f['betay'] = betay
        f['alphax'] = alphax
        f['alphay'] = alphay
        f.create_dataset('nuxs', data=nuxs, **_kwargs)
        f.create_dataset('nuys', data=nuys, **_kwargs)
        f.create_dataset('Axs', data=Axs, **_kwargs)
        f.create_dataset('Ays', data=Ays, **_kwargs)
        f.create_dataset('time_domain_Axs', data=time_domain_Axs, **_kwargs)
        f.create_dataset('time_domain_Ays', data=time_domain_Ays, **_kwargs)
        f['timestamp_fin'] = timestamp_fin
        f.close()

    elif output_file_type == 'pgz':
        d = dict(
            x0s=x0_array, y0s=y0_array, xtbt=xtbt, ytbt=ytbt,
            nux0=nux0, nuy0=nuy0,
            betax=betax, betay=betay, alphax=alphax, alphay=alphay,
            nuxs=nuxs, nuys=nuys, Axs=Axs, Ays=Ays,
            time_domain_Axs=time_domain_Axs, time_domain_Ays=time_domain_Ays,
            input=input_dict, timestamp_fin=timestamp_fin)
        if xptbt is not None:
            d['xptbt'] = xptbt
        if yptbt is not None:
            d['yptbt'] = yptbt
        if fft_nus is not None:
            d['fft_nus'] = fft_nus
            d['fft_hAxs'] = fft_hAxs
            d['fft_hAys'] = fft_hAys
        d['betax'] = betax
        d['betay'] = betay
        d['alphax'] = alphax
        d['alphay'] = alphay
        util.robust_pgz_file_write(output_filepath, d, nMaxTry=10, sleep=10.0)
    else:
        raise ValueError()

def plot_tswa(
    output_filepath, title='', fit_abs_xmax=None, fit_abs_ymax=None,
    plot_xy0=True, x0lim=None, y0lim=None,
    plot_Axy=False, use_time_domain_amplitude=True, Axlim=None, Aylim=None,
    nuxlim=None, nuylim=None, max_resonance_line_order=5,
    ax_nu_vs_xy0=None, ax_nu_vs_A=None, ax_nuy_vs_nux=None,
    plot_fft=False, ax_fft_hx=None, ax_fft_hy=None):
    """"""

    assert max_resonance_line_order <= 5

    is_nuxlim_frac = False
    if nuxlim:
        if (0.0 <= nuxlim[0] <= 1.0) and (0.0 <= nuxlim[1] <= 1.0):
            is_nuxlim_frac = True

    is_nuylim_frac = False
    if nuylim:
        if (0.0 <= nuylim[0] <= 1.0) and (0.0 <= nuylim[1] <= 1.0):
            is_nuylim_frac = True

    if is_nuxlim_frac and is_nuylim_frac:
        _plot_nu_frac = True
    elif (not is_nuxlim_frac) and (not is_nuylim_frac):
        _plot_nu_frac = False
    else:
        raise ValueError('"nuxlim" and "nuylim" must be either both fractional or both non-fractional')

    try:
        d = util.load_pgz_file(output_filepath)
        scan_plane = d['input']['scan_plane']
        x0s, y0s = d['x0s'], d['y0s']
        nuxs, nuys = d['nuxs'], d['nuys']
        Axs, Ays = d['Axs'], d['Ays']
        time_domain_Axs, time_domain_Ays = d['time_domain_Axs'], d['time_domain_Ays']
        nux0, nuy0 = d['nux0'], d['nuy0']
        betax, betay = d['betax'], d['betay']
        #alphax, alphay = d['alphax'], d['alphay']
        if 'fft_nus' in d:
            fft_d = {k: d[k] for k in ['fft_nus', 'fft_hAxs', 'fft_hAys']}
        else:
            fft_d = None
    except:
        f = h5py.File(output_filepath, 'r')
        scan_plane = f['input']['scan_plane'][()]
        x0s = f['x0s'][()]
        y0s = f['y0s'][()]
        nuxs = f['nuxs'][()]
        nuys = f['nuys'][()]
        Axs = f['Axs'][()]
        Ays = f['Ays'][()]
        time_domain_Axs = f['time_domain_Axs'][()]
        time_domain_Ays = f['time_domain_Ays'][()]
        nux0 = f['nux0'][()]
        nuy0 = f['nuy0'][()]
        betax = f['betax'][()]
        betay = f['betay'][()]
        #alphax = f['alphax'][()]
        #alphay = f['alphay'][()]
        if 'fft_nus' in f:
            fft_d = {k: f[k][()] for k in ['fft_nus', 'fft_hAxs', 'fft_hAys']}
        else:
            fft_d = None
        f.close()

    use_time_domain_amplitude = True
    if use_time_domain_amplitude:
        Axs = time_domain_Axs
        Ays = time_domain_Ays

    nux0_int = np.floor(nux0)
    nuy0_int = np.floor(nuy0)

    nuxs += nux0_int
    nuys += nuy0_int

    twoJxs = Axs**2 / betax
    twoJys = Ays**2 / betay
    Jxs = twoJxs / 2
    Jys = twoJys / 2

    twoJx0s = x0s**2 / betax
    twoJy0s = y0s**2 / betay
    Jx0s = twoJx0s / 2
    Jy0s = twoJy0s / 2

    beta_str = fr'\{{(\beta_x, \beta_y) [\mathrm{{m}}] = ({betax:.2f}, {betay:.2f})\}}'

    if scan_plane == 'x':
        if np.sign(x0s[-1]) > 0:
            fit_roi = (x0s <= fit_abs_xmax)
            scan_sign = +1.0
            scan_sign_str = ''
            scan_sign_beta_str = (
                '(x_0 > 0)$' '\n'
                fr'$(\beta_x, \beta_y) [\mathrm{{m}}] = ({betax:.2f}, {betay:.2f})')
        else:
            fit_roi = (x0s >= fit_abs_xmax * (-1))
            scan_sign = -1.0
            scan_sign_str = '-'
            scan_sign_beta_str = (
                '(x_0 < 0)$' '\n'
                fr'$(\beta_x, \beta_y) [\mathrm{{m}}] = ({betax:.2f}, {betay:.2f})')

        if plot_xy0:
            coeffs = np.polyfit(Jx0s[fit_roi], nuxs[fit_roi], 1)
            dnux_dJx0 = coeffs[0]
            nux_fit0 = np.poly1d(coeffs)

            coeffs = np.polyfit(Jx0s[fit_roi], nuys[fit_roi], 1)
            dnuy_dJx0 = coeffs[0]
            nuy_fit0 = np.poly1d(coeffs)

            if False: # Not being used and will generate warning about poor fit
                # because the elements of Jy0s are all the same
                dnux_dJy0 = np.polyfit(Jy0s[fit_roi], nuxs[fit_roi], 1)[0]
                dnuy_dJy0 = np.polyfit(Jy0s[fit_roi], nuys[fit_roi], 1)[0]

            x0_fit = np.linspace(np.min(np.abs(x0s)), np.max(np.abs(x0s)), 101)
            Jx0_fit = (x0_fit**2 / betax) / 2

            dnux_dJx0_str = util.pprint_sci_notation(dnux_dJx0, '+.3g')
            dnuy_dJx0_str = util.pprint_sci_notation(dnuy_dJx0, '+.3g')

            fit0_label = dict(
                nux=fr'$d\nu_x / d J_x = {dnux_dJx0_str}\, [\mathrm{{m}}^{{-1}}]$',
                nuy=fr'$d\nu_y / d J_x = {dnuy_dJx0_str}\, [\mathrm{{m}}^{{-1}}]$',
            )

        if plot_Axy:
            coeffs = np.polyfit(Jxs[fit_roi], nuxs[fit_roi], 1)
            dnux_dJx = coeffs[0]
            nux_fit = np.poly1d(coeffs)

            coeffs = np.polyfit(Jxs[fit_roi], nuys[fit_roi], 1)
            dnuy_dJx = coeffs[0]
            nuy_fit = np.poly1d(coeffs)

            if False: # Not being used
                dnux_dJy = np.polyfit(Jys[fit_roi], nuxs[fit_roi], 1)[0]
                dnuy_dJy = np.polyfit(Jys[fit_roi], nuys[fit_roi], 1)[0]

            Ax_fit = np.linspace(np.min(Axs), np.max(Axs), 101)
            Jx_fit = (Ax_fit**2 / betax) / 2

            dnux_dJx_str = util.pprint_sci_notation(dnux_dJx, '+.3g')
            dnuy_dJx_str = util.pprint_sci_notation(dnuy_dJx, '+.3g')

            fit_label = dict(
                nux=fr'$d\nu_x / d J_x = {dnux_dJx_str}\, [\mathrm{{m}}^{{-1}}]$',
                nuy=fr'$d\nu_y / d J_x = {dnuy_dJx_str}\, [\mathrm{{m}}^{{-1}}]$',
            )

    elif scan_plane == 'y':
        if np.sign(y0s[-1]) > 0:
            fit_roi = (y0s <= fit_abs_ymax)
            scan_sign = +1.0
            scan_sign_str = ''
            scan_sign_beta_str = (
                '(y_0 > 0)$' '\n'
                fr'$(\beta_x, \beta_y) [\mathrm{{m}}] = ({betax:.2f}, {betay:.2f})')
        else:
            fit_roi = (y0s >= fit_abs_ymax * (-1))
            scan_sign = -1.0
            scan_sign_str = '-'
            scan_sign_beta_str = (
                '(y_0 < 0)$' '\n'
                fr'$(\beta_x, \beta_y) [\mathrm{{m}}] = ({betax:.2f}, {betay:.2f})')

        if plot_xy0:
            coeffs = np.polyfit(Jy0s[fit_roi], nuxs[fit_roi], 1)
            dnux_dJy0 = coeffs[0]
            nux_fit0 = np.poly1d(coeffs)

            coeffs = np.polyfit(Jy0s[fit_roi], nuys[fit_roi], 1)
            dnuy_dJy0 = coeffs[0]
            nuy_fit0 = np.poly1d(coeffs)

            if False: # Not being used and will generate warning about poor fit
                # because the elements of Jy0s are all the same
                dnux_dJx0 = np.polyfit(Jx0s[fit_roi], nuxs[fit_roi], 1)[0]
                dnuy_dJx0 = np.polyfit(Jx0s[fit_roi], nuys[fit_roi], 1)[0]

            y0_fit = np.linspace(np.min(np.abs(y0s)), np.max(np.abs(y0s)), 101)
            Jy0_fit = (y0_fit**2 / betay) / 2

            dnux_dJy0_str = util.pprint_sci_notation(dnux_dJy0, '+.3g')
            dnuy_dJy0_str = util.pprint_sci_notation(dnuy_dJy0, '+.3g')

            fit0_label = dict(
                nux=fr'$d\nu_x / d J_y = {dnux_dJy0_str}\, [\mathrm{{m}}^{{-1}}]$',
                nuy=fr'$d\nu_y / d J_y = {dnuy_dJy0_str}\, [\mathrm{{m}}^{{-1}}]$',
            )

        if plot_Axy:
            coeffs = np.polyfit(Jys[fit_roi], nuxs[fit_roi], 1)
            dnux_dJy = coeffs[0]
            nux_fit = np.poly1d(coeffs)

            coeffs = np.polyfit(Jys[fit_roi], nuys[fit_roi], 1)
            dnuy_dJy = coeffs[0]
            nuy_fit = np.poly1d(coeffs)

            if False: # Not being used
                dnux_dJx = np.polyfit(Jxs[fit_roi], nuxs[fit_roi], 1)[0]
                dnuy_dJx = np.polyfit(Jxs[fit_roi], nuys[fit_roi], 1)[0]

            Ay_fit = np.linspace(np.min(Ays), np.max(Ays), 101)
            Jy_fit = (Ay_fit**2 / betay) / 2

            dnux_dJy_str = util.pprint_sci_notation(dnux_dJy, '+.3g')
            dnuy_dJy_str = util.pprint_sci_notation(dnuy_dJy, '+.3g')

            fit_label = dict(
                nux=fr'$d\nu_x / d J_y = {dnux_dJy_str}\, [\mathrm{{m}}^{{-1}}]$',
                nuy=fr'$d\nu_y / d J_y = {dnuy_dJy_str}\, [\mathrm{{m}}^{{-1}}]$',
            )

    else:
        raise ValueError


    if _plot_nu_frac:
        if plot_xy0:
            offset0 = dict(nux = np.floor(nuxs), nuy = np.floor(nuys))
            if scan_plane == 'x':
                offset0['fit_nux'] = np.floor(nux_fit0(Jx0_fit))
                offset0['fit_nuy'] = np.floor(nuy_fit0(Jx0_fit))
            elif scan_plane == 'y':
                offset0['fit_nux'] = np.floor(nux_fit0(Jy0_fit))
                offset0['fit_nuy'] = np.floor(nuy_fit0(Jy0_fit))
        if plot_Axy:
            offset = dict(nux = np.floor(nuxs), nuy = np.floor(nuys))
            if scan_plane == 'x':
                offset['fit_nux'] = np.floor(nux_fit(Jx_fit))
                offset['fit_nuy'] = np.floor(nuy_fit(Jx_fit))
            elif scan_plane == 'y':
                offset['fit_nux'] = np.floor(nux_fit(Jy_fit))
                offset['fit_nuy'] = np.floor(nuy_fit(Jy_fit))
    else:
        if plot_xy0:
            offset0 = dict(nux = np.zeros(nuxs.shape), nuy = np.zeros(nuys.shape))
            if scan_plane == 'x':
                offset0['fit_nux'] = offset0['fit_nuy'] = np.zeros(Jx0_fit.shape)
            elif scan_plane == 'y':
                offset0['fit_nux'] = offset0['fit_nuy'] = np.zeros(Jy0_fit.shape)
        if plot_Axy:
            offset = dict(nux = np.zeros(nuxs.shape), nuy = np.zeros(nuys.shape))
            if scan_plane == 'x':
                offset['fit_nux'] = offset['fit_nuy'] = np.zeros(Jx_fit.shape)
            elif scan_plane == 'y':
                offset['fit_nux'] = offset['fit_nuy'] = np.zeros(Jy_fit.shape)



    A_font_sz = 18
    font_sz = 22
    fit_x_line_style = 'b-'
    fit_y_line_style = 'r-'

    if plot_xy0:

        if ax_nu_vs_xy0:
            ax1 = ax_nu_vs_xy0
        else:
            fig, ax1 = plt.subplots()
        #
        if scan_plane == 'x':
            lines1 = ax1.plot(scan_sign * x0s * 1e3, nuxs - offset0['nux'], 'b.', label=r'$\nu_x$')
            fit_lines1 = ax1.plot(
                x0_fit * 1e3, nux_fit0(Jx0_fit) - offset0['fit_nux'],
                fit_x_line_style, label=fit0_label['nux'])
            ax2 = ax1.twinx()
            lines2 = ax2.plot(scan_sign * x0s * 1e3, nuys - offset0['nuy'], 'r.', label=r'$\nu_y$')
            fit_lines2 = ax2.plot(
                x0_fit * 1e3, nuy_fit0(Jx0_fit) - offset0['fit_nuy'],
                fit_y_line_style, label=fit0_label['nuy'])
            ax1.set_xlabel(fr'${scan_sign_str}x_0\, [\mathrm{{mm}}]\, {beta_str}$', size=A_font_sz)
            if x0lim is not None:
                ax1.set_xlim([v * 1e3 for v in x0lim])
        elif scan_plane == 'y':
            lines1 = ax1.plot(scan_sign * y0s * 1e3, nuxs - offset0['nux'], 'b.', label=r'$\nu_x$')
            fit_lines1 = ax1.plot(
                y0_fit * 1e3, nux_fit0(Jy0_fit) - offset0['fit_nux'],
                fit_x_line_style, label=fit0_label['nux'])
            ax2 = ax1.twinx()
            lines2 = ax2.plot(scan_sign * y0s * 1e3, nuys - offset0['nuy'], 'r.', label=r'$\nu_y$')
            fit_lines2 = ax2.plot(
                y0_fit * 1e3, nuy_fit0(Jy0_fit) - offset0['fit_nuy'],
                fit_y_line_style, label=fit0_label['nuy'])
            ax1.set_xlabel(fr'${scan_sign_str}y_0\, [\mathrm{{mm}}]\, {beta_str}$', size=A_font_sz)
            if y0lim is not None:
                ax1.set_xlim([v * 1e3 for v in y0lim])
        ax1.set_ylabel(r'$\nu_x$', size=font_sz, color='b')
        ax2.set_ylabel(r'$\nu_y$', size=font_sz, color='r')
        if nuxlim is not None:
            ax1.set_ylim(nuxlim)
        else:
            nuxlim = np.array(ax1.get_ylim())
            # The fitted lines for nux & nuy will often overlap each other
            # with default ylim. So, here, nux range is slided up.
            nuxlim[0] -= (nuxlim[1] - nuxlim[0]) * 0.1
            ax1.set_ylim(nuxlim)
        if nuylim is not None:
            ax2.set_ylim(nuylim)
        else:
            nuylim = np.array(ax2.get_ylim())
            # The fitted lines for nux & nuy will often overlap each other
            # with default ylim. So, here, nuy range is slided down.
            nuylim[1] += (nuylim[1] - nuylim[0]) * 0.1
            ax2.set_ylim(nuylim)
        if title != '':
            ax1.set_title(title, size=font_sz, pad=60)
        combined_lines = fit_lines1 + fit_lines2
        leg = ax2.legend(combined_lines, [L.get_label() for L in combined_lines],
                         loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.2),
                         fancybox=True, shadow=True, prop=dict(size=12))
        plt.sca(ax1)
        plt.tight_layout()

    if plot_Axy:

        if ax_nu_vs_A:
            ax1 = ax_nu_vs_A
        else:
            fig, ax1 = plt.subplots()
        #
        if scan_plane == 'x':
            lines1 = ax1.plot(Axs * 1e3, nuxs - offset['nux'], 'b.', label=r'$\nu_x$')
            fit_lines1 = ax1.plot(
                Ax_fit * 1e3, nux_fit(Jx_fit) - offset['fit_nux'],
                fit_x_line_style, label=fit_label['nux'])
            ax2 = ax1.twinx()
            lines2 = ax2.plot(Axs * 1e3, nuys - offset['nuy'], 'r.', label=r'$\nu_y$')
            fit_lines2 = ax2.plot(
                Ax_fit * 1e3, nuy_fit(Jx_fit) - offset['fit_nuy'],
                fit_y_line_style, label=fit_label['nuy'])
            ax1.set_xlabel(fr'$A_x\, [\mathrm{{mm}}]\, {scan_sign_beta_str}$', size=A_font_sz)
            if Axlim is not None:
                ax1.set_xlim([v * 1e3 for v in Axlim])
        elif scan_plane == 'y':
            lines1 = ax1.plot(Ays * 1e3, nuxs - offset['nux'], 'b.', label=r'$\nu_x$')
            fit_lines1 = ax1.plot(
                Ay_fit * 1e3, nux_fit(Jy_fit) - offset['fit_nux'],
                fit_x_line_style, label=fit_label['nux'])
            ax2 = ax1.twinx()
            lines2 = ax2.plot(Ays * 1e3, nuys - offset['nuy'], 'r.', label=r'$\nu_y$')
            fit_lines2 = ax2.plot(
                Ay_fit * 1e3, nuy_fit(Jy_fit) - offset['fit_nuy'],
                fit_y_line_style, label=fit_label['nuy'])
            ax1.set_xlabel(fr'$A_y\, [\mathrm{{mm}}]\, {scan_sign_beta_str}$', size=A_font_sz)
            if Aylim is not None:
                ax1.set_xlim([v * 1e3 for v in Aylim])
        ax1.set_ylabel(r'$\nu_x$', size=font_sz, color='b')
        ax2.set_ylabel(r'$\nu_y$', size=font_sz, color='r')
        if nuxlim is not None:
            ax1.set_ylim(nuxlim)
        else:
            nuxlim = np.array(ax1.get_ylim())
            # The fitted lines for nux & nuy will often overlap each other
            # with default ylim. So, here, nux range is slided up.
            nuxlim[0] -= (nuxlim[1] - nuxlim[0]) * 0.1
            ax1.set_ylim(nuxlim)
        if nuylim is not None:
            ax2.set_ylim(nuylim)
        else:
            nuylim = np.array(ax2.get_ylim())
            # The fitted lines for nux & nuy will often overlap each other
            # with default ylim. So, here, nuy range is slided down.
            nuylim[1] += (nuylim[1] - nuylim[0]) * 0.1
            ax2.set_ylim(nuylim)
        if title != '':
            ax1.set_title(title, size=font_sz, pad=60)
        combined_lines = fit_lines1 + fit_lines2
        leg = ax2.legend(combined_lines, [L.get_label() for L in combined_lines],
                         loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.2),
                         fancybox=True, shadow=True, prop=dict(size=12))
        plt.sca(ax1)
        plt.tight_layout()



    if scan_plane == 'x':
        As = Axs
    else:
        As = Ays

    if _plot_nu_frac:
        _nuxs = nuxs - nux0_int
        _nuys = nuys - nuy0_int

        frac_nuxlim = nuxlim
        frac_nuylim = nuylim
    else:
        _nuxs = nuxs
        _nuys = nuys

        frac_nuxlim = nuxlim - nux0_int
        frac_nuylim = nuylim - nuy0_int

    if ax_nuy_vs_nux:
        ax = ax_nuy_vs_nux
    else:
        _, ax = plt.subplots()
    sc_obj = ax.scatter(_nuxs, _nuys, s=10, c=As * 1e3, marker='o', cmap='jet')
    if nuxlim is not None:
        ax.set_xlim(nuxlim)
    if nuylim is not None:
        ax.set_ylim(nuylim)
    ax.set_xlabel(r'$\nu_x$', size=font_sz)
    ax.set_ylabel(r'$\nu_y$', size=font_sz)
    #
    rd = util.ResonanceDiagram()
    lineprops = dict(
        color    =['k',  'k', 'g', 'm', 'm'],
        linestyle=['-', '--', '-', '-', ':'],
        linewidth=[  2,    2, 0.5, 0.5, 0.5],
    )
    for n in range(1, max_resonance_line_order):
        d = rd.getResonanceCoeffsAndLines(
            n, np.array(frac_nuxlim) + nux0_int, frac_nuylim + nuy0_int) # <= CRITICAL: Must pass tunes w/ integer parts
        prop = {k: lineprops[k][n-1] for k in ['color', 'linestyle', 'linewidth']}
        assert len(d['lines']) == len(d['coeffs'])
        for ((nux1, nuy1), (nux2, nuy2)), (nx, ny, _) in zip(d['lines'], d['coeffs']):
            if _plot_nu_frac:
                _x = np.array([nux1 - nux0_int, nux2 - nux0_int])
                _y = np.array([nuy1 - nuy0_int, nuy2 - nuy0_int])
            else:
                _x = np.array([nux1, nux2])
                _y = np.array([nuy1, nuy2])
            ax.plot(_x, _y, label=rd.getResonanceCoeffLabelString(nx, ny), **prop)
    #leg = plt.legend(loc='best')
    #leg.set_draggable(True, use_blit=True)
    #
    cb = plt.colorbar(sc_obj, ax=ax)
    cb.ax.set_title(fr'$A_{scan_plane}\, [\mathrm{{mm}}]$', size=16)
    if title != '':
        ax.set_title(title, size=font_sz)
    plt.sca(ax)
    plt.tight_layout()


    if (fft_d is not None) and plot_fft:

        use_log = True

        font_sz = 18
        if use_log:
            EQ_STR = r'$\rm{log}_{10}(A/\mathrm{max}A)$'
        else:
            EQ_STR = r'$A/\mathrm{max}A$'

        if scan_plane == 'x':
            v1array = np.abs(x0s)
        else:
            v1array = np.abs(y0s)

        for _nu_plane in ['x', 'y']:

            v2array = fft_d['fft_nus'].copy()
            v2array[v2array < 0.0] += 1

            if _nu_plane == 'x':
                v2array += nux0_int

                if ax_fft_hx:
                    ax1 = ax_fft_hx
                else:
                    fig, ax1 = plt.subplots()

                norm_fft_hAs = fft_d['fft_hAxs'] / np.max(fft_d['fft_hAxs'], axis=0)

                ylim = nuxlim
            else:
                v2array += nuy0_int

                if ax_fft_hy:
                    ax1 = ax_fft_hy
                else:
                    fig, ax1 = plt.subplots()

                norm_fft_hAs = fft_d['fft_hAys'] / np.max(fft_d['fft_hAys'], axis=0)

                ylim = nuylim

            V1, V2 = np.meshgrid(v1array, v2array)

            if not use_log:
                plt.pcolor(V1 * 1e3, V2, norm_fft_hAs, cmap='jet')
            else:
                plt.pcolor(V1 * 1e3, V2, np.log10(norm_fft_hAs), cmap='jet')
            plt.xlabel(fr'${scan_sign_str}{scan_plane}\, [\mathrm{{mm}}]$', size=font_sz)
            plt.ylabel(fr'$\nu_{_nu_plane}$', size=font_sz)
            ax1.set_ylim(ylim)
            cb = plt.colorbar()
            cb.ax.set_title(EQ_STR)
            cb.ax.title.set_position((0.5, 1.02))
            plt.tight_layout()


def plot_tswa_both_sides(
    output_filepath_positive, output_filepath_negative, title='',
    fit_xmax=None, fit_xmin=None, fit_ymax=None, fit_ymin=None,
    Axlim=None, Aylim=None, nuxlim=None, nuylim=None, max_resonance_line_order=5,
    ax_nu_vs_A=None, ax_nuy_vs_nux=None):
    """"""

    assert max_resonance_line_order <= 5

    is_nuxlim_frac = False
    if nuxlim:
        if (0.0 <= nuxlim[0] <= 1.0) and (0.0 <= nuxlim[1] <= 1.0):
            is_nuxlim_frac = True

    is_nuylim_frac = False
    if nuylim:
        if (0.0 <= nuylim[0] <= 1.0) and (0.0 <= nuylim[1] <= 1.0):
            is_nuylim_frac = True

    if is_nuxlim_frac and is_nuylim_frac:
        _plot_nu_frac = True
    elif (not is_nuxlim_frac) and (not is_nuylim_frac):
        _plot_nu_frac = False
    else:
        raise ValueError('"nuxlim" and "nuylim" must be either both fractional or both non-fractional')

    scan_plane, x0s, y0s, nuxs, nuys, Axs, Ays, nux0, nuy0, betax, betay = [
        {} for _ in range(11)]

    for side, output_filepath in [('+', output_filepath_positive),
                                  ('-', output_filepath_negative)]:
        try:
            d = util.load_pgz_file(output_filepath)
            scan_plane[side] = d['input']['scan_plane']
            x0s[side], y0s[side] = d['x0s'], d['y0s']
            nuxs[side], nuys[side] = d['nuxs'], d['nuys']
            Axs[side], Ays[side] = d['Axs'], d['Ays']
            nux0[side], nuy0[side] = d['nux0'], d['nuy0']
            betax[side], betay[side] = d['betax'], d['betay']
        except:
            f = h5py.File(output_filepath, 'r')
            scan_plane[side] = f['input']['scan_plane'][()]
            x0s[side] = f['x0s'][()]
            y0s[side] = f['y0s'][()]
            nuxs[side] = f['nuxs'][()]
            nuys[side] = f['nuys'][()]
            Axs[side] = f['Axs'][()]
            Ays[side] = f['Ays'][()]
            nux0[side] = f['nux0'][()]
            nuy0[side] = f['nuy0'][()]
            betax[side] = f['betax'][()]
            betay[side] = f['betay'][()]
            f.close()

    # Check consistency between the positive and negative results (i.e.,
    # see if the results are likley from the same lattice)
    assert scan_plane['+'] == scan_plane['-']
    scan_plane = scan_plane['+']
    np.testing.assert_almost_equal(nux0['+'], nux0['-'], decimal=12)
    np.testing.assert_almost_equal(nuy0['+'], nuy0['-'], decimal=12)
    nux0, nuy0 = nux0['+'], nuy0['+']
    np.testing.assert_almost_equal(betax['+'], betax['-'], decimal=12)
    np.testing.assert_almost_equal(betay['+'], betay['-'], decimal=12)
    betax, betay = betax['+'], betay['+']

    if scan_plane == 'x':
        assert np.all(x0s['+'] >= 0.0)
        assert np.all(x0s['-'] <= 0.0)
    elif scan_plane == 'y':
        assert np.all(y0s['+'] >= 0.0)
        assert np.all(y0s['-'] <= 0.0)
    else:
        raise ValueError

    nux0_int = np.floor(nux0)
    nuy0_int = np.floor(nuy0)

    twoJxs, twoJys, Jxs, Jys, nux_fit, nuy_fit, fit_label = [{} for _ in range(7)]
    if scan_plane == 'x':
        Ax_fit, Jx_fit = {}, {}
    elif scan_plane == 'y':
        Ay_fit , Jy_fit = {}, {}

    beta_str = fr'(\beta_x, \beta_y) [\mathrm{{m}}] = ({betax:.2f}, {betay:.2f})'

    for side in ['+', '-']:
        nuxs[side] += nux0_int
        nuys[side] += nuy0_int

        twoJxs[side] = Axs[side]**2 / betax
        twoJys[side] = Ays[side]**2 / betay
        Jxs[side] = twoJxs[side] / 2
        Jys[side] = twoJys[side] / 2

        if scan_plane == 'x':
            if side == '+':
                fit_roi = (x0s[side] <= fit_xmax)
            else:
                fit_roi = (x0s[side] >= fit_xmin)

            coeffs = np.polyfit(Jxs[side][fit_roi], nuxs[side][fit_roi], 1)
            dnux_dJx = coeffs[0]
            nux_fit[side] = np.poly1d(coeffs)

            coeffs = np.polyfit(Jxs[side][fit_roi], nuys[side][fit_roi], 1)
            dnuy_dJx = coeffs[0]
            nuy_fit[side] = np.poly1d(coeffs)

            dnux_dJy = np.polyfit(Jys[side][fit_roi], nuxs[side][fit_roi], 1)[0]
            dnuy_dJy = np.polyfit(Jys[side][fit_roi], nuys[side][fit_roi], 1)[0]

            Ax_fit[side] = np.linspace(np.min(Axs[side]), np.max(Axs[side]), 101)
            Jx_fit[side] = (Ax_fit[side]**2 / betax) / 2

            dnux_dJx_str = util.pprint_sci_notation(dnux_dJx, '+.3g')
            dnuy_dJx_str = util.pprint_sci_notation(dnuy_dJx, '+.3g')

            fit_label[side] = dict(nux=dnux_dJx_str, nuy=dnuy_dJx_str,)

        elif scan_plane == 'y':
            if side == '+':
                fit_roi = (y0s[side] <= fit_ymax)
            else:
                fit_roi = (y0s[side] >= fit_ymin)

            coeffs = np.polyfit(Jys[side][fit_roi], nuxs[side][fit_roi], 1)
            dnux_dJy = coeffs[0]
            nux_fit[side] = np.poly1d(coeffs)

            coeffs = np.polyfit(Jys[side][fit_roi], nuys[side][fit_roi], 1)
            dnuy_dJy = coeffs[0]
            nuy_fit[side] = np.poly1d(coeffs)

            dnux_dJx = np.polyfit(Jxs[side][fit_roi], nuxs[side][fit_roi], 1)[0]
            dnuy_dJx = np.polyfit(Jxs[side][fit_roi], nuys[side][fit_roi], 1)[0]

            Ay_fit[side] = np.linspace(np.min(Ays[side]), np.max(Ays[side]), 101)
            Jy_fit[side] = (Ay_fit[side]**2 / betay) / 2

            dnux_dJy_str = util.pprint_sci_notation(dnux_dJy, '+.3g')
            dnuy_dJy_str = util.pprint_sci_notation(dnuy_dJy, '+.3g')

            fit_label[side] = dict(nux=dnux_dJy_str, nuy=dnuy_dJy_str,)

    if scan_plane == 'x':
        fit_label_combo = dict(
            nux=r'$d\nu_x / d J_x [\mathrm{{m}}^{{-1}}] = {} (-); {} (+)$'.format(
                fit_label['-']['nux'], fit_label['+']['nux']),
            nuy=r'$d\nu_y / d J_x [\mathrm{{m}}^{{-1}}] = {} (-); {} (+)$'.format(
                fit_label['-']['nuy'], fit_label['+']['nuy']),
        )
    elif scan_plane == 'y':
        fit_label_combo = dict(
            nux=r'$d\nu_x / d J_y [\mathrm{{m}}^{{-1}}] = {} (-); {} (+)$'.format(
                fit_label['-']['nux'], fit_label['+']['nux']),
            nuy=r'$d\nu_y / d J_y [\mathrm{{m}}^{{-1}}] = {} (-); {} (+)$'.format(
                fit_label['-']['nuy'], fit_label['+']['nuy']),
        )


    A_font_sz = 18
    font_sz = 22

    if ax_nu_vs_A:
        ax1 = ax_nu_vs_A
    else:
        fig, ax1 = plt.subplots()
    #
    nuxs_combo = np.append(nuxs['-'][::-1], nuxs['+'])
    nuys_combo = np.append(nuys['-'][::-1], nuys['+'])
    #
    if scan_plane == 'x':
        nux_fit_combo = np.append(nux_fit['-'](Jx_fit['-'])[::-1], nux_fit['+'](Jx_fit['+']))
        nuy_fit_combo = np.append(nuy_fit['-'](Jx_fit['-'])[::-1], nuy_fit['+'](Jx_fit['+']))
    elif scan_plane == 'y':
        nux_fit_combo = np.append(nux_fit['-'](Jy_fit['-'])[::-1], nux_fit['+'](Jy_fit['+']))
        nuy_fit_combo = np.append(nuy_fit['-'](Jy_fit['-'])[::-1], nuy_fit['+'](Jy_fit['+']))
    #
    if _plot_nu_frac:
        offset_combo = dict(nux = np.floor(nuxs_combo), nuy = np.floor(nuys_combo))
        if scan_plane == 'x':
            offset_combo['fit_nux'] = np.floor(nux_fit_combo)
            offset_combo['fit_nuy'] = np.floor(nuy_fit_combo)
        elif scan_plane == 'y':
            offset_combo['fit_nux'] = np.floor(nux_fit_combo)
            offset_combo['fit_nuy'] = np.floor(nuy_fit_combo)
    else:
        offset_combo = dict(nux = np.zeros(nuxs_combo.shape), nuy = np.zeros(nuys_combo.shape))
        if scan_plane == 'x':
            offset_combo['fit_nux'] = offset_combo['fit_nuy'] = np.zeros(
                np.append(Jx_fit['-'][::-1], Jx_fit['+']).shape)
        elif scan_plane == 'y':
            offset_combo['fit_nux'] = offset_combo['fit_nuy'] = np.zeros(
                np.append(Jy_fit['-'][::-1], Jy_fit['+']).shape)
    #
    if scan_plane == 'x':
        Axs_combo = np.append(Axs['-'][::-1] * (-1), Axs['+'])
        Ax_fit_combo = np.append(Ax_fit['-'][::-1] * (-1), Ax_fit['+'])
        lines1 = ax1.plot(Axs_combo * 1e3, nuxs_combo - offset_combo['nux'], 'b.',
                          label=r'$\nu_x$')
        fit_lines1 = ax1.plot(
            Ax_fit_combo * 1e3, nux_fit_combo - offset_combo['fit_nux'], 'b-',
            label=fit_label_combo['nux'])
        ax2 = ax1.twinx()
        lines2 = ax2.plot(Axs_combo * 1e3, nuys_combo - offset_combo['nuy'], 'r.',
                          label=r'$\nu_y$')
        fit_lines2 = ax2.plot(
            Ax_fit_combo * 1e3, nuy_fit_combo - offset_combo['fit_nuy'], 'r-',
            label=fit_label_combo['nuy'])
        ax1.set_xlabel(fr'$A_x\, [\mathrm{{mm}}];\, {beta_str}$', size=A_font_sz)
        if Axlim is not None:
            ax1.set_xlim([v * 1e3 for v in Axlim])
    elif scan_plane == 'y':
        Ays_combo = np.append(Ays['-'][::-1] * (-1), Ays['+'])
        Ay_fit_combo = np.append(Ay_fit['-'][::-1] * (-1), Ay_fit['+'])
        lines1 = ax1.plot(Ays_combo * 1e3, nuxs_combo - offset_combo['nux'], 'b.',
                          label=r'$\nu_x$')
        fit_lines1 = ax1.plot(
            Ay_fit_combo * 1e3, nux_fit_combo - offset_combo['fit_nux'], 'b-',
            label=fit_label_combo['nux'])
        ax2 = ax1.twinx()
        lines2 = ax2.plot(Ays_combo * 1e3, nuys_combo - offset_combo['nuy'], 'r.',
                          label=r'$\nu_y$')
        fit_lines2 = ax2.plot(
            Ay_fit_combo * 1e3, nuy_fit_combo - offset_combo['fit_nuy'], 'r-',
            label=fit_label_combo['nuy'])
        ax1.set_xlabel(fr'$A_y\, [\mathrm{{mm}}];\, {beta_str}$', size=A_font_sz)
        if Aylim is not None:
            ax1.set_xlim([v * 1e3 for v in Aylim])
    ax1.set_ylabel(r'$\nu_x$', size=font_sz, color='b')
    ax2.set_ylabel(r'$\nu_y$', size=font_sz, color='r')
    if nuxlim:
        ax1.set_ylim(nuxlim)
    else:
        nuxlim = ax1.get_ylim()
    if nuylim:
        ax2.set_ylim(nuylim)
    else:
        nuylim = ax2.get_ylim()
    if title != '':
        ax1.set_title(title, size=font_sz, pad=60)
    combined_lines = fit_lines1 + fit_lines2
    leg = ax2.legend(combined_lines, [L.get_label() for L in combined_lines],
                     loc='upper center', ncol=1, bbox_to_anchor=(0.5, 1.3),
                     fancybox=True, shadow=True, prop=dict(size=12))
    plt.sca(ax1)
    plt.tight_layout()


    if scan_plane == 'x':
        As = np.append(Axs['-'][::-1] * (-1), Axs['+'])
    else:
        As = np.append(Ays['-'][::-1] * (-1), Ays['+'])

    if _plot_nu_frac:
        _nuxs = nuxs_combo - nux0_int
        _nuys = nuys_combo - nuy0_int

        frac_nuxlim = nuxlim
        frac_nuylim = nuylim
    else:
        _nuxs = nuxs_combo
        _nuys = nuys_combo

        frac_nuxlim = nuxlim - nux0_int
        frac_nuylim = nuylim - nuy0_int

    if ax_nuy_vs_nux:
        ax = ax_nuy_vs_nux
    else:
        _, ax = plt.subplots()
    sc_obj = ax.scatter(_nuxs, _nuys, s=10, c=As * 1e3, marker='o', cmap='jet')
    if nuxlim:
        ax.set_xlim(nuxlim)
    if nuylim:
        ax.set_ylim(nuylim)
    ax.set_xlabel(r'$\nu_x$', size=font_sz)
    ax.set_ylabel(r'$\nu_y$', size=font_sz)
    #
    rd = util.ResonanceDiagram()
    lineprops = dict(
        color    =['k',  'k', 'g', 'm', 'm'],
        linestyle=['-', '--', '-', '-', ':'],
        linewidth=[  2,    2, 0.5, 0.5, 0.5],
    )
    for n in range(1, max_resonance_line_order):
        d = rd.getResonanceCoeffsAndLines(
            n, np.array(frac_nuxlim) + nux0_int, frac_nuylim + nuy0_int) # <= CRITICAL: Must pass tunes w/ integer parts
        prop = {k: lineprops[k][n-1] for k in ['color', 'linestyle', 'linewidth']}
        assert len(d['lines']) == len(d['coeffs'])
        for ((nux1, nuy1), (nux2, nuy2)), (nx, ny, _) in zip(d['lines'], d['coeffs']):
            if _plot_nu_frac:
                _x = np.array([nux1 - nux0_int, nux2 - nux0_int])
                _y = np.array([nuy1 - nuy0_int, nuy2 - nuy0_int])
            else:
                _x = np.array([nux1, nux2])
                _y = np.array([nuy1, nuy2])
            ax.plot(_x, _y, label=rd.getResonanceCoeffLabelString(nx, ny), **prop)
    #leg = plt.legend(loc='best')
    #leg.set_draggable(True, use_blit=True)
    #
    cb = plt.colorbar(sc_obj, ax=ax)
    cb.ax.set_title(fr'$A_{scan_plane}\, [\mathrm{{mm}}]$', size=16)
    if title != '':
        ax.set_title(title, size=font_sz)
    plt.sca(ax)
    plt.tight_layout()



def track(
    output_filepath, LTE_filepath, E_MeV, n_turns,
    x0=0.0, xp0=0.0, y0=0.0, yp0=0.0, delta0=0.0,
    output_coordinates=('x', 'xp', 'y', 'yp', 'delta'),
    use_beamline=None, N_KICKS=None, transmute_elements=None, ele_filepath=None,
    output_file_type=None, del_tmp_files=True, print_cmd=False,
    run_local=True, remote_opts=None):
    """"""

    LTE_file_pathobj = Path(LTE_filepath)

    file_contents = LTE_file_pathobj.read_text()

    input_dict = dict(
        LTE_filepath=str(LTE_file_pathobj.resolve()), E_MeV=E_MeV,
        n_turns=n_turns, x0=x0, xp0=xp0, y0=y0, yp0=yp0, delta0=delta0,
        output_coordinates=output_coordinates,
        use_beamline=use_beamline, N_KICKS=N_KICKS, transmute_elements=transmute_elements,
        ele_filepath=ele_filepath, del_tmp_files=del_tmp_files,
        run_local=run_local, remote_opts=remote_opts,
        lattice_file_contents=file_contents,
        timestamp_ini=util.get_current_local_time_str(),
    )

    output_file_type = util.auto_check_output_file_type(output_filepath, output_file_type)
    input_dict['output_file_type'] = output_file_type

    if output_file_type in ('hdf5', 'h5'):
        util.save_input_to_hdf5(output_filepath, input_dict)

    if ele_filepath is None:
        tmp = tempfile.NamedTemporaryFile(
            dir=Path.cwd(), delete=False, prefix=f'tmpTrack_', suffix='.ele')
        ele_pathobj = Path(tmp.name)
        ele_filepath = str(ele_pathobj.resolve())
        tmp.close()

    watch_pathobj = ele_pathobj.with_suffix('.wc')

    ed = elebuilder.EleDesigner(ele_filepath, double_format='.12g')

    elebuilder.add_transmute_blocks(ed, transmute_elements)

    ed.add_newline()

    ed.add_block('run_setup',
        lattice=LTE_filepath, p_central_mev=E_MeV, use_beamline=use_beamline,
    )

    ed.add_newline()

    temp_watch_elem_name = 'ELEGANT_TRACK_WATCH'
    if run_local:
        watch_filepath = str(watch_pathobj.resolve())
    else:
        watch_filepath = watch_pathobj.name
    temp_watch_elem_def = (
        f'{temp_watch_elem_name}: WATCH, FILENAME="{watch_filepath}", '
        'MODE=coordinate')

    ed.add_block('insert_elements',
        name='*', exclude='*', add_at_start=True, element_def=temp_watch_elem_def
    )

    nWatch = 1

    ed.add_newline()

    elebuilder.add_N_KICKS_alter_elements_blocks(ed, N_KICKS)

    ed.add_newline()

    ed.add_block('run_control', n_passes=n_turns)

    ed.add_newline()

    centroid = {}
    centroid[0] = x0
    centroid[1] = xp0
    centroid[2] = y0
    centroid[3] = yp0
    centroid[5] = delta0
    #
    ed.add_block(
        'bunched_beam', n_particles_per_bunch=1, centroid=centroid)

    ed.add_newline()

    ed.add_block('track')

    ed.write()
    #print(ed.actual_output_filepath_list)

    tbt = dict(x = np.full((n_turns, nWatch), np.nan),
               y = np.full((n_turns, nWatch), np.nan),
               xp = np.full((n_turns, nWatch), np.nan),
               yp = np.full((n_turns, nWatch), np.nan),
               delta = np.full((n_turns, nWatch), np.nan),
               t = np.full((n_turns, nWatch), np.nan),
               dt = np.full((n_turns, nWatch), np.nan),
               )

    # Run Elegant
    if run_local:
        run(ele_filepath, print_cmd=print_cmd,
            print_stdout=std_print_enabled['out'],
            print_stderr=std_print_enabled['err'])
    else:

        if remote_opts is None:
            remote_opts = dict(use_sbatch=False)

        if ('pelegant' in remote_opts) and (remote_opts['pelegant'] is not False):
            print('"pelegant" option in `remote_opts` must be False for nonlin.track()')
            remote_opts['pelegant'] = False
        else:
            remote_opts['pelegant'] = False

        remote_opts['ntasks'] = 1
        # ^ If this is more than 1, you will likely see an error like "Unable to
        #   access file /.../tmp*.twi--file is locked (SDDS_InitializeOutput)"

        remote.run(remote_opts, ele_filepath, print_cmd=print_cmd,
                   print_stdout=std_print_enabled['out'],
                   print_stderr=std_print_enabled['err'],
                   output_filepaths=None)
    #
    output, _ = sdds.sdds2dicts(watch_pathobj)
    #
    cols = output['columns']
    for k in list(tbt):
        if k == 'delta':
            _delta = cols['p'] / output['params']['pCentral'] - 1.0
            tbt[k][:len(cols['p']), :] = _delta.reshape((-1,1))
        else:
            tbt[k][:len(cols[k]), :] = cols[k].reshape((-1,1))

    timestamp_fin = util.get_current_local_time_str()

    if output_file_type in ('hdf5', 'h5'):
        _kwargs = dict(compression='gzip')
        f = h5py.File(output_filepath)
        for coord in output_coordinates:
            f.create_dataset(coord, data=tbt[coord], **_kwargs)
        f['timestamp_fin'] = timestamp_fin
        f.close()

    elif output_file_type == 'pgz':
        d = dict(input=input_dict, timestamp_fin=timestamp_fin)
        for coord in output_coordinates:
            d[coord] = tbt[coord]
        util.robust_pgz_file_write(output_filepath, d, nMaxTry=10, sleep=10.0)
    else:
        raise ValueError()

    if del_tmp_files:
        util.delete_temp_files(
            ed.actual_output_filepath_list + [ele_filepath, str(watch_pathobj)])

    return output_filepath

