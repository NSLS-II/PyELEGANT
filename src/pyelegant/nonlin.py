import os
import numpy as np
import matplotlib.pylab as plt
import tempfile
import h5py

from .local import run
from .remote import remote
from . import std_print_enabled
from . import elebuilder
from . import util
from . import sdds


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
        ele_filepath=ele_filepath, output_file_type=output_file_type,
        del_tmp_files=del_tmp_files, run_local=run_local,
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

    if output_file_type in ('hdf5', 'h5'):
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

    if ele_filepath is None:
        tmp = tempfile.NamedTemporaryFile(
            dir=os.getcwd(), delete=False, prefix=f'tmpFMA{plane}_', suffix='.ele')
        ele_filepath = os.path.abspath(tmp.name)
        tmp.close()

    ed = elebuilder.EleDesigner(double_format='.12g')

    default_transmute_elems = dict(
        SBEN='CSBEN', RBEN='CSBEN', QUAD='KQUAD', SEXT='KSEXT',
        RFCA='MARK', SREFFECTS='MARK')

    if transmute_elements is None:
        actual_transmute_elems = default_transmute_elems
    else:
        actual_transmute_elems = {}
        for old_type, new_type in default_transmute_elems.items():
            actual_transmute_elems[old_type] = new_type

    for old_type, new_type in actual_transmute_elems.items():
        ed.add_block('transmute_elements',
                     name='*', type=old_type, new_type=new_type)


    ed.add_block('run_setup',
        lattice=LTE_filepath, p_central_mev=E_MeV, use_beamline=use_beamline,
        semaphore_file='%s.done')

    ed.add_newline()

    ed.add_block('run_control', n_passes=n_turns)

    ed.add_newline()

    if N_KICKS is None:
        N_KICKS = dict(KQUAD=20, KSEXT=20, CSBEND=20)

    for k, v in N_KICKS.items():
        if k.upper() not in ('KQUAD', 'KSEXT', 'CSBEND'):
            raise ValueError(f'The key "{k}" in N_KICKS dict is invalid. '
                             f'Must be one of KQUAD, KSEXT, or CSBEND')
        ed.add_block('alter_elements',
                     name='*', type=k.upper(), item='N_KICKS', value=v)

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

    output_file_type = output_file_type.lower()

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

