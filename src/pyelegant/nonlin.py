import os
import numpy as np
import matplotlib.pylab as plt
import tempfile
import h5py

from .local import run
from .remote import remote
from . import elebuilder
from . import util
from . import sdds

def calc_fma_xy(
    output_filepath, LTE_filepath, E_MeV, xmin, xmax, ymin, ymax, nx, ny,
    n_turns=1024, delta_offset=0.0, quadratic_spacing=False, full_grid_output=False,
    use_beamline=None, N_KICKS=None, transmute_elements=None, ele_filepath=None,
    output_file_type=None, del_tmp_files=True,
    run_local=False, remote_opts=None, print_stdout=True, print_stderr=True):
    """"""

    with open(LTE_filepath, 'r') as f:
        file_contents = f.read()

    input_dict = dict(
        LTE_filepath=os.path.abspath(LTE_filepath), E_MeV=E_MeV,
        xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, nx=nx, ny=ny,
        n_turns=n_turns, delta_offset=delta_offset,
        quadratic_spacing=quadratic_spacing, use_beamline=use_beamline,
        N_KICKS=N_KICKS, transmute_elements=transmute_elements,
        ele_filepath=ele_filepath, output_file_type=output_file_type,
        del_tmp_files=del_tmp_files, run_local=run_local,
        remote_opts=remote_opts,
        lattice_file_contents=file_contents,
        timestamp_ini=util.get_current_local_time_str(),
    )

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
            dir=os.getcwd(), delete=False, prefix='tmpFMAxy_', suffix='.ele')
        ele_filepath = os.path.abspath(tmp.name)
        tmp.close()

    eb = elebuilder.EleContents(double_format='.12g')

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
        eb.transmute_elements(name='*', type=old_type, new_type=new_type)


    eb.run_setup(
        lattice=LTE_filepath, p_central_mev=E_MeV, use_beamline=use_beamline,
        semaphore_file='%s.done')

    eb.newline()

    eb.run_control(n_passes=n_turns)

    eb.newline()

    if N_KICKS is None:
        N_KICKS = dict(KQUAD=20, KSEXT=20, CSBEND=20)

    for k, v in N_KICKS.items():
        if k.upper() not in ('KQUAD', 'KSEXT', 'CSBEND'):
            raise ValueError(f'The key "{k}" in N_KICKS dict is invalid. '
                             f'Must be one of KQUAD, KSEXT, or CSBEND')
        eb.alter_elements(name='*', type=k.upper(), item='N_KICKS', value=v)

    eb.bunched_beam(n_particles_per_bunch=1)

    eb.newline()

    eb.frequency_map(
        output='%s.fma', xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
        delta_min=delta_offset, delta_max=delta_offset, nx=nx, ny=ny,
        ndelta=1, include_changes=True, quadratic_spacing=quadratic_spacing,
        full_grid_output=full_grid_output,
    )

    eb.write(ele_filepath)

    eb.update_output_filepaths(ele_filepath[:-4]) # Remove ".ele"
    #print(eb.actual_output_filepath_list)

    for fp in eb.actual_output_filepath_list:
        if fp.endswith('.fma'):
            fma_output_filepath = fp
        elif fp.endswith('.done'):
            done_filepath = fp
        else:
            raise ValueError('This line should not be reached.')

    # Run Elegant
    if run_local:
        run(ele_filepath, print_cmd=False,
            print_stdout=print_stdout, print_stderr=print_stderr)
    else:
        if remote_opts is None:
            remote_opts = dict(
                use_sbatch=False, pelegant=True, job_name='fma',
                output='fma.%J.out', error='fma.%J.err',
                partition='normal', ntasks=50)

        remote.run(remote_opts, ele_filepath, print_cmd=True,
                   print_stdout=print_stdout, print_stderr=print_stderr,
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
        for fp in eb.actual_output_filepath_list + [ele_filepath]:
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

    try:
        d = util.load_pgz_file(output_filepath)
        g = d['data']['fma']['arrays']
        x = g['x']
        y = g['y']
        if is_diffusion:
            diffusion = g['diffusion']
        else:
            diffusionRate = g['diffusionRate']
        if not scatter:
            g = d['input']
            xmax = g['xmax']
            xmin = g['xmin']
            ymax = g['ymax']
            ymin = g['ymin']
            nx = g['nx']
            ny = g['ny']
            quadratic_spacing = g['quadratic_spacing']
    except:
        f = h5py.File(output_filepath, 'r')
        g = f['fma']['arrays']
        x = g['x'][()]
        y = g['y'][()]
        if is_diffusion:
            diffusion = g['diffusion'][()]
        else:
            diffusionRate = g['diffusionRate'][()]
        if not scatter:
            g = f['input']
            xmax = g['xmax'][()]
            xmin = g['xmin'][()]
            ymax = g['ymax'][()]
            ymin = g['ymin'][()]
            nx = g['nx'][()]
            ny = g['ny'][()]
            quadratic_spacing = g['quadratic_spacing'][()]
        f.close()


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
        plt.scatter(x * 1e3, y * 1e3, s=14, c=values, cmap='jet', vmin=LB, vmax=UB)
        plt.xlabel(r'$x\, [\mathrm{mm}]$', size=font_sz)
        plt.ylabel(r'$y\, [\mathrm{mm}]$', size=font_sz)
        if xlim is not None:
            plt.xlim([v * 1e3 for v in xlim])
        if ylim is not None:
            plt.ylim([v * 1e3 for v in ylim])
        if title != '':
            plt.title(title, size=font_sz)
        cb = plt.colorbar()
        cb.set_ticks(range(LB,UB+1))
        cb.set_ticklabels([str(i) for i in range(LB,UB+1)])
        cb.ax.set_title(DIFFUSION_EQ_STR)
        cb.ax.title.set_position((0.5, 1.02))
        plt.tight_layout()

    else:

        font_sz = 18

        if not quadratic_spacing:
            x0array = np.linspace(xmin, xmax, nx)
            y0array = np.linspace(ymin, ymax, ny)
        else:

            dx = xmax - max([0.0, xmin])
            x0array = np.sqrt(np.linspace((dx**2) / nx, dx**2, nx))
            #x0array - np.unique(x)
            #plt.figure()
            #plt.plot(np.unique(x), 'b-', x0array, 'r-')

            dy = ymax - max([0.0, ymin])
            y0array = ymin + np.sqrt(np.linspace((dy**2) / ny, dy**2, ny))
            #y0array - np.unique(y)
            #plt.figure()
            #plt.plot(np.unique(y), 'b-', y0array, 'r-')

        X, Y = np.meshgrid(x0array, y0array)
        D = X * np.nan

        xinds = np.argmin(np.abs(
            x0array.reshape((-1,1)) @ np.ones((1,x.size)) - x), axis=0)
        yinds = np.argmin(np.abs(
            y0array.reshape((-1,1)) @ np.ones((1,y.size)) - y), axis=0)
        flatinds = np.ravel_multi_index((xinds, yinds), X.shape, order='F')
        D_flat = D.flatten()
        D_flat[flatinds] = values
        D = D_flat.reshape(D.shape)

        D = np.ma.masked_array(D, np.isnan(D))

        plt.figure()
        ax = plt.subplot(111)
        plt.pcolor(X*1e3, Y*1e3, D, cmap='jet', vmin=LB, vmax=UB)
        plt.xlabel(r'$x\, [\mathrm{mm}]$', size=font_sz)
        plt.ylabel(r'$y\, [\mathrm{mm}]$', size=font_sz)
        if xlim is not None:
            plt.xlim([v * 1e3 for v in xlim])
        if ylim is not None:
            plt.ylim([v * 1e3 for v in ylim])
        if title != '':
            plt.title(title, size=font_sz)
        cb = plt.colorbar()
        cb.set_ticks(range(LB,UB+1))
        cb.set_ticklabels([str(i) for i in range(LB,UB+1)])
        cb.ax.set_title(DIFFUSION_EQ_STR)
        cb.ax.title.set_position((0.5, 1.02))
        plt.tight_layout()
