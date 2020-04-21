import os
import tempfile
import numpy as np
import matplotlib.pylab as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as patches
import h5py

from .local import run
from .remote import remote
from . import __version__, std_print_enabled
from . import elebuilder
from . import util
from . import sdds

def calc_ring_twiss(
    output_filepath, LTE_filepath, E_MeV, use_beamline=None, radiation_integrals=False,
    compute_driving_terms=False, concat_order=1, higher_order_chromaticity=False,
    calc_matrix_lin_chrom=False,
    ele_filepath=None, twi_filepath='%s.twi', rootname=None, magnets=None,
    semaphore_file=None, parameters='%s.param', element_divisions=0, macros=None,
    alter_elements_list=None,
    output_file_type=None, del_tmp_files=True,
    run_local=True, remote_opts=None):
    """"""

    matched = True

    return _calc_twiss(
        output_filepath, matched, LTE_filepath, E_MeV, ele_filepath=ele_filepath,
        twi_filepath=twi_filepath, use_beamline=use_beamline,
        radiation_integrals=radiation_integrals,
        compute_driving_terms=compute_driving_terms, concat_order=concat_order,
        higher_order_chromaticity=higher_order_chromaticity,
        rootname=rootname, magnets=magnets, semaphore_file=semaphore_file,
        parameters=parameters, element_divisions=element_divisions,
        macros=macros, alter_elements_list=alter_elements_list,
        calc_matrix_lin_chrom=calc_matrix_lin_chrom,
        output_file_type=output_file_type, del_tmp_files=del_tmp_files,
        run_local=run_local, remote_opts=remote_opts)


def calc_line_twiss(
    output_filepath, LTE_filepath, E_MeV, betax0, betay0, alphax0=0.0, alphay0=0.0,
    etax0=0.0, etay0=0.0, etaxp0=0.0, etayp0=0.0,
    use_beamline=None, radiation_integrals=False, compute_driving_terms=False,
    concat_order=1, calc_matrix_lin_chrom=False,
    ele_filepath=None, twi_filepath='%s.twi', rootname=None, magnets=None,
    semaphore_file=None, parameters='%s.param', element_divisions=0, macros=None,
    alter_elements_list=None,
    output_file_type=None, del_tmp_files=True,
    run_local=True, remote_opts=None):
    """"""

    matched = False

    return _calc_twiss(
        output_filepath, matched, LTE_filepath, E_MeV, ele_filepath=ele_filepath,
        twi_filepath=twi_filepath, betax0=betax0, betay0=betay0,
        alphax0=alphax0, alphay0=alphay0, etax0=etax0, etay0=etay0,
        etaxp0=etaxp0, etayp0=etayp0, use_beamline=use_beamline,
        radiation_integrals=radiation_integrals,
        compute_driving_terms=compute_driving_terms, concat_order=concat_order,
        higher_order_chromaticity=False, rootname=rootname, magnets=magnets,
        semaphore_file=semaphore_file, parameters=parameters,
        element_divisions=element_divisions, macros=macros,
        alter_elements_list=alter_elements_list,
        calc_matrix_lin_chrom=calc_matrix_lin_chrom,
        output_file_type=output_file_type, del_tmp_files=del_tmp_files,
        run_local=run_local, remote_opts=remote_opts)

def _calc_twiss(
    output_filepath, matched, LTE_filepath, E_MeV,
    ele_filepath=None, twi_filepath='%s.twi',
    betax0=1.0, betay0=1.0, alphax0=0.0, alphay0=0.0, etax0=0.0, etay0=0.0,
    etaxp0=0.0, etayp0=0.0, use_beamline=None, radiation_integrals=False,
    compute_driving_terms=False, concat_order=1, higher_order_chromaticity=False,
    rootname=None, magnets=None, semaphore_file=None, parameters='%s.param',
    element_divisions=0, macros=None, alter_elements_list=None,
    calc_matrix_lin_chrom=False,
    output_file_type=None, del_tmp_files=True,
    run_local=True, remote_opts=None):
    """"""

    if calc_matrix_lin_chrom:
        assert parameters is not None

    if higher_order_chromaticity:
        if concat_order != 3:
            sys.stderr(('WARNING: When computing higher-order chromaticity, '
                        '"concat_order" should be set to 3.'))

    with open(LTE_filepath, 'r') as f:
        file_contents = f.read()

    input_dict = dict(
        matched=matched, LTE_filepath=os.path.abspath(LTE_filepath), E_MeV=E_MeV,
        use_beamline=use_beamline, radiation_integrals=radiation_integrals,
        compute_driving_terms=compute_driving_terms, concat_order=concat_order,
        higher_order_chromaticity=higher_order_chromaticity, rootname=rootname,
        magnets=magnets, semaphore_file=semaphore_file, parameters=parameters,
        element_divisions=element_divisions, macros=macros,
        alter_elements_list=alter_elements_list,
        calc_matrix_lin_chrom=calc_matrix_lin_chrom,
        twi_filepath=twi_filepath, ele_filepath=ele_filepath,
        del_tmp_files=del_tmp_files, run_local=run_local,
        remote_opts=remote_opts,
        lattice_file_contents=file_contents,
        timestamp_ini=util.get_current_local_time_str(),
    )
    if not matched:
        input_dict['betax0'] = betax0
        input_dict['betay0'] = betay0
        input_dict['alphax0'] = alphax0
        input_dict['alphay0'] = alphay0
        input_dict['etax0'] = etax0
        input_dict['etay0'] = etay0
        input_dict['etaxp0'] = etaxp0
        input_dict['etayp0'] = etayp0

    output_file_type = util.auto_check_output_file_type(output_filepath, output_file_type)
    input_dict['output_file_type'] = output_file_type

    if output_file_type in ('hdf5', 'h5'):
        util.save_input_to_hdf5(output_filepath, input_dict)

    if ele_filepath is None:
        tmp = tempfile.NamedTemporaryFile(
            dir=os.getcwd(), delete=False, prefix='tmpCalcTwi_', suffix='.ele')
        ele_filepath = os.path.abspath(tmp.name)
        tmp.close()

    ed = elebuilder.EleDesigner(ele_filepath, double_format='.12g')

    ed.add_block('run_setup',
        lattice=LTE_filepath, p_central_mev=E_MeV, use_beamline=use_beamline,
        rootname=rootname, magnets=magnets, semaphore_file=semaphore_file,
        parameters=parameters, element_divisions=element_divisions)

    ed.add_newline()

    ed.add_block('run_control')

    ed.add_newline()

    disable_watch_elem_d = dict(
        name='*', type='WATCH', item='DISABLE', value=True,
        allow_missing_elements=True)

    if alter_elements_list is not None:
        alter_elements_list += [disable_watch_elem_d]
    else:
        alter_elements_list = [disable_watch_elem_d]

    for block in alter_elements_list:
        ed.add_block('alter_elements', **block)

    ed.add_newline()

    twi_kwargs = dict(matched=matched, filename=twi_filepath,
        radiation_integrals=radiation_integrals,
        compute_driving_terms=compute_driving_terms, concat_order=concat_order,
        higher_order_chromaticity=higher_order_chromaticity)
    if not matched:
        twi_kwargs['beta_x'] = betax0
        twi_kwargs['beta_y'] = betay0
        twi_kwargs['alpha_x'] = alphax0
        twi_kwargs['alpha_y'] = alphay0
        twi_kwargs['eta_x'] = etax0
        twi_kwargs['eta_y'] = etay0
        twi_kwargs['etap_x'] = etaxp0
        twi_kwargs['etap_y'] = etayp0
    ed.add_block('twiss_output', **twi_kwargs)

    ed.add_newline()

    ed.add_block('bunched_beam')

    ed.add_newline()

    ed.add_block('track')

    ed.write()
    #print(ed.actual_output_filepath_list)

    # Run Elegant
    if run_local:
        run(ele_filepath, macros=macros, print_cmd=False,
            print_stdout=std_print_enabled['out'],
            print_stderr=std_print_enabled['err'])
    else:
        if remote_opts is None:
            remote_opts = dict(use_sbatch=False)

        if ('pelegant' in remote_opts) and (remote_opts['pelegant'] is not False):
            print('"pelegant" option in `remote_opts` must be False for Twiss calculation')
            remote_opts['pelegant'] = False
        else:
            remote_opts['pelegant'] = False

        remote_opts['ntasks'] = 1
        # ^ If this is more than 1, you will likely see an error like "Unable to
        #   access file /.../tmp*.twi--file is locked (SDDS_InitializeOutput)"

        remote.run(remote_opts, ele_filepath, macros=macros, print_cmd=True,
                   print_stdout=std_print_enabled['out'],
                   print_stderr=std_print_enabled['err'],
                   output_filepaths=None)

    #tmp_filepaths = ed.actual_output_filepath_list
    #output, meta = {}, {}
    #for k, v in tmp_filepaths.items():
        #try:
            #output[k], meta[k] = sdds.sdds2dicts(v)
        #except:
            #continue
    _d = ed.load_sdds_output_files()
    output, meta = _d['data'], _d['meta']

    if calc_matrix_lin_chrom:
        lin_chrom_nat = _calc_matrix_elem_linear_natural_chrom(
            output['twi']['columns'], output['parameters']['columns'])

        output['lin_chrom_nat'] = dict(columns={})
        meta['lin_chrom_nat'] = dict(columns={})
        for k, v in lin_chrom_nat.items():
            output['lin_chrom_nat']['columns'][k] = v
            meta['lin_chrom_nat']['columns'][k] = {}

    timestamp_fin = util.get_current_local_time_str()

    if output_file_type in ('hdf5', 'h5'):
        util.robust_sdds_hdf5_write(
            output_filepath, [output, meta], nMaxTry=10, sleep=10.0, mode='a')
        f = h5py.File(output_filepath, 'a')
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
            output_filepath, dict(
                data=mod_output, meta=mod_meta,
                input=input_dict, timestamp_fin=timestamp_fin,
                _version_PyELEGANT=__version__['PyELEGANT'],
                _version_ELEGANT=__version__['ELEGANT']),
            nMaxTry=10, sleep=10.0)
    else:
        raise ValueError()

    tmp_filepaths = {'ele': ele_filepath}
    for sdds_fp in ed.actual_output_filepath_list:
        if sdds_fp.startswith('/dev/'):
            continue
        ext = sdds_fp.split('.')[-1]
        tmp_filepaths[ext] = sdds_fp

    if del_tmp_files:
        for fp in tmp_filepaths.values():
            if fp.startswith('/dev'):
                continue
            else:
                try:
                    os.remove(fp)
                except:
                    print(f'Failed to delete "{fp}"')

        return
    else:
        return tmp_filepaths

def _calc_matrix_elem_linear_natural_chrom(twi_arrays, parameters_arrays):
    """
    ELEGANT's linear natural chromaticity calculation sometimes cannot be trusted,
    particularly for transport lines.

    This function should correctly compute linear chromaticity, based on
    matrix elements for quads and combined-function bends.

    TODO: Currently, only sector bends can be handled (i.e., E1 = E2 = 0)
    Deal with any arbitrary edge angles in the future.
    """

    twi = twi_arrays
    para = parameters_arrays

    quads_bends = np.logical_or(twi['ElementType'] == 'QUAD',
                                twi['ElementType'] == 'SBEN')
    quads_bends = np.logical_or(
        quads_bends, twi['ElementType'] == 'KQUAD')
    quads_bends = np.logical_or(
        quads_bends, twi['ElementType'] == 'CSBEND')
    quads_bends = np.where(quads_bends)[0]
    #
    elem_names = twi['ElementName'][quads_bends]
    #
    sbs = twi['s'][quads_bends - 1]
    ses = twi['s'][quads_bends]
    #
    entrance_betaxs = twi['betax'][quads_bends - 1]
    entrance_betays = twi['betay'][quads_bends - 1]
    entrance_alphaxs = twi['alphax'][quads_bends - 1]
    entrance_alphays = twi['alphay'][quads_bends - 1]


    quads_bends = np.logical_or(para['ElementType'] == 'QUAD',
                                para['ElementType'] == 'SBEN')
    quads_bends = np.logical_or(
        quads_bends, para['ElementType'] == 'KQUAD')
    quads_bends = np.logical_or(
        quads_bends, para['ElementType'] == 'CSBEND')
    param_inds = {}
    for k in ['K1', 'L', 'ANGLE']:
        param_inds[k] = np.logical_and(quads_bends, para['ElementParameter'] == k)
    #
    param_vals = dict(K1=[], L=[], rhoinv_sqr=[])
    for name in elem_names:
        name_matches = (para['ElementName'] == name)

        for k in ['K1', 'L', 'ANGLE']: # <= CRITICAL: "L" must come
            # before "ANGLE" as the value "L" will be used to compute
            # the value "rhoinv_sqr".
            sel = np.logical_and(param_inds[k], name_matches)
            vals = np.unique(para['ParameterValue'][sel])
            if k == 'ANGLE':
                if len(vals) == 0:
                    param_vals['rhoinv_sqr'].append(0.0)
                elif len(vals) == 1:
                    one_over_rho = vals[0] / param_vals['L'][-1]
                    param_vals['rhoinv_sqr'].append(one_over_rho**2)
                else:
                    raise RuntimeError(
                        'Found non-unique bending angle for elements with same names')
            else:
                assert len(vals) == 1
                param_vals[k].append(vals[0])
    for k, v in param_vals.items():
        param_vals[k] = np.array(v)

    output = dict(
        elem_name=elem_names, sb=sbs, se=ses, K1=param_vals['K1'],
        L=param_vals['L'], rhoinv_sqr=param_vals['rhoinv_sqr'],
        nat_ksi_x=[], nat_ksi_y=[])

    n = len(elem_names)
    assert n == len(sbs) == len(ses)
    assert n == len(param_vals['K1']) == len(param_vals['L']) == len(param_vals['rhoinv_sqr'])
    assert n == len(entrance_betaxs) == len(entrance_alphaxs)
    assert n == len(entrance_betays) == len(entrance_alphays)
    for name, sb, se, K1, L, rhoinv_sqr, bx0, ax0, by0, ay0 in zip(
        elem_names, sbs, ses, param_vals['K1'], param_vals['L'],
        param_vals['rhoinv_sqr'], entrance_betaxs, entrance_alphaxs,
        entrance_betays, entrance_alphays):

        if K1 > -rhoinv_sqr:
            kx = np.sqrt(K1 + rhoinv_sqr)
            dksi = (-1) * K1 / (16 * np.pi * bx0 * kx**3) * (
                2 * kx * L * (ax0**2 + (bx0 * kx)**2 + 1)
                - (ax0**2 - (bx0 * kx)**2 + 1) * np.sin(2 * kx * L)
                - 4 * ax0 * bx0 * kx * (np.sin(kx * L)**2)
            )
        else:
            kx = np.sqrt(-K1 - rhoinv_sqr)
            dksi = (-1) * K1 / (16 * np.pi * bx0 * kx**3) * (
                2 * kx * L * (-(ax0**2) + (bx0 * kx)**2 - 1)
                + (ax0**2 + (bx0 * kx)**2 + 1) * np.sinh(2 * kx * L)
                - 4 * ax0 * bx0 * kx * (np.sinh(kx * L)**2)
            )
        output['nat_ksi_x'].append(dksi)

        if K1 < 0.0:
            ky = np.sqrt(-K1)
            dksi = (+1) * K1 / (16 * np.pi * by0 * ky**3) * (
                2 * ky * L * (ay0**2 + (by0 * ky)**2 + 1)
                - (ay0**2 - (by0 * ky)**2 + 1) * np.sin(2 * ky * L)
                - 4 * ay0 * by0 * ky * (np.sin(ky * L)**2)
            )
        else:
            ky = np.sqrt(K1)
            dksi = (+1) * K1 / (16 * np.pi * by0 * ky**3) * (
                2 * ky * L * (-(ay0**2) + (by0 * ky)**2 - 1)
                + (ay0**2 + (by0 * ky)**2 + 1) * np.sinh(2 * ky * L)
                - 4 * ay0 * by0 * ky * (np.sinh(ky * L)**2)
            )
        output['nat_ksi_y'].append(dksi)

    for k in ['nat_ksi_x', 'nat_ksi_y']:
        output[k] = np.array(output[k])

    return output

def get_visible_inds(all_s_array, slim, s_margin_m=0.1):
    """
    s_margin_m [m]
    """

    shifted_slim = slim - slim[0]

    _visible = np.logical_and(
        all_s_array - slim[0] >= shifted_slim[0] - s_margin_m,
        all_s_array - slim[0] <= shifted_slim[1] + s_margin_m
    )

    return _visible

def _get_param_val(param_name, parameters_dict, elem_name, elem_occur):
    """"""

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

def plot_twiss(
    result_filepath, result_file_type=None, slim=None, s_margin_m=0.1, s0_m=0.0,
    etax_unit='mm', right_margin_adj=0.88, print_scalars=None,
    disp_elem_names=None):
    """"""

    if result_file_type in ('hdf5', 'h5'):
        d, meta, version = util.load_sdds_hdf5_file(result_filepath)
    elif result_file_type == 'pgz':
        _d = util.load_pgz_file(result_filepath)
        d = _d['data']
        meta = _d['meta']
    elif result_file_type is None:
        try:
            d, meta, version = util.load_sdds_hdf5_file(result_filepath)
        except OSError:
            _d = util.load_pgz_file(result_filepath)
            d = _d['data']
            meta = _d['meta']
    else:
        raise ValueError('"result_file_type" must be one of "hdf5", "h5", "pgz"')

    twi_ar = d['twi']['arrays']
    twi_sc = d['twi']['scalars']

    meta_twi_ar = meta['twi']['arrays']
    meta_twi_sc = meta['twi']['scalars']

    # Print out scalar values
    scalar_lines = []
    for k in sorted(list(twi_sc)):
        if (print_scalars is not None) and (k not in print_scalars):
            continue
        v = twi_sc[k]
        dtype = meta_twi_sc[k]['TYPE']
        units = meta_twi_sc[k]['UNITS']
        descr = meta_twi_sc[k]['DESCRIPTION']
        if dtype == 'double':
            scalar_lines.append(f'{k} = {v:.9g} [{units}] ({descr})')
        elif dtype == 'string':
            scalar_lines.append(f'{k} = {v} [{units}] ({descr})')
        elif dtype == 'long':
            scalar_lines.append(f'{k} = {int(v):d} [{units}] ({descr})')
        else:
            raise ValueError((k, dtype, units))
    if scalar_lines:
        separator_line = '################################################################'
        scalar_lines = [separator_line] + scalar_lines + [separator_line]
        print('\n'.join(scalar_lines))


    if slim is None:
        slim = [np.min(twi_ar['s']), np.max(twi_ar['s'])]

    slim = np.array(slim)

    shifted_slim = slim - s0_m
    _vis = get_visible_inds(twi_ar['s'], slim, s_margin_m=s_margin_m)

    if 'parameters' in d:
        parameters = d['parameters']['arrays']
        show_mag_prof = True
    else:
        parameters = {}
        show_mag_prof = False

    _kw_label = dict(size=16)
    leg_prop = dict(size=14)
    maj_ticks = dict(labelsize=12, length=7)
    min_ticks = dict(labelsize=5, length=2)

    fig = plt.figure()
    ax1 = plt.subplot2grid((4,1), (0,0), colspan=1, rowspan=4)
    [hbx,hby] = ax1.plot(
        (twi_ar['s'] - s0_m)[_vis], twi_ar['betax'][_vis], 'b.-',
        (twi_ar['s'] - s0_m)[_vis], twi_ar['betay'][_vis], 'r.-')
    ax1.set_xlim(shifted_slim)
    if s0_m == 0.0:
        ax1.set_xlabel(r'$s\, [\mathrm{m}]$', **_kw_label)
    else:
        ax1.set_xlabel(r'$s\, - {:.6g}\, [\mathrm{{m}}]$'.format(s0_m), **_kw_label)
    #ax1.set_xticks([]) # hide x-ticks
    #
    # Make the y-axis label and tick labels match the line color.
    ax1.set_ylabel(r'$\beta_x,\, \beta_y\, [\mathrm{m}]$', color='k', **_kw_label)
    for tl in ax1.get_yticklabels():
        tl.set_color('k')
    ax1.tick_params(axis='both', which='major', **maj_ticks)
    ax1.tick_params(axis='both', which='minor', **min_ticks)
    if show_mag_prof or disp_elem_names:
        extra_dy_frac = 0.2

        ylim = ax1.get_ylim()
        extra_dy = (ylim[1] - ylim[0]) * extra_dy_frac
        ax1.set_ylim([ylim[0] - extra_dy, ylim[1]])

        prof_center_y = - extra_dy * 0.6
        quad_height = extra_dy / 5.0
        sext_height = quad_height * 1.5
        oct_height = quad_height * 1.75
        bend_half_height = quad_height/3.0
    ax1.axhline(0.0, color='k', linestyle='-.')
    ax2 = ax1.twinx()
    if etax_unit == 'mm':
        etax = twi_ar['etax'] * 1e3
        etax_ylabel = r'$\eta_x\, [\mathrm{mm}]$'
    elif etax_unit == 'm':
        etax = twi_ar['etax']
        etax_ylabel = r'$\eta_x\, [\mathrm{m}]$'
    else:
        raise ValueError()
    _ex_c = 'g'
    hex, = ax2.plot((twi_ar['s'] - s0_m)[_vis], etax[_vis], _ex_c + '.-')
    ax2.set_xlim(shifted_slim)
    ax2.set_ylabel(etax_ylabel, color='k', **_kw_label)
    for tl in ax2.get_yticklabels():
        tl.set_color('k')
    ax2.tick_params(axis='both', which='major', **maj_ticks)
    ax2.tick_params(axis='both', which='minor', **min_ticks)
    ax2.axhline(0.0, color=_ex_c, linestyle='-.')
    plt.legend([hbx,hby,hex], [r'$\beta_x$', r'$\beta_y$', r'$\eta_x$'],
               bbox_to_anchor=(0, 1.04, 1., 0.102),
               loc='upper center', mode='expand',
               ncol=3, borderaxespad=0., prop=leg_prop)
    #ax1.grid(True)
    if show_mag_prof:

        ylim = ax2.get_ylim()
        extra_dy = (ylim[1] - ylim[0]) * extra_dy_frac
        ax2.set_ylim([ylim[0] - extra_dy, ylim[1]])

        # --- Add magnet profiles

        ax = ax1

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

    if disp_elem_names:

        vis_elem_types = twi_ar['ElementType'][_vis]
        vis_elem_names = twi_ar['ElementName'][_vis]
        vis_s = twi_ar['s'][_vis]

        elem_names_to_show = disp_elem_names.get('elem_names', [])
        if disp_elem_names.get('bends', False):
            inds = np.where(vis_elem_types == 'RBEND')[0].tolist()
            inds += np.where(vis_elem_types == 'RBEN')[0].tolist()
            inds += np.where(vis_elem_types == 'SBEND')[0].tolist()
            inds += np.where(vis_elem_types == 'SBEN')[0].tolist()
            inds += np.where(vis_elem_types == 'CSBEND')[0].tolist()
            elem_names_to_show += vis_elem_names[inds].tolist()
        if disp_elem_names.get('quads', False):
            inds = np.where(vis_elem_types == 'QUAD')[0].tolist()
            inds += np.where(vis_elem_types == 'KQUAD')[0].tolist()
            elem_names_to_show += vis_elem_names[inds].tolist()
        if disp_elem_names.get('sexts', False):
            inds = np.where(vis_elem_types == 'SEXT')[0].tolist()
            inds += np.where(vis_elem_types == 'KSEXT')[0].tolist()
            elem_names_to_show += vis_elem_names[inds].tolist()
        if disp_elem_names.get('octs', False):
            inds = np.where(vis_elem_types == 'OCTU')[0].tolist()
            inds += np.where(vis_elem_types == 'KOCT')[0].tolist()
            elem_names_to_show += vis_elem_names[inds].tolist()
        elem_names_to_show = np.unique(elem_names_to_show).tolist()

        font_size = disp_elem_names.get('font_size', 10)
        extra_dy_frac = disp_elem_names.get('extra_dy_frac', 0.1)

        ylim = ax1.get_ylim()
        extra_dy = (ylim[1] - ylim[0]) * extra_dy_frac
        ax1.set_ylim([ylim[0] - extra_dy, ylim[1]])

        ylim = ax2.get_ylim()
        extra_dy = (ylim[1] - ylim[0]) * extra_dy_frac
        ax2.set_ylim([ylim[0] - extra_dy, ylim[1]])

        ax = ax1
        char_top = prof_center_y - sext_height
        for elem_name in elem_names_to_show:
            s_inds = np.where(elem_name == vis_elem_names)[0]
            sb = None
            for _i in s_inds:
                if sb is None:
                    sb = vis_s[_i-1]

                se = vis_s[_i]
                if _i + 1 in s_inds:
                    continue

                s = (sb + se) / 2
                s -= s0_m
                ax.text(s, char_top, elem_name, rotation='vertical',
                        rotation_mode='anchor', ha='right', va='center',
                        fontdict=dict(size=font_size))

                sb = None

    plt.tight_layout()
    plt.subplots_adjust(right=right_margin_adj)

    if False:
        plt.figure()
        plt.plot((twi_ar['s'] - s0_m)[_vis], twi_ar['psix'][_vis] / (2 * np.pi), 'b-',
                 (twi_ar['s'] - s0_m)[_vis], twi_ar['psiy'][_vis] / (2 * np.pi), 'r-')
        plt.legend([r'$\nu_x$', r'$\nu_y$'], prop=leg_prop)
        #plt.grid(True)
        if s0_m == 0.0:
            plt.xlabel(r'$s\, [\mathrm{m}]$', **_kw_label)
        else:
            plt.xlabel(r'$s\, - {:.6g}\, [\mathrm{{m}}]$'.format(s0_m), **_kw_label)
        plt.ylabel(r'$\mathrm{Phase\, Advance\, [2\pi]}$', **_kw_label)
        plt.tight_layout()

