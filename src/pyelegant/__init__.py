#from __future__ import print_function, division, absolute_import
#from __future__ import unicode_literals

import os
from subprocess import Popen, PIPE
import json
import importlib
import tempfile
import numpy as np
import h5py
import gzip
import pickle
import matplotlib.pylab as plt
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as patches

from . import util
from . import sdds

this_folder = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(this_folder, 'facility.json'), 'r') as f:
    facility_name = json.load(f)['name']

#if False:
    #from nsls2apcluster import *
#else:
    ## Based on "placeholder"'s answer on
    ##   https://stackoverflow.com/questions/44492803/python-dynamic-import-how-to-import-from-module-name-from-variable/44492879#44492879
    #facility_specific_module = importlib.import_module('.'+facility_name, __name__)
    #globals().update(
        #{n: getattr(facility_specific_module, n) for n in facility_specific_module.__all__}
        #if hasattr(facility_specific_module, '__all__')
        #else
        #{k: v for (k, v) in facility_specific_module.__dict__.items()
         #if not k.startswith('_')}
    #)

try:
    remote = importlib.import_module('.'+facility_name, __name__)
except:
    print('\n## WARNING ##')
    print('Failed to load remote run setup for "{}"'.format(facility_name))
    print('All the Elegant commands will only be run locally.')
    remote = None

def run(ele_filepath, macros=None, print_cmd=False, print_stdout=True, print_stderr=True):
    """"""

    cmd_list = ['elegant', ele_filepath]

    if macros is not None:
        macro_str_list = []
        for k, v in macros.items():
            macro_str_list.append('='.join([k, v]))
        cmd_list.append('-macro=' + ','.join(macro_str_list))

    if print_cmd:
        print('$ ' + ' '.join(cmd_list))

    p = Popen(cmd_list, stdout=PIPE, stderr=PIPE)
    out, err = p.communicate()
    out, err = out.decode('utf-8'), err.decode('utf-8')

    if out and print_stdout:
        print(out)

    if err and print_stderr:
        print('ERROR:')
        print(err)

def calc_ring_twiss(
    output_filepath, LTE_filepath, E_MeV, use_beamline=None, radiation_integrals=False,
    compute_driving_terms=False, concat_order=1, higher_order_chromaticity=False,
    ele_filepath=None, twi_filepath='%s.twi', rootname=None, magnets=None,
    semaphore_file=None, parameters='%s.param', element_divisions=0, macros=None,
    alter_elements_list=None,
    output_file_type='hdf5', del_tmp_files=True,
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
        calc_correct_transport_line_linear_chrom=False,
        output_file_type=output_file_type, del_tmp_files=del_tmp_files,
        run_local=run_local, remote_opts=remote_opts)


def calc_line_twiss(
    output_filepath, LTE_filepath, E_MeV, betax0, betay0, alphax0=0.0, alphay0=0.0,
    etax0=0.0, etay0=0.0, etaxp0=0.0, etayp0=0.0,
    use_beamline=None, radiation_integrals=False, compute_driving_terms=False,
    concat_order=1, calc_correct_transport_line_linear_chrom=True,
    ele_filepath=None, twi_filepath='%s.twi', rootname=None, magnets=None,
    semaphore_file=None, parameters='%s.param', element_divisions=0, macros=None,
    alter_elements_list=None,
    output_file_type='hdf5', del_tmp_files=True,
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
        calc_correct_transport_line_linear_chrom=calc_correct_transport_line_linear_chrom,
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
    calc_correct_transport_line_linear_chrom=False,
    output_file_type='hdf5', del_tmp_files=True,
    run_local=True, remote_opts=None):
    """"""

    if output_file_type.lower() not in ('hdf5', 'h5', 'pgz'):
        raise ValueError('Invalid output file type: {}'.format(output_file_type))

    ele_contents = ''

    ele_contents += _build_block_run_setup(
        LTE_filepath, E_MeV, use_beamline=use_beamline, rootname=rootname,
        magnets=magnets, semaphore_file=semaphore_file, parameters=parameters,
        element_divisions=element_divisions)

    ele_contents += '''
&run_control &end
'''

    if alter_elements_list is not None:
        ele_contents += _build_block_alter_elements(alter_elements_list)

    ele_contents += _build_block_twiss_output(
        matched, twi_filepath=twi_filepath, radiation_integrals=radiation_integrals,
        compute_driving_terms=compute_driving_terms, concat_order=concat_order,
        higher_order_chromaticity=higher_order_chromaticity,
        beta_x=betax0, alpha_x=alphax0, eta_x=etax0, etap_x=etaxp0,
        beta_y=betay0, alpha_y=alphay0, eta_y=etay0, etap_y=etayp0)

    ele_contents += '''
&bunched_beam &end

&track &end
'''

    if ele_filepath is None:
        tmp = tempfile.NamedTemporaryFile(
            dir=os.getcwd(), delete=False, prefix='tmpCalcTwi_', suffix='.ele')
        ele_filepath = os.path.abspath(tmp.name)

    util.robust_text_file_write(ele_filepath, ele_contents, nMaxTry=1)

    tmp_filepaths = dict(
        ele=ele_filepath,
        twi=util.get_abspath(twi_filepath, ele_filepath, rootname=rootname)
    )
    tmp_filepaths.update(util.get_run_setup_output_abspaths(
        ele_filepath, rootname=rootname,
        magnets=magnets, semaphore_file=semaphore_file, parameters=parameters))

    # Run Elegant
    if run_local:
        run(ele_filepath, macros=macros, print_cmd=False,
            print_stdout=True, print_stderr=True)
    else:
        remote.run(remote_opts, ele_filepath, macros=macros, print_cmd=False,
                   print_stdout=True, print_stderr=True, output_filepaths=None)

    if calc_correct_transport_line_linear_chrom:
        # TODO
        raise NotImplementedError('TODO: calc_correct_transport_line_linear_chrom')

    output, meta = {}, {}
    for k, v in tmp_filepaths.items():
        try:
            meta_params, meta_columns = sdds.query(v, suppress_err_msg=True)
            params, columns = sdds.printout(
                v, param_name_list=None, column_name_list=None,
                str_format='', show_output=False, show_cmd=False,
                suppress_err_msg=True)

            meta[k] = {}
            if meta_params != {}:
                meta[k]['params'] = meta_params
            if meta_columns != {}:
                meta[k]['columns'] = meta_columns

            output[k] = {}
            if params != {}:
                output[k]['params'] = params
            if columns != {}:
                for _k, _v in columns.items():
                    columns[_k] = np.array(_v)
                output[k]['columns'] = columns
        except:
            continue

    output_file_type = output_file_type.lower()

    if output_file_type in ('hdf5', 'h5'):
        util.robust_sdds_hdf5_write(
            output_filepath, [output, meta], nMaxTry=10, sleep=10.0)
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
            output_filepath, [mod_output, mod_meta], nMaxTry=10, sleep=10.0)
    else:
        raise ValueError()

    if del_tmp_files:
        for k, fp in tmp_filepaths.items():
            try:
                os.remove(fp)
            except:
                print(f'Failed to delete "{fp}"')

        return
    else:
        return tmp_filepaths

def _build_block_run_setup(
    LTE_filepath, E_MeV, use_beamline=None, rootname=None, magnets=None,
    semaphore_file=None, parameters=None, element_divisions=0):
    """"""

    block = []
    block += ['lattice = "{}"'.format(LTE_filepath)]
    if use_beamline is not None:
        block += ['use_beamline = "{}"'.format(use_beamline)]
    if rootname is not None:
        block += ['rootname = "{}"'.format(rootname)]
    if magnets is not None:
        block += ['magnets = "{}"'.format(magnets)]
    if semaphore_file is not None:
        block += ['semaphore_file = "{}"'.format(semaphore_file)]
    if parameters is not None:
        block += ['parameters = "{}"'.format(parameters)]
    block += ['p_central_mev = {:.9g}'.format(E_MeV)]
    if element_divisions != 0:
        block += ['element_divisions = {:d}'.format(element_divisions)]

    ele_contents = '''
&run_setup
{}
&end
'''.format('\n'.join([' ' * 4 + line for line in block]))

    return ele_contents

def _build_block_twiss_output(
    matched, twi_filepath='%s.twi', radiation_integrals=False,
    compute_driving_terms=False, concat_order=3, higher_order_chromaticity=False,
    beta_x=1.0, alpha_x=0.0, eta_x=0.0, etap_x=0.0, beta_y=1.0, alpha_y=0.0,
    eta_y=0.0, etap_y=0.0):
    """"""

    block = []
    block += ['filename = "{}"'.format(twi_filepath)]
    block += ['matched = {:d}'.format(1 if matched else 0)]
    block += ['radiation_integrals = {:d}'.format(1 if radiation_integrals else 0)]
    block += ['compute_driving_terms = {:d}'.format(1 if compute_driving_terms else 0)]
    block += ['concat_order = {:d}'.format(concat_order)]
    if higher_order_chromaticity:
        block += ['higher_order_chromaticity = 1']
        if concat_order != 3:
            print('WARNING: When computing higher-order chromaticity, "concat_order" should be set to 3.')
    if not matched:
        block += ['beta_x = {:.9g}'.format(beta_x)]
        block += ['alpha_x = {:.9g}'.format(alpha_x)]
        block += ['eta_x = {:.9g}'.format(eta_x)]
        block += ['etap_x = {:.9g}'.format(etap_x)]
        block += ['beta_y = {:.9g}'.format(beta_y)]
        block += ['alpha_y = {:.9g}'.format(alpha_y)]
        block += ['eta_y = {:.9g}'.format(eta_y)]
        block += ['etap_y = {:.9g}'.format(etap_y)]

    ele_contents = '''
&twiss_output
{}
&end
'''.format('\n'.join([' ' * 4 + line for line in block]))

    return ele_contents

def _build_block_alter_elements(alter_elements_list):
    """"""

    block = []

    for d in alter_elements_list:
        block += [
            '&alter_elements name = {name}, type = {type}, item = {item}, '
            'value = {value:.9g} &end'.format(**d)]

    ele_contents = '\n' + '\n'.join(block) + '\n'

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

def plot_twiss(
    result_filepath, result_file_type, slim=None, s_margin_m=0.1, s0_m=0.0,
    etax_unit='mm', right_margin_adj=0.88, print_scalars=None):
    """"""

    if result_file_type in ('hdf5', 'h5'):
        d, meta = util.load_sdds_hdf5_file(result_filepath)
    elif result_file_type == 'pgz':
        d, meta = util.load_pgz_file(result_filepath)
    else:
        raise ValueError('"result_file_type" must be one of "hdf5", "h5", "pgz"')

    twi_ar = d['twi']['arrays']
    twi_sc = d['twi']['scalars']

    meta_twi_ar = meta['twi']['arrays']
    meta_twi_sc = meta['twi']['scalars']

    # Print out scalar values
    print('################################################################')
    for k in sorted(list(twi_sc)):
        if (print_scalars is not None) and (k not in print_scalars):
            continue
        v = twi_sc[k]
        dtype = meta_twi_sc[k]['TYPE']
        units = meta_twi_sc[k]['UNITS']
        descr = meta_twi_sc[k]['DESCRIPTION']
        if dtype == 'double':
            print(f'{k} = {v:.9g} [{units}] ({descr})')
        elif dtype == 'string':
            print(f'{k} = {v} [{units}] ({descr})')
        elif dtype == 'long':
            print(f'{k} = {int(v):d} [{units}] ({descr})')
        else:
            raise ValueError((k, dtype, units))
    print('################################################################')


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
    if show_mag_prof:
        extra_dy_frac = 0.2

        ylim = ax1.get_ylim()
        extra_dy = (ylim[1] - ylim[0]) * extra_dy_frac
        ax1.set_ylim([ylim[0] - extra_dy, ylim[1]])

        prof_center_y = - extra_dy * 0.6
        quad_height = extra_dy / 5.0
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
                m = np.logical_and(parameters['ElementName'] == elem_name,
                                   parameters['ElementOccurence'] == elem_occur)
                m = np.logical_and(m, parameters['ElementParameter'] == 'K1')
                assert np.sum(m) == 1
                K1 = parameters['ParameterValue'][m]
                if K1 >= 0.0: # Focusing Quad
                    bottom, top = 0.0, quad_height
                    c = 'r'
                else: # Defocusing Quad
                    bottom, top = -quad_height, 0.0
                    c = 'b'

                # Shift vertically
                bottom += prof_center_y
                top += prof_center_y

                width = cur_s - prev_s
                height = top - bottom

                p = patches.Rectangle((prev_s, bottom), width, height, fill=True, color=c)
                ax.add_patch(p)

            elif elem_type in ('SEXT', 'KSEXT'):
                ax.plot([prev_s, cur_s], np.array([0.0, 0.0]) + prof_center_y, 'k-')

            elif elem_type in ('RBEND', 'SBEND', 'CSBEND'):
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
    plt.tight_layout()
    plt.subplots_adjust(right=right_margin_adj)

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
