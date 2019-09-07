from __future__ import print_function, division, absolute_import
from __future__ import unicode_literals

import os
from subprocess import Popen, PIPE
import json
import importlib
import tempfile

from . import util

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
    LTE_filepath, E_MeV, use_beamline=None, radiation_integrals=False,
    compute_driving_terms=False, concat_order=1, higher_order_chromaticity=False,
    ele_filepath=None, twi_filepath='%s.twi', rootname=None, magnets=None,
    semaphore_file=None, parameters=None, element_divisions=0, macros=None,
    alter_elements_list=None,
    run_local=True, remote_opts=None):
    """"""

    matched = True

    return _calc_twiss(
        matched, LTE_filepath, E_MeV, ele_filepath=ele_filepath,
        twi_filepath=twi_filepath, use_beamline=use_beamline,
        radiation_integrals=radiation_integrals,
        compute_driving_terms=compute_driving_terms, concat_order=concat_order,
        higher_order_chromaticity=higher_order_chromaticity,
        rootname=rootname, magnets=magnets, semaphore_file=semaphore_file,
        parameters=parameters, element_divisions=element_divisions,
        macros=macros, alter_elements_list=alter_elements_list,
        calc_correct_transport_line_linear_chrom=False,
        run_local=run_local, remote_opts=remote_opts)


def calc_line_twiss(
    LTE_filepath, E_MeV, betax0, betay0, alphax0=0.0, alphay0=0.0,
    etax0=0.0, etay0=0.0, etaxp0=0.0, etayp0=0.0,
    use_beamline=None, radiation_integrals=False, compute_driving_terms=False,
    concat_order=1, calc_correct_transport_line_linear_chrom=True,
    ele_filepath=None, twi_filepath='%s.twi', rootname=None, magnets=None,
    semaphore_file=None, parameters=None, element_divisions=0, macros=None,
    alter_elements_list=None,
    run_local=True, remote_opts=None):
    """"""

    matched = False

    return _calc_twiss(
        matched, LTE_filepath, E_MeV, ele_filepath=ele_filepath,
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
        run_local=run_local, remote_opts=remote_opts)

def _calc_twiss(
    matched, LTE_filepath, E_MeV, ele_filepath=None, twi_filepath='%s.twi',
    betax0=1.0, betay0=1.0, alphax0=0.0, alphay0=0.0, etax0=0.0, etay0=0.0,
    etaxp0=0.0, etayp0=0.0, use_beamline=None, radiation_integrals=False,
    compute_driving_terms=False, concat_order=1, higher_order_chromaticity=False,
    rootname=None, magnets=None, semaphore_file=None, parameters=None,
    element_divisions=0, macros=None, alter_elements_list=None,
    calc_correct_transport_line_linear_chrom=False,
    run_local=True, remote_opts=None):
    """"""

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

    output_filepaths = dict(
        ele=ele_filepath,
        twi=util.get_abspath(twi_filepath, ele_filepath, rootname=rootname)
    )
    output_filepaths.update(util.get_run_setup_output_abspaths(
        ele_filepath, rootname=rootname,
        magnets=magnets, semaphore_file=semaphore_file, parameters=parameters))
    #if other_output_filepaths is not None:
        #for k, v in other_output_filepaths.items():
            #assert k not in output_filepaths
            #output_filepaths[k] = util.get_abspath(v, ele_filepath, rootname=rootname)

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

    return output_filepaths

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

def plot_twiss(output_filepaths_dict):
    """"""


