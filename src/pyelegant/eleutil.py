import os
import tempfile
from typing import Dict, List, Union, Optional

from . import std_print_enabled
from . import elebuilder
from . import sdds
from .local import run

def save_lattice_after_load_parameters(
    input_LTE_filepath: str, new_LTE_filepath: str, load_parameters: Dict) -> None:
    """"""

    tmp = tempfile.NamedTemporaryFile(
        dir=os.getcwd(), delete=False, prefix=f'tmp_', suffix='.ele')
    ele_filepath = os.path.abspath(tmp.name)
    tmp.close()

    ed = elebuilder.EleDesigner(ele_filepath, double_format='.12g',
                                auto_print_on_add=False)

    ed.add_block('run_setup', lattice=input_LTE_filepath, p_central_mev=1.0)
    # ^ The value for "p_central_mev" being used here is just an arbitrary
    #   non-zero value. If it stays as the default value of 0, then ELEGANT will
    #   complain.

    ed.add_newline()

    if 'change_defined_values' in load_parameters:
        if not load_parameters['change_defined_values']:
            raise ValueError('load_parameters["change_defined_values"] cannot be False')
    else:
        load_parameters['change_defined_values'] = True
    load_parameters.setdefault('allow_missing_elements', True)
    load_parameters.setdefault('allow_missing_parameters', True)
    ed.add_block('load_parameters', **load_parameters)

    ed.add_newline()

    ed.add_block('save_lattice', filename=new_LTE_filepath)

    ed.write()

    run(ele_filepath, print_cmd=False,
        print_stdout=std_print_enabled['out'],
        print_stderr=std_print_enabled['err'])

    try:
        os.remove(ele_filepath)
    except:
        print(f'Failed to delete "{ele_filepath}"')

def save_lattice_after_alter_elements(
    input_LTE_filepath: str, new_LTE_filepath: str,
    alter_elements: Union[Dict, List]) -> None:
    """"""

    tmp = tempfile.NamedTemporaryFile(
        dir=os.getcwd(), delete=False, prefix=f'tmp_', suffix='.ele')
    ele_filepath = os.path.abspath(tmp.name)
    tmp.close()

    ed = elebuilder.EleDesigner(ele_filepath, double_format='.12g',
                                auto_print_on_add=False)

    ed.add_block('run_setup', lattice=input_LTE_filepath, p_central_mev=1.0)
    # ^ The value for "p_central_mev" being used here is just an arbitrary
    #   non-zero value. If it stays as the default value of 0, then ELEGANT will
    #   complain.

    ed.add_newline()

    if isinstance(alter_elements, dict):
        ed.add_block('alter_elements', **alter_elements)
    else:
        for block in alter_elements:
            ed.add_block('alter_elements', **block)

    ed.add_newline()

    ed.add_block('save_lattice', filename=new_LTE_filepath)

    ed.write()

    run(ele_filepath, print_cmd=False,
        print_stdout=std_print_enabled['out'],
        print_stderr=std_print_enabled['err'])

    try:
        os.remove(ele_filepath)
    except:
        print(f'Failed to delete "{ele_filepath}"')

def get_transport_matrices(
    input_LTE_filepath: str, use_beamline: Optional[str] = None,
    individual_matrices: bool = False, del_tmp_files: bool = True) -> None:
    """"""

    tmp = tempfile.NamedTemporaryFile(
        dir=os.getcwd(), delete=False, prefix=f'tmp_', suffix='.ele')
    ele_filepath = os.path.abspath(tmp.name)
    tmp.close()

    ed = elebuilder.EleDesigner(ele_filepath, double_format='.12g',
                                auto_print_on_add=False)

    ed.add_block(
        'run_setup', lattice=input_LTE_filepath, use_beamline=use_beamline,
        p_central_mev=1.0)
    # ^ The value for "p_central_mev" being used here is just an arbitrary
    #   non-zero value. If it stays as the default value of 0, then ELEGANT will
    #   complain.

    ed.add_newline()

    ed.add_block('matrix_output', printout=None, SDDS_output='%s.mat',
                 individual_matrices=individual_matrices)

    ed.write()

    run(ele_filepath, print_cmd=False,
        print_stdout=std_print_enabled['out'],
        print_stderr=std_print_enabled['err'])

    for fp in ed.actual_output_filepath_list:
        if fp.endswith('.mat'):
            data, meta = sdds.sdds2dicts(fp)

    if del_tmp_files:
        for fp in ed.actual_output_filepath_list + [ele_filepath]:
            if fp.startswith('/dev'):
                continue
            else:
                try:
                    os.remove(fp)
                except:
                    print(f'Failed to delete "{fp}"')

    return data['columns']
