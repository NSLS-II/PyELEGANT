from typing import Union, Optional, Tuple
import os
from pathlib import Path
import numpy as np
import shlex
from subprocess import Popen, PIPE
import tempfile

from .local import run
from .remote import remote
from . import std_print_enabled
from . import elebuilder
from . import util
from . import sdds

def get_closed_orbit(
    ele_filepath: str, clo_output_filepath: str,
    run_local: bool = True, remote_opts: Optional[dict] = None,
    ) -> Tuple[dict]:
    """"""

    # Run Elegant
    if run_local:
        run(ele_filepath, print_cmd=False,
            print_stdout=std_print_enabled['out'],
            print_stderr=std_print_enabled['err'])
    else:
        if remote_opts is None:
            remote_opts = dict(
                use_sbatch=False, pelegant=False, job_name='clo',
                output='clo.%J.out', error='clo.%J.err',
                partition='normal', ntasks=1)

        remote.run(remote_opts, ele_filepath, print_cmd=True,
                   print_stdout=std_print_enabled['out'],
                   print_stderr=std_print_enabled['err'],
                   output_filepaths=None)

    tmp_filepaths = dict(clo=clo_output_filepath)
    output, meta = {}, {}
    for k, v in tmp_filepaths.items():
        try:
            output[k], meta[k] = sdds.sdds2dicts(v)
        except:
            continue

    return output, meta

def calc_closed_orbit(
    output_filepath: str, LTE_filepath: str, E_MeV: float,
    fixed_length: bool = True, output_monitors_only: bool = False,
    closed_orbit_accuracy: float = 1e-12, closed_orbit_iterations: int = 40,
    iteration_fraction: float = 0.9, n_turns: int = 1,
    load_parameters: Optional[dict] = None, reuse_ele: bool = False,
    use_beamline: Optional[str] = None, N_KICKS: Optional[dict] = None,
    transmute_elements: Optional[dict] = None, ele_filepath: Optional[str] = None,
    output_file_type: Optional[str] = None, del_tmp_files: bool = True,
    run_local: bool = True, remote_opts: Optional[dict] = None,
    ) -> Union[str, dict]:
    """"""

    assert n_turns >= 1
    assert iteration_fraction <= 1.0

    with open(LTE_filepath, 'r') as f:
        file_contents = f.read()

    input_dict = dict(
        LTE_filepath=os.path.abspath(LTE_filepath), E_MeV=E_MeV,
        fixed_length=fixed_length, output_monitors_only=output_monitors_only,
        closed_orbit_accuracy=closed_orbit_accuracy,
        closed_orbit_iterations=closed_orbit_iterations,
        n_turns=n_turns, load_parameters=load_parameters, reuse_ele=reuse_ele,
        use_beamline=use_beamline, N_KICKS=N_KICKS,
        transmute_elements=transmute_elements, ele_filepath=ele_filepath,
        del_tmp_files=del_tmp_files, run_local=run_local,
        remote_opts=remote_opts,
        lattice_file_contents=file_contents,
        timestamp_ini=util.get_current_local_time_str(),
    )

    output_file_type = util.auto_check_output_file_type(output_filepath, output_file_type)
    input_dict['output_file_type'] = output_file_type

    if output_file_type in ('hdf5', 'h5'):
        util.save_input_to_hdf5(output_filepath, input_dict)

    if ele_filepath is None:
        tmp = tempfile.NamedTemporaryFile(
            dir=os.getcwd(), delete=False, prefix=f'tmpCO_', suffix='.ele')
        ele_filepath = os.path.abspath(tmp.name)
        tmp.close()

    ed = elebuilder.EleDesigner(double_format='.12g')

    if transmute_elements is not None:
        elebuilder.add_transmute_blocks(ed, transmute_elements)

        ed.add_newline()

    ed.add_newline()

    ed.add_block('run_setup',
        lattice=LTE_filepath, p_central_mev=E_MeV, use_beamline=use_beamline)

    ed.add_newline()

    if load_parameters is not None:
        load_parameters['change_defined_values'] = True
        load_parameters.setdefault('allow_missing_elements', True)
        load_parameters.setdefault('allow_missing_parameters', True)
        ed.add_block('load_parameters', **load_parameters)

        ed.add_newline()

    ed.add_block('run_control', n_passes=n_turns)

    ed.add_newline()

    if N_KICKS is not None:
        elebuilder.add_N_KICKS_alter_elements_blocks(ed, N_KICKS)

        ed.add_newline()

    _block_opts = dict(
        output='%s.clo', tracking_turns=(False if n_turns == 1 else True),
        fixed_length=fixed_length, output_monitors_only=output_monitors_only,
        closed_orbit_accuracy=closed_orbit_accuracy,
        closed_orbit_iterations=closed_orbit_iterations,
        iteration_fraction=iteration_fraction,
    )
    ed.add_block('closed_orbit', **_block_opts)

    ed.add_newline()

    ed.add_block('bunched_beam')

    ed.add_newline()

    ed.add_block('track')

    ed.write(ele_filepath)

    ed.update_output_filepaths(ele_filepath[:-4]) # Remove ".ele"
    #print(ed.actual_output_filepath_list)

    for fp in ed.actual_output_filepath_list:
        if fp.endswith('.clo'):
            clo_output_filepath = fp
        elif fp.endswith('.done'):
            done_filepath = fp
        else:
            raise ValueError('This line should not be reached.')

    if False:
        # Run Elegant
        if run_local:
            run(ele_filepath, print_cmd=False,
                print_stdout=std_print_enabled['out'],
                print_stderr=std_print_enabled['err'])
        else:
            if remote_opts is None:
                remote_opts = dict(
                    use_sbatch=False, pelegant=False, job_name='clo',
                    output='clo.%J.out', error='clo.%J.err',
                    partition='normal', ntasks=1)

            remote.run(remote_opts, ele_filepath, print_cmd=True,
                       print_stdout=std_print_enabled['out'],
                       print_stderr=std_print_enabled['err'],
                       output_filepaths=None)

        tmp_filepaths = dict(clo=clo_output_filepath)
        output, meta = {}, {}
        for k, v in tmp_filepaths.items():
            try:
                output[k], meta[k] = sdds.sdds2dicts(v)
            except:
                continue
    else:
        output, meta = get_closed_orbit(
            ele_filepath, clo_output_filepath, run_local=run_local,
            remote_opts=remote_opts)

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

        files_to_be_deleted = ed.actual_output_filepath_list[:]
        if not reuse_ele:
            files_to_be_deleted += [ele_filepath]

        for fp in files_to_be_deleted:
            if fp.startswith('/dev'):
                continue
            else:
                try:
                    os.remove(fp)
                except:
                    print(f'Failed to delete "{fp}"')

    if reuse_ele:
        return dict(
            output_filepath=output_filepath,
            ele_filepath=ele_filepath, clo_output_filepath=clo_output_filepath)
    else:
        return output_filepath

