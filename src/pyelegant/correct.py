import os
import tempfile

from . import std_print_enabled
from . import elebuilder
from .local import run
from .remote import remote

def tunes(
    corrected_LTE_filepath, init_LTE_filepath, E_MeV, use_beamline=None,
    ele_filepath=None, del_tmp_files=True,
    quadrupoles=None, exclude=None, tune_x=0.0, tune_y=0.0, n_iterations=5,
    correction_fraction=0.9, tolerance=0.0,
    run_local=True, remote_opts=None):
    """"""

    if quadrupoles is None:
        raise ValueError('"quadrupoles" must be a list of strings.')
    used_quads_str = ' '.join(quadrupoles)

    if ele_filepath is None:
        tmp = tempfile.NamedTemporaryFile(
            dir=os.getcwd(), delete=False, prefix='tmpTunes_', suffix='.ele')
        ele_filepath = os.path.abspath(tmp.name)
        tmp.close()

    ed = elebuilder.EleDesigner(double_format='.12g')

    ed.add_block('run_setup',
        lattice=init_LTE_filepath, p_central_mev=E_MeV, use_beamline=use_beamline)

    ed.add_newline()

    ed.add_block('run_control')

    ed.add_newline()

    ed.add_block('correct_tunes',
        quadrupoles=used_quads_str, exclude=exclude, tune_x=tune_x, tune_y=tune_y,
        n_iterations=n_iterations, correction_fraction=correction_fraction,
        tolerance=tolerance, change_defined_values=True)

    ed.add_newline()

    ed.add_block('twiss_output')

    ed.add_newline()

    ed.add_block('bunched_beam')

    ed.add_newline()

    ed.add_block('track')

    ed.add_block('save_lattice', filename=corrected_LTE_filepath)

    ed.write(ele_filepath)

    ed.update_output_filepaths(ele_filepath[:-4]) # Remove ".ele"
    #print(ed.actual_output_filepath_list)

    # Run Elegant
    if run_local:
        run(ele_filepath, print_cmd=False,
            print_stdout=std_print_enabled['out'],
            print_stderr=std_print_enabled['err'])
    else:
        if remote_opts is None:
            remote_opts = dict(use_sbatch=False)

        if ('pelegant' in remote_opts) and (remote_opts['pelegant'] is not False):
            print('"pelegant" option in `remote_opts` must be False.')
            remote_opts['pelegant'] = False
        else:
            remote_opts['pelegant'] = False

        remote_opts['ntasks'] = 1
        # ^ If this is more than 1, you will likely see an error like "Unable to
        #   access file /.../tmp*.twi--file is locked (SDDS_InitializeOutput)"

        remote.run(remote_opts, ele_filepath, print_cmd=True,
                   print_stdout=std_print_enabled['out'],
                   print_stderr=std_print_enabled['err'],
                   output_filepaths=None)

    if del_tmp_files:
        for fp in ed.actual_output_filepath_list + [ele_filepath]:
            if fp.startswith('/dev'):
                continue
            elif fp == corrected_LTE_filepath:
                continue
            else:
                try:
                    os.remove(fp)
                except:
                    print(f'Failed to delete "{fp}"')

def chroms(corrected_LTE_filepath, init_LTE_filepath, E_MeV, use_beamline=None,
    ele_filepath=None, del_tmp_files=True,
    sextupoles=None, exclude=None, dnux_dp=0.0, dnuy_dp=0.0, n_iterations=5,
    correction_fraction=0.9, tolerance=0.0,
    run_local=True, remote_opts=None):
    """"""

    if sextupoles is None:
        raise ValueError('"sextupoles" must be a list of strings.')
    used_sexts_str = ' '.join(sextupoles)

    if ele_filepath is None:
        tmp = tempfile.NamedTemporaryFile(
            dir=os.getcwd(), delete=False, prefix='tmpChroms_', suffix='.ele')
        ele_filepath = os.path.abspath(tmp.name)
        tmp.close()

    ed = elebuilder.EleDesigner(double_format='.12g')

    ed.add_block('run_setup',
        lattice=init_LTE_filepath, p_central_mev=E_MeV, use_beamline=use_beamline)

    ed.add_newline()

    ed.add_block('run_control')

    ed.add_newline()

    ed.add_block('chromaticity',
        sextupoles=used_sexts_str, exclude=exclude,
        dnux_dp=dnux_dp, dnuy_dp=dnuy_dp,
        n_iterations=n_iterations, correction_fraction=correction_fraction,
        tolerance=tolerance, change_defined_values=True)

    ed.add_newline()

    ed.add_block('twiss_output')

    ed.add_newline()

    ed.add_block('bunched_beam')

    ed.add_newline()

    ed.add_block('track')

    ed.add_block('save_lattice', filename=corrected_LTE_filepath)

    ed.write(ele_filepath)

    ed.update_output_filepaths(ele_filepath[:-4]) # Remove ".ele"
    #print(ed.actual_output_filepath_list)

    # Run Elegant
    if run_local:
        run(ele_filepath, print_cmd=False,
            print_stdout=std_print_enabled['out'],
            print_stderr=std_print_enabled['err'])
    else:
        if remote_opts is None:
            remote_opts = dict(use_sbatch=False)

        if ('pelegant' in remote_opts) and (remote_opts['pelegant'] is not False):
            print('"pelegant" option in `remote_opts` must be False.')
            remote_opts['pelegant'] = False
        else:
            remote_opts['pelegant'] = False

        remote_opts['ntasks'] = 1
        # ^ If this is more than 1, you will likely see an error like "Unable to
        #   access file /.../tmp*.twi--file is locked (SDDS_InitializeOutput)"

        remote.run(remote_opts, ele_filepath, print_cmd=True,
                   print_stdout=std_print_enabled['out'],
                   print_stderr=std_print_enabled['err'],
                   output_filepaths=None)

    if del_tmp_files:
        for fp in ed.actual_output_filepath_list + [ele_filepath]:
            if fp.startswith('/dev'):
                continue
            elif fp == corrected_LTE_filepath:
                continue
            else:
                try:
                    os.remove(fp)
                except:
                    print(f'Failed to delete "{fp}"')
