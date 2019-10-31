import os
import tempfile

from . import elebuilder
from .local import run
from .remote import remote

def tunes(
    corrected_LTE_filepath, init_LTE_filepath, E_MeV, use_beamline=None,
    macros=None, ele_filepath=None, del_tmp_files=True,
    quadrupoles=None, exclude=None, tune_x=0.0, tune_y=0.0, n_iterations=5,
    correction_fraction=0.9, tolerance=0.0,
    run_local=True, remote_opts=None, print_stdout=True, print_stderr=True):
    """"""

    if quadrupoles is None:
        raise ValueError('"quadrupoles" must be a list of strings.')
    used_quads_str = ' '.join(quadrupoles)

    if ele_filepath is None:
        tmp = tempfile.NamedTemporaryFile(
            dir=os.getcwd(), delete=False, prefix='tmpTunes_', suffix='.ele')
        ele_filepath = os.path.abspath(tmp.name)
        tmp.close()

    eb = elebuilder.EleContents(double_format='.12g')

    eb.run_setup(
        lattice=init_LTE_filepath, p_central_mev=E_MeV, use_beamline=use_beamline,
        parameters='%s.param')

    eb.newline()

    eb.run_control()

    eb.newline()

    eb.correct_tunes(
        quadrupoles=used_quads_str, exclude=exclude, tune_x=tune_x, tune_y=tune_y,
        n_iterations=n_iterations, correction_fraction=correction_fraction,
        tolerance=tolerance, change_defined_values=True)

    eb.newline()

    eb.twiss_output()

    eb.newline()

    eb.bunched_beam()

    eb.newline()

    eb.track()

    #eb.newline()

    #eb.run_setup(
        #lattice=init_LTE_filepath, p_central_mev=E_MeV, use_beamline=use_beamline)

    #eb.newline()

    #eb.load_parameters(
        #filename='%s.param', change_defined_values=True,
        #include_item_pattern='K1', include_name_pattern=used_quads_str)

    #eb.newline()

    eb.save_lattice(filename=corrected_LTE_filepath)

    eb.write(ele_filepath)

    eb.update_output_filepaths(ele_filepath[:-4]) # Remove ".ele"
    #print(eb.actual_output_filepath_list)

    # Run Elegant
    if run_local:
        run(ele_filepath, macros=macros, print_cmd=False,
            print_stdout=print_stdout, print_stderr=print_stderr)
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
                   print_stdout=print_stdout, print_stderr=print_stderr,
                   output_filepaths=None)

    if del_tmp_files:
        for fp in eb.actual_output_filepath_list + [ele_filepath]:
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
    macros=None, ele_filepath=None, del_tmp_files=True,
    sextupoles=None, exclude=None, dnux_dp=0.0, dnuy_dp=0.0, n_iterations=5,
    correction_fraction=0.9, tolerance=0.0,
    run_local=True, remote_opts=None, print_stdout=True, print_stderr=True):
    """"""

    if sextupoles is None:
        raise ValueError('"sextupoles" must be a list of strings.')
    used_sexts_str = ' '.join(sextupoles)

    if ele_filepath is None:
        tmp = tempfile.NamedTemporaryFile(
            dir=os.getcwd(), delete=False, prefix='tmpChroms_', suffix='.ele')
        ele_filepath = os.path.abspath(tmp.name)
        tmp.close()

    eb = elebuilder.EleContents(double_format='.12g')

    eb.run_setup(
        lattice=init_LTE_filepath, p_central_mev=E_MeV, use_beamline=use_beamline,
        parameters='%s.param')

    eb.newline()

    eb.run_control()

    eb.newline()

    eb.chromaticity(
        sextupoles=used_sexts_str, exclude=exclude,
        dnux_dp=dnux_dp, dnuy_dp=dnuy_dp,
        n_iterations=n_iterations, correction_fraction=correction_fraction,
        tolerance=tolerance, change_defined_values=True)

    eb.newline()

    eb.twiss_output()

    eb.newline()

    eb.bunched_beam()

    eb.newline()

    eb.track()

    eb.save_lattice(filename=corrected_LTE_filepath)

    eb.write(ele_filepath)

    eb.update_output_filepaths(ele_filepath[:-4]) # Remove ".ele"
    #print(eb.actual_output_filepath_list)

    # Run Elegant
    if run_local:
        run(ele_filepath, macros=macros, print_cmd=False,
            print_stdout=print_stdout, print_stderr=print_stderr)
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
                   print_stdout=print_stdout, print_stderr=print_stderr,
                   output_filepaths=None)

    if del_tmp_files:
        for fp in eb.actual_output_filepath_list + [ele_filepath]:
            if fp.startswith('/dev'):
                continue
            elif fp == corrected_LTE_filepath:
                continue
            else:
                try:
                    os.remove(fp)
                except:
                    print(f'Failed to delete "{fp}"')
