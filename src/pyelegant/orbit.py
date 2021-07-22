from typing import Union, Optional, Tuple, Iterable, Dict
import os
from pathlib import Path
import numpy as np
import matplotlib.pylab as plt
import tempfile
from copy import deepcopy

from .local import run
from .remote import remote
from . import __version__, std_print_enabled
from . import elebuilder
from . import util
from . import sdds
from . import twiss

def get_closed_orbit(
    ele_filepath: str, clo_output_filepath: str, param_output_filepath: str = '',
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
                sbatch={'use': False}, pelegant=False, job_name='clo',
                output='clo.%J.out', error='clo.%J.err',
                partition='normal', ntasks=1)

        remote.run(remote_opts, ele_filepath, print_cmd=True,
                   print_stdout=std_print_enabled['out'],
                   print_stderr=std_print_enabled['err'],
                   output_filepaths=None)

    tmp_filepaths = dict(clo=clo_output_filepath)
    if Path(param_output_filepath).exists():
        tmp_filepaths['param'] = param_output_filepath
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

    ed = elebuilder.EleDesigner(ele_filepath, double_format='.12g')

    if transmute_elements is not None:
        elebuilder.add_transmute_blocks(ed, transmute_elements)

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

    ed.add_block('track', soft_failure=False)

    ed.write()
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
                    sbatch={'use': False}, pelegant=False, job_name='clo',
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

    if output == {}:
        print('\n*** Closed orbit file could NOT be found, '
              'possibly due to closed orbit finding convergence failure. **\n')

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
                _version_ELEGANT=__version__['ELEGANT'],
                ),
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

def plot_closed_orbit(clo_columns: dict, clo_params: dict) -> None:
    """"""

    col = clo_columns
    par = clo_params

    plt.figure()
    plt.subplot(211)
    plt.plot(col['s'], col['x'] * 1e3, '.-')
    plt.grid(True)
    plt.ylabel(r'$x\, [\mathrm{mm}]$', size=20)
    plt.title((
        r'$\delta = {dp}\, (\mathrm{{Errors\, [m]:}}\, \Delta L = {dL},$'
        '\n'
        r'$\Delta x = {dx}, \Delta y = {dy})$'
        ).format(
            dp = util.pprint_sci_notation(par['delta'], '.3g'),
            dL = util.pprint_sci_notation(par['lengthError'], '.3g'),
            dx = util.pprint_sci_notation(par['xError'], '.3g'),
            dy = util.pprint_sci_notation(par['yError'], '.3g')),
    size=16)
    plt.subplot(212)
    plt.plot(col['s'], col['y'] * 1e3, '.-')
    plt.grid(True)
    plt.xlabel(r'$s\, [\mathrm{m}]$', size=20)
    plt.ylabel(r'$y\, [\mathrm{mm}]$', size=20)
    plt.tight_layout()

class ClosedOrbitCalculator:
    """
    If you observe too large closed orbit distortion (particularly horizontally)
    without any orbit kicks even after "closed_orbit_accuracy" is reduced, it is
    recommended to increase N_KICKS for CSBEND elements (e.g., 40).

    Also, "closed_orbit_iterations" being too small may generate an error message
    like the following:
      error: closed orbit did not converge to better than 4.494233e+307 after 40 iteration
    In this case, try to raise this value to a larger one like 200.
    """

    def __init__(
        self, LTE_filepath: str, E_MeV: float, fixed_length: bool = True,
        output_monitors_only: bool = True, closed_orbit_accuracy: float = 1e-12,
        closed_orbit_iterations: int = 40, iteration_fraction: float = 0.9,
        n_turns: int = 0, use_beamline: Optional[str] = None,
        N_KICKS: Optional[dict] = None, transmute_elements: Optional[dict] = None,
        ele_filepath: Optional[str] = None, tempdir_path: Optional[str] = None,
        fixed_lattice: bool = False) -> None:
        """Constructor
        If "fixed_lattice" is False, this object is meant to compute the closed
        orbit with the lattice defined in the LTE file "as is", without altering
        any corrector strengths.
        """

        assert n_turns >= 0
        assert iteration_fraction <= 1.0

        self.hcors = {}
        self.vcors = {}

        self._mod_props = {}
        self._checkpoints = {}

        self.make_tempdir(tempdir_path=tempdir_path)

        if ele_filepath is None:
            tmp = tempfile.NamedTemporaryFile(
                dir=self.tempdir.name, delete=False, prefix=f'tmpCO_',
                suffix='.ele')
            self.ele_filepath = os.path.abspath(tmp.name)
            tmp.close()
        else:
            self.ele_filepath = ele_filepath

        if self.ele_filepath.endswith('.ele'):
            self.param_filepath = self.ele_filepath[:-4] + '.param'
        else:
            self.param_filepath = self.ele_filepath + '.param'
        self._param_output = {}

        self.ed = ed = elebuilder.EleDesigner(self.ele_filepath, double_format='.12g')

        if transmute_elements is not None:
            elebuilder.add_transmute_blocks(ed, transmute_elements)

            ed.add_newline()

        ed.add_block('run_setup',
            lattice=Path(LTE_filepath).resolve(),
            p_central_mev=E_MeV,
            use_beamline=use_beamline,
            parameters=self.param_filepath)

        ed.add_newline()

        self.fixed_lattice = fixed_lattice

        if not fixed_lattice:
            load_parameters = dict(
                change_defined_values=True, allow_missing_elements=True,
                allow_missing_parameters=True)
            tmp = tempfile.NamedTemporaryFile(
                dir=self.tempdir.name, delete=False, prefix=f'tmpCorrSetpoints_',
                suffix='.sdds')
            load_parameters['filename'] = os.path.abspath(tmp.name)
            tmp.close()

            self.corrector_params_filepath = load_parameters['filename']

            ed.add_block('load_parameters', **load_parameters)

            ed.add_newline()

        ed.add_block('run_control', n_passes=1)

        ed.add_newline()

        if N_KICKS is not None:
            elebuilder.add_N_KICKS_alter_elements_blocks(ed, N_KICKS)

            ed.add_newline()

        _block_opts = dict(
            output='%s.clo', tracking_turns=n_turns,
            fixed_length=fixed_length, output_monitors_only=output_monitors_only,
            closed_orbit_accuracy=closed_orbit_accuracy,
            closed_orbit_iterations=closed_orbit_iterations,
            iteration_fraction=iteration_fraction,
        )
        ed.add_block('closed_orbit', **_block_opts)

        ed.add_newline()

        ed.add_block('bunched_beam')

        ed.add_newline()

        ed.add_block('track', soft_failure=False)

        ed.write()
        #print(ed.actual_output_filepath_list)

        for fp in ed.actual_output_filepath_list:
            if fp.endswith('.clo'):
                self.clo_output_filepath = fp
            elif fp.endswith('.param'):
                pass
            else:
                raise ValueError('This line should not be reached.')

    def __del__(self):
        """"""

        self.remove_tempdir()

    def make_tempdir(self, tempdir_path=None):
        """"""

        self.tempdir = tempfile.TemporaryDirectory(
            prefix='tmpClosedOrb_', dir=tempdir_path)

    def remove_tempdir(self):
        """"""

        self.tempdir.cleanup()

    def getLattice(self):
        """"""

        return self.ed.getLattice()

    def get_all_available_kickers(self, spos_sorted=True):
        """"""

        return self.ed.get_LTE_all_kickers(spos_sorted=spos_sorted)

    def select_kickers(self, plane: str, cor_names: Iterable[str]) -> None:
        """"""

        if self.fixed_lattice:
            raise RuntimeError(
                'This object was created with "fixed_lattice" set to True. So, '
                'this function is disabled.')

        if plane.lower() == 'h':

            self.hcors['kick_prop_names'] = []
            for elem_name in cor_names:
                elem_type = self.ed.get_LTE_elem_info(elem_name)['elem_type']

                if elem_type is None:
                    raise ValueError(
                        f'Element named "{elem_name}" does NOT exist in loaded LTE file.')

                if elem_type in ('HKICK', 'EHKICK'):
                    self.hcors['kick_prop_names'].append('KICK')
                elif elem_type in ('KICK', 'KICKER', 'EKICKER'):
                    self.hcors['kick_prop_names'].append('HKICK')
                else:
                    raise ValueError(
                        (f'Element "{elem_name}" is of type "{elem_type}". '
                        'Must be one of "KICK", "HKICK", "EHKICK", "KICKER", "EKICKER".'))

            self.hcors['names'] = np.array(cor_names)
            self.hcors['rads'] = np.zeros(len(cor_names))

        elif plane.lower() == 'v':

            self.vcors['kick_prop_names'] = []
            for elem_name in cor_names:
                elem_type = self.ed.get_LTE_elem_info(elem_name)['elem_type']

                if elem_type is None:
                    raise ValueError(
                        f'Element named "{elem_name}" does NOT exist in loaded LTE file.')

                if elem_type in ('VKICK', 'EVKICK'):
                    self.vcors['kick_prop_names'].append('KICK')
                elif elem_type in ('KICK', 'KICKER', 'EKICKER'):
                    self.vcors['kick_prop_names'].append('VKICK')
                else:
                    raise ValueError(
                        (f'Element "{elem_name}" is of type "{elem_type}". '
                        'Must be one of "KICK", "VKICK", "EVKICK", "KICKER", "EKICKER".'))

            self.vcors['names'] = np.array(cor_names)
            self.vcors['rads'] = np.zeros(len(cor_names))
        else:
            raise ValueError('"plane" must be either "h" or "v".')

    def get_selected_kickers(self, plane: str) -> dict:
        """"""

        if plane.lower() == 'h':
            return self.hcors
        elif plane.lower() == 'v':
            return self.vcors
        else:
            raise ValueError('"plane" must be either "h" or "v".')

    def set_kick_angles(self, hkick_rads, vkick_rads) -> None:
        """"""

        if self.fixed_lattice:
            raise RuntimeError(
                'This object was created with "fixed_lattice" set to True. So, '
                'this function is disabled.')

        assert len(hkick_rads) == len(self.hcors['names'])
        self.hcors['rads'] = hkick_rads

        assert len(vkick_rads) == len(self.vcors['rads'])
        self.vcors['rads'] = vkick_rads

        elem_names = np.append(self.hcors['names'], self.vcors['names'])
        elem_prop_names = self.hcors['kick_prop_names'] + self.vcors['kick_prop_names']
        elem_prop_vals = np.append(self.hcors['rads'], self.vcors['rads'])

        assert len(elem_names) == len(elem_prop_names) == len(elem_prop_vals)

        for elem_name, param_name, param_val in zip(
            elem_names, elem_prop_names, elem_prop_vals):

            self._mod_props[(elem_name, param_name)] = param_val

        self._write_to_parameters_file()

    def get_kick_angles(self) -> Dict:
        """"""

        return dict(h=self.hcors['rads'], v=self.vcors['rads'])

    def set_elem_properties(self, elem_names, elem_prop_names, elem_prop_vals):
        """"""

        if self.fixed_lattice:
            raise RuntimeError(
                'This object was created with "fixed_lattice" set to True. So, '
                'this function is disabled.')

        assert len(elem_names) == len(elem_prop_names) == len(elem_prop_vals)

        for elem_name, param_name, param_val in zip(
            elem_names, elem_prop_names, elem_prop_vals):

            k = (elem_name, param_name)
            self._mod_props[k] = param_val

        self._write_to_parameters_file()

    def get_modified_elem_properties(self, elem_names, elem_prop_names,
                                     ret_dict=False):
        """"""

        assert len(elem_names) == len(elem_prop_names)

        if ret_dict:
            out = {}
        else:
            out = np.full((len(elem_names),), np.nan)

        for i, (elem_name, param_name) in enumerate(zip(
            elem_names, elem_prop_names)):

            k = (elem_name, param_name)

            if k in self._mod_props:
                val = self._mod_props[k]
            else:
                val = np.nan

            if ret_dict:
                out[k] = val
            else:
                out[i] = val

        return out

    def get_elem_properties(self, elem_names, elem_prop_names, ret_dict=False):
        """"""

        assert len(elem_names) == len(elem_prop_names)

        if ret_dict:
            out = {}
        else:
            out = np.full((len(elem_names),), np.nan)

        for i, (elem_name, param_name) in enumerate(zip(
            elem_names, elem_prop_names)):

            k = (elem_name, param_name)

            try:
                val = self._param_output['ParameterValue'][
                    (self._param_output['ElementName'] == elem_name) &
                    (self._param_output['ElementParameter'] == param_name)][0]
            except:
                val = np.nan

            if ret_dict:
                out[k] = val
            else:
                out[i] = val

        return out

    def _write_to_parameters_file(self):
        """"""

        col = dict(ElementName=[], ElementParameter=[], ParameterValue=[])
        for (elem_name, param_name), param_val in self._mod_props.items():
            col['ElementName'].append(elem_name)
            col['ElementParameter'].append(param_name)
            col['ParameterValue'].append(param_val)

        sdds.dicts2sdds(
            self.corrector_params_filepath, params=None, columns=col,
            outputMode='binary', suppress_err_msg=True)

    def get_elem_names_by_regex(self, pattern, spos_sorted=False):
        """"""

        return self.ed.get_LTE_elem_names_by_regex(pattern,
                                                   spos_sorted=spos_sorted)

    def get_elem_names_types_by_regex(self, pattern, spos_sorted=False):
        """"""

        return self.ed.get_LTE_elem_names_types_by_regex(
            pattern, spos_sorted=spos_sorted)

    def get_elem_names_for_elem_type(self, sel_elem_type, spos_sorted=False):
        """"""

        return self.ed.get_LTE_elem_names_for_elem_type(sel_elem_type,
                                                        spos_sorted=spos_sorted)

    def calc(self, run_local: bool = True, remote_opts: Optional[dict] = None,
             ) -> dict:
        """"""

        data, meta= get_closed_orbit(
            self.ele_filepath, self.clo_output_filepath,
            param_output_filepath=self.param_filepath,
            run_local=run_local, remote_opts=remote_opts)

        if data == {}:
            # *** Closed orbit file could NOT be found,
            #    'possibly due to closed orbit finding convergence failure. ***
            self.clo_columns = {}
            self.clo_params = {}
        else:
            self.clo_columns = data['clo']['columns']
            self.clo_params = data['clo']['params']

            self._param_output = data['param']['columns']

        return dict(columns=self.clo_columns, params=self.clo_params)

    def plot(self):
        """"""

        plot_closed_orbit(self.clo_columns, self.clo_params)

    def save_checkpoint(self, name='last'):
        """"""

        self._checkpoints[name] = deepcopy(self._mod_props)

    def load_checkpoint(self, name='last'):
        """"""

        self._mod_props.clear()
        self._mod_props.update(self._checkpoints[name])

        self._write_to_parameters_file()

def calcAnalyticalORM(
    LTE_filepath, E_MeV, bpm_elem_inds_or_name_occur_tups,
    hcor_elem_inds_or_name_occur_tups, vcor_elem_inds_or_name_occur_tups,
    use_beamline=None):
    """
    No transverse coupling assumed
    """

    with tempfile.NamedTemporaryFile(suffix='.pgz') as f:
        output_filepath = f.name
        twiss.calc_ring_twiss(
            output_filepath, LTE_filepath, E_MeV, use_beamline=use_beamline)

        twi = util.load_pgz_file(output_filepath)

    twi_arrays = twi['data']['twi']['arrays']
    twi_scalars = twi['data']['twi']['scalars']

    betax = twi_arrays['betax']
    betay = twi_arrays['betay']
    phix = twi_arrays['psix'] # [rad]
    phiy = twi_arrays['psiy'] # [rad]
    etax = twi_arrays['etax']
    etay = twi_arrays['etay']
    nux, nuy = twi_scalars['nux'], twi_scalars['nuy']
    alphac = twi_scalars['alphac']

    circumf = twi_arrays['s'][-1]

    if isinstance(bpm_elem_inds_or_name_occur_tups[0], (int, np.integer)):
        bpm_inds = bpm_elem_inds_or_name_occur_tups
    else:
        bpm_inds = np.array([
            np.where((twi_arrays['ElementName'] == name) &
                     (twi_arrays['ElementOccurence'] == occur))[0][0]
            for name, occur in bpm_elem_inds_or_name_occur_tups])
    nBPM = len(bpm_inds)

    if isinstance(hcor_elem_inds_or_name_occur_tups[0], (int, np.integer)):
        hcor_inds = np.array(hcor_elem_inds_or_name_occur_tups)
    else:
        hcor_inds = np.array([
            np.where((twi_arrays['ElementName'] == name) &
                     (twi_arrays['ElementOccurence'] == occur))[0][0]
            for name, occur in hcor_elem_inds_or_name_occur_tups])
    nHCOR = len(hcor_inds)

    if isinstance(vcor_elem_inds_or_name_occur_tups[0], (int, np.integer)):
        vcor_inds = np.array(vcor_elem_inds_or_name_occur_tups)
    else:
        vcor_inds = np.array([
            np.where((twi_arrays['ElementName'] == name) &
                     (twi_arrays['ElementOccurence'] == occur))[0][0]
            for name, occur in vcor_elem_inds_or_name_occur_tups])
    nVCOR = len(vcor_inds)

    M = np.zeros((2*nBPM, nHCOR+nVCOR))

    for i in range(nBPM):
        cor_betax_avg = (betax[hcor_inds-1] + betax[hcor_inds]) / 2
        cor_phix_avg = (phix[hcor_inds-1] + phix[hcor_inds]) / 2
        cor_etax_avg = (etax[hcor_inds-1] + etax[hcor_inds]) / 2
        M[i,:nHCOR] = np.sqrt(cor_betax_avg * betax[bpm_inds[i]]) * np.cos(
            np.pi * nux - np.abs(cor_phix_avg - phix[bpm_inds[i]])) / (
                2 * np.sin(np.pi * nux)
            ) + cor_etax_avg * etax[bpm_inds[i]] / alphac / circumf

        cor_betay_avg = (betay[vcor_inds-1] + betay[vcor_inds]) / 2
        cor_phiy_avg = (phiy[vcor_inds-1] + phiy[vcor_inds]) / 2
        cor_etay_avg = (etay[vcor_inds-1] + etay[vcor_inds]) / 2
        M[i+nBPM,nHCOR:] = np.sqrt(cor_betay_avg * betay[bpm_inds[i]]) * np.cos(
            np.pi * nuy - np.abs(cor_phiy_avg - phiy[bpm_inds[i]])) / (
                2 * np.sin(np.pi * nuy)
            ) + cor_etay_avg * etay[bpm_inds[i]] / alphac / circumf

    return M