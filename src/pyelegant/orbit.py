from collections import defaultdict
import contextlib
from copy import deepcopy
import os
from pathlib import Path
from subprocess import PIPE, Popen
import sys
import tempfile
import time
from typing import Dict, Iterable, List, Optional, Tuple, Union

import h5py
import matplotlib.pylab as plt
import numpy as np
from scipy import constants, optimize

from . import __version__, elebuilder, sdds, std_print_enabled, twiss, util
from .local import run
from .ltemanager import Lattice, get_ELEGANT_element_dictionary
from .remote import remote
from .respmat import calcSVD, calcTruncSVMatrix


class DummyFile(object):
    def write(self, x):
        pass


@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout


def get_closed_orbit(
    ele_filepath: str,
    clo_output_filepath: str,
    param_output_filepath: str = "",
    run_local: bool = True,
    remote_opts: Optional[dict] = None,
) -> Tuple[dict]:
    """"""

    # Run Elegant
    if run_local:
        run(
            ele_filepath,
            print_cmd=False,
            print_stdout=std_print_enabled["out"],
            print_stderr=std_print_enabled["err"],
        )
    else:
        if remote_opts is None:
            remote_opts = dict(
                sbatch={"use": False},
                pelegant=False,
                job_name="clo",
                output="clo.%J.out",
                error="clo.%J.err",
                partition="normal",
                ntasks=1,
            )

        remote.run(
            remote_opts,
            ele_filepath,
            print_cmd=True,
            print_stdout=std_print_enabled["out"],
            print_stderr=std_print_enabled["err"],
            output_filepaths=None,
        )

    tmp_filepaths = dict(clo=clo_output_filepath)
    if Path(param_output_filepath).exists():
        tmp_filepaths["param"] = param_output_filepath
    output, meta = {}, {}
    for k, v in tmp_filepaths.items():
        try:
            output[k], meta[k] = sdds.sdds2dicts(v)
        except:
            continue

    return output, meta


def calc_closed_orbit(
    output_filepath: str,
    LTE_filepath: str,
    E_MeV: float,
    fixed_length: bool = True,
    output_monitors_only: bool = False,
    closed_orbit_accuracy: float = 1e-12,
    closed_orbit_iterations: int = 40,
    iteration_fraction: float = 0.9,
    n_turns: int = 1,
    load_parameters: Optional[dict] = None,
    reuse_ele: bool = False,
    use_beamline: Optional[str] = None,
    N_KICKS: Optional[dict] = None,
    transmute_elements: Optional[dict] = None,
    ele_filepath: Optional[str] = None,
    output_file_type: Optional[str] = None,
    del_tmp_files: bool = True,
    run_local: bool = True,
    remote_opts: Optional[dict] = None,
) -> Union[str, dict]:
    """"""

    assert n_turns >= 1
    assert iteration_fraction <= 1.0

    with open(LTE_filepath, "r") as f:
        file_contents = f.read()

    input_dict = dict(
        LTE_filepath=os.path.abspath(LTE_filepath),
        E_MeV=E_MeV,
        fixed_length=fixed_length,
        output_monitors_only=output_monitors_only,
        closed_orbit_accuracy=closed_orbit_accuracy,
        closed_orbit_iterations=closed_orbit_iterations,
        n_turns=n_turns,
        load_parameters=load_parameters,
        reuse_ele=reuse_ele,
        use_beamline=use_beamline,
        N_KICKS=N_KICKS,
        transmute_elements=transmute_elements,
        ele_filepath=ele_filepath,
        del_tmp_files=del_tmp_files,
        run_local=run_local,
        remote_opts=remote_opts,
        lattice_file_contents=file_contents,
        timestamp_ini=util.get_current_local_time_str(),
    )

    output_file_type = util.auto_check_output_file_type(
        output_filepath, output_file_type
    )
    input_dict["output_file_type"] = output_file_type

    if output_file_type in ("hdf5", "h5"):
        util.save_input_to_hdf5(output_filepath, input_dict)

    if ele_filepath is None:
        tmp = tempfile.NamedTemporaryFile(
            dir=os.getcwd(), delete=False, prefix=f"tmpCO_", suffix=".ele"
        )
        ele_filepath = os.path.abspath(tmp.name)
        tmp.close()

    ed = elebuilder.EleDesigner(ele_filepath, double_format=".12g")

    if transmute_elements is not None:
        elebuilder.add_transmute_blocks(ed, transmute_elements)

        ed.add_newline()

    ed.add_block(
        "run_setup",
        lattice=LTE_filepath,
        p_central_mev=E_MeV,
        use_beamline=use_beamline,
    )

    ed.add_newline()

    if load_parameters is not None:
        load_parameters["change_defined_values"] = True
        load_parameters.setdefault("allow_missing_elements", True)
        load_parameters.setdefault("allow_missing_parameters", True)
        ed.add_block("load_parameters", **load_parameters)

        ed.add_newline()

    ed.add_block("run_control", n_passes=n_turns)

    ed.add_newline()

    if N_KICKS is not None:
        elebuilder.add_N_KICKS_alter_elements_blocks(ed, N_KICKS)

        ed.add_newline()

    _block_opts = dict(
        output="%s.clo",
        tracking_turns=(False if n_turns == 1 else True),
        fixed_length=fixed_length,
        output_monitors_only=output_monitors_only,
        closed_orbit_accuracy=closed_orbit_accuracy,
        closed_orbit_iterations=closed_orbit_iterations,
        iteration_fraction=iteration_fraction,
    )
    ed.add_block("closed_orbit", **_block_opts)

    ed.add_newline()

    ed.add_block("bunched_beam")

    ed.add_newline()

    ed.add_block("track", soft_failure=False)

    ed.write()
    # print(ed.actual_output_filepath_list)

    for fp in ed.actual_output_filepath_list:
        if fp.endswith(".clo"):
            clo_output_filepath = fp
        elif fp.endswith(".done"):
            done_filepath = fp
        else:
            raise ValueError("This line should not be reached.")

    if False:
        # Run Elegant
        if run_local:
            run(
                ele_filepath,
                print_cmd=False,
                print_stdout=std_print_enabled["out"],
                print_stderr=std_print_enabled["err"],
            )
        else:
            if remote_opts is None:
                remote_opts = dict(
                    sbatch={"use": False},
                    pelegant=False,
                    job_name="clo",
                    output="clo.%J.out",
                    error="clo.%J.err",
                    partition="normal",
                    ntasks=1,
                )

            remote.run(
                remote_opts,
                ele_filepath,
                print_cmd=True,
                print_stdout=std_print_enabled["out"],
                print_stderr=std_print_enabled["err"],
                output_filepaths=None,
            )

        tmp_filepaths = dict(clo=clo_output_filepath)
        output, meta = {}, {}
        for k, v in tmp_filepaths.items():
            try:
                output[k], meta[k] = sdds.sdds2dicts(v)
            except:
                continue
    else:
        output, meta = get_closed_orbit(
            ele_filepath,
            clo_output_filepath,
            run_local=run_local,
            remote_opts=remote_opts,
        )

    timestamp_fin = util.get_current_local_time_str()

    if output == {}:
        print(
            "\n*** Closed orbit file could NOT be found, "
            "possibly due to closed orbit finding convergence failure. **\n"
        )

    if output_file_type in ("hdf5", "h5"):
        util.robust_sdds_hdf5_write(
            output_filepath, [output, meta], nMaxTry=10, sleep=10.0, mode="a"
        )
        f = h5py.File(output_filepath, "a")
        f["timestamp_fin"] = timestamp_fin
        f.close()

    elif output_file_type == "pgz":
        mod_output = {}
        for k, v in output.items():
            mod_output[k] = {}
            if "params" in v:
                mod_output[k]["scalars"] = v["params"]
            if "columns" in v:
                mod_output[k]["arrays"] = v["columns"]
        mod_meta = {}
        for k, v in meta.items():
            mod_meta[k] = {}
            if "params" in v:
                mod_meta[k]["scalars"] = v["params"]
            if "columns" in v:
                mod_meta[k]["arrays"] = v["columns"]
        util.robust_pgz_file_write(
            output_filepath,
            dict(
                data=mod_output,
                meta=mod_meta,
                input=input_dict,
                timestamp_fin=timestamp_fin,
                _version_PyELEGANT=__version__["PyELEGANT"],
                _version_ELEGANT=__version__["ELEGANT"],
            ),
            nMaxTry=10,
            sleep=10.0,
        )
    else:
        raise ValueError()

    if del_tmp_files:
        files_to_be_deleted = ed.actual_output_filepath_list[:]
        if not reuse_ele:
            files_to_be_deleted += [ele_filepath]

        for fp in files_to_be_deleted:
            if fp.startswith("/dev"):
                continue
            else:
                try:
                    os.remove(fp)
                except:
                    print(f'Failed to delete "{fp}"')

    if reuse_ele:
        return dict(
            output_filepath=output_filepath,
            ele_filepath=ele_filepath,
            clo_output_filepath=clo_output_filepath,
        )
    else:
        return output_filepath


def plot_closed_orbit(clo_columns: dict, clo_params: dict) -> None:
    """"""

    col = clo_columns
    par = clo_params

    plt.figure()
    plt.subplot(211)
    plt.plot(col["s"], col["x"] * 1e3, ".-")
    plt.grid(True)
    plt.ylabel(r"$x\, [\mathrm{mm}]$", size=20)
    plt.title(
        (
            r"$\delta = {dp}\, (\mathrm{{Errors\, [m]:}}\, \Delta L = {dL},$"
            "\n"
            r"$\Delta x = {dx}, \Delta y = {dy})$"
        ).format(
            dp=util.pprint_sci_notation(par["delta"], ".3g"),
            dL=util.pprint_sci_notation(par["lengthError"], ".3g"),
            dx=util.pprint_sci_notation(par["xError"], ".3g"),
            dy=util.pprint_sci_notation(par["yError"], ".3g"),
        ),
        size=16,
    )
    plt.subplot(212)
    plt.plot(col["s"], col["y"] * 1e3, ".-")
    plt.grid(True)
    plt.xlabel(r"$s\, [\mathrm{m}]$", size=20)
    plt.ylabel(r"$y\, [\mathrm{mm}]$", size=20)
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
        self,
        LTE_filepath: Union[Path, str],
        E_MeV: float,
        fixed_length: bool = True,
        output_monitors_only: bool = True,
        closed_orbit_accuracy: float = 1e-12,
        closed_orbit_iterations: int = 40,
        iteration_fraction: float = 0.9,
        n_turns: int = 0,
        use_beamline: Optional[str] = None,
        N_KICKS: Optional[dict] = None,
        transmute_elements: Optional[dict] = None,
        ele_filepath: Optional[str] = None,
        tempdir_path: Optional[str] = None,
        fixed_lattice: bool = False,
    ) -> None:
        """Constructor
        If "fixed_lattice" is False, this object is meant to compute the closed
        orbit with the lattice defined in the LTE file "as is", without altering
        any corrector strengths.
        """

        LTE_filepath = Path(LTE_filepath)

        assert n_turns >= 0
        assert iteration_fraction <= 1.0

        self.hcors = {}
        self.vcors = {}

        self._mod_props = {}
        self._checkpoints = {}

        self.make_tempdir(tempdir_path=tempdir_path)

        if ele_filepath is None:
            tmp = tempfile.NamedTemporaryFile(
                dir=self.tempdir.name, delete=False, prefix=f"tmpCO_", suffix=".ele"
            )
            self.ele_filepath = os.path.abspath(tmp.name)
            tmp.close()
        else:
            self.ele_filepath = ele_filepath

        if self.ele_filepath.endswith(".ele"):
            self.param_filepath = self.ele_filepath[:-4] + ".param"
        else:
            self.param_filepath = self.ele_filepath + ".param"
        self._param_output = {}

        self.ed = ed = elebuilder.EleDesigner(self.ele_filepath, double_format=".16g")

        if transmute_elements is not None:
            elebuilder.add_transmute_blocks(ed, transmute_elements)

            ed.add_newline()

        ed.add_block(
            "run_setup",
            lattice=str(LTE_filepath.resolve()),
            p_central_mev=E_MeV,
            use_beamline=use_beamline,
            parameters=self.param_filepath,
        )

        ed.add_newline()

        self.fixed_lattice = fixed_lattice

        if not fixed_lattice:
            load_parameters = dict(
                change_defined_values=True,
                allow_missing_elements=True,
                allow_missing_parameters=True,
            )
            tmp = tempfile.NamedTemporaryFile(
                dir=self.tempdir.name,
                delete=False,
                prefix=f"tmpCorrSetpoints_",
                suffix=".sdds",
            )
            load_parameters["filename"] = os.path.abspath(tmp.name)
            tmp.close()

            self.corrector_params_filepath = load_parameters["filename"]

            ed.add_block("load_parameters", **load_parameters)

            ed.add_newline()

        ed.add_block("run_control", n_passes=1)

        ed.add_newline()

        if N_KICKS is not None:
            elebuilder.add_N_KICKS_alter_elements_blocks(ed, N_KICKS)

            ed.add_newline()

        _block_opts = dict(
            output="%s.clo",
            tracking_turns=n_turns,
            fixed_length=fixed_length,
            output_monitors_only=output_monitors_only,
            closed_orbit_accuracy=closed_orbit_accuracy,
            closed_orbit_iterations=closed_orbit_iterations,
            iteration_fraction=iteration_fraction,
        )
        ed.add_block("closed_orbit", **_block_opts)

        ed.add_newline()

        ed.add_block("bunched_beam")

        ed.add_newline()

        ed.add_block("track", soft_failure=False)

        ed.write()
        # print(ed.actual_output_filepath_list)

        for fp in ed.actual_output_filepath_list:
            if fp.endswith(".clo"):
                self.clo_output_filepath = fp
            elif fp.endswith(".param"):
                pass
            else:
                raise ValueError("This line should not be reached.")

    def __del__(self):
        """"""

        self.remove_tempdir()

    def make_tempdir(self, tempdir_path=None):
        """"""

        self.tempdir = tempfile.TemporaryDirectory(
            prefix="tmpClosedOrb_", dir=tempdir_path
        )

    def remove_tempdir(self):
        """"""

        if not hasattr(self, "tempdir"):
            return

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
                "this function is disabled."
            )

        if plane.lower() in ("x", "h"):  # "h" kept for backward compatibility
            self.hcors["kick_prop_names"] = []
            for elem_name in cor_names:
                elem_type = self.ed.get_LTE_elem_info(elem_name)["elem_type"]

                if elem_type is None:
                    raise ValueError(
                        f'Element named "{elem_name}" does NOT exist in loaded LTE file.'
                    )

                if elem_type in ("HKICK", "EHKICK"):
                    self.hcors["kick_prop_names"].append("KICK")
                elif elem_type in ("KICK", "KICKER", "EKICKER"):
                    self.hcors["kick_prop_names"].append("HKICK")
                else:
                    raise ValueError(
                        (
                            f'Element "{elem_name}" is of type "{elem_type}". '
                            'Must be one of "KICK", "HKICK", "EHKICK", "KICKER", "EKICKER".'
                        )
                    )

            self.hcors["names"] = np.array(cor_names)
            self.hcors["rads"] = np.zeros(len(cor_names))

        elif plane.lower() in ("y", "v"):  # "v" kept for backward compatibility
            self.vcors["kick_prop_names"] = []
            for elem_name in cor_names:
                elem_type = self.ed.get_LTE_elem_info(elem_name)["elem_type"]

                if elem_type is None:
                    raise ValueError(
                        f'Element named "{elem_name}" does NOT exist in loaded LTE file.'
                    )

                if elem_type in ("VKICK", "EVKICK"):
                    self.vcors["kick_prop_names"].append("KICK")
                elif elem_type in ("KICK", "KICKER", "EKICKER"):
                    self.vcors["kick_prop_names"].append("VKICK")
                else:
                    raise ValueError(
                        (
                            f'Element "{elem_name}" is of type "{elem_type}". '
                            'Must be one of "KICK", "VKICK", "EVKICK", "KICKER", "EKICKER".'
                        )
                    )

            self.vcors["names"] = np.array(cor_names)
            self.vcors["rads"] = np.zeros(len(cor_names))
        else:
            raise ValueError('"plane" must be either "x" or "y".')

    def get_selected_kickers(self, plane: str) -> dict:
        """"""

        if plane.lower() in ("x", "h"):  # "h" kept for backward compatibility
            return self.hcors
        elif plane.lower() in ("y", "v"):  # "v" kept for backward compatibility:
            return self.vcors
        else:
            raise ValueError('"plane" must be either "x" or "y".')

    def set_kick_angles(self, hkick_rads, vkick_rads) -> None:
        """"""

        if self.fixed_lattice:
            raise RuntimeError(
                'This object was created with "fixed_lattice" set to True. So, '
                "this function is disabled."
            )

        assert len(hkick_rads) == len(self.hcors["names"])
        self.hcors["rads"] = hkick_rads

        assert len(vkick_rads) == len(self.vcors["rads"])
        self.vcors["rads"] = vkick_rads

        elem_names = np.append(self.hcors["names"], self.vcors["names"])
        elem_prop_names = self.hcors["kick_prop_names"] + self.vcors["kick_prop_names"]
        elem_prop_vals = np.append(self.hcors["rads"], self.vcors["rads"])

        assert len(elem_names) == len(elem_prop_names) == len(elem_prop_vals)

        for elem_name, param_name, param_val in zip(
            elem_names, elem_prop_names, elem_prop_vals
        ):
            self._mod_props[(elem_name, param_name)] = param_val

        self._write_to_parameters_file()

    def get_kick_angles(self) -> Dict:
        """"""

        return dict(x=self.hcors["rads"], y=self.vcors["rads"])

    def set_elem_properties(self, elem_names, elem_prop_names, elem_prop_vals):
        """"""

        if self.fixed_lattice:
            raise RuntimeError(
                'This object was created with "fixed_lattice" set to True. So, '
                "this function is disabled."
            )

        assert len(elem_names) == len(elem_prop_names) == len(elem_prop_vals)

        for elem_name, param_name, param_val in zip(
            elem_names, elem_prop_names, elem_prop_vals
        ):
            k = (elem_name, param_name)
            self._mod_props[k] = param_val

        self._write_to_parameters_file()

    def get_modified_elem_properties(self, elem_names, elem_prop_names, ret_dict=False):
        """"""

        assert len(elem_names) == len(elem_prop_names)

        if ret_dict:
            out = {}
        else:
            out = np.full((len(elem_names),), np.nan)

        for i, (elem_name, param_name) in enumerate(zip(elem_names, elem_prop_names)):
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

        for i, (elem_name, param_name) in enumerate(zip(elem_names, elem_prop_names)):
            k = (elem_name, param_name)

            try:
                val = self._param_output["ParameterValue"][
                    (self._param_output["ElementName"] == elem_name)
                    & (self._param_output["ElementParameter"] == param_name)
                ][0]
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
            col["ElementName"].append(elem_name)
            col["ElementParameter"].append(param_name)
            col["ParameterValue"].append(param_val)

        sdds.dicts2sdds(
            self.corrector_params_filepath,
            params=None,
            columns=col,
            outputMode="binary",
            suppress_err_msg=True,
        )

    def get_elem_names_by_regex(self, pattern, spos_sorted=False):
        """"""

        return self.ed.get_LTE_elem_names_by_regex(pattern, spos_sorted=spos_sorted)

    def get_elem_names_types_by_regex(self, pattern, spos_sorted=False):
        """"""

        return self.ed.get_LTE_elem_names_types_by_regex(
            pattern, spos_sorted=spos_sorted
        )

    def get_elem_names_for_elem_type(self, sel_elem_type, spos_sorted=False):
        """"""

        return self.ed.get_LTE_elem_names_for_elem_type(
            sel_elem_type, spos_sorted=spos_sorted
        )

    def calc(
        self,
        run_local: bool = True,
        remote_opts: Optional[Dict] = None,
    ) -> Dict:
        """"""

        data, meta = get_closed_orbit(
            self.ele_filepath,
            self.clo_output_filepath,
            param_output_filepath=self.param_filepath,
            run_local=run_local,
            remote_opts=remote_opts,
        )

        if "param" in data:
            self._param_output = data["param"]["columns"]
        else:
            self._param_output = {}

        if "clo" in data:
            self.clo_columns = data["clo"]["columns"]
            self.clo_params = data["clo"]["params"]
        else:
            # *** Closed orbit file could NOT be found,
            #    'possibly due to closed orbit finding convergence failure. ***
            self.clo_columns = {}
            self.clo_params = {}

        return dict(columns=self.clo_columns, params=self.clo_params)

    def plot(self):
        """"""

        plot_closed_orbit(self.clo_columns, self.clo_params)

    def save_checkpoint(self, name="last"):
        """"""

        self._checkpoints[name] = deepcopy(self._mod_props)

    def load_checkpoint(self, name="last"):
        """"""

        self._mod_props.clear()
        self._mod_props.update(self._checkpoints[name])

        self._write_to_parameters_file()


class ClosedOrbitCalculatorViaTraj:
    def __init__(
        self,
        LTE: Lattice,
        E_MeV: float,
        N_KICKS: Optional[dict] = None,
        x0: float = 0.0,
        xp0: float = 0.0,
        y0: float = 0.0,
        yp0: float = 0.0,
        dp0: float = 0.0,
        fixed_length=True,
        n_iter_max: int = 200,
        transv_cor_frac: float = 0.5,
        alphac: Union[None, float] = None,
        dp_cor_frac: float = 0.8,
        dp_change_thresh: float = 1e-7,
        tempdir_path: Optional[str] = None,
        print_stdout=False,
        print_stderr=True,
    ) -> None:
        std_print_enabled["out"] = print_stdout
        std_print_enabled["err"] = print_stderr

        assert isinstance(LTE, Lattice)
        self.LTE = LTE
        self.LTE_filepath = LTE.LTE_filepath
        self.N_KICKS = N_KICKS

        self.E_MeV = E_MeV

        self.ini_inj_coords = np.array([x0, xp0, y0, yp0, dp0])

        self.fixed_length = fixed_length
        if self.fixed_length:
            self.alphac = alphac
            self.dp_cor_frac = dp_cor_frac
            self.dp_change_thresh = dp_change_thresh

        self.n_iter_max = n_iter_max

        self.transv_cor_frac = transv_cor_frac

        self.n_turns = 2

        self.make_tempdir(tempdir_path=tempdir_path)

    def make_tempdir(self, tempdir_path=None):
        if isinstance(tempdir_path, Path):
            if tempdir_path.exists():
                self.tempdir = tempdir_path
                return
            else:
                tempdir_path = str(tempdir_path)
        elif isinstance(tempdir_path, str):
            tempdir_path = Path(tempdir_path)
            if tempdir_path.exists():
                self.tempdir = tempdir_path
                return
            else:
                tempdir_path = str(tempdir_path)

        self.tempdir = tempfile.TemporaryDirectory(prefix="tmpCOThr_", dir=tempdir_path)

        self._write_ele_files()

    def remove_tempdir(self):

        if not hasattr(self, "tempdir"):
            return

        if isinstance(self.tempdir, Path):
            try:
                self.tempdir.rmdir()
            except:
                pass
        else:
            self.tempdir.cleanup()

    def __del__(self):
        self.remove_tempdir()

    def _write_ele_files(self):
        """
        Write ELE files for
        1) computing trajectory
        """
        tmp = tempfile.NamedTemporaryFile(
            dir=self.tempdir.name, delete=False, prefix=f"tmpTraj_", suffix=".ele"
        )
        self.traj_calc_ele_path = Path(tmp.name).resolve()
        tmp.close()

        self._write_traj_calc_ele()

    def _write_traj_calc_ele(self):
        ed = elebuilder.EleDesigner(self.traj_calc_ele_path, double_format=".16g")

        ed.add_block(
            "run_setup",
            lattice=str(self.LTE_filepath),
            p_central_mev=self.E_MeV,
            use_beamline=self.LTE.used_beamline_name,
        )

        ed.add_newline()

        # Note that the output of the WATCH element data will be always with
        # respect to the design orbit. This means the even if MONI elements are
        # defined with non-zero DX and/or DY values, those values will be ignored.
        name = self.LTE.flat_used_elem_names[0]
        assert self.LTE.is_unique_elem_name(name)
        watch_pathobj = self.traj_calc_ele_path.with_suffix(".wc000")
        temp_watch_elem_name = f"ELEGANT_WATCH_000"
        watch_filepath = watch_pathobj.resolve()
        temp_watch_elem_def = (
            f'{temp_watch_elem_name}: WATCH, FILENAME="{watch_filepath}", '
            "MODE=coordinate"
        )
        ed.add_block(
            "insert_elements",
            name=name,
            element_def=temp_watch_elem_def,
        )

        ed.add_newline()

        if self.N_KICKS is not None:
            elebuilder.add_N_KICKS_alter_elements_blocks(ed, self.N_KICKS)

            ed.add_newline()

        ed.add_block("run_control", n_passes=self.n_turns)

        ed.add_newline()

        centroid = {}
        centroid[0] = "<x0>"
        centroid[1] = "<xp0>"
        centroid[2] = "<y0>"
        centroid[3] = "<yp0>"
        centroid[5] = "<dp0>"
        #
        ed.add_block("bunched_beam", n_particles_per_bunch=1, centroid=centroid)

        ed.add_newline()

        ed.add_block("track")

        ed.write()

    def _calc_traj(
        self, x0=0.0, y0=0.0, xp0=0.0, yp0=0.0, dp0=0.0, dt=False, debug_print=False
    ):
        assert self.n_turns == 2

        run(
            self.traj_calc_ele_path,
            print_cmd=False,
            macros=dict(
                x0=f"{x0:.16g}",
                y0=f"{y0:.16g}",
                xp0=f"{xp0:.16g}",
                yp0=f"{yp0:.16g}",
                dp0=f"{dp0:.16g}",
            ),
            print_stdout=std_print_enabled["out"],
            print_stderr=std_print_enabled["err"],
        )

        watch_filepaths = list(self.traj_calc_ele_path.parent.glob("*.wc*"))
        assert len(watch_filepaths) == 1

        data, meta = sdds.sdds2dicts(watch_filepaths[0])
        col = data["columns"]

        traj = {k: col[k] for k in ["x", "xp", "y", "yp"]}
        if dt:
            traj["dt"] = col["dt"]

        return traj

    def _calc_2turn_diff(self, x0_xp0_y0_yp0, dp0: float, dt: bool):
        traj = self._calc_traj(
            x0=x0_xp0_y0_yp0[0],
            xp0=x0_xp0_y0_yp0[1],
            y0=x0_xp0_y0_yp0[2],
            yp0=x0_xp0_y0_yp0[3],
            dp0=dp0,
            dt=dt,
            debug_print=False,
        )

        if traj["x"].size == 1:
            return np.nan
        assert traj["x"].size == self.n_turns == 2

        if dt:
            self.dt = traj["dt"][1]

        diff_sq = np.diff(traj["x"])[0] ** 2
        diff_sq += np.diff(traj["xp"])[0] ** 2
        diff_sq += np.diff(traj["y"])[0] ** 2
        diff_sq += np.diff(traj["yp"])[0] ** 2

        return np.sqrt(diff_sq)

    def calc(self, debug_print: bool = False) -> Dict:
        inj_coords = self.ini_inj_coords.copy()

        if self.fixed_length:
            circumf = self.LTE.get_circumference()
            c = constants.c  # speed of light [m/s]

        if self.alphac is None:
            x0, xp0, y0, yp0, dp0 = inj_coords
            base_traj = self._calc_traj(
                x0=x0, xp0=xp0, y0=y0, yp0=yp0, dp0=dp0, dt=True
            )
            base_diff = {
                coord: np.diff(base_traj[coord])[0] for coord in base_traj.keys()
            }

            dp_change = 1e-6

            traj = self._calc_traj(
                x0=x0, y0=y0, xp0=xp0, yp0=yp0, dp0=dp0 + dp_change, dt=True
            )
            self.alphac = (
                (np.diff(traj["dt"])[0] - np.diff(base_traj["dt"])[0])
                * c
                / circumf
                / dp_change
            )

        use_svd = True

        if not use_svd:  # This is super slow. Don't use it.
            COD_converged = True
            for i_iter in range(self.n_iter_max):
                if debug_print:
                    print(f"Iteration #{i_iter+1}/{self.n_iter_max}")

                if not self.fixed_length:
                    dp0 = inj_coords[4]
                else:
                    if i_iter != 0:
                        dp0 += full_dp_change * self.dp_cor_frac
                    else:
                        dp0 = inj_coords[4]

                res = optimize.fmin(
                    self._calc_2turn_diff,
                    inj_coords[:4],
                    (dp0, self.fixed_length),
                    xtol=1e-7,
                    ftol=1e-9,
                    maxiter=200,
                    full_output=True,
                    disp=False,
                )

                inj_coords, _fopt, _n_iter, _n_func_calls, _warnflag = res
                if _warnflag == 0:
                    if self.fixed_length:
                        dC_over_C = (c * self.dt) / circumf
                        full_dp_change = dC_over_C / self.alphac * (-1)

                        dp_converged = np.abs(full_dp_change) < self.dp_change_thresh

                        if dp_converged:
                            break
                    else:
                        break
            else:
                COD_converged = False

            inj_COD = dict(
                x=inj_coords[0],
                xp=inj_coords[1],
                y=inj_coords[2],
                yp=inj_coords[3],
                dp=dp0,
            )

        else:
            x0, xp0, y0, yp0, dp0 = inj_coords
            base_traj = self._calc_traj(
                x0=x0, xp0=xp0, y0=y0, yp0=yp0, dp0=dp0, dt=True
            )
            self.dt = traj["dt"][1]
            base_diff = {
                coord: np.diff(base_traj[coord])[0] for coord in base_traj.keys()
            }

            inj_angle = 1e-6  # [rad]
            inj_offset = 1e-6  # [m]

            M = np.zeros((4, 4))

            # print("* Calculating traj for positive x inj. offset...")
            traj = self._calc_traj(
                x0=x0 + inj_offset, y0=y0, xp0=xp0, yp0=yp0, dp0=dp0, dt=False
            )
            traj_diff = {coord: np.diff(traj[coord])[0] for coord in traj.keys()}
            diff = np.array(
                [
                    traj_diff[coord] - base_diff[coord]
                    for coord in ["x", "xp", "y", "yp"]
                ]
            )
            M[:, 0] = diff / inj_offset

            # print("* Calculating traj for positive x inj. angle...")
            traj = self._calc_traj(
                x0=x0, y0=y0, xp0=xp0 + inj_angle, yp0=yp0, dp0=dp0, dt=False
            )
            traj_diff = {coord: np.diff(traj[coord])[0] for coord in traj.keys()}
            diff = np.array(
                [
                    traj_diff[coord] - base_diff[coord]
                    for coord in ["x", "xp", "y", "yp"]
                ]
            )
            M[:, 1] = diff / inj_angle

            # print("* Calculating traj for positive y inj. offset...")
            traj = self._calc_traj(
                x0=x0, y0=y0 + inj_offset, xp0=xp0, yp0=yp0, dp0=dp0, dt=False
            )
            traj_diff = {coord: np.diff(traj[coord])[0] for coord in traj.keys()}
            diff = np.array(
                [
                    traj_diff[coord] - base_diff[coord]
                    for coord in ["x", "xp", "y", "yp"]
                ]
            )
            M[:, 2] = diff / inj_offset

            # print("* Calculating traj for positive y inj. angle...")
            traj = self._calc_traj(
                x0=x0, y0=y0, xp0=xp0, yp0=yp0 + inj_angle, dp0=dp0, dt=False
            )
            traj_diff = {coord: np.diff(traj[coord])[0] for coord in traj.keys()}
            diff = np.array(
                [
                    traj_diff[coord] - base_diff[coord]
                    for coord in ["x", "xp", "y", "yp"]
                ]
            )
            M[:, 3] = diff / inj_angle

            U, sv, Vt = calcSVD(M)
            if False:
                plt.figure()
                plt.semilogy(sv / sv[0], ".-")
            assert np.all(sv / sv[0] > 1e-4)

            # Use all singular values by setting "rcond=1e-4"
            Sinv_trunc = calcTruncSVMatrix(sv, rcond=1e-4, nsv=None, disp=0)

            M_inv = Vt.T @ Sinv_trunc @ U.T

            traj = base_traj

            COD_converged = True
            for i_iter in range(self.n_iter_max):
                if debug_print:
                    print(f"Iteration #{i_iter+1}/{self.n_iter_max}")

                if self.fixed_length:
                    dC_over_C = (c * self.dt) / circumf
                    full_dp_change = dC_over_C / self.alphac * (-1)

                    dp_converged = np.abs(full_dp_change) < self.dp_change_thresh

                traj_diff = np.array(
                    [np.diff(traj[coord])[0] for coord in ["x", "xp", "y", "yp"]]
                )

                if np.std(traj_diff) < 1e-9:
                    if self.fixed_length:
                        if dp_converged:
                            break
                    else:
                        break

                delta_trans_inj_coords = M_inv @ (-traj_diff)
                inj_coords[:4] += delta_trans_inj_coords * self.transv_cor_frac

                x0, xp0, y0, yp0 = inj_coords[:4]
                if self.fixed_length:
                    dp0 += full_dp_change * self.dp_cor_frac
                traj = self._calc_traj(x0=x0, xp0=xp0, y0=y0, yp0=yp0, dp0=dp0, dt=True)
                if self.fixed_length:
                    self.dt = traj["dt"][1]
            else:
                COD_converged = False

            inj_COD = dict(
                x=inj_coords[0],
                xp=inj_coords[1],
                y=inj_coords[2],
                yp=inj_coords[3],
                dp=dp0,
            )

        return dict(COD_converged=COD_converged, inj_COD=inj_COD)


class ClosedOrbitThreader:
    def __init__(
        self,
        LTE: Lattice,
        E_MeV: float,
        bpmx_names,
        bpmy_names,
        hcor_names,
        vcor_names,
        zero_orbit_type="BBA",
        BBA_elem_type="KQUAD",
        BBA_elem_names=None,
        TRM_filepath: Union[Path, str] = "",
        N_KICKS: Optional[dict] = None,
        tempdir_path: Union[None, Path, str] = None,
        print_stdout=False,
        print_stderr=True,
    ) -> None:
        """
        method = 0:

        This method can get stuck if the particle dies somewhere in the 1st
        turn, without making any progress to propagate further down the ring.

        method = 1:

        This method works!

        obs_incl_dxy_thresh [m]
        """

        std_print_enabled["out"] = print_stdout
        std_print_enabled["err"] = print_stderr

        assert isinstance(LTE, Lattice)
        self.LTE = LTE
        self.LTE_filepath = LTE.LTE_filepath
        self.N_KICKS = N_KICKS

        if TRM_filepath == "":
            self.TRM_filepath = None
        else:
            self.TRM_filepath = Path(TRM_filepath)
            if self.TRM_filepath.exists():
                with h5py.File(self.TRM_filepath, "r") as f:
                    self._M_inj = f["M_inj"][()]
                    self._M_cor = f["M_cor"][()]
                    self._alphac = f["alphac"][()]
                    self._TRM_bpm_names = f["bpm_names"][()]
                    self._TRM_bpm_fields = f["bpm_fields"][()]
                    self._TRM_cor_names = f["cor_names"][()]
                    self._TRM_cor_fields = f["cor_fields"][()]

        self.E_MeV = E_MeV

        self.validate_BPM_selection(bpmx_names, bpmy_names)

        self.extract_BPM_zeros()

        # Need to only track 2 turns
        self.n_turns = 2
        # First-pass trajectory will try to correct up to 2nd BPM of 2nd turn
        self.second_turn_n_bpms = 2

        assert zero_orbit_type in ("design", "BPM_zero", "BBA")
        self.zero_orbit_type = zero_orbit_type
        self.BBA_elem_type = BBA_elem_type
        self.BBA_elem_names = BBA_elem_names
        self.setup_target_orbit_traj()

        self.validate_corrector_selection(hcor_names, vcor_names)

        self.nBPM = {k: len(self.bpm_names[k]) for k in self.bpm_names.keys()}
        self.nCOR = {k: len(self.cor_names[k]) for k in self.cor_names.keys()}

        self.iter_opts = {}
        self.iter_opts["fixed_energy"] = {
            "full_traj": dict(
                n_iter_max=500,
                cor_frac=0.7,
                rms_thresh={"x": 0.1e-6, "y": 0.1e-6},
                cor_frac_divider_upon_loss=0.5,
            ),
            "partial_traj": dict(
                n_iter_max=500,
                cor_frac=0.7,
                obs_incl_dxy_thresh=1e-3,
            ),
        }
        self.iter_opts["fixed_length"] = {
            "method1": dict(
                dp_cor_frac=0.8,
                dp_change_thresh=1e-7,
                full_traj=self.iter_opts["fixed_energy"]["full_traj"].copy(),
            )
        }

        self.make_tempdir(tempdir_path=tempdir_path)

        self._write_ele_files()

        self._write_cor_setpoints_file()

    def make_tempdir(self, tempdir_path=None):
        if isinstance(tempdir_path, Path):
            if tempdir_path.exists():
                self.tempdir = tempdir_path
                return
            else:
                tempdir_path = str(tempdir_path)
        elif isinstance(tempdir_path, str):
            tempdir_path = Path(tempdir_path)
            if tempdir_path.exists():
                self.tempdir = tempdir_path
                return
            else:
                tempdir_path = str(tempdir_path)

        self.tempdir = tempfile.TemporaryDirectory(prefix="tmpCOThr_", dir=tempdir_path)

    def remove_tempdir(self):

        if not hasattr(self, "tempdir"):
            return

        if isinstance(self.tempdir, Path):
            try:
                self.tempdir.rmdir()
            except:
                pass
        else:
            self.tempdir.cleanup()

    def __del__(self):
        self.remove_tempdir()

    def setup_target_orbit_traj(self):
        self.target_orbit = {}
        self.target_traj = {}

        if self.zero_orbit_type == "design":
            for plane in "xy":
                self.target_orbit[plane] = np.zeros(len(self.bpm_names[plane]))
        elif self.zero_orbit_type == "BPM_zero":
            for plane in "xy":
                self.target_orbit[plane] = self.bpm_zeros[plane]
        elif self.zero_orbit_type == "BBA":
            if self.BBA_elem_names is None:
                self.BBA_elem_names = {}
                for bpm_name in self.bpm_names["xy"]:
                    sel_elem_names = self.LTE.get_closest_names_from_ref_name(
                        bpm_name, self.BBA_elem_type, n=1
                    )
                    assert len(sel_elem_names) == 1  # only 1 elem for "name"
                    assert len(sel_elem_names[0]) == 1  # only 1 elem as closest elem
                    sel_elem_name = sel_elem_names[0][0]
                    self.BBA_elem_names[bpm_name] = sel_elem_name

            self.extract_BBA_zeros()
            for plane in "xy":
                self.target_orbit[plane] = self.bba_zeros[plane]
        else:
            raise ValueError

        for plane, orbit in self.target_orbit.items():
            # First-pass trajectory will try to correct up to 2nd BPM of 2nd turn
            self.target_traj[plane] = np.append(orbit, orbit[: self.second_turn_n_bpms])

        self.comb_target_traj = np.append(self.target_traj["x"], self.target_traj["y"])

    def extract_BPM_zeros(self):
        all_bpm_names = self.bpm_names["xy"].tolist()

        all_bpm_props = self.LTE.get_elem_props_from_names(all_bpm_names)

        self.bpm_zeros = {}
        bpm_zeros = defaultdict(list)

        for plane in "xy":
            prop_name = f"D{plane.upper()}"
            for name in self.bpm_names[plane]:
                bpm_zeros[plane].append(
                    all_bpm_props[name]["properties"].get(prop_name, 0.0)
                )

            self.bpm_zeros[plane] = np.array(bpm_zeros[plane])

    def extract_BBA_zeros(self):
        all_bba_elem_names = list(self.BBA_elem_names.values())

        all_bba_elem_props = self.LTE.get_elem_props_from_names(all_bba_elem_names)

        self.bba_zeros = {}
        bba_zeros = defaultdict(list)

        for plane in "xy":
            prop_name = f"D{plane.upper()}"
            for bpm_name in self.bpm_names[plane]:
                bba_elem_name = self.BBA_elem_names[bpm_name]
                bba_zeros[plane].append(
                    all_bba_elem_props[bba_elem_name]["properties"].get(prop_name, 0.0)
                )

            self.bba_zeros[plane] = np.array(bba_zeros[plane])

    def validate_BPM_selection(self, bpmx_names, bpmy_names):
        s = self.LTE.get_s_mid_array()
        self.bpm_s = {}

        sorted_x_elem_inds = self.LTE.get_elem_inds_from_names(bpmx_names)
        sorted_y_elem_inds = self.LTE.get_elem_inds_from_names(bpmy_names)

        self.bpm_names = dict(
            x=self.LTE.get_names_from_elem_inds(sorted_x_elem_inds),
            y=self.LTE.get_names_from_elem_inds(sorted_y_elem_inds),
        )

        self.bpm_s["x"] = s[sorted_x_elem_inds]
        self.bpm_s["y"] = s[sorted_y_elem_inds]

        u_bpm_xy_names = np.unique(
            self.bpm_names["x"].tolist() + self.bpm_names["y"].tolist()
        )
        sorted_elem_inds = self.LTE.get_elem_inds_from_names(u_bpm_xy_names)
        self.bpm_names["xy"] = self.LTE.get_names_from_elem_inds(sorted_elem_inds)

        sorted_u_bpm_names = self.bpm_names["xy"].tolist()

        self.u_bpm_names_to_bpm_inds = {
            plane: [sorted_u_bpm_names.index(name) for name in self.bpm_names[plane]]
            for plane in "xy"
        }

        self.bpm_s["xy"] = s[sorted_elem_inds]

    def validate_corrector_selection(self, hcor_names, vcor_names):
        if False:  # for educational purpose
            ELEM_d = get_ELEGANT_element_dictionary()

            for elem_type, d in ELEM_d["elements"].items():
                for line in d["table"][1:]:
                    if ("KICK" in line[0]) and (line[0] != "N_KICKS"):
                        print(f"{elem_type}: {line[0]}")

            # From this output, only the following element type / properties
            # should be accepated as H/V correctors:
            #   EHKICK: KICK
            #   EVKICK: KICK
            #   EKICKER: HKICK & VKICK
            #   HKICK: KICK
            #   VKICK: KICK
            #   KICKER: HKICK & VKICK

        valid_elem_types_props = {
            "EHKICK": dict(x="KICK"),
            "EVKICK": dict(y="KICK"),
            "EKICKER": dict(x="HKICK", y="VKICK"),
            "HKICK": dict(x="KICK"),
            "VKICK": dict(y="KICK"),
            "KICKER": dict(x="HKICK", y="VKICK"),
        }

        self.cor_names = {}
        self.cor_props = {}

        for plane, plane_str, name_list in [
            ("x", "horiz.", hcor_names),
            ("y", "vert.", vcor_names),
        ]:
            self.cor_names[plane] = []
            self.cor_props[plane] = {}

            sorted_elem_inds = self.LTE.get_elem_inds_from_names(name_list)
            sorted_names = self.LTE.get_names_from_elem_inds(sorted_elem_inds)

            for name in sorted_names:
                elem_type = self.LTE.get_elem_type_from_name(name)

                try:
                    kick_prop_name = valid_elem_types_props[elem_type][plane]
                except KeyError:
                    raise AssertionError(
                        f"'{elem_type}' not a valid {plane_str} cor. elem."
                    )

                self.cor_names[plane].append(name)
                self.cor_props[plane][name] = dict(name=kick_prop_name, value=0.0)

        s = self.LTE.get_s_mid_array()
        self.cor_s = {}

        for plane, names in self.cor_names.items():
            self.cor_s[plane] = s[self.LTE.get_elem_inds_from_names(names)]

    def _write_traj_calc_ele(self):
        ed = elebuilder.EleDesigner(self.traj_calc_ele_path, double_format=".16g")

        ed.add_block(
            "run_setup",
            lattice=str(self.LTE_filepath),
            p_central_mev=self.E_MeV,
            use_beamline=self.LTE.used_beamline_name,
        )

        ed.add_newline()

        # Note that the output of the WATCH element data will be always with
        # respect to the design orbit. This means the even if MONI elements are
        # defined with non-zero DX and/or DY values, those values will be ignored.
        for bpm_index, name in enumerate(self.bpm_names["xy"]):
            watch_pathobj = self.traj_calc_ele_path.with_suffix(f".wc{bpm_index:03d}")
            temp_watch_elem_name = f"ELEGANT_WATCH_{bpm_index:03d}"
            watch_filepath = watch_pathobj.resolve()
            temp_watch_elem_def = (
                f'{temp_watch_elem_name}: WATCH, FILENAME="{watch_filepath}", '
                "MODE=coordinate"
            )
            if name == "_BEG_":
                name = self.LTE.flat_used_elem_names[0]
            assert self.LTE.is_unique_elem_name(name)
            ed.add_block(
                "insert_elements",
                name=name,
                element_def=temp_watch_elem_def,
            )

        ed.add_newline()

        if self.N_KICKS is not None:
            elebuilder.add_N_KICKS_alter_elements_blocks(ed, self.N_KICKS)

            ed.add_newline()

        load_parameters = dict(
            change_defined_values=True,
            allow_missing_elements=True,
            allow_missing_parameters=True,
            filename=str(self.cor_setpoints_path.resolve()),
        )
        ed.add_block("load_parameters", **load_parameters)

        ed.add_block("run_control", n_passes=self.n_turns)

        ed.add_newline()

        centroid = {}
        centroid[0] = "<x0>"
        centroid[1] = "<xp0>"
        centroid[2] = "<y0>"
        centroid[3] = "<yp0>"
        centroid[5] = "<dp0>"
        #
        ed.add_block("bunched_beam", n_particles_per_bunch=1, centroid=centroid)

        ed.add_newline()

        ed.add_block("track")

        ed.write()

    def _write_lte_save_ele(self):
        ed = elebuilder.EleDesigner(self.lte_save_ele_path, double_format=".16g")

        ed.add_block(
            "run_setup",
            lattice=str(self.LTE_filepath),
            p_central_mev=self.E_MeV,
            use_beamline=self.LTE.used_beamline_name,
        )

        ed.add_newline()

        if self.N_KICKS is not None:
            elebuilder.add_N_KICKS_alter_elements_blocks(ed, self.N_KICKS)

            ed.add_newline()

        load_parameters = dict(
            change_defined_values=True,
            allow_missing_elements=True,
            allow_missing_parameters=True,
            filename=str(self.cor_setpoints_path.resolve()),
        )
        ed.add_block("load_parameters", **load_parameters)

        ed.add_newline()

        ed.add_block("save_lattice", filename="<new_filepath_str>")

        ed.write()

    def _write_ele_files(self):
        """
        Write ELE files for
        1) computing trajectory
        2) saving LTE file that includes horizontal/vertical kicks after correction
        """
        tmp = tempfile.NamedTemporaryFile(
            dir=self.tempdir.name, delete=False, prefix="tmpTraj_", suffix=".ele"
        )
        self.traj_calc_ele_path = Path(tmp.name).resolve()
        tmp.close()

        tmp = tempfile.NamedTemporaryFile(
            dir=self.tempdir.name, delete=False, prefix="tmpLteSave_", suffix=".ele"
        )
        self.lte_save_ele_path = Path(tmp.name).resolve()
        tmp.close()

        self.cor_setpoints_path = self.traj_calc_ele_path.with_suffix(".corsp")

        self._write_traj_calc_ele()
        self._write_lte_save_ele()

    def save_current_lattice_to_LTE_file(self, new_LTE_filepath: Union[Path, str]):
        new_LTE_filepath = Path(new_LTE_filepath).resolve()

        run(
            self.lte_save_ele_path,
            print_cmd=False,
            macros=dict(new_filepath_str=str(new_LTE_filepath)),
            print_stdout=std_print_enabled["out"],
            print_stderr=std_print_enabled["err"],
        )

    def save_current_lattice_to_LTEZIP_file(
        self, new_LTEZIP_filepath: Union[Path, str]
    ):
        tmp = tempfile.NamedTemporaryFile(
            dir=self.tempdir.name, delete=False, prefix=f"tmpLte_", suffix=".lte"
        )
        new_LTE_filepath = Path(tmp.name).resolve()
        tmp.close()

        run(
            self.lte_save_ele_path,
            print_cmd=False,
            macros=dict(new_filepath_str=str(new_LTE_filepath)),
            print_stdout=std_print_enabled["out"],
            print_stderr=std_print_enabled["err"],
        )

        temp_LTE = Lattice(
            LTE_filepath=new_LTE_filepath,
            used_beamline_name=self.LTE.used_beamline_name,
        )

        new_LTEZIP_filepath = Path(new_LTEZIP_filepath)
        temp_LTE.zip_lte(new_LTEZIP_filepath)
        print(f"\nSaved current lattice to {new_LTEZIP_filepath.resolve()}\n")

        try:
            new_LTE_filepath.unlink()
        except:
            pass

    def calc_traj(
        self, x0=0.0, y0=0.0, xp0=0.0, yp0=0.0, dp0=0.0, dt=False, debug_print=False
    ):
        if self._uncommited_cor_change:
            self._write_cor_setpoints_file()

        n_turns = self.n_turns
        assert n_turns == 2

        nBPM = self.nBPM["xy"]

        t0 = time.perf_counter()

        run(
            self.traj_calc_ele_path,
            print_cmd=False,
            macros=dict(
                x0=f"{x0:.16g}",
                y0=f"{y0:.16g}",
                xp0=f"{xp0:.16g}",
                yp0=f"{yp0:.16g}",
                dp0=f"{dp0:.16g}",
            ),
            print_stdout=std_print_enabled["out"],
            print_stderr=std_print_enabled["err"],
        )

        t1 = time.perf_counter()

        if False:  # This takes 10's of seconds!
            tbt = dict(x=[], y=[])
            if dt:
                tbt["dt"] = []
            for bpm_index in range(nBPM):
                watch_pathobj = self.traj_calc_ele_path.with_suffix(
                    f".wc{bpm_index:03d}"
                )
                with nostdout():
                    output, _ = sdds.sdds2dicts(watch_pathobj)
                for coord in list(tbt):
                    tbt[coord].append(output["columns"][coord])

            t2 = time.perf_counter()

            tbt_ar = dict(
                x=np.full((n_turns, nBPM), np.nan), y=np.full((n_turns, nBPM), np.nan)
            )
            for coord, v_list in tbt.items():
                for iBPM, v in enumerate(v_list):
                    tbt_ar[coord][: v.size, iBPM] = v
                tbt_ar[coord] = tbt_ar[coord].flatten()

            t3 = time.perf_counter()

            if False and debug_print:
                print(
                    f"calc_traj() [s]: ELE={t1-t0:.1f}, SDDS={t2-t1:.1f}, Format={t3-t2:.1f}"
                )

        else:  # This takes only a fraction of a second!
            comb_watch_filepath = self.traj_calc_ele_path.parent / "all.w1"
            cmd = f"sddscombine -overWrite *.wc??? {comb_watch_filepath.name}"
            p = Popen(
                cmd,
                stdout=PIPE,
                stderr=PIPE,
                cwd=self.traj_calc_ele_path.parent,
                encoding="utf-8",
                shell=True,
            )
            out, err = p.communicate()
            try:
                assert (out == "") and (err == "")
            except AssertionError:
                print(f"*** out = {out}")
                print(f"*** err = {err}")
                raise
            output, _ = sdds.sdds2dicts(comb_watch_filepath)
            tbt_ar = {}
            coord_list = ["x", "y"]
            if dt:
                coord_list.append("dt")
            for coord in coord_list:
                ps_ar = output["columns"][coord]
                n_survived = ps_ar.size
                n_missing = nBPM * n_turns - n_survived
                if n_missing == 0:
                    tbt_ar[coord] = ps_ar.reshape((nBPM, n_turns)).T.flatten()
                else:
                    tbt_ar[coord] = np.full((nBPM * n_turns,), np.nan)
                    if n_survived <= nBPM:  # Didn't go to 2nd turn
                        tbt_ar[coord][:n_survived] = ps_ar
                    else:
                        assert nBPM > n_missing
                        twoTurn_nBPM = nBPM - n_missing
                        twoTurn_bpm_data = ps_ar[: twoTurn_nBPM * n_turns]
                        oneTurnOnly_bpm_data = ps_ar[twoTurn_nBPM * n_turns :]
                        temp = twoTurn_bpm_data.reshape((twoTurn_nBPM, n_turns))
                        turn1_partial = temp[:, 0]
                        turn2_partial = temp[:, 1]
                        turn1_full = np.append(turn1_partial, oneTurnOnly_bpm_data)
                        tbt_ar[coord][:n_survived] = np.append(
                            turn1_full, turn2_partial
                        )

            t2 = time.perf_counter()

            if False and debug_print:
                print(f"calc_traj() [s]: ELE={t1-t0:.1f}, SDDS={t2-t1:.1f}")

        return tbt_ar

    def _write_cor_setpoints_file(self):
        col = dict(ElementName=[], ElementParameter=[], ParameterValue=[])

        for plane in "xy":
            for elem_name, prop_d in self.cor_props[plane].items():
                param_name = prop_d["name"]
                param_val = prop_d["value"]
                col["ElementName"].append(elem_name)
                col["ElementParameter"].append(param_name)
                col["ParameterValue"].append(param_val)

        sdds.dicts2sdds(
            self.cor_setpoints_path,
            params=None,
            columns=col,
            outputMode="binary",
            suppress_err_msg=True,
        )

        self._uncommited_cor_change = False

    def start_fixed_length_orbit_correction(
        self, method=1, init_inj_coords=None, debug_print=True, debug_plot=False
    ):
        """Fixed-length COD (closed orbit distortion) correction.

        method = 0:

        Using ELEGANT's "&closed_orbit" and with "fixed_length=True".
        For some cases, the correction settles with a relatively large max orbit
        deviation, so not ideal.

        method = 1:

        Apply the full-trajectory corrections while adjusting the energy so that the
        particle pathlength becomes closer to the circumference, while maintaining
        the closed orbit.
        """

        tStart = time.perf_counter()

        if method == 0:
            raise NotImplementedError
        elif method == 1:
            assert init_inj_coords is not None

            if debug_print:
                print("\n* Starting fixed-length closed orbit corrections...")

            tStart_fixed_len = time.perf_counter()

            alphac = self._alphac
            circumf = self.LTE.get_circumference()
            c = constants.c  # speed of light [m/s]

            inj_coords = init_inj_coords

            traj = self.calc_traj(
                x0=inj_coords[0],
                xp0=inj_coords[1],
                y0=inj_coords[2],
                yp0=inj_coords[3],
                dp0=0.0,  # start from on-referene energy
                dt=True,
                debug_print=debug_print,
            )

            traj_dt_1st = traj["dt"][: self.nBPM["xy"]]
            traj_dt_2nd = traj["dt"][self.nBPM["xy"] :]

            opts = self.iter_opts["fixed_length"]["method1"]

            dp_cor_frac = opts["dp_cor_frac"]
            dp_change_thresh = opts["dp_change_thresh"]

            n_iter_max = opts["full_traj"]["n_iter_max"]

            full_traj_cor_opts = deepcopy(self.iter_opts["fixed_energy"]["full_traj"])

            i_iter = 0
            cod_calc_failed = False
            cor_dp = 0.0
            inj_coords_hist = [inj_coords]
            cor_dp_hist = [cor_dp]
            dt_diffs_hist = [traj_dt_2nd - traj_dt_1st]
            avg_dt_hist = []
            prev_full_dp_change = None
            while True:
                dt = np.mean(traj_dt_2nd - traj_dt_1st)
                avg_dt_hist.append(dt)
                dC_over_C = (c * dt) / circumf
                full_dp_change = dC_over_C / alphac

                if i_iter >= n_iter_max:
                    cod_calc_failed = True
                    break

                if np.abs(full_dp_change) < dp_change_thresh:
                    break

                if debug_print:
                    msg = f"\nFixed-length COD correction #{i_iter+1}/{n_iter_max}"
                    msg += f" (delta_dp = {full_dp_change:.3e}, avg_dt = {dt:.3e})"
                    print(msg)

                if prev_full_dp_change is not None:
                    # print(f"{np.abs(full_dp_change):.3e}, {np.abs(prev_full_dp_change):.3e}")
                    if np.abs(full_dp_change) > np.abs(prev_full_dp_change):
                        inj_coords = prev_inj_coords.copy()
                        cor_dp = prev_dp0
                        full_dp_change = prev_full_dp_change

                        full_traj_cor_opts["rms_thresh"]["x"] /= 2
                        full_traj_cor_opts["rms_thresh"]["y"] /= 2
                        _thresh = full_traj_cor_opts["rms_thresh"]

                        # If the "rms_thresh" for self._full_traj_correction() is too high,
                        # dt vs. dp plot will be like a saw tooth with the each tooth slope
                        # direction being the opposite of the general trend. This results,
                        # in non-converging beheavior for energy adjustment.
                        # It appears RMS threshold of 100 nm is too high. 10 nm should be good enough.
                        # Too small a threshold value will result in longer correction computation.
                        msg = "* dp correction got worse. Rolling back and reducing fixed-energy, "
                        msg += "full-traj. RMS threshold (x, y) to "
                        msg += f"({_thresh['x']*1e9:.6g}, {_thresh['y']*1e9:.6g}) [nm]"
                        print(msg)

                        cor_dp += full_dp_change * dp_cor_frac

                        out = self._full_traj_correction(
                            inj_coords=inj_coords,
                            iter_opts=full_traj_cor_opts,
                            dp0=cor_dp,  # adjust energy
                            debug_print=False,
                            debug_plot=False,
                        )

                        if not out["success"]:
                            cod_calc_failed = True
                            raise RuntimeError("Somehow COD calc failed")

                        inj_coords = out["inj_coords_list"][-1]

                        traj = self.calc_traj(
                            x0=inj_coords[0],
                            xp0=inj_coords[1],
                            y0=inj_coords[2],
                            yp0=inj_coords[3],
                            dp0=cor_dp,
                            dt=True,
                            debug_print=debug_print,
                        )

                        traj_dt_1st = traj["dt"][: self.nBPM["xy"]]
                        traj_dt_2nd = traj["dt"][self.nBPM["xy"] :]

                        dt = np.mean(traj_dt_2nd - traj_dt_1st)
                        dC_over_C = (c * dt) / circumf
                        full_dp_change = dC_over_C / alphac

                        # Only `full_dp_change` needs updated with smaller RMS threshold.
                        # `inj_coords` and `cor_dp` must be reset again here
                        inj_coords = prev_inj_coords.copy()
                        cor_dp = prev_dp0

                prev_full_dp_change = full_dp_change
                prev_inj_coords = inj_coords.copy()
                prev_dp0 = cor_dp

                cor_dp += full_dp_change * dp_cor_frac

                out = self._full_traj_correction(
                    inj_coords=inj_coords,
                    iter_opts=full_traj_cor_opts,
                    dp0=cor_dp,  # adjust energy
                    debug_print=False,
                    debug_plot=False,
                )

                if not out["success"]:
                    cod_calc_failed = True
                    break

                inj_coords_list = out["inj_coords_list"]
                hkick_rads = out["hkicks_hist"][-1]
                vkick_rads = out["vkicks_hist"][-1]

                inj_coords = inj_coords_list[-1]

                traj = self.calc_traj(
                    x0=inj_coords[0],
                    xp0=inj_coords[1],
                    y0=inj_coords[2],
                    yp0=inj_coords[3],
                    dp0=cor_dp,
                    dt=True,
                    debug_print=debug_print,
                )

                traj_dt_1st = traj["dt"][: self.nBPM["xy"]]
                traj_dt_2nd = traj["dt"][self.nBPM["xy"] :]

                cor_dp_hist.append(cor_dp)
                inj_coords_hist.append(inj_coords)
                dt_diffs_hist.append(traj_dt_2nd - traj_dt_1st)

                i_iter += 1

            tEnd_fixed_len = time.perf_counter()
            fixed_len_cod_cor_tElapsed = tEnd_fixed_len - tStart_fixed_len

            if debug_print:
                print(
                    f"\n* Finished fixed-length closed orbit corrections ({fixed_len_cod_cor_tElapsed:.1f} [s])."
                )

            traj_1st = {plane: traj[plane][: self.nBPM[plane]] for plane in "xy"}
            traj_2nd = {plane: traj[plane][self.nBPM[plane] :] for plane in "xy"}

            cod = {
                plane: np.mean([traj_1st[plane], traj_2nd[plane]], axis=0)
                for plane in "xy"
            }

            fin_diff_orbs = {
                plane: cod[plane] - self.target_orbit[plane] for plane in "xy"
            }

            fin_kicks = dict(x=hkick_rads, y=vkick_rads)

            if debug_plot:
                clo_calc = ClosedOrbitCalculator(
                    self.LTE.LTE_filepath,
                    self.E_MeV,
                    fixed_length=True,
                    output_monitors_only=False,
                    closed_orbit_accuracy=1e-12,  # 1e-9,
                    closed_orbit_iterations=100,  # 200,
                    iteration_fraction=0.9,
                    n_turns=0,
                    use_beamline=self.LTE.used_beamline_name,
                    N_KICKS=self.N_KICKS,
                )

                clo_calc.select_kickers("x", self.cor_names["x"])
                clo_calc.select_kickers("y", self.cor_names["y"])

                clo_calc.set_kick_angles(hkick_rads, vkick_rads)

                d = clo_calc.calc(run_local=True)
                if False:
                    plot_closed_orbit(d["columns"], d["params"])
                cod_ele = d["columns"]

            tEnd = time.perf_counter()
            print(f"* Took {tEnd-tStart:.1f} [s].")
            sys.stdout.flush()

            if debug_plot:
                plt.figure()
                plt.plot(np.array(cor_dp_hist), ".-")

                plt.figure()
                plt.plot(np.array(inj_coords_hist), ".-")

                plt.figure()
                plt.plot(np.array(dt_diffs_hist), ".-")

                plt.figure()
                plt.subplot(211)
                plt.plot(traj_2nd["x"] - traj_1st["x"], "b.-")
                plt.subplot(212)
                plt.plot(traj_2nd["y"] - traj_1st["y"], "b.-")
                plt.tight_layout()

                plt.figure()
                plt.subplot(211)
                plt.plot(fin_diff_orbs["x"], "b.-")
                plt.subplot(212)
                plt.plot(fin_diff_orbs["y"], "b.-")
                plt.tight_layout()

                plt.figure()
                plt.plot(self.cor_s["x"], hkick_rads * 1e3, "b.-", label="x")
                plt.plot(self.cor_s["y"], vkick_rads * 1e3, "r.-", label="y")
                leg = plt.legend(loc="best", prop=dict(size=18))
                plt.ylabel(r"$\theta\; [\mathrm{mrad}]$", size="large")
                plt.xlabel(r"$s\; [\mathrm{m}]$", size="large")
                plt.tight_layout()

                plt.figure()
                plt.subplot(211)
                plt.plot(self.bpm_s["x"], self.target_orbit["x"] * 1e6, "k^:")
                plt.plot(cod_ele["s"], cod_ele["x"] * 1e6, "m-")
                plt.ylabel(r"$x\; [\mu\mathrm{m}]$", size="large")
                plt.subplot(212)
                plt.plot(self.bpm_s["y"], self.target_orbit["y"] * 1e6, "k^:")
                plt.plot(cod_ele["s"], cod_ele["y"] * 1e6, "m-")
                plt.ylabel(r"$y\; [\mu\mathrm{m}]$", size="large")
                plt.xlabel(r"$s\; [\mathrm{m}]$", size="large")
                plt.tight_layout()

        else:
            raise NotImplementedError

        result = dict(
            cod_calc_failed=cod_calc_failed,
            fin_diff_orbs=fin_diff_orbs,
            fin_kicks=fin_kicks,
            elapsed={"fixed_len_cod_cor": fixed_len_cod_cor_tElapsed},
        )

        if method == 1:
            result["fin_inj_coords"] = inj_coords
            result["fin_dp"] = cor_dp

        return result

    def _full_traj_correction(
        self,
        inj_coords=None,
        iter_opts: Union[Dict, None] = None,
        dp0: float = 0.0,
        debug_print: bool = False,
        debug_plot: bool = False,
    ):
        """2nd Stage: Correct the trajectory at all observations"""

        if iter_opts is None:
            iter_opts = self.iter_opts["fixed_energy"]["full_traj"]
        n_iter_max = iter_opts["n_iter_max"]
        cor_frac = iter_opts["cor_frac"]
        rms_thresh = iter_opts["rms_thresh"]
        cor_frac_divider_upon_loss = iter_opts["cor_frac_divider_upon_loss"]

        assert cor_frac_divider_upon_loss <= 1.0

        nx = self.target_traj["x"].size  # == self.nBPM["x"] + self.second_turn_n_bpms
        ny = self.target_traj["y"].size  # == self.nBPM["y"] + self.second_turn_n_bpms

        n_inj_coords = 4
        if inj_coords is None:
            inj_coords = np.zeros(n_inj_coords)
        assert inj_coords.shape == (n_inj_coords,)

        inj_coords_list = []
        traj_list = []
        hkicks_list = []
        vkicks_list = []

        # Backup data placeholder. Capture backup data only when the particle fully survives.
        # If the particle is lost, roll back to the backup data.
        backup = dict(inj_coords=None, cor_props=None)

        i_iter = 0
        i_loss = 0
        while i_iter < n_iter_max:
            if debug_print:
                print(f"Iteration #{i_iter+1}/{n_iter_max}")

            inj_coords_list.append(inj_coords.copy())
            hkicks_list.append([_d["value"] for _d in self.cor_props["x"].values()])
            vkicks_list.append([_d["value"] for _d in self.cor_props["y"].values()])

            traj = self.calc_traj(
                x0=inj_coords[0],
                xp0=inj_coords[1],
                y0=inj_coords[2],
                yp0=inj_coords[3],
                dp0=dp0,
                debug_print=debug_print,
            )

            if debug_plot:
                plt.figure()
                plt.subplot(211)
                plt.plot(traj["x"][:nx])
                plt.subplot(212)
                plt.plot(traj["y"][:ny])

            comb_traj = np.append(traj["x"][:nx], traj["y"][:ny])
            traj_list.append(comb_traj)

            survived_inds = self._get_survived_inds(traj)

            dx_rms = np.std(
                self.target_traj["x"][: survived_inds["x"]]
                - traj["x"][: survived_inds["x"]]
            )
            dy_rms = np.std(
                self.target_traj["y"][: survived_inds["y"]]
                - traj["y"][: survived_inds["y"]]
            )

            if debug_print:
                print(f"Residual RMS (x,y) [um] = ({dx_rms*1e6:.3f}, {dy_rms*1e6:.3f})")

            incl_nObs = {}
            incl_nKnobs = {}
            survived_str = {}
            for plane in "xy":
                i = survived_inds[plane] - 1
                incl_nObs[plane] = i + 1
                if i < self.nBPM[plane]:
                    if debug_print:
                        survived_str[plane] = f"#{i+1} at Turn #1"

                    incl_nKnobs[plane] = np.sum(
                        self.cor_s[plane] < self.bpm_s[plane][i]
                    )
                else:
                    if debug_print:
                        survived_str[plane] = f"#{i+1 - self.nBPM[plane]} at Turn #2"

                    incl_nKnobs[plane] = self.nCOR[plane]

            if debug_print:
                header = "Particle survived up to BPM (x, y)"
                print(f"{header} = ({survived_str['x']}, {survived_str['y']})")

            if i_iter == 0:
                assert (survived_inds["x"] == nx) and (survived_inds["y"] == ny)

                _d = self._construct_full_TRM(incl_nObs, incl_nKnobs)
                M = _d["M"]

                U, sv, Vt = calcSVD(M)
                if False:
                    plt.figure()
                    plt.semilogy(sv / sv[0], ".-")
                assert np.all(sv / sv[0] > 1e-4)

                # Use all singular values by setting "rcond=1e-4"
                Sinv_trunc = calcTruncSVMatrix(sv, rcond=1e-4, nsv=None, disp=0)

                M_full_inv = Vt.T @ Sinv_trunc @ U.T

            if (survived_inds["x"] == nx) and (survived_inds["y"] == ny):
                # The particle fully survived

                if dx_rms < rms_thresh["x"] and dy_rms < rms_thresh["y"]:
                    if debug_print:
                        print("Corrections converged.")
                    success = True
                    break

                i_loss = 0

                _target_traj = self.comb_target_traj
                _current_traj = comb_traj

                dv = _target_traj - _current_traj

                dI = M_full_inv @ dv
                if False:
                    plt.figure()
                    plt.plot(dI, ".-")

                dI *= cor_frac

                dI_inj = dI[:n_inj_coords]
                dI_cor = dI[n_inj_coords:]

                backup["inj_coords"] = inj_coords.copy()
                inj_coords += dI_inj

                assert incl_nKnobs["x"] == self.nCOR["x"]
                assert incl_nKnobs["y"] == self.nCOR["y"]
                slice_x_cor_knobs = np.s_[: self.nCOR["x"]]
                slice_y_cor_knobs = np.s_[self.nCOR["x"] :]

                backup["cor_props"] = dict(x={}, y={})
                dtheta_H = dI_cor[slice_x_cor_knobs]
                dtheta_V = dI_cor[slice_y_cor_knobs]
                assert len(self.cor_names["x"]) == len(dtheta_H)
                for elem_name, dtheta in zip(self.cor_names["x"], dtheta_H):
                    backup["cor_props"]["x"][elem_name] = self.cor_props["x"][
                        elem_name
                    ]["value"]
                    self.change_corrector_setpoint_by(elem_name, "x", dtheta)
                assert len(self.cor_names["y"]) == len(dtheta_V)
                for elem_name, dtheta in zip(self.cor_names["y"], dtheta_V):
                    backup["cor_props"]["y"][elem_name] = self.cor_props["y"][
                        elem_name
                    ]["value"]
                    self.change_corrector_setpoint_by(elem_name, "y", dtheta)

            else:  # The particle didn't fully survive.
                if cor_frac_divider_upon_loss == 1.0:
                    _target_traj = np.append(
                        self.target_traj["x"][: survived_inds["x"]],
                        self.target_traj["y"][: survived_inds["y"]],
                    )

                    _current_traj = np.append(
                        traj["x"][: survived_inds["x"]], traj["y"][: survived_inds["y"]]
                    )

                    dv = _target_traj - _current_traj

                    _d = self._construct_full_TRM(incl_nObs, incl_nKnobs)
                    M_partial = _d["M"]

                    U, sv, Vt = calcSVD(M_partial)
                    if False:
                        plt.figure()
                        plt.semilogy(sv / sv[0], ".-")
                    assert np.all(sv / sv[0] > 1e-4)

                    # Use all singular values by setting "rcond=1e-4"
                    Sinv_trunc = calcTruncSVMatrix(sv, rcond=1e-4, nsv=None, disp=0)

                    M_partial_inv = Vt.T @ Sinv_trunc @ U.T

                    dI = M_partial_inv @ dv
                    if False:
                        plt.figure()
                        plt.plot(dI, ".-")

                    dI *= cor_frac

                    dI_inj = dI[:n_inj_coords]
                    dI_cor = dI[n_inj_coords:]

                    inj_coords += dI_inj

                    slice_x_cor_knobs = np.s_[: incl_nKnobs["x"]]
                    slice_y_cor_knobs = np.s_[incl_nKnobs["x"] :]

                    dtheta_H = dI_cor[slice_x_cor_knobs]
                    dtheta_V = dI_cor[slice_y_cor_knobs]
                    assert len(self.cor_names["x"][: incl_nKnobs["x"]]) == len(dtheta_H)
                    for elem_name, dtheta in zip(
                        self.cor_names["x"][: incl_nKnobs["x"]], dtheta_H
                    ):
                        self.change_corrector_setpoint_by(elem_name, "x", dtheta)
                    assert len(self.cor_names["y"][: incl_nKnobs["y"]]) == len(dtheta_V)
                    for elem_name, dtheta in zip(
                        self.cor_names["y"][: incl_nKnobs["y"]], dtheta_V
                    ):
                        self.change_corrector_setpoint_by(elem_name, "y", dtheta)

                else:
                    dI = M_full_inv @ dv

                    i_loss += 1
                    actual_cor_frac = cor_frac * (cor_frac_divider_upon_loss**i_loss)
                    dI *= actual_cor_frac
                    if debug_print:
                        print(
                            f"Applying a smaller change (frac = {actual_cor_frac:.3g})."
                        )

                    dI_inj = dI[:n_inj_coords]
                    dI_cor = dI[n_inj_coords:]

                    # First restore the previous changes
                    inj_coords = backup["inj_coords"].copy()
                    for plane in "xy":
                        for elem_name, prev_val in backup["cor_props"][plane].items():
                            self.cor_props[plane][elem_name]["value"] = prev_val

                    # Then apply the new smaller changes, hoping it will not lead to beam loss
                    inj_coords += dI_inj

                    slice_x_cor_knobs = np.s_[: self.nCOR["x"]]
                    slice_y_cor_knobs = np.s_[self.nCOR["x"] :]

                    dtheta_H = dI_cor[slice_x_cor_knobs]
                    dtheta_V = dI_cor[slice_y_cor_knobs]
                    assert len(self.cor_names["x"]) == len(dtheta_H)
                    for elem_name, dtheta in zip(self.cor_names["x"], dtheta_H):
                        self.change_corrector_setpoint_by(elem_name, "x", dtheta)
                    assert len(self.cor_names["y"]) == len(dtheta_V)
                    for elem_name, dtheta in zip(self.cor_names["y"], dtheta_V):
                        self.change_corrector_setpoint_by(elem_name, "y", dtheta)

            i_iter += 1
        else:
            success = False

        return dict(
            success=success,
            inj_coords_list=np.array(inj_coords_list),
            hkicks_hist=np.array(hkicks_list),
            vkicks_hist=np.array(vkicks_list),
            traj_hist=np.array(hkicks_list),
        )

    def _check_no_NaN_observations(self, traj):
        """Check if the particle reached the 2nd BPM in 2nd turn"""

        i2n = self.second_turn_n_bpms - 1

        nx = self.target_traj["x"].size  # == self.nBPM["x"] + self.second_turn_n_bpms
        ny = self.target_traj["y"].size  # == self.nBPM["y"] + self.second_turn_n_bpms

        ix = nx - 1
        iy = ny - 1

        if self.bpm_names["x"][i2n] == self.bpm_names["y"][i2n]:
            if not np.isnan(traj["x"][ix]):
                assert not np.isnan(traj["y"][iy])
                return True
            else:
                return False
        else:
            if np.isnan(traj["x"][ix]) or np.isnan(traj["y"][iy]):
                return False
            else:
                return True

    def _get_survived_inds(self, traj):
        nx = self.target_traj["x"].size  # == self.nBPM["x"] + self.second_turn_n_bpms
        ny = self.target_traj["y"].size  # == self.nBPM["y"] + self.second_turn_n_bpms

        survived_inds = dict(
            x=nx - np.sum(np.isnan(traj["x"][:nx])),
            y=ny - np.sum(np.isnan(traj["y"][:ny])),
        )

        if (self.nBPM["x"] == self.nBPM["y"]) and np.all(
            self.bpm_names["x"] == self.bpm_names["y"]
        ):
            assert survived_inds["x"] == survived_inds["y"]

        return survived_inds

    def start_fixed_energy_orbit_correction(self, debug_print=False, debug_plot=False):
        """
        Fixed-energy trajectory correction using first- and second-turn trajctories

        1st Stage: Only use BPMs and correctors for the positions within 1 mm
        at those BPMs in order to propagate beam till the end of the ring.

        2nd Stage: Try trajectory correction at all BPMs.
        """

        inj_coords = np.zeros(4)

        nx = self.target_traj["x"].size  # == self.nBPM["x"] + self.second_turn_n_bpms
        ny = self.target_traj["y"].size  # == self.nBPM["y"] + self.second_turn_n_bpms

        traj_list = []

        for i_stage, stage_label in enumerate(
            [
                '"Before Partial Traj. Corrections"',
                '"After Partial Traj. Corrections"',
            ]
        ):
            if debug_plot:
                plt.figure()
                plt.subplot(211)
                theta_x = np.array(
                    [
                        self.cor_props["x"][elem_name]["value"]
                        for elem_name in self.cor_names["x"]
                    ]
                )
                plt.plot(theta_x, ".-")
                plt.ylabel(r"$\theta_x$", size="large")
                plt.subplot(212)
                theta_y = np.array(
                    [
                        self.cor_props["y"][elem_name]["value"]
                        for elem_name in self.cor_names["y"]
                    ]
                )
                plt.plot(theta_y, ".-")
                plt.ylabel(r"$\theta_y$", size="large")
                plt.tight_layout()

            traj = self.calc_traj(
                x0=inj_coords[0],
                xp0=inj_coords[1],
                y0=inj_coords[2],
                yp0=inj_coords[3],
                dp0=0.0,  # "fixed-energy" at ref. energy in this function
                debug_print=debug_print,
            )

            if debug_plot:
                plt.figure()
                plt.subplot(211)
                plt.plot(traj["x"][:nx] - self.target_traj["x"], ".-")
                plt.ylabel(r"$\Delta_x$", size="large")
                plt.subplot(212)
                plt.plot(traj["y"][:ny] - self.target_traj["y"], ".-")
                plt.ylabel(r"$\Delta_y$", size="large")
                plt.tight_layout()

            comb_traj = np.append(traj["x"][:nx], traj["y"][:ny])
            traj_list.append(comb_traj)

            survived_inds = self._get_survived_inds(traj)

            dx = (
                self.target_traj["x"][: survived_inds["x"]]
                - traj["x"][: survived_inds["x"]]
            )
            dy = (
                self.target_traj["y"][: survived_inds["y"]]
                - traj["y"][: survived_inds["y"]]
            )

            _, incl_nKnobs = self._get_indexes_selected_for_partial_traj_cor(
                dx, dy, debug_print=debug_print
            )

            if (incl_nKnobs["x"] == self.nCOR["x"]) and (
                incl_nKnobs["y"] == self.nCOR["y"]
            ):
                if debug_print:
                    print(
                        f"\n### Starting full trajectory corrections {stage_label} ###\n"
                    )

                return self._full_traj_correction(
                    inj_coords=inj_coords,
                    iter_opts=self.iter_opts["fixed_energy"]["full_traj"],
                    dp0=0.0,  # "fixed-energy" at ref. energy in this function
                    debug_print=debug_print,
                    debug_plot=debug_plot,
                )

            if i_stage == 1:
                raise RuntimeError(
                    "Could not reach the stage where fixed-energy full traj. correction can be performed."
                )

            self._partial_traj_correction(
                traj,
                inj_coords=inj_coords,
                debug_print=debug_print,
                debug_plot=debug_plot,
            )

    def _get_well_corrected_observ_index(self, dxy):
        opts = self.iter_opts["fixed_energy"]["partial_traj"]
        obs_incl_dxy_thresh = opts["obs_incl_dxy_thresh"]  # [m]

        invalid_inds = np.where(np.abs(dxy) > obs_incl_dxy_thresh)[0]

        if invalid_inds.size != 0:
            i = invalid_inds[0] - 1
        else:
            i = dxy.size - 1

        assert i >= 0

        return i

    def _get_indexes_selected_for_partial_traj_cor(self, dx, dy, debug_print=False):
        incl_nObs = {}
        incl_nKnobs = {}
        for plane, dxy in [("x", dx), ("y", dy)]:
            i = self._get_well_corrected_observ_index(dxy)

            incl_nObs[plane] = i + 1

            if i < self.nBPM[plane]:
                incl_nKnobs[plane] = np.sum(self.cor_s[plane] < self.bpm_s[plane][i])
            else:
                incl_nKnobs[plane] = self.nCOR[plane]

        return incl_nObs, incl_nKnobs

    def _construct_TRM_for_next_bpm_cor_pair(
        self, incl_nObs: Dict[str, int], incl_nKnobs: Dict[str, int]
    ) -> np.ndarray:
        """ " TRM for the next pair of BPM and corrector right after the
        BPMs and correctors included in the current iteration."""

        # `M_cor` are assumed to contain 2 full turns.
        M_cor = self._M_cor

        nBPM = self.nBPM
        nCOR = self.nCOR
        n_turns = 2

        bpm_x_i = incl_nObs["x"]
        bpm_y_i = incl_nObs["y"]

        bpm_x_comb_i = bpm_x_i
        bpm_y_comb_i = nBPM["x"] * n_turns + bpm_y_i

        if incl_nKnobs["x"] < nCOR["x"]:
            cor_x_i = incl_nKnobs["x"]
            cor_x_comb_i = cor_x_i
            while M_cor[bpm_x_comb_i, cor_x_comb_i] == 0.0:
                # Since this corrector is downstream of the BPM,
                # go back by one corrector
                cor_x_i -= 1

                cor_x_comb_i = cor_x_i

                if cor_x_i < 0:
                    raise RuntimeError(
                        f"No horiz. corrector that can correct horiz. BPM index #{bpm_x_i} was found."
                    )
        elif incl_nKnobs["x"] == nCOR["x"]:
            # There is no next corrector. Partial traj. corr. is done in this plane.
            cor_x_i = cor_x_comb_i = None
        else:
            raise RuntimeError("This should not be reached")

        if incl_nKnobs["y"] < nCOR["y"]:
            cor_y_i = incl_nKnobs["y"]
            cor_y_comb_i = nCOR["x"] + cor_y_i
            while M_cor[bpm_y_comb_i, cor_y_comb_i] == 0.0:
                # Since this corrector is downstream of the BPM,
                # go back by one corrector
                cor_y_i -= 1

                cor_y_comb_i = nCOR["x"] + cor_y_i

                if cor_y_i < 0:
                    raise RuntimeError(
                        f"No vert. corrector that can correct vert. BPM index #{bpm_y_i} was found."
                    )
        elif incl_nKnobs["y"] == nCOR["y"]:
            # There is no next corrector. Partial traj. corr. is done in this plane.
            cor_y_i = cor_y_comb_i = None
        else:
            raise RuntimeError("This should not be reached")

        if cor_x_comb_i and cor_y_comb_i:
            M_next = np.vstack(
                (
                    [
                        M_cor[bpm_x_comb_i, cor_x_comb_i],
                        M_cor[bpm_y_comb_i, cor_x_comb_i],
                    ],
                    [
                        M_cor[bpm_x_comb_i, cor_y_comb_i],
                        M_cor[bpm_y_comb_i, cor_y_comb_i],
                    ],
                )
            ).T

            assert M_next.shape == (2, 2)
        elif cor_x_comb_i:
            M_next = np.vstack(([M_cor[bpm_x_comb_i, cor_x_comb_i]])).T

            assert M_next.shape == (1, 1)
        elif cor_x_comb_i:
            M_next = np.array([M_cor[bpm_x_comb_i, cor_x_comb_i]]).reshape((1, 1))
        elif cor_y_comb_i:
            M_next = np.array([M_cor[bpm_y_comb_i, cor_y_comb_i]]).reshape((1, 1))
        else:
            raise RuntimeError(
                (
                    "All of both horiz. and vert. correctors are included in correction, "
                    "which means partial traj. correction should have been terminated "
                    "before reaching this point."
                )
            )

        return dict(
            M=M_next,
            next_bpm_index={"x": bpm_x_i, "y": bpm_y_i},
            next_cor_index={"x": cor_x_i, "y": cor_y_i},
        )

    def _construct_full_TRM(
        self, incl_nObs: Dict[str, int], incl_nKnobs: Dict[str, int]
    ) -> np.ndarray:
        nBPM = self.nBPM
        nCOR = self.nCOR

        n_inj_coords = 4
        n_turns = 2

        # `M_inj` and `M_cor` are assumed to contain 2 full turns.
        M_inj = self._M_inj
        M_cor = self._M_cor

        slice_x = np.s_[: (n_turns * nBPM["x"])]
        slice_y = np.s_[(n_turns * nBPM["x"]) :]
        slice_h = np.s_[: nCOR["x"]]
        slice_v = np.s_[nCOR["x"] :]

        slice_incl_x = np.s_[: incl_nObs["x"]]
        slice_incl_y = np.s_[incl_nObs["x"] :]

        M_cor_xx = M_cor[slice_x, slice_h]
        M_cor_xy = M_cor[slice_x, slice_v]
        M_cor_yx = M_cor[slice_y, slice_h]
        M_cor_yy = M_cor[slice_y, slice_v]

        M_inj_x = M_inj[slice_x, :]
        M_inj_y = M_inj[slice_y, :]

        # For NSLS-II:
        #   nBPM["x"] = nBPM["y"] = 180
        #   X-observs: 1st-turn 180 Horiz. BPMs + 2nd-turn 2 Horiz. BPMs = 182
        #   Y-observs: 1st-turn 180 Vert.  BPMs + 2nd-turn 2 Vert.  BPMs = 182
        #   X-knobs: 1 (inj. offset) + 1 (inj. angle) + 180 Horiz. Cors = 182
        #   Y-knobs: 1 (inj. offset) + 1 (inj. angle) + 180 Vert.  Cors = 182
        #     => M.shape = (364, 364)
        assert incl_nObs["x"] <= nBPM["x"] + self.second_turn_n_bpms
        assert incl_nObs["y"] <= nBPM["y"] + self.second_turn_n_bpms
        assert incl_nKnobs["x"] <= nCOR["x"]
        assert incl_nKnobs["y"] <= nCOR["y"]

        M = np.zeros(
            (
                incl_nObs["x"] + incl_nObs["y"],
                incl_nKnobs["x"] + incl_nKnobs["y"] + 2 + 2,
            )
        )

        # (1st-turn 180 & 2nd-turn 2) H-BPMs vs. (180 H-Cors + 180 V-Cors)
        M[slice_incl_x, n_inj_coords:] = np.hstack(
            (
                M_cor_xx[: incl_nObs["x"], : incl_nKnobs["x"]],
                M_cor_xy[: incl_nObs["x"], : incl_nKnobs["y"]],
            )
        )

        # (1st-turn 180 & 2nd-turn 2) V-BPMs vs. (180 H-Cors + 180 V-Cors)
        M[slice_incl_y, n_inj_coords:] = np.hstack(
            (
                M_cor_yx[: incl_nObs["y"], : incl_nKnobs["x"]],
                M_cor_yy[: incl_nObs["y"], : incl_nKnobs["y"]],
            )
        )

        # (1st-turn 180 & 2nd-turn 2) H-BPMs vs. X inj. offset
        M[slice_incl_x, 0] = M_inj_x[: incl_nObs["x"], 0]
        # (1st-turn 180 & 2nd-turn 2) V-BPMs vs. X inj. offset
        M[slice_incl_y, 0] = M_inj_y[: incl_nObs["y"], 0]

        # (1st-turn 180 & 2nd-turn 2) H-BPMs vs. X inj. angle
        M[slice_incl_x, 1] = M_inj_x[: incl_nObs["x"], 1]
        # (1st-turn 180 & 2nd-turn 2) V-BPMs vs. X inj. angle
        M[slice_incl_y, 1] = M_inj_y[: incl_nObs["y"], 1]

        # (1st-turn 180 & 2nd-turn 2) H-BPMs vs. Y inj. offset
        M[slice_incl_x, 2] = M_inj_x[: incl_nObs["x"], 2]
        # (1st-turn 180 & 2nd-turn 2) V-BPMs vs. Y inj. offset
        M[slice_incl_y, 2] = M_inj_y[: incl_nObs["y"], 2]

        # (1st-turn 180 & 2nd-turn 2) H-BPMs vs. Y inj. angle
        M[slice_incl_x, 3] = M_inj_x[: incl_nObs["x"], 3]
        # (1st-turn 180 & 2nd-turn 2) V-BPMs vs. Y inj. angle
        M[slice_incl_y, 3] = M_inj_y[: incl_nObs["y"], 3]

        # Keep only the non-zero rows
        M = M[~np.all(M == 0.0, axis=1), :]

        # Keep only the non-zero columns
        M = M[:, ~np.all(M == 0.0, axis=0)]

        return dict(M=M)

    def _partial_traj_correction(
        self, traj, inj_coords=None, n_iter_max=100, debug_print=False, debug_plot=False
    ):
        opts = self.iter_opts["fixed_energy"]["partial_traj"]
        n_iter_max = opts["n_iter_max"]
        cor_frac = opts["cor_frac"]

        n_inj_coords = 4
        if inj_coords is None:
            inj_coords = np.zeros(n_inj_coords)
        assert inj_coords.shape == (n_inj_coords,)

        i_iter_partial_traj_cor = 0
        while i_iter_partial_traj_cor < n_iter_max:
            if debug_print:
                print(
                    f"\n# Partial Traj. Cor. Iteration #{i_iter_partial_traj_cor+1}/{n_iter_max}"
                )

            survived_inds = self._get_survived_inds(traj)

            dxy = {}
            for plane in "xy":
                s_ = np.s_[: survived_inds[plane]]
                dxy[plane] = self.target_traj[plane][s_] - traj[plane][s_]

            incl_nObs, incl_nKnobs = self._get_indexes_selected_for_partial_traj_cor(
                dxy["x"], dxy["y"], debug_print=debug_print
            )

            _partial_opts = self.iter_opts["fixed_energy"]["partial_traj"]
            orig_obs_incl_dxy_thresh = _partial_opts["obs_incl_dxy_thresh"]
            while (
                (survived_inds["x"] < self.nBPM["x"] + self.second_turn_n_bpms)
                and (survived_inds["x"] == incl_nObs["x"])
            ) or (
                (survived_inds["y"] < self.nBPM["y"] + self.second_turn_n_bpms)
                and (survived_inds["y"] == incl_nObs["y"])
            ):
                _partial_opts["obs_incl_dxy_thresh"] /= 2

                msg = "* Bad correction detected. Temporarily reducing the RMS "
                msg += "threshold for 'well-corrected' BPMs by half to "
                msg += f"{_partial_opts['obs_incl_dxy_thresh']*1e3:.3f} [mm]."
                print(msg)

                (
                    incl_nObs,
                    incl_nKnobs,
                ) = self._get_indexes_selected_for_partial_traj_cor(
                    dxy["x"], dxy["y"], debug_print=debug_print
                )
            _partial_opts["obs_incl_dxy_thresh"] = orig_obs_incl_dxy_thresh  # restore

            assert survived_inds["x"] >= incl_nObs["x"]
            assert survived_inds["y"] >= incl_nObs["y"]

            if debug_print:
                survived_str = {}
                for plane in "xy":
                    _i = survived_inds[plane]
                    if _i <= self.nBPM[plane]:
                        survived_str[plane] = f"#{_i} at Turn #1"
                    else:
                        survived_str[plane] = f"#{_i - self.nBPM[plane]} at Turn #2"
                header = "Particle survived up to BPM (x, y)"
                print(f"{header} = ({survived_str['x']}, {survived_str['y']})")
                _nObsX = self.nBPM["x"] + self.second_turn_n_bpms
                _nObsY = self.nBPM["y"] + self.second_turn_n_bpms
                header = "Next iter. includes (x, y)"
                print(
                    f"{header} = ({incl_nObs['x']}/{_nObsX}, {incl_nObs['y']}/{_nObsY}) BPMs."
                )
                print(
                    f"{header} = ({incl_nKnobs['x']}/{self.nCOR['x']}, {incl_nKnobs['y']}/{self.nCOR['y']}) correctors."
                )

            if (incl_nKnobs["x"] == self.nCOR["x"]) and (
                incl_nKnobs["y"] == self.nCOR["y"]
            ):
                if debug_print:
                    print(
                        "\n*** Partial trajectory corrections successfully completed.\n"
                    )
                break

            _target_traj = np.append(
                self.target_traj["x"][: incl_nObs["x"]],
                self.target_traj["y"][: incl_nObs["y"]],
            )

            _current_traj = np.append(
                traj["x"][: incl_nObs["x"]], traj["y"][: incl_nObs["y"]]
            )

            if debug_print:
                _d_rms = {}
                for plane in "xy":
                    s_ = np.s_[: incl_nObs[plane]]
                    _d_rms[plane] = np.std(
                        self.target_traj[plane][s_] - traj[plane][s_]
                    )
                print(
                    f"Residual RMS [correctable] (x,y) [um] = ({_d_rms['x']*1e6:.3f}, {_d_rms['y']*1e6:.3f})"
                )

            dv = _target_traj - _current_traj

            _d = self._construct_full_TRM(incl_nObs, incl_nKnobs)
            M = _d["M"]

            U, sv, Vt = calcSVD(M)
            if False:
                plt.figure()
                plt.semilogy(sv / sv[0], ".-")
            assert np.all(sv / sv[0] > 1e-4)

            # Use all singular values by setting "rcond=1e-4"
            Sinv_trunc = calcTruncSVMatrix(sv, rcond=1e-4, nsv=None, disp=0)

            _Minv = Vt.T @ Sinv_trunc @ U.T

            dI = _Minv @ dv
            if debug_plot:
                plt.figure(figsize=(18, 6))
                plt.subplot(231)
                plt.plot(dI, ".-")
                plt.ylabel(r"$\Delta_I$", size="large")
                plt.tight_layout()

            dI *= cor_frac

            dI_inj = dI[:n_inj_coords]
            dI_cor = dI[n_inj_coords:]

            for _i, _dI in enumerate(dI_inj):
                inj_coords[_i] += _dI

            slice_x_cor_knobs = np.s_[: incl_nKnobs["x"]]
            slice_y_cor_knobs = np.s_[incl_nKnobs["x"] :]

            dtheta_H = dI_cor[slice_x_cor_knobs]
            dtheta_V = dI_cor[slice_y_cor_knobs]

            incl_cor_names_x = self.cor_names["x"][: incl_nKnobs["x"]]
            assert len(incl_cor_names_x) == len(dtheta_H)
            for elem_name, dtheta in zip(incl_cor_names_x, dtheta_H):
                self.change_corrector_setpoint_by(elem_name, "x", dtheta)
            incl_cor_names_y = self.cor_names["y"][: incl_nKnobs["y"]]
            assert len(incl_cor_names_y) == len(dtheta_V)
            for elem_name, dtheta in zip(incl_cor_names_y, dtheta_V):
                self.change_corrector_setpoint_by(elem_name, "y", dtheta)

            if debug_plot:
                plt.subplot(232)
                plt.plot(
                    np.array(
                        [
                            self.cor_props["x"][elem_name]["value"]
                            for elem_name in incl_cor_names_x
                        ]
                    ),
                    ".-",
                )
                plt.ylabel(r"$\theta_x$", size="large")
                plt.subplot(235)
                plt.plot(
                    np.array(
                        [
                            self.cor_props["y"][elem_name]["value"]
                            for elem_name in incl_cor_names_y
                        ]
                    ),
                    ".-",
                )
                plt.ylabel(r"$\theta_y$", size="large")
                plt.tight_layout()

            traj = self.calc_traj(
                x0=inj_coords[0],
                xp0=inj_coords[1],
                y0=inj_coords[2],
                yp0=inj_coords[3],
                dp0=0.0,
                debug_print=debug_print,
            )

            # Now adjust the next horizontal corrector and vertical corrector
            # right after the included correctors to correct the trajectory at the
            # next horizontal and vertical BPMs is corrected such that a good
            # trajectory direction is established for next iteration.

            _d = self._construct_TRM_for_next_bpm_cor_pair(incl_nObs, incl_nKnobs)
            M_next = _d["M"]
            next_bpm_index = _d["next_bpm_index"]
            next_cor_index = _d["next_cor_index"]

            U, sv, Vt = calcSVD(M_next)
            if False:
                plt.figure()
                plt.semilogy(sv / sv[0], ".-")
            assert np.all(sv / sv[0] > 1e-4)

            Sinv_trunc = calcTruncSVMatrix(sv, rcond=1e-4, nsv=None, disp=0)

            _Mnext_inv = Vt.T @ Sinv_trunc @ U.T

            next_bpm_dv = {}
            for plane, cor_i in next_cor_index.items():
                if cor_i is None:
                    continue

                bpm_i = next_bpm_index[plane]
                next_bpm_dv[plane] = self.target_traj[plane][bpm_i] - traj[plane][bpm_i]

            dI_next = _Mnext_inv @ np.array(list(next_bpm_dv.values()))
            dI_next *= cor_frac

            assert len(next_bpm_dv) == dI_next.size

            for plane, _dI in zip(list(next_bpm_dv), dI_next):
                elem_name = self.cor_names[plane][next_cor_index[plane]]
                self.change_corrector_setpoint_by(elem_name, plane, _dI)

            traj = self.calc_traj(
                x0=inj_coords[0],
                xp0=inj_coords[1],
                y0=inj_coords[2],
                yp0=inj_coords[3],
                dp0=0.0,
                debug_print=debug_print,
            )

            if debug_plot:
                plt.subplot(233)
                s_ = np.s_[: self.nBPM["x"] + 2]
                plt.plot(traj["x"][s_] - self.target_traj["x"][s_], ".-")
                plt.ylabel(r"$\Delta_x$", size="large")
                plt.subplot(236)
                s_ = np.s_[: self.nBPM["y"] + 2]
                plt.plot(traj["y"][s_] - self.target_traj["y"][s_], ".-")
                plt.ylabel(r"$\Delta_y$", size="large")

                plt.tight_layout()

            i_iter_partial_traj_cor += 1

    def change_corrector_setpoint(self, elem_name: str, plane: str, value: float):
        self.cor_props[plane][elem_name]["value"] = value
        self._uncommited_cor_change = True

    def change_corrector_setpoint_by(
        self, elem_name: str, plane: str, delta_value: float
    ):
        self.cor_props[plane][elem_name]["value"] += delta_value
        self._uncommited_cor_change = True


def _get_twiss_for_analytical_resp_mat(
    LTE: Lattice,
    bpm_x_elem_inds: Union[Union[List[int], np.ndarray], None] = None,
    bpm_y_elem_inds: Union[Union[List[int], np.ndarray], None] = None,
    cor_x_elem_inds: Union[Union[List[int], np.ndarray], None] = None,
    cor_y_elem_inds: Union[Union[List[int], np.ndarray], None] = None,
    elem_name_verification: Union[None, Dict] = None,
):
    assert isinstance(LTE, Lattice)

    LTE_filepath = LTE.LTE_filepath

    with tempfile.NamedTemporaryFile(suffix=".pgz", dir=None, delete=True) as f:
        output_filepath = f.name
        E_MeV = 3e3  # This value doesn't matter for this computation.
        twiss.calc_ring_twiss(
            output_filepath, LTE_filepath, E_MeV, use_beamline=LTE.used_beamline_name
        )

        twi = util.load_pgz_file(output_filepath)

    twi_arrays = twi["data"]["twi"]["arrays"]
    twi_scalars = twi["data"]["twi"]["scalars"]

    beta = {plane: twi_arrays[f"beta{plane}"] for plane in "xy"}
    phi = {plane: twi_arrays[f"psi{plane}"] for plane in "xy"}  # [rad]
    eta = {plane: twi_arrays[f"eta{plane}"] for plane in "xy"}
    nu = {plane: twi_scalars[f"nu{plane}"] for plane in "xy"}
    alphac = twi_scalars["alphac"]

    circumf = twi_arrays["s"][-1]

    bpm_elem_inds = {}

    if bpm_x_elem_inds is None:
        bpm_elem_inds["x"] = []
    else:
        bpm_elem_inds["x"] = bpm_x_elem_inds

    if bpm_y_elem_inds is None:
        bpm_elem_inds["y"] = []
    else:
        bpm_elem_inds["y"] = bpm_y_elem_inds

    cor_elem_inds = {}

    if cor_x_elem_inds is None:
        cor_elem_inds["x"] = []
    else:
        cor_elem_inds["x"] = cor_x_elem_inds

    if cor_y_elem_inds is None:
        cor_elem_inds["y"] = []
    else:
        cor_elem_inds["y"] = cor_y_elem_inds

    bpm_elem_inds = {plane: np.array(bpm_elem_inds[plane]) for plane in "xy"}
    cor_elem_inds = {plane: np.array(cor_elem_inds[plane]) for plane in "xy"}

    n_bpm = {plane: len(bpm_elem_inds[plane]) for plane in "xy"}
    n_cor = {plane: len(cor_elem_inds[plane]) for plane in "xy"}

    assert n_bpm["x"] + n_bpm["y"] >= 1
    assert n_cor["x"] + n_cor["y"] >= 1
    for plane in "xy":
        if n_bpm[plane] == 0:
            assert n_cor[plane] == 0
        if n_cor[plane] == 0:
            assert n_bpm[plane] == 0

    if elem_name_verification:
        for elem_type, expected_elem_names in elem_name_verification.items():
            if elem_type == "bpm_x":
                if n_bpm["x"] == 0:
                    raise AssertionError("Element indexes for 'bpm_x' was empty.")
                try:
                    twi_elem_names = twi_arrays["ElementName"][bpm_elem_inds["x"]]
                    assert np.all(twi_elem_names == expected_elem_names)
                except:
                    raise AssertionError(
                        "Provided element names do not match with the given element indexes for 'bpm_x'."
                    )

            if elem_type == "bpm_y":
                if n_bpm["y"] == 0:
                    raise AssertionError("Element indexes for 'bpm_y' was empty.")
                try:
                    twi_elem_names = twi_arrays["ElementName"][bpm_elem_inds["y"]]
                    assert np.all(twi_elem_names == expected_elem_names)
                except:
                    raise AssertionError(
                        "Provided element names do not match with the given element indexes for 'bpm_y'."
                    )

            if elem_type == "cor_x":
                if n_cor["x"] == 0:
                    raise AssertionError("Element indexes for 'cor_x' was empty.")
                try:
                    twi_elem_names = twi_arrays["ElementName"][cor_elem_inds["x"]]
                    assert np.all(twi_elem_names == expected_elem_names)
                except:
                    raise AssertionError(
                        "Provided element names do not match with the given element indexes for 'cor_x'."
                    )

            if elem_type == "cor_y":
                if n_cor["y"] == 0:
                    raise AssertionError("Element indexes for 'cor_y' was empty.")
                try:
                    twi_elem_names = twi_arrays["ElementName"][cor_elem_inds["y"]]
                    assert np.all(twi_elem_names == expected_elem_names)
                except:
                    raise AssertionError(
                        "Provided element names do not match with the given element indexes for 'cor_y'."
                    )

    beta_avg_at_cors = {}
    phi_avg_at_cors = {}
    eta_avg_at_cors = {}
    for plane in "xy":
        if n_cor[plane] == 0:
            continue
        b_s_ = cor_elem_inds[plane] - 1
        e_s_ = cor_elem_inds[plane]

        beta_avg_at_cors[plane] = (beta[plane][b_s_] + beta[plane][e_s_]) / 2
        phi_avg_at_cors[plane] = (phi[plane][b_s_] + phi[plane][e_s_]) / 2
        eta_avg_at_cors[plane] = (eta[plane][b_s_] + eta[plane][e_s_]) / 2

    beta_at_bpms = {}
    phi_at_bpms = {}
    eta_at_bpms = {}
    for plane in "xy":
        if n_bpm[plane] == 0:
            continue

        inds = bpm_elem_inds[plane]

        beta_at_bpms[plane] = beta[plane][inds]
        phi_at_bpms[plane] = phi[plane][inds]
        eta_at_bpms[plane] = eta[plane][inds]

    return dict(
        element_names=twi_arrays["ElementName"],
        bpm_elem_inds=bpm_elem_inds,
        cor_elem_inds=cor_elem_inds,
        beta=beta,
        phi=phi,
        eta=eta,
        nu=nu,
        alphac=alphac,
        circumf=circumf,
        n_bpm=n_bpm,
        n_cor=n_cor,
        beta_at_bpms=beta_at_bpms,
        phi_at_bpms=phi_at_bpms,
        eta_at_bpms=eta_at_bpms,
        beta_avg_at_cors=beta_avg_at_cors,
        phi_avg_at_cors=phi_avg_at_cors,
        eta_avg_at_cors=eta_avg_at_cors,
    )


def calc_analytical_uncoupled_ORM(
    LTE: Lattice,
    bpm_x_elem_inds: Union[Union[List[int], np.ndarray], None] = None,
    bpm_y_elem_inds: Union[Union[List[int], np.ndarray], None] = None,
    cor_x_elem_inds: Union[Union[List[int], np.ndarray], None] = None,
    cor_y_elem_inds: Union[Union[List[int], np.ndarray], None] = None,
    twiss_dict: Union[Dict, None] = None,
) -> Dict:
    """
    Calculates the analytical orbit response matrix (ORM) for a given lattice.
    No transverse coupling is assumed. The unit of the returned ORM is [m/rad].

    The wrong sign for the horizontal dispersion term in Eq. (2.34) of Minty &
    Zimmermann has been corrected. (For vertical orbit, the sign still needs to
    be checked.)
    """

    if twiss_dict is None:
        twiss_dict = _get_twiss_for_analytical_resp_mat(
            LTE, bpm_x_elem_inds, bpm_y_elem_inds, cor_x_elem_inds, cor_y_elem_inds
        )
    nu = twiss_dict["nu"]
    alphac = twiss_dict["alphac"]
    circumf = twiss_dict["circumf"]
    n_bpm = twiss_dict["n_bpm"]
    n_cor = twiss_dict["n_cor"]
    beta_at_bpms = twiss_dict["beta_at_bpms"]
    phi_at_bpms = twiss_dict["phi_at_bpms"]
    eta_at_bpms = twiss_dict["eta_at_bpms"]
    beta_avg_at_cors = twiss_dict["beta_avg_at_cors"]
    phi_avg_at_cors = twiss_dict["phi_avg_at_cors"]
    eta_avg_at_cors = twiss_dict["eta_avg_at_cors"]

    M = np.zeros((n_bpm["x"] + n_bpm["y"], n_cor["x"] + n_cor["y"]))

    beta_c = beta_avg_at_cors["x"]
    beta_b = beta_at_bpms["x"]
    phi_c = phi_avg_at_cors["x"]
    phi_b = phi_at_bpms["x"]
    eta_c = eta_avg_at_cors["x"]
    eta_b = eta_at_bpms["x"]
    tune = nu["x"]

    for i in range(n_bpm["x"]):
        M[i, : n_cor["x"]] = (
            np.sqrt(beta_c * beta_b[i])
            * np.cos(np.pi * tune - np.abs(phi_c - phi_b[i]))
            / (2 * np.sin(np.pi * tune))
            - eta_c * eta_b[i] / alphac / circumf
        )

    beta_c = beta_avg_at_cors["y"]
    beta_b = beta_at_bpms["y"]
    phi_c = phi_avg_at_cors["y"]
    phi_b = phi_at_bpms["y"]
    eta_c = eta_avg_at_cors["x"]
    eta_b = eta_at_bpms["x"]
    tune = nu["y"]

    for i in range(n_bpm["y"]):
        M[i + n_bpm["x"], n_cor["x"] :] = (
            np.sqrt(beta_c * beta_b[i])
            * np.cos(np.pi * tune - np.abs(phi_c - phi_b[i]))
            / (2 * np.sin(np.pi * tune))
            - eta_c * eta_b[i] / alphac / circumf
        )

    return dict(M=M, twiss_dict=twiss_dict)


def calc_analytical_uncoupled_TRM(
    LTE: Lattice,
    bpm_x_elem_inds: Union[Union[List[int], np.ndarray], None] = None,
    bpm_y_elem_inds: Union[Union[List[int], np.ndarray], None] = None,
    cor_x_elem_inds: Union[Union[List[int], np.ndarray], None] = None,
    cor_y_elem_inds: Union[Union[List[int], np.ndarray], None] = None,
    twiss_dict: Union[Dict, None] = None,
    n_turns: int = 2,
) -> Dict:
    """
    Calculates the analytical trajectory response matrix (TRM) for a given lattice.
    No transverse coupling is assumed. The unit of the returned TRM is [m/rad].

    The wrong sign for the horizontal dispersion term in Eq. (2.34) of Minty &
    Zimmermann has been corrected. (For vertical orbit, the sign still needs to
    be checked.)
    """

    assert n_turns >= 1

    if twiss_dict is None:
        twiss_dict = _get_twiss_for_analytical_resp_mat(
            LTE, bpm_x_elem_inds, bpm_y_elem_inds, cor_x_elem_inds, cor_y_elem_inds
        )
    nu = twiss_dict["nu"]
    n_bpm = twiss_dict["n_bpm"]
    n_cor = twiss_dict["n_cor"]
    beta_at_bpms = twiss_dict["beta_at_bpms"]
    phi_at_bpms = twiss_dict["phi_at_bpms"]
    beta_avg_at_cors = twiss_dict["beta_avg_at_cors"]
    phi_avg_at_cors = twiss_dict["phi_avg_at_cors"]

    M = {plane: np.zeros((n_turns, n_bpm[plane], n_cor[plane])) for plane in "xy"}

    M_flat = {}
    for plane in "xy":
        beta_c = beta_avg_at_cors[plane]
        beta_b = beta_at_bpms[plane]
        phi_c = phi_avg_at_cors[plane]
        phi_b = phi_at_bpms[plane]
        tune = nu[plane]

        for i_turn in range(n_turns):
            for i in range(n_bpm[plane]):
                dM = np.sqrt(beta_c * beta_b[i]) * np.sin(
                    (phi_b[i] + 2 * np.pi * tune * i_turn) - phi_c
                )
                if i_turn == 0:
                    M[plane][i_turn, i, :] = dM
                else:
                    M[plane][i_turn, i, :] = M[plane][i_turn - 1, i, :] + dM

            if i_turn == 0:
                # Just for the first turn, zero out the responses at BPMs
                # upstream of each corrector.
                for j in range(n_cor[plane]):
                    upstream = phi_b < phi_c[j]
                    M[plane][i_turn, upstream, j] = 0.0

        M_flat[plane] = M[plane].reshape((n_bpm[plane] * n_turns, n_cor[plane]))

    n_rows = (n_bpm["x"] + n_bpm["y"]) * n_turns
    n_cols = n_cor["x"] + n_cor["y"]
    M = np.zeros((n_rows, n_cols))
    M[: (n_bpm["x"] * n_turns), : n_cor["x"]] = M_flat["x"]
    M[(n_bpm["x"] * n_turns) :, n_cor["x"] :] = M_flat["y"]

    if False:
        U, sv, Vt = calcSVD(M)

        plt.figure()
        plt.semilogy(sv / sv[0], ".-")

        plt.figure()
        plt.plot(M[:, 0])

        plt.figure()
        plt.plot(M[:, 130])

        plt.figure()
        plt.plot(M[:, 180 + 179])

    return dict(M=M, twiss_dict=twiss_dict)


def calc_numerical_corrector_TRM(
    LTE: Lattice,
    bpm_x_elem_inds: Union[Union[List[int], np.ndarray], None] = None,
    bpm_y_elem_inds: Union[Union[List[int], np.ndarray], None] = None,
    cor_x_elem_inds: Union[Union[List[int], np.ndarray], None] = None,
    cor_y_elem_inds: Union[Union[List[int], np.ndarray], None] = None,
    twiss_dict: Union[Dict, None] = None,
    n_turns: int = 2,
    **kwargs,
) -> Dict:
    """
    Calculates the numerical trajectory response matrix (TRM) for a given lattice.
    The unit of the returned TRM is [m/rad].
    """

    std_print_enabled["out"] = False

    assert n_turns >= 1

    if twiss_dict is None:
        twiss_dict = _get_twiss_for_analytical_resp_mat(
            LTE, bpm_x_elem_inds, bpm_y_elem_inds, cor_x_elem_inds, cor_y_elem_inds
        )
    n_bpm = twiss_dict["n_bpm"]
    n_cor = twiss_dict["n_cor"]
    elem_names = twiss_dict["element_names"]
    bpm_elem_inds = twiss_dict["bpm_elem_inds"]
    cor_elem_inds = twiss_dict["cor_elem_inds"]

    bpmx_names = elem_names[bpm_elem_inds["x"]]
    bpmy_names = elem_names[bpm_elem_inds["y"]]
    hcor_names = elem_names[cor_elem_inds["x"]]
    vcor_names = elem_names[cor_elem_inds["y"]]

    E_MeV = kwargs.get("E_MeV", 3e3)  # This value shoudn't matter for this calculation.
    N_KICKS = kwargs.get("N_KICKS", None)
    tempdir_path = kwargs.get("tempdir_path", None)

    run_mode = kwargs.get("mode", "serial")
    nCPUs = kwargs.get("nCPUs", 1)
    partition = kwargs.get("partition", "normal")
    time_limit_str = kwargs.get("time_limit_str", "6:00:00")

    cot_args = (LTE, E_MeV, bpmx_names, bpmy_names, hcor_names, vcor_names)
    cot_kwargs = dict(
        zero_orbit_type="design",
        N_KICKS=N_KICKS,
        tempdir_path=tempdir_path,
    )

    cot = ClosedOrbitThreader(*cot_args, **cot_kwargs)

    kick = 1e-6  # [rad]; 1 urad for trim kicks

    cor_names_planes = [(name, "x") for name in hcor_names]
    cor_names_planes += [(name, "y") for name in vcor_names]

    if run_mode == "serial":
        M_cols = _calc_numerical_corrector_TRM_columns(
            cor_names_planes, kick, cot=cot, verbose=True
        )

        M = np.vstack(M_cols).T
        assert M.shape == ((n_bpm["x"] + n_bpm["y"]) * n_turns, n_cor["x"] + n_cor["y"])

    elif run_mode == "mpi":
        # This folder must be different for all worker nodes, as WATCH files
        # generated by different workers will collide. So, if it must be
        # set to `None`.
        cot_kwargs["tempdir_path"] = None

        if nCPUs > len(cor_names_planes):
            nCPUs = len(cor_names_planes)

        remote_opts = dict(
            job_name="job",
            partition=partition,
            ntasks=nCPUs,
            time=time_limit_str,
            qos="long",
        )

        chunked_list, reverse_mapping = util.chunk_list(cor_names_planes, nCPUs)

        module_name = "pyelegant.orbit"
        func_name = "_calc_numerical_corrector_TRM_columns"
        _cot = None
        verbose = False
        func_args = (kick, _cot, cot_args, cot_kwargs, verbose)
        chunked_output = remote.run_mpi_python(
            remote_opts,
            module_name,
            func_name,
            chunked_list,
            func_args,
            err_log_check=dict(funcs=[remote.check_remote_err_log_exit_code]),
            paths_to_prepend=[str(Path(__file__).parent)],
        )

        unchunked = util.unchunk_list_of_lists(chunked_output, reverse_mapping)
        M = np.array(unchunked).T

    return dict(M=M, twiss_dict=twiss_dict)


def _calc_numerical_corrector_TRM_columns(
    sel_cor_names_planes: Union[List, Tuple],
    kick: float,
    cot: ClosedOrbitThreader = None,
    cot_args: Union[Tuple, None] = None,
    cot_kwargs: Union[Dict, None] = None,
    verbose=False,
):
    std_print_enabled["out"] = False

    if cot is None:
        cot = ClosedOrbitThreader(*cot_args, **cot_kwargs)
    else:
        assert isinstance(cot, ClosedOrbitThreader)

    bpm_map = defaultdict(list)
    for plane in "xy":
        inds = np.array(cot.u_bpm_names_to_bpm_inds[plane])
        for i_turn in range(cot.n_turns):
            bpm_map[plane].append(cot.nBPM["xy"] * i_turn + inds)
        bpm_map[plane] = np.hstack(bpm_map[plane])

    M_cols = []

    if verbose:
        N = len(sel_cor_names_planes)

    ini_coords = dict(x0=0.0, y0=0.0, xp0=0.0, yp0=0.0, dp0=0.0)

    for i_cor, (elem_name, plane) in enumerate(sel_cor_names_planes):
        if verbose:
            print(f"Corrector #{i_cor+1} / {N}")

        # Positive horizontal/vertical kick
        cot.change_corrector_setpoint(elem_name, plane, kick)
        traj = cot.calc_traj(**ini_coords)
        traj_xP = traj["x"][bpm_map["x"]]
        traj_yP = traj["y"][bpm_map["y"]]

        # Negative horizontal/vertical kick
        cot.change_corrector_setpoint(elem_name, plane, -kick)
        traj = cot.calc_traj(**ini_coords)
        traj_xN = traj["x"][bpm_map["x"]]
        traj_yN = traj["y"][bpm_map["y"]]

        # Reset kick
        cot.change_corrector_setpoint(elem_name, plane, 0.0)

        M_cols.append(np.append(traj_xP - traj_xN, traj_yP - traj_yN) / (2.0 * kick))

    return M_cols


def _calc_inj_offset_angle_TRM(
    cot: ClosedOrbitThreader = None,
    cot_args: Union[Tuple, None] = None,
    cot_kwargs: Union[Dict, None] = None,
):
    std_print_enabled["out"] = False

    if cot is None:
        cot = ClosedOrbitThreader(*cot_args, **cot_kwargs)
    else:
        assert isinstance(cot, ClosedOrbitThreader)

    bpm_map = defaultdict(list)
    for plane in "xy":
        inds = np.array(cot.u_bpm_names_to_bpm_inds[plane])
        for i_turn in range(cot.n_turns):
            bpm_map[plane].append(cot.nBPM["xy"] * i_turn + inds)
        bpm_map[plane] = np.hstack(bpm_map[plane])

    # "4" is for x0, y0, xp0, yp0
    M = np.zeros(((cot.nBPM["x"] + cot.nBPM["y"]) * cot.n_turns, 4))

    base_traj = cot.calc_traj(x0=0.0, y0=0.0, xp0=0.0, yp0=0.0, dp0=0.0)
    assert np.all(np.abs(base_traj["x"]) < 10e-9)
    assert np.all(np.abs(base_traj["y"]) < 1e-9)

    inj_angle = 1e-6  # [rad]
    inj_offset = 1e-6  # [m]

    print("* Calculating traj for positive x inj. offset...")
    traj = cot.calc_traj(x0=inj_offset, y0=0.0, xp0=0.0, yp0=0.0, dp0=0.0)
    traj_xP = traj["x"][bpm_map["x"]]
    traj_yP = traj["y"][bpm_map["y"]]

    print("* Calculating traj for negative x inj. offset...")
    traj = cot.calc_traj(x0=-inj_offset, y0=0.0, xp0=0.0, yp0=0.0, dp0=0.0)
    traj_xN = traj["x"][bpm_map["x"]]
    traj_yN = traj["y"][bpm_map["y"]]

    M[:, 0] = np.append(traj_xP - traj_xN, traj_yP - traj_yN) / (2.0 * inj_offset)

    print("* Calculating traj for positive x inj. angle...")
    traj = cot.calc_traj(x0=0.0, y0=0.0, xp0=inj_angle, yp0=0.0, dp0=0.0)
    traj_xP = traj["x"][bpm_map["x"]]
    traj_yP = traj["y"][bpm_map["y"]]

    print("* Calculating traj for negative x inj. angle...")
    traj = cot.calc_traj(x0=0.0, y0=0.0, xp0=-inj_angle, yp0=0.0, dp0=0.0)
    traj_xN = traj["x"][bpm_map["x"]]
    traj_yN = traj["y"][bpm_map["y"]]

    M[:, 1] = np.append(traj_xP - traj_xN, traj_yP - traj_yN) / (2.0 * inj_angle)

    print("* Calculating traj for positive y inj. offset...")
    traj = cot.calc_traj(x0=0.0, y0=inj_offset, xp0=0.0, yp0=0.0, dp0=0.0)
    traj_xP = traj["x"][bpm_map["x"]]
    traj_yP = traj["y"][bpm_map["y"]]

    print("* Calculating traj for negative y inj. offset...")
    traj = cot.calc_traj(x0=0.0, y0=-inj_offset, xp0=0.0, yp0=0.0, dp0=0.0)
    traj_xN = traj["x"][bpm_map["x"]]
    traj_yN = traj["y"][bpm_map["y"]]

    M[:, 2] = np.append(traj_xP - traj_xN, traj_yP - traj_yN) / (2.0 * inj_offset)

    print("* Calculating traj for positive y inj. angle...")
    traj = cot.calc_traj(x0=0.0, y0=0.0, xp0=0.0, yp0=inj_angle, dp0=0.0)
    traj_xP = traj["x"][bpm_map["x"]]
    traj_yP = traj["y"][bpm_map["y"]]

    print("* Calculating traj for negative y inj. angle...")
    traj = cot.calc_traj(x0=0.0, y0=0.0, xp0=0.0, yp0=-inj_angle, dp0=0.0)
    traj_xN = traj["x"][bpm_map["x"]]
    traj_yN = traj["y"][bpm_map["y"]]

    M[:, 3] = np.append(traj_xP - traj_xN, traj_yP - traj_yN) / (2.0 * inj_angle)

    return M


def calc_numerical_inj_phase_space_TRM(
    LTE: Lattice,
    bpm_x_elem_inds: Union[Union[List[int], np.ndarray], None] = None,
    bpm_y_elem_inds: Union[Union[List[int], np.ndarray], None] = None,
    cor_x_elem_inds: Union[Union[List[int], np.ndarray], None] = None,
    cor_y_elem_inds: Union[Union[List[int], np.ndarray], None] = None,
    twiss_dict: Union[Dict, None] = None,
    n_turns: int = 2,
    **kwargs,
) -> np.ndarray:
    """
    Calculates the numerical trajectory response matrix (TRM) for a given lattice
    with respect to the transverse phase-space coordinates (x, xp, y, yp) at the
    injection point (i.e., s=0).

    The unit of the returned TRM is either [m/rad] for x (1st col.) and y (3rd col.),
    and [rad/rad] for xp (2nd col.) and yp (4th col.).
    """

    std_print_enabled["out"] = False

    assert n_turns >= 1

    if twiss_dict is None:
        twiss_dict = _get_twiss_for_analytical_resp_mat(
            LTE, bpm_x_elem_inds, bpm_y_elem_inds, cor_x_elem_inds, cor_y_elem_inds
        )
    elem_names = twiss_dict["element_names"]
    bpm_elem_inds = twiss_dict["bpm_elem_inds"]
    cor_elem_inds = twiss_dict["cor_elem_inds"]

    bpmx_names = elem_names[bpm_elem_inds["x"]]
    bpmy_names = elem_names[bpm_elem_inds["y"]]
    hcor_names = elem_names[cor_elem_inds["x"]]
    vcor_names = elem_names[cor_elem_inds["y"]]

    E_MeV = kwargs.get("E_MeV", 3e3)  # This value shoudn't matter for this calculation.
    N_KICKS = kwargs.get("N_KICKS", None)
    tempdir_path = kwargs.get("tempdir_path", None)

    cot_args = (LTE, E_MeV, bpmx_names, bpmy_names, hcor_names, vcor_names)
    cot_kwargs = dict(
        zero_orbit_type="design",
        N_KICKS=N_KICKS,
        tempdir_path=tempdir_path,
    )

    cot = ClosedOrbitThreader(*cot_args, **cot_kwargs)

    M_inj = _calc_inj_offset_angle_TRM(cot=cot)

    return dict(M=M_inj, twiss_dict=twiss_dict)


def generate_TRM_file(
    LTE: Lattice,
    bpm_x_elem_inds,
    bpm_y_elem_inds,
    cor_x_elem_inds,
    cor_y_elem_inds,
    n_turns=2,
    N_KICKS=None,
):
    if LTE.LTEZIP_filepath:
        LTE_source = dict(
            LTEZIP_filepath=str(LTE.LTEZIP_filepath.resolve()),
            hash_val=util.calculate_file_hash(LTE.LTEZIP_filepath),
        )
        trm_filepath = f"{LTE.LTEZIP_filepath.stem}_TRM.h5"
    else:
        LTE_source = dict(
            LTE_filepath=str(LTE.LTE_filepath.resolve()),
            used_beamline_name=LTE.used_beamline_name,
            hash_val=util.calculate_file_hash(LTE.LTE_filepath),
        )
        trm_filepath = f"{LTE.LTE_filepath.stem}_TRM.h5"

    out = calc_numerical_inj_phase_space_TRM(
        LTE,
        bpm_x_elem_inds,
        bpm_y_elem_inds,
        cor_x_elem_inds,
        cor_y_elem_inds,
        n_turns=n_turns,
        N_KICKS=N_KICKS,  # CRITICAL
    )
    M_inj = out["M"]
    twiss_dict = out["twiss_dict"]

    out = calc_analytical_uncoupled_TRM(
        LTE,
        bpm_x_elem_inds,
        bpm_y_elem_inds,
        cor_x_elem_inds,
        cor_y_elem_inds,
        n_turns=n_turns,
        twiss_dict=twiss_dict,
    )
    M_ana = out["M"]

    if False:
        M_num = calc_numerical_corrector_TRM(
            LTE,
            bpm_x_elem_inds,
            bpm_y_elem_inds,
            cor_x_elem_inds,
            cor_y_elem_inds,
            twiss_dict=twiss_dict,
            n_turns=n_turns,
            N_KICKS=N_KICKS,  # CRITICAL
            # mode="serial",
            mode="mpi",
            # nCPUs=40,
            # partition="debug",
            nCPUs=360,
            partition="normal",
            # time_limit_str="10:00",
            time_limit_str="30:00",
        )

        plt.figure()
        plt.imshow(M_num)

        plt.figure()
        plt.imshow(M_ana)

        plt.figure()
        i_cor = 0
        plt.plot(M_num[:, i_cor], "b.-")
        plt.plot(M_ana[:, i_cor], "r.-")

        plt.figure()
        i_cor = 0
        plt.subplot(211)
        plt.hist(M_num[:, i_cor])
        plt.subplot(212)
        plt.hist(M_ana[:, i_cor] - M_num[:, i_cor])

        plt.figure()
        i_cor = 0
        ref_mag_thresh = np.max(np.abs(M_num[:, i_cor])) * 0.1
        above_thresh = np.abs(M_num[:, i_cor]) > ref_mag_thresh
        below_thresh = ~above_thresh
        plt.subplot(211)
        plt.plot(M_ana[:, i_cor][above_thresh] / M_num[:, i_cor][above_thresh], "b.-")
        plt.subplot(212)
        plt.plot(M_ana[:, i_cor][below_thresh] - M_num[:, i_cor][below_thresh], "b.-")

        std_ratios = []
        max_abs_ratios = []
        for i_cor in range(M_num.shape[1]):
            dM = M_ana[:, i_cor] - M_num[:, i_cor]
            std_ratios.append(np.std(dM) / np.std(M_num[:, i_cor]))
            max_abs_ratios.append(np.max(np.abs(dM)) / np.max(np.abs(M_num[:, i_cor])))

        plt.figure()
        plt.plot(std_ratios, "b.-")
        plt.plot(max_abs_ratios, "r.-")

    kw = dict(compression="gzip")
    with h5py.File(trm_filepath, "w") as f:
        f.create_dataset("M_inj", data=M_inj, **kw)
        f.create_dataset("M_cor", data=M_ana, **kw)
        f["alphac"] = twiss_dict["alphac"]  # momentum compaction
        bpm_x_names = LTE.get_names_from_elem_inds(bpm_x_elem_inds).tolist()
        bpm_y_names = LTE.get_names_from_elem_inds(bpm_y_elem_inds).tolist()
        f["bpm_names"] = bpm_x_names + bpm_y_names
        f["bpm_fields"] = ["x"] * len(bpm_x_names) + ["y"] * len(bpm_y_names)
        cor_x_names = LTE.get_names_from_elem_inds(cor_x_elem_inds).tolist()
        cor_y_names = LTE.get_names_from_elem_inds(cor_y_elem_inds).tolist()
        f["cor_names"] = cor_x_names + cor_y_names
        f["cor_fields"] = ["x"] * len(cor_x_names) + ["y"] * len(cor_y_names)
        g = f.create_group("LTE_source")
        for k, v in LTE_source.items():
            g[k] = v
