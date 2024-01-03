import base64
from collections import defaultdict
from itertools import product
import os
from pathlib import Path
import pickle
from subprocess import PIPE, Popen
import tempfile
import time
from typing import Dict, List, Tuple, Union

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from . import elebuilder, eleutil, sdds, sigproc, std_print_enabled
from .local import run
from .ltemanager import Lattice
from .orbit import ClosedOrbitCalculatorViaTraj
from .remote import remote
from .respmat import calcSVD, calcTruncSVMatrix
from .sigproc import unwrap_montonically_increasing
from .twiss import calc_ring_twiss
from .util import load_pgz_file


def tunes(
    corrected_LTE_filepath,
    init_LTE_filepath,
    E_MeV,
    use_beamline=None,
    ele_filepath=None,
    del_tmp_files=True,
    quadrupoles=None,
    exclude=None,
    tune_x=0.0,
    tune_y=0.0,
    n_iterations=5,
    correction_fraction=0.9,
    tolerance=0.0,
    run_local=True,
    remote_opts=None,
):
    """"""

    if quadrupoles is None:
        raise ValueError('"quadrupoles" must be a list of strings.')
    used_quads_str = " ".join(quadrupoles)

    if ele_filepath is None:
        tmp = tempfile.NamedTemporaryFile(
            dir=os.getcwd(), delete=False, prefix="tmpTunes_", suffix=".ele"
        )
        ele_filepath = os.path.abspath(tmp.name)
        tmp.close()

    ed = elebuilder.EleDesigner(ele_filepath, double_format=".12g")

    ed.add_block(
        "run_setup",
        lattice=init_LTE_filepath,
        p_central_mev=E_MeV,
        use_beamline=use_beamline,
    )

    ed.add_newline()

    ed.add_block("run_control")

    ed.add_newline()

    ed.add_block(
        "correct_tunes",
        quadrupoles=used_quads_str,
        exclude=exclude,
        tune_x=tune_x,
        tune_y=tune_y,
        n_iterations=n_iterations,
        correction_fraction=correction_fraction,
        tolerance=tolerance,
        change_defined_values=True,
    )

    ed.add_newline()

    ed.add_block("twiss_output")

    ed.add_newline()

    ed.add_block("bunched_beam")

    ed.add_newline()

    ed.add_block("track")

    ed.add_block("save_lattice", filename=corrected_LTE_filepath)

    ed.write()
    # print(ed.actual_output_filepath_list)

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
            remote_opts = dict(sbatch={"use": False})

        if ("pelegant" in remote_opts) and (remote_opts["pelegant"] is not False):
            print('"pelegant" option in `remote_opts` must be False.')
            remote_opts["pelegant"] = False
        else:
            remote_opts["pelegant"] = False

        remote_opts["ntasks"] = 1
        # ^ If this is more than 1, you will likely see an error like "Unable to
        #   access file /.../tmp*.twi--file is locked (SDDS_InitializeOutput)"

        remote.run(
            remote_opts,
            ele_filepath,
            print_cmd=True,
            print_stdout=std_print_enabled["out"],
            print_stderr=std_print_enabled["err"],
            output_filepaths=None,
        )

    if del_tmp_files:
        for fp in ed.actual_output_filepath_list + [ele_filepath]:
            if fp.startswith("/dev"):
                continue
            elif fp == corrected_LTE_filepath:
                continue
            else:
                try:
                    os.remove(fp)
                except:
                    print(f'Failed to delete "{fp}"')


def chroms(
    corrected_LTE_filepath,
    init_LTE_filepath,
    E_MeV,
    use_beamline=None,
    ele_filepath=None,
    del_tmp_files=True,
    sextupoles=None,
    exclude=None,
    dnux_dp=0.0,
    dnuy_dp=0.0,
    n_iterations=5,
    correction_fraction=0.9,
    tolerance=0.0,
    max_abs_K2=None,
    run_local=True,
    remote_opts=None,
):
    """"""

    if sextupoles is None:
        raise ValueError('"sextupoles" must be a list of strings.')
    used_sexts_str = " ".join(sextupoles)

    if ele_filepath is None:
        tmp = tempfile.NamedTemporaryFile(
            dir=os.getcwd(), delete=False, prefix="tmpChroms_", suffix=".ele"
        )
        ele_filepath = os.path.abspath(tmp.name)
        tmp.close()

    ed = elebuilder.EleDesigner(ele_filepath, double_format=".12g")

    ed.add_block(
        "run_setup",
        lattice=init_LTE_filepath,
        p_central_mev=E_MeV,
        use_beamline=use_beamline,
    )

    ed.add_newline()

    ed.add_block("run_control")

    ed.add_newline()

    ed.add_block(
        "chromaticity",
        sextupoles=used_sexts_str,
        exclude=exclude,
        dnux_dp=dnux_dp,
        dnuy_dp=dnuy_dp,
        n_iterations=n_iterations,
        correction_fraction=correction_fraction,
        tolerance=tolerance,
        change_defined_values=True,
        strength_limit=(0.0 if max_abs_K2 is None else max_abs_K2),
    )

    ed.add_newline()

    ed.add_block("twiss_output")

    ed.add_newline()

    ed.add_block("bunched_beam")

    ed.add_newline()

    ed.add_block("track")

    ed.add_block("save_lattice", filename=corrected_LTE_filepath)

    ed.write()
    # print(ed.actual_output_filepath_list)

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
            remote_opts = dict(sbatch={"use": False})

        if ("pelegant" in remote_opts) and (remote_opts["pelegant"] is not False):
            print('"pelegant" option in `remote_opts` must be False.')
            remote_opts["pelegant"] = False
        else:
            remote_opts["pelegant"] = False

        remote_opts["ntasks"] = 1
        # ^ If this is more than 1, you will likely see an error like "Unable to
        #   access file /.../tmp*.twi--file is locked (SDDS_InitializeOutput)"

        remote.run(
            remote_opts,
            ele_filepath,
            print_cmd=True,
            print_stdout=std_print_enabled["out"],
            print_stderr=std_print_enabled["err"],
            output_filepaths=None,
        )

    if del_tmp_files:
        for fp in ed.actual_output_filepath_list + [ele_filepath]:
            if fp.startswith("/dev"):
                continue
            elif fp == corrected_LTE_filepath:
                continue
            else:
                try:
                    os.remove(fp)
                except:
                    print(f'Failed to delete "{fp}"')


class TbTLinOptCorrector:
    def __init__(
        self,
        actual_LTE: Lattice,
        E_MeV: float,
        bpmx_names: List[str],
        bpmy_names: List[str],
        normal_quad_names: List[Union[str, List[str]]],
        skew_quad_names: List[Union[str, List[str]]],
        use_x1_y1_re_im: bool = False,
        n_turns: int = 256,
        actual_inj_CO: Union[Dict, None] = None,
        tbt_ps_offset_wrt_CO: Union[Dict, None] = None,
        design_LTE: Union[Lattice, None] = None,
        RM_filepath: Union[Path, str] = "",
        RM_obs_weights: Union[Dict, None] = None,
        rcond: float = 1e-4,
        N_KICKS: Union[None, Dict] = None,
        tempdir_path: Union[None, Path, str] = None,
        print_stdout=False,
        print_stderr=True,
    ) -> None:
        std_print_enabled["out"] = print_stdout
        std_print_enabled["err"] = print_stderr

        self.LTE_d = {}
        self.LTE_filepath_d = {}

        assert isinstance(actual_LTE, Lattice)
        self.LTE_d["actual"] = actual_LTE
        self.LTE_filepath_d["actual"] = actual_LTE.LTE_filepath

        self.E_MeV = E_MeV
        self.n_turns = n_turns
        self.N_KICKS = N_KICKS

        self._full_nu_vec = np.fft.fftfreq(self.n_turns)

        if actual_inj_CO is None:
            actual_inj_CO = dict(x0=0.0, xp0=0.0, y0=0.0, yp0=0.0, dp0=0.0)
        assert isinstance(actual_inj_CO, dict)
        self._co_calculator = ClosedOrbitCalculatorViaTraj(
            self.LTE_d["actual"],
            self.E_MeV,
            N_KICKS=self.N_KICKS,
            fixed_length=True,
            **actual_inj_CO,
        )

        self.calc_actual_inj_CO()

        self.tbt_ps_offset_wrt_CO = dict(x0=1e-6, xp0=0.0, y0=1e-6, yp0=0.0, dp0=0.0)
        if tbt_ps_offset_wrt_CO is not None:
            assert isinstance(tbt_ps_offset_wrt_CO, dict)
            self.tbt_ps_offset_wrt_CO.update(tbt_ps_offset_wrt_CO)

        self.validate_BPM_selection(bpmx_names, bpmy_names)

        self.validate_quad_selection(normal_quad_names, skew_quad_names)

        self.nBPM = {k: len(v) for k, v in self.bpm_names.items()}
        self.nQUAD = {k: len(v) for k, v in self.quad_flat_names.items()}

        assert self.bpm_names["x"].tolist() == self.bpm_names["y"].tolist()
        assert self.nBPM["x"] == self.nBPM["y"]

        self.use_x1_y1_re_im = use_x1_y1_re_im
        if self.use_x1_y1_re_im:
            self._avail_obs_keys = ["x1_re", "x1_im", "y1_re", "y1_im"]
        else:
            self._avail_obs_keys = ["x1_bbeat", "x1_dphi", "y1_bbeat", "y1_dphi"]
        self._avail_obs_keys += [
            "x2_re",
            "x2_im",
            "y2_re",
            "y2_im",
            "etax",
            "etay",
            "nux",
            "nuy",
        ]

        assert isinstance(design_LTE, Lattice)
        self.LTE_d["design"] = design_LTE
        self.LTE_filepath_d["design"] = design_LTE.LTE_filepath

        self.make_tempdir(tempdir_path=tempdir_path)

        self._write_ele_files()

        self._write_quad_setpoints_file()

        self.twiss = dict(design=None, actual=None)
        self.tune_above_half = dict(design=None, actual=None)
        self._update_design_twiss()

        self.lin_comp = {}
        self.norm_lin_comp = {}
        self.tbt_avg_nu = {}

        self._actual_design_diff_history = defaultdict(list)
        self._dK1s_history = []

        if RM_filepath != "":
            self._construct_RM(RM_filepath, obs_weights=RM_obs_weights, rcond=rcond)

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

    def _write_actual_tbt_calc_ele(self):
        ed = elebuilder.EleDesigner(self.actual_tbt_calc_ele_path, double_format=".16g")

        LTE_filepath = self.LTE_filepath_d["actual"]
        LTE = self.LTE_d["actual"]

        ed.add_block(
            "run_setup",
            lattice=str(LTE_filepath),
            p_central_mev=self.E_MeV,
            use_beamline=LTE.used_beamline_name,
        )

        ed.add_newline()

        # Note that the output of the WATCH element data will be always with
        # respect to the design orbit. This means the even if MONI elements are
        # defined with non-zero DX and/or DY values, those values will be ignored.
        for bpm_index, name in enumerate(self.bpm_names["xy"]):
            watch_pathobj = self.actual_tbt_calc_ele_path.with_suffix(
                f".wc{bpm_index:03d}"
            )
            temp_watch_elem_name = f"ELEGANT_WATCH_{bpm_index:03d}"
            watch_filepath = watch_pathobj.resolve()
            temp_watch_elem_def = (
                f'{temp_watch_elem_name}: WATCH, FILENAME="{watch_filepath}", '
                "MODE=coordinate"
            )
            if name == "_BEG_":
                name = LTE.flat_used_elem_names[0]
            assert LTE.is_unique_elem_name(name)
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
            filename=str(self.quad_setpoints_path.resolve()),
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

    def _write_design_tbt_calc_ele(self):
        ed = elebuilder.EleDesigner(self.design_tbt_calc_ele_path, double_format=".16g")

        LTE_filepath = self.LTE_filepath_d["design"]
        LTE = self.LTE_d["design"]

        ed.add_block(
            "run_setup",
            lattice=str(LTE_filepath),
            p_central_mev=self.E_MeV,
            use_beamline=LTE.used_beamline_name,
        )

        ed.add_newline()

        # Note that the output of the WATCH element data will be always with
        # respect to the design orbit. This means the even if MONI elements are
        # defined with non-zero DX and/or DY values, those values will be ignored.
        for bpm_index, name in enumerate(self.bpm_names["xy"]):
            watch_pathobj = self.design_tbt_calc_ele_path.with_suffix(
                f".wc{bpm_index:03d}"
            )
            temp_watch_elem_name = f"ELEGANT_WATCH_{bpm_index:03d}"
            watch_filepath = watch_pathobj.resolve()
            temp_watch_elem_def = (
                f'{temp_watch_elem_name}: WATCH, FILENAME="{watch_filepath}", '
                "MODE=coordinate"
            )
            if name == "_BEG_":
                name = LTE.flat_used_elem_names[0]
            assert LTE.is_unique_elem_name(name)
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

    def _write_lte_save_ele(self):
        ed = elebuilder.EleDesigner(self.lte_save_ele_path, double_format=".16g")

        LTE_filepath = self.LTE_filepath_d["actual"]
        LTE = self.LTE_d["actual"]

        ed.add_block(
            "run_setup",
            lattice=str(LTE_filepath),
            p_central_mev=self.E_MeV,
            use_beamline=LTE.used_beamline_name,
        )

        ed.add_newline()

        if self.N_KICKS is not None:
            elebuilder.add_N_KICKS_alter_elements_blocks(ed, self.N_KICKS)

            ed.add_newline()

        load_parameters = dict(
            change_defined_values=True,
            allow_missing_elements=True,
            allow_missing_parameters=True,
            filename=str(self.quad_setpoints_path.resolve()),
        )
        ed.add_block("load_parameters", **load_parameters)

        ed.add_newline()

        ed.add_block("save_lattice", filename="<new_filepath_str>")

        ed.write()

    def _write_ele_files(self):
        """
        Write ELE files for
        1) computing TbT data for the "actual" lattice
        2) computing TbT data for the "design/model" lattice
        3) saving LTE file that includes quad settings after correction
        """
        tmp = tempfile.NamedTemporaryFile(
            dir=self.tempdir.name, delete=False, prefix="tmpTbtActual_", suffix=".ele"
        )
        self.actual_tbt_calc_ele_path = Path(tmp.name).resolve()
        tmp.close()

        tmp = tempfile.NamedTemporaryFile(
            dir=self.tempdir.name, delete=False, prefix="tmpTbtDesign_", suffix=".ele"
        )
        self.design_tbt_calc_ele_path = Path(tmp.name).resolve()
        tmp.close()

        tmp = tempfile.NamedTemporaryFile(
            dir=self.tempdir.name, delete=False, prefix="tmpLteSave_", suffix=".ele"
        )
        self.lte_save_ele_path = Path(tmp.name).resolve()
        tmp.close()

        self.quad_setpoints_path = self.actual_tbt_calc_ele_path.with_suffix(".quadsp")

        self._write_actual_tbt_calc_ele()
        self._write_design_tbt_calc_ele()
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

    def _write_quad_setpoints_file(self):
        print("* Writing quad setpoints to file")

        col = dict(ElementName=[], ElementParameter=[], ParameterValue=[])

        for kind in ["normal", "skew"]:
            for elem_name, prop_d in self.quad_props[kind].items():
                param_name = prop_d["name"]
                param_val = prop_d["value"]
                col["ElementName"].append(elem_name)
                col["ElementParameter"].append(param_name)
                col["ParameterValue"].append(param_val)

        sdds.dicts2sdds(
            self.quad_setpoints_path,
            params=None,
            columns=col,
            outputMode="binary",
            suppress_err_msg=False,
        )

        self._uncommited_quad_change = False

    def _extract_twiss_from_twiss_pgz(self, output_filepath):
        d = load_pgz_file(output_filepath)
        twi = d["data"]["twi"]
        twi_sca = twi["scalars"]
        twi_arr = twi["arrays"]

        assert self.bpm_names["x"].tolist() == self.bpm_names["y"].tolist()
        bpm_names = self.bpm_names["x"]

        bpm_inds = np.array(
            [np.where(twi_arr["ElementName"] == name)[0][0] for name in bpm_names]
        )

        s_bpms = twi_arr["s"][bpm_inds]

        betax_bpms = (twi_arr["betax"][bpm_inds - 1] + twi_arr["betax"][bpm_inds]) / 2
        betay_bpms = (twi_arr["betay"][bpm_inds - 1] + twi_arr["betay"][bpm_inds]) / 2

        etax_bpms = (twi_arr["etax"][bpm_inds - 1] + twi_arr["etax"][bpm_inds]) / 2
        etay_bpms = (twi_arr["etay"][bpm_inds - 1] + twi_arr["etay"][bpm_inds]) / 2

        phix_bpms = (
            (twi_arr["psix"][bpm_inds - 1] + twi_arr["psix"][bpm_inds])
            / 2
            / (2 * np.pi)
        )  # [2\pi]
        phiy_bpms = (
            (twi_arr["psiy"][bpm_inds - 1] + twi_arr["psiy"][bpm_inds])
            / 2
            / (2 * np.pi)
        )  # [2\pi]

        betax_inj = twi_arr["betax"][0]
        betay_inj = twi_arr["betay"][0]

        alphax_inj = twi_arr["alphax"][0]
        alphay_inj = twi_arr["alphay"][0]

        nux = twi_sca["nux"]
        nuy = twi_sca["nuy"]

        beta = dict(
            bpms=dict(x=betax_bpms, y=betay_bpms), inj=dict(x=betax_inj, y=betay_inj)
        )
        eta = dict(bpms=dict(x=etax_bpms, y=etay_bpms))
        alpha = dict(inj=dict(x=alphax_inj, y=alphay_inj))
        phi = dict(bpms=dict(x=phix_bpms, y=phiy_bpms))  # [2\pi]

        return dict(
            s_bpms=s_bpms, beta=beta, eta=eta, alpha=alpha, phi=phi, nux=nux, nuy=nuy
        )

    def _update_design_twiss(self):
        LTE = self.LTE_d["design"]

        tmp = tempfile.NamedTemporaryFile(
            dir=self.tempdir.name, delete=False, prefix="tmpLinOpt_", suffix=".pgz"
        )
        output_filepath = Path(tmp.name).resolve()
        tmp.close()

        calc_ring_twiss(
            output_filepath,
            LTE.LTE_filepath,
            self.E_MeV,
            use_beamline=LTE.used_beamline_name,
            alter_elements_list=[
                dict(
                    name="*",
                    type=elem_type,
                    item="N_KICKS",
                    value=v,
                    allow_missing_elements=True,
                )
                for elem_type, v in self.N_KICKS.items()
            ],
        )

        self.twiss["design"] = self._extract_twiss_from_twiss_pgz(output_filepath)

        self.tune_above_half["design"] = {}
        for plane in "xy":
            nu = self.twiss["design"][f"nu{plane}"]
            frac_nu = nu - np.floor(nu)
            self.tune_above_half["design"][plane] = frac_nu > 0.5

    def update_actual_twiss(self):
        if not hasattr(self, "_actual_twiss_LTE_filepath"):
            tmp = tempfile.NamedTemporaryFile(
                dir=self.tempdir.name,
                delete=False,
                prefix="tmpLinOptActual_",
                suffix=".lte",
            )
            self._actual_twiss_LTE_filepath = Path(tmp.name).resolve()
            tmp.close()

            tmp = tempfile.NamedTemporaryFile(
                dir=self.tempdir.name,
                delete=False,
                prefix="tmpLinOptActual_",
                suffix=".pgz",
            )
            self._actual_twiss_output_filepath = Path(tmp.name).resolve()
            tmp.close()

        if self._uncommited_quad_change:
            self._write_quad_setpoints_file()

            # Re-compute injection closed-orbit (CO) phase space coordinates
            self.calc_actual_inj_CO()

        self.save_current_lattice_to_LTE_file(self._actual_twiss_LTE_filepath)

        LTE = self.LTE_d["actual"]

        calc_ring_twiss(
            self._actual_twiss_output_filepath,
            self._actual_twiss_LTE_filepath,
            self.E_MeV,
            use_beamline=LTE.used_beamline_name,
            alter_elements_list=[
                dict(
                    name="*",
                    type=elem_type,
                    item="N_KICKS",
                    value=v,
                    allow_missing_elements=True,
                )
                for elem_type, v in self.N_KICKS.items()
            ],
        )

        self.twiss["actual"] = self._extract_twiss_from_twiss_pgz(
            self._actual_twiss_output_filepath
        )

        self.tune_above_half["actual"] = {}
        for plane in "xy":
            nu = self.twiss["actual"][f"nu{plane}"]
            frac_nu = nu - np.floor(nu)
            self.tune_above_half["actual"][plane] = frac_nu > 0.5

    def calc_actual_inj_CO(self):
        res = self._co_calculator.calc()
        self.actual_inj_CO = {f"{coord}0": v for coord, v in res["inj_COD"].items()}

        msg = ", ".join(
            [f"{coord[:-1]}={v:+.3e}" for coord, v in self.actual_inj_CO.items()]
        )
        print(f"* Inj. CO computed for actual lattice to be ({msg})")

    def calc_actual_tbt(self, dt=False, verbose=0):
        if self._uncommited_quad_change:
            self._write_quad_setpoints_file()

            # Re-compute injection closed-orbit (CO) phase space coordinates
            self.calc_actual_inj_CO()

        # Initial injection phase space offset based upon CO and
        # self.self.tbt_ps_offset_wrt_CO
        x0 = self.actual_inj_CO["x0"] + self.tbt_ps_offset_wrt_CO["x0"]
        y0 = self.actual_inj_CO["y0"] + self.tbt_ps_offset_wrt_CO["y0"]
        xp0 = self.actual_inj_CO["xp0"] + self.tbt_ps_offset_wrt_CO["xp0"]
        yp0 = self.actual_inj_CO["yp0"] + self.tbt_ps_offset_wrt_CO["yp0"]
        dp0 = self.actual_inj_CO["dp0"] + self.tbt_ps_offset_wrt_CO["dp0"]

        n_turns = self.n_turns

        nBPM = self.nBPM["xy"]

        t0 = time.perf_counter()

        run(
            self.actual_tbt_calc_ele_path,
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

        comb_watch_filepath = self.actual_tbt_calc_ele_path.parent / "actual_all.w1"
        # cmd = f"sddscombine -overWrite *.wc??? {comb_watch_filepath.name}"
        _prefix = self.actual_tbt_calc_ele_path.stem
        cmd = f"sddscombine -overWrite {_prefix}.wc??? {comb_watch_filepath.name}"
        p = Popen(
            cmd,
            stdout=PIPE,
            stderr=PIPE,
            cwd=self.actual_tbt_calc_ele_path.parent,
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
                tbt_ar[coord] = ps_ar.reshape((nBPM, n_turns))
            else:
                raise RuntimeError("Particle lost in TbT calculation")
        t2 = time.perf_counter()

        if False and (verbose == 1):
            print(f"calc_actual_tbt() [s]: ELE={t1-t0:.1f}, SDDS={t2-t1:.1f}")

        return tbt_ar

    def calc_design_tbt(self, dt=False, verbose=0):
        n_turns = self.n_turns

        nBPM = self.nBPM["xy"]

        x0 = self.tbt_ps_offset_wrt_CO["x0"]
        y0 = self.tbt_ps_offset_wrt_CO["y0"]
        xp0 = self.tbt_ps_offset_wrt_CO["xp0"]
        yp0 = self.tbt_ps_offset_wrt_CO["yp0"]
        dp0 = self.tbt_ps_offset_wrt_CO["dp0"]

        t0 = time.perf_counter()

        run(
            self.design_tbt_calc_ele_path,
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

        comb_watch_filepath = self.design_tbt_calc_ele_path.parent / "design_all.w1"
        # cmd = f"sddscombine -overWrite *.wc??? {comb_watch_filepath.name}"
        _prefix = self.design_tbt_calc_ele_path.stem
        cmd = f"sddscombine -overWrite {_prefix}.wc??? {comb_watch_filepath.name}"
        p = Popen(
            cmd,
            stdout=PIPE,
            stderr=PIPE,
            cwd=self.design_tbt_calc_ele_path.parent,
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
                tbt_ar[coord] = ps_ar.reshape((nBPM, n_turns))
            else:
                raise RuntimeError("Particle lost in TbT calculation")
        t2 = time.perf_counter()

        if False and (verbose == 1):
            print(f"calc_design_tbt() [s]: ELE={t1-t0:.1f}, SDDS={t2-t1:.1f}")

        return tbt_ar

    def extract_lin_freq_components_from_multi_BPM_tbt(
        self,
        xtbt,
        ytbt,
        start_turn_index=0,
        n_turn_to_use=None,
        window="sine_squared",
        max_sync_tune=1e-3,
        min_nu_distance=0.02,
        nu_resolution=1e-5,
        nux0_range=None,
        nuy0_range=None,
        fxy1_closed=True,
    ):
        """
        If `fxy1_closed` is True, for a given m-by-n TbT data ("m" turns for "n" BPMs),
        the first BPM data from second turn to the last (Turn Index 1 thru "m-1") will
        be added as the virtual last BPM (BPM Index "n") TbT data with a total of
        "m-1" turns. To match the total number of turns for all the BPMs, the last
        turn data (Turn Index "m-1") for the BPMs that actually exist (Index 0 thru
        "n-1") will be all discarded such that all the real (BPM Index 0 thru "n-1"
        and virtual (BPM Index "n") BPM TbT data will have only "m-1" turns. In
        other words, the final BPM TbT data shape will be "m-1"-by-"n+1".
        """

        # X, Y shape: (n_turns x nBPM)
        X = xtbt.T
        Y = ytbt.T

        if fxy1_closed:
            # Add 1-st BPM data from 2nd turn to the last as an additional BPM
            # TbT data, while reomving the last TbT data from the existing BPM
            # TbT data to maintain the same number of turns for all the BPM data
            X = np.hstack((X[:-1, :], X[1:, 0].reshape((-1, 1))))
            Y = np.hstack((Y[:-1, :], Y[1:, 0].reshape((-1, 1))))

        if n_turn_to_use is None:
            s_ = np.s_[start_turn_index:]
        else:
            s_ = np.s_[start_turn_index : (start_turn_index + n_turn_to_use)]
        X = X[s_, :]
        Y = Y[s_, :]

        X = signal.detrend(X, type="constant", axis=0)
        Y = signal.detrend(Y, type="constant", axis=0)

        return sigproc.get_linear_freq_components_from_xy_matrices(
            X,
            Y,
            kick_type="hv",
            window=window,
            max_sync_tune=max_sync_tune,
            min_nu_distance=min_nu_distance,
            nu_resolution=nu_resolution,
            nux0_range=nux0_range,
            nuy0_range=nuy0_range,
        )

    @staticmethod
    def normalize_phase_adj_lin_freq_components(
        lin_comp_d, model_bpm_beta, tune_above_half
    ):
        """Assuming `fxy1_closed=True`, i.e., the first and the last data in
        `lin_comp_d` are both from the first BPM (the first data from the first
        turn while the last data from the second turn)"""

        assert (nBPM := model_bpm_beta["x"].size) == model_bpm_beta["y"].size

        assert lin_comp_d["x1amp"].size == lin_comp_d["y1amp"].size == nBPM + 1

        # The last data has the redundant amplitude informtion for the first BPM,
        # so excluding it.
        x_sq = lin_comp_d["x1amp"][:-1] ** 2
        y_sq = lin_comp_d["y1amp"][:-1] ** 2

        est_twoJx_arr = x_sq / model_bpm_beta["x"]
        est_twoJy_arr = y_sq / model_bpm_beta["y"]
        base_est_twoJx = np.mean(est_twoJx_arr)
        base_est_twoJy = np.mean(est_twoJy_arr)
        if False:
            print(f"Base est 2Jx = {base_est_twoJx:.6g}")
            print(f"Base est 2Jy = {base_est_twoJy:.6g}")

            base_est_twoJx_rms = np.std(est_twoJx_arr)
            base_est_twoJy_rms = np.std(est_twoJy_arr)
            print(f"RMS est 2Jx = {base_est_twoJx_rms:.6g}")
            print(f"RMS est 2Jy = {base_est_twoJy_rms:.6g}")

        x1phi_rad = unwrap_montonically_increasing(
            lin_comp_d["x1phi"], tune_above_half=tune_above_half["x"]
        )
        y1phi_rad = unwrap_montonically_increasing(
            lin_comp_d["y1phi"], tune_above_half=tune_above_half["y"]
        )
        if False:
            plt.figure()
            v = model_bpm_phix
            plt.plot(v - v[0], "b.-")
            v = np.unwrap(1 - lin_comp_d["x1phi"]) / (2 * np.pi)
            plt.plot(v - v[0], "r.-")
            v = x1phi_rad / (2 * np.pi)
            plt.plot(v - v[0], "m.-")

        norm_x1C = lin_comp_d["x1C"] / np.sqrt(base_est_twoJx)
        norm_y1C = lin_comp_d["y1C"] / np.sqrt(base_est_twoJy)
        norm_x2C = lin_comp_d["x2C"] / np.sqrt(base_est_twoJy)
        norm_y2C = lin_comp_d["y2C"] / np.sqrt(base_est_twoJx)

        # Shift the phases so that the primary components of the first BPM will
        # have always zero phase.
        if tune_above_half["x"]:
            phix_adj = np.exp(1j * (+x1phi_rad[0]))
        else:
            phix_adj = np.exp(1j * (-x1phi_rad[0]))
        if tune_above_half["y"]:
            phiy_adj = np.exp(1j * (+y1phi_rad[0]))
        else:
            phiy_adj = np.exp(1j * (-y1phi_rad[0]))
        norm_x1C *= phix_adj
        norm_y1C *= phiy_adj
        norm_x2C *= phix_adj
        norm_y2C *= phiy_adj
        np.testing.assert_almost_equal(np.angle(norm_x1C[0]), 0.0, decimal=15)
        np.testing.assert_almost_equal(np.angle(norm_y1C[0]), 0.0, decimal=15)
        # np.angle(norm_x2C[0]) & np.angle(norm_y2C[0]) are unlikely to be zero.

        # Because of duplicity of the first and last BPM for the primary amplitude
        # (due to fxy1_closed=True) and zeroing of the primary phase for the first
        # BPM means that the primary data at the first BPM is useless. So, removing
        # this here.
        norm_x1C = norm_x1C[1:]
        norm_y1C = norm_y1C[1:]
        # On the otherhand, the secondary components at the first BPM are not
        # useless. So, we're retaining them here. This means:
        assert norm_x1C.size == norm_y1C.size == nBPM
        assert norm_x2C.size == norm_y2C.size == nBPM + 1

        return dict(x1=norm_x1C, y1=norm_y1C, x2=norm_x2C, y2=norm_y2C)

    @staticmethod
    def normalize_and_phase_diff_lin_freq_components(
        lin_comp_d, model_bpm_beta, tune_above_half
    ):
        """Assuming `fxy1_closed=True`, i.e., the first and the last data in
        `lin_comp_d` are both from the first BPM (the first data from the first
        turn while the last data from the second turn)"""

        assert (nBPM := model_bpm_beta["x"].size) == model_bpm_beta["y"].size

        assert lin_comp_d["x1amp"].size == lin_comp_d["y1amp"].size == nBPM + 1

        # The last data has the redundant amplitude informtion for the first BPM,
        # so excluding it.
        x_sq = lin_comp_d["x1amp"][:-1] ** 2
        y_sq = lin_comp_d["y1amp"][:-1] ** 2

        est_twoJx_arr = x_sq / model_bpm_beta["x"]
        est_twoJy_arr = y_sq / model_bpm_beta["y"]
        base_est_twoJx = np.mean(est_twoJx_arr)
        base_est_twoJy = np.mean(est_twoJy_arr)
        if False:
            print(f"Base est 2Jx = {base_est_twoJx:.6g}")
            print(f"Base est 2Jy = {base_est_twoJy:.6g}")

            base_est_twoJx_rms = np.std(est_twoJx_arr)
            base_est_twoJy_rms = np.std(est_twoJy_arr)
            print(f"RMS est 2Jx = {base_est_twoJx_rms:.6g}")
            print(f"RMS est 2Jy = {base_est_twoJy_rms:.6g}")

        x1phi_rad = unwrap_montonically_increasing(
            lin_comp_d["x1phi"], tune_above_half=tune_above_half["x"]
        )
        y1phi_rad = unwrap_montonically_increasing(
            lin_comp_d["y1phi"], tune_above_half=tune_above_half["y"]
        )
        if False:
            plt.figure()
            v = model_bpm_phix
            plt.plot(v - v[0], "b.-")
            v = np.unwrap(1 - lin_comp_d["x1phi"]) / (2 * np.pi)
            plt.plot(v - v[0], "r.-")
            v = x1phi_rad / (2 * np.pi)
            plt.plot(v - v[0], "m.-")

        norm_x1C = lin_comp_d["x1C"] / np.sqrt(base_est_twoJx)
        norm_y1C = lin_comp_d["y1C"] / np.sqrt(base_est_twoJy)
        norm_x2C = lin_comp_d["x2C"] / np.sqrt(base_est_twoJy)
        norm_y2C = lin_comp_d["y2C"] / np.sqrt(base_est_twoJx)

        beta_x1 = (np.abs(lin_comp_d["x1C"]) ** 2) / base_est_twoJx
        beta_y1 = (np.abs(lin_comp_d["y1C"]) ** 2) / base_est_twoJy

        # Shift the phases so that the primary components of the first BPM will
        # have always zero phase.
        if tune_above_half["x"]:
            phix_adj = np.exp(1j * (+x1phi_rad[0]))
        else:
            phix_adj = np.exp(1j * (-x1phi_rad[0]))
        if tune_above_half["y"]:
            phiy_adj = np.exp(1j * (+y1phi_rad[0]))
        else:
            phiy_adj = np.exp(1j * (-y1phi_rad[0]))
        norm_x1C *= phix_adj
        norm_y1C *= phiy_adj
        norm_x2C *= phix_adj
        norm_y2C *= phiy_adj
        np.testing.assert_almost_equal(np.angle(norm_x1C[0]), 0.0, decimal=15)
        np.testing.assert_almost_equal(np.angle(norm_y1C[0]), 0.0, decimal=15)
        # np.angle(norm_x2C[0]) & np.angle(norm_y2C[0]) are unlikely to be zero.

        # Calculate phase diff.
        phix1 = np.angle(norm_x1C)
        phiy1 = np.angle(norm_y1C)
        dphix1 = np.diff(phix1)
        dphiy1 = np.diff(phiy1)
        while not np.all(dphix1 > 0.0):
            dphix1[dphix1 < 0.0] += 2 * np.pi
        while not np.all(dphiy1 > 0.0):
            dphiy1[dphiy1 < 0.0] += 2 * np.pi

        # Because of duplicity of the first and last BPM for the primary amplitude
        # (due to fxy1_closed=True), the primary beta data at the last BPM is
        # useless. So, removing this here.
        beta_x1 = beta_x1[:-1]
        beta_y1 = beta_y1[:-1]
        beta_beat_x1 = beta_x1 / model_bpm_beta["x"] - 1
        beta_beat_y1 = beta_y1 / model_bpm_beta["y"] - 1
        # The phase diff. of the primary components are already donwn by 1.
        # On the otherhand, the secondary components at the first BPM are not
        # useless. So, we're retaining them here. This means:
        assert beta_x1.size == beta_y1.size == dphix1.size == dphiy1.size == nBPM
        assert norm_x2C.size == norm_y2C.size == nBPM + 1

        # `x1_dphi` and `y1_dphi` are in [rad]
        return dict(
            x1_bbeat=beta_beat_x1,
            x1_dphi=dphix1,
            y1_bbeat=beta_beat_y1,
            y1_dphi=dphiy1,
            x2=norm_x2C,
            y2=norm_y2C,
        )

    def calc_design_lin_comp(self):
        tbt = self.calc_design_tbt()

        self.lin_comp["design"] = self.extract_lin_freq_components_from_multi_BPM_tbt(
            tbt["x"], tbt["y"], fxy1_closed=True
        )

        model_twi = self.twiss["design"]
        model_bpm_beta = model_twi["beta"]["bpms"]
        if self.use_x1_y1_re_im:
            normalizer = self.normalize_phase_adj_lin_freq_components
        else:
            normalizer = self.normalize_and_phase_diff_lin_freq_components
        self.norm_lin_comp["design"] = normalizer(
            self.lin_comp["design"], model_bpm_beta, self.tune_above_half["design"]
        )

        self.tbt_avg_nu["design"] = dict(
            x=np.mean(self.lin_comp["design"]["nux"]),
            y=np.mean(self.lin_comp["design"]["nuy"]),
        )

    def calc_actual_lin_comp(self):
        tbt = self.calc_actual_tbt()

        self.lin_comp["actual"] = self.extract_lin_freq_components_from_multi_BPM_tbt(
            tbt["x"], tbt["y"], fxy1_closed=True
        )

        model_twi = self.twiss["design"]
        model_bpm_beta = model_twi["beta"]["bpms"]
        if self.use_x1_y1_re_im:
            normalizer = self.normalize_phase_adj_lin_freq_components
        else:
            normalizer = self.normalize_and_phase_diff_lin_freq_components
        self.norm_lin_comp["actual"] = normalizer(
            self.lin_comp["actual"], model_bpm_beta, self.tune_above_half["actual"]
        )

        self.tbt_avg_nu["actual"] = dict(
            x=np.mean(self.lin_comp["actual"]["nux"]),
            y=np.mean(self.lin_comp["actual"]["nuy"]),
        )

    def calc_response(self, quad_names, dK1, verbose=0):
        resp = {}

        if "design" not in self.lin_comp:
            self.calc_design_lin_comp()
        model_tbt_avg_nu = self.tbt_avg_nu["design"]

        model_twi = self.twiss["design"]
        model_bpm_beta = model_twi["beta"]["bpms"]
        model_bpm_eta = model_twi["eta"]["bpms"]

        if verbose == 1:
            print(f"Changing K1 for [{quad_names}] by {dK1:.6g}")
        self.change_K1_setpoint_by(quad_names, dK1)

        self.update_actual_twiss()
        twi = self.twiss["actual"]

        tbt = self.calc_actual_tbt()
        lin_comp = self.extract_lin_freq_components_from_multi_BPM_tbt(
            tbt["x"], tbt["y"], fxy1_closed=True
        )

        tbt_avg_nu = {plane: np.mean(lin_comp[f"nu{plane}"]) for plane in "xy"}
        tbt_rms_nu = {plane: np.std(lin_comp[f"nu{plane}"]) for plane in "xy"}
        if verbose == 1:
            contents = ", ".join(
                [
                    f"{tbt_avg_nu['x']:.6f}+/-{tbt_rms_nu['x']:.3g}",
                    f"{tbt_avg_nu['y']:.6f}+/-{tbt_rms_nu['y']:.3g}",
                ]
            )
            print(f"TbT (nux, nuy) = ({contents})")

        resp["nu_tbt"] = {}
        resp["nu_twi"] = {}
        resp["eta"] = {}
        for plane in "xy":
            resp["nu_tbt"][plane] = (tbt_avg_nu[plane] - model_tbt_avg_nu[plane]) / dK1
            resp["nu_twi"][plane] = (twi[f"nu{plane}"] - model_twi[f"nu{plane}"]) / dK1

            _base_eta = model_bpm_eta[plane]
            _eta = twi["eta"]["bpms"][plane]
            resp["eta"][plane] = (_eta - _base_eta) / dK1

        if self.use_x1_y1_re_im:
            normalizer = self.normalize_phase_adj_lin_freq_components
        else:
            normalizer = self.normalize_and_phase_diff_lin_freq_components
        norm_lin_comp = normalizer(
            lin_comp, model_bpm_beta, self.tune_above_half["design"]
        )
        resp["_extra"] = dict(norm_lin_comp=norm_lin_comp)

        if self.use_x1_y1_re_im:
            resp["norm_lin_comp"] = {
                comp: (norm_lin_comp[comp] - self.norm_lin_comp["design"][comp]) / dK1
                for comp in ["x1", "y1", "x2", "y2"]
            }
        else:
            resp["norm_lin_comp"] = {
                comp: (norm_lin_comp[comp] - self.norm_lin_comp["design"][comp]) / dK1
                for comp in ["x1_bbeat", "y1_bbeat", "x1_dphi", "y1_dphi", "x2", "y2"]
            }

        # Reset the quad strength change
        if verbose == 1:
            print(f"Changing K1 for [{quad_names}] by {-dK1:.6g}")
        self.change_K1_setpoint_by(quad_names, -dK1)

        return resp

    def validate_BPM_selection(self, bpmx_names, bpmy_names):
        LTE = self.LTE_d["actual"]

        s = LTE.get_s_mid_array()
        self.bpm_s = {}

        sorted_x_elem_inds = LTE.get_elem_inds_from_names(bpmx_names)
        sorted_y_elem_inds = LTE.get_elem_inds_from_names(bpmy_names)

        self.bpm_names = dict(
            x=LTE.get_names_from_elem_inds(sorted_x_elem_inds),
            y=LTE.get_names_from_elem_inds(sorted_y_elem_inds),
        )

        self.bpm_s["x"] = s[sorted_x_elem_inds]
        self.bpm_s["y"] = s[sorted_y_elem_inds]

        u_bpm_xy_names = np.unique(
            self.bpm_names["x"].tolist() + self.bpm_names["y"].tolist()
        )
        sorted_elem_inds = LTE.get_elem_inds_from_names(u_bpm_xy_names)
        self.bpm_names["xy"] = LTE.get_names_from_elem_inds(sorted_elem_inds)

        sorted_u_bpm_names = self.bpm_names["xy"].tolist()

        self.u_bpm_names_to_bpm_inds = {
            plane: [sorted_u_bpm_names.index(name) for name in self.bpm_names[plane]]
            for plane in "xy"
        }

        self.bpm_s["xy"] = s[sorted_elem_inds]

    def validate_quad_selection(
        self,
        normal_quad_names: List[Union[str, List[str]]],
        skew_quad_names: List[Union[str, List[str]]],
    ):
        valid_elem_types_props = {"KQUAD": dict(normal="K1", skew="K1")}

        LTE = self.LTE_d["actual"]

        self.quad_flat_names = {}
        self.quad_props = {}
        self.quad_col2names = {}

        for kind, kind_str, name_list in [
            ("normal", "normal", normal_quad_names),
            ("skew", "skew", skew_quad_names),
        ]:
            self.quad_flat_names[kind] = []
            self.quad_props[kind] = {}
            self.quad_col2names[kind] = []

            flat_name_list = []
            name2arb_column_id = {}
            for i, name_or_names in enumerate(name_list):
                if isinstance(name_or_names, str):
                    _name = name_or_names
                    flat_name_list.append(_name)
                    name2arb_column_id[_name] = i
                else:
                    _names = name_or_names
                    flat_name_list.extend(list(_names))
                    for _name in _names:
                        name2arb_column_id[_name] = i

            if flat_name_list == []:
                continue

            sorted_elem_inds = LTE.get_elem_inds_from_names(flat_name_list)
            sorted_names = LTE.get_names_from_elem_inds(sorted_elem_inds)
            elem_props = LTE.get_elem_props_from_names(flat_name_list)

            already_incl_ids = []
            col2names = [None] * len(name_list)

            i_col = 0
            for name in sorted_names:
                elem_type = LTE.get_elem_type_from_name(name)

                try:
                    prop_name = valid_elem_types_props[elem_type][kind]
                except KeyError:
                    raise AssertionError(
                        f"'{elem_type}' not a valid {kind_str} quad elem."
                    )

                self.quad_flat_names[kind].append(name)
                self.quad_props[kind][name] = dict(
                    name=prop_name,
                    value=elem_props[name]["properties"].get(prop_name, 0.0),
                )

                if name2arb_column_id[name] not in already_incl_ids:
                    col2names[i_col] = [name]
                    already_incl_ids.append(name2arb_column_id[name])
                    i_col += 1
                else:
                    lumped_col_index = already_incl_ids.index(name2arb_column_id[name])
                    col2names[lumped_col_index].append(name)

            assert all([_v is not None for _v in col2names])
            self.quad_col2names[kind] = [tuple(v) for v in col2names]

        s = LTE.get_s_mid_array()
        self.quad_flat_s = {}

        for kind, names in self.quad_flat_names.items():
            if len(names) == 0:
                self.quad_flat_s[kind] = np.array([])
            else:
                self.quad_flat_s[kind] = s[LTE.get_elem_inds_from_names(names)]

    def change_K1_setpoint(
        self, elem_name: Union[str, Tuple, List, np.ndarray], value: float
    ):
        if isinstance(elem_name, str):
            if elem_name in self.quad_props["normal"]:
                kind = "normal"
            elif elem_name in self.quad_props["skew"]:
                kind = "skew"
            else:
                raise ValueError(f'Element "{elem_name}" not in `self.quad_props`.')

            self.quad_props[kind][elem_name]["value"] = value

        elif isinstance(elem_name, (tuple, list, np.ndarray)):
            for _name in elem_name:
                if _name in self.quad_props["normal"]:
                    kind = "normal"
                elif _name in self.quad_props["skew"]:
                    kind = "skew"
                else:
                    raise ValueError(f'Element "{_name}" not in `self.quad_props`.')

                self.quad_props[kind][_name]["value"] = value
        else:
            raise TypeError

        self._uncommited_quad_change = True

    def change_K1_setpoint_by(
        self, elem_name: Union[str, Tuple, List, np.ndarray], delta_value: float
    ):
        if isinstance(elem_name, str):
            if elem_name in self.quad_props["normal"]:
                kind = "normal"
            elif elem_name in self.quad_props["skew"]:
                kind = "skew"
            else:
                raise ValueError(f'Element "{elem_name}" not in `self.quad_props`.')

            self.quad_props[kind][elem_name]["value"] += delta_value

        elif isinstance(elem_name, (tuple, list, np.ndarray)):
            for _name in elem_name:
                if _name in self.quad_props["normal"]:
                    kind = "normal"
                elif _name in self.quad_props["skew"]:
                    kind = "skew"
                else:
                    raise ValueError(f'Element "{_name}" not in `self.quad_props`.')

                self.quad_props[kind][_name]["value"] += delta_value
        else:
            raise TypeError

        self._uncommited_quad_change = True

    def _construct_RM(self, RM_filepath, obs_weights=None, rcond=1e-4):
        RM_filepath = Path(RM_filepath)
        assert RM_filepath.exists()

        if obs_weights is None:
            obs_weights = {k: 1.0 for k in self._avail_obs_keys}

        self.obs_keys = []
        for k in self._avail_obs_keys:
            if k in obs_weights:
                self.obs_keys.append(k)

        for k in obs_weights.keys():
            if k not in self.obs_keys:
                raise ValueError(f"Following observation key is not valid: {k}")

        self.obs_weights = obs_weights

        self.RM = {}
        with h5py.File(RM_filepath, "r") as f:

            _base64_data = f["input_dict"][()]
            _binary_data = base64.b64decode(_base64_data)
            input_dict = pickle.loads(_binary_data)

            quad_names_in_file = input_dict["quad_names"]
            for quad_type in ["normal", "skew"]:
                names = quad_names_in_file[quad_type]
                # Make sure RM was constructed for flat (i.e., non-lumped)
                # elements
                assert all([isinstance(v, str) for v in names])
                quad_names_in_file[quad_type] = list(names)

            if self.use_x1_y1_re_im:
                for comp in ["x1", "y1", "x2", "y2"]:
                    temp = defaultdict(list)
                    for quad_type in ["normal", "skew"]:

                        if len(self.quad_col2names[quad_type]) == 0:
                            continue

                        src_q_names = quad_names_in_file[quad_type]
                        src_C = f["M_lin_comp"][quad_type][comp][()]

                        actual_C_cols = []
                        for name_set in self.quad_col2names[quad_type]:
                            temp_col = np.zeros(src_C.shape[0], dtype=complex)
                            for name in name_set:
                                i = src_q_names.index(name)
                                temp_col += src_C[:, i]
                            actual_C_cols.append(temp_col)
                        actual_C = np.vstack(actual_C_cols).T

                        temp["re"].append(actual_C.real)
                        temp["im"].append(actual_C.imag)
                    for re_or_im, v in temp.items():
                        self.RM[f"{comp}_{re_or_im}"] = np.hstack(v)
            else:
                model_twi = self.twiss["design"]
                model_bpm_beta = model_twi["beta"]["bpms"]

                model_bpm_phi = model_twi["phi"]["bpms"]
                model_bpm_dphi_rad = {}
                for plane in "xy":
                    dphi = np.diff(
                        np.append(
                            model_bpm_phi[plane],
                            model_twi[f"nu{plane}"] + model_bpm_phi[plane][0],
                        )
                    )
                    model_bpm_dphi_rad[plane] = dphi * (2 * np.pi)

                mat_list = defaultdict(list)
                for comp, bbeat_or_dphi in product(["x1", "y1"], ["bbeat", "dphi"]):
                    plane = comp[0]
                    obs_key = f"{comp}_{bbeat_or_dphi}"
                    for quad_type in ["normal", "skew"]:

                        if len(self.quad_col2names[quad_type]) == 0:
                            continue

                        src_q_names = quad_names_in_file[quad_type]
                        src_M = f["M_lin_comp"][quad_type][obs_key][()]

                        actual_cols = []
                        for name_set in self.quad_col2names[quad_type]:
                            temp_col = np.zeros(src_M.shape[0])
                            for name in name_set:
                                i = src_q_names.index(name)
                                if bbeat_or_dphi == "bbeat":
                                    temp_col += src_M[:, i]
                                else:
                                    temp_col += src_M[:, i] / model_bpm_dphi_rad[plane]
                            actual_cols.append(temp_col)

                        actual_mat = np.vstack(actual_cols).T

                        mat_list[obs_key].append(actual_mat)

                for obs_key, v in mat_list.items():
                    self.RM[obs_key] = np.hstack(v)

                for comp in ["x2", "y2"]:
                    temp = defaultdict(list)
                    for quad_type in ["normal", "skew"]:

                        if len(self.quad_col2names[quad_type]) == 0:
                            continue

                        src_q_names = quad_names_in_file[quad_type]
                        src_C = f["M_lin_comp"][quad_type][comp][()]

                        actual_C_cols = []
                        for name_set in self.quad_col2names[quad_type]:
                            temp_col = np.zeros(src_C.shape[0], dtype=complex)
                            for name in name_set:
                                i = src_q_names.index(name)
                                temp_col += src_C[:, i]
                            actual_C_cols.append(temp_col)
                        actual_C = np.vstack(actual_C_cols).T

                        temp["re"].append(actual_C.real)
                        temp["im"].append(actual_C.imag)
                    for re_or_im, v in temp.items():
                        self.RM[f"{comp}_{re_or_im}"] = np.hstack(v)

            eta = defaultdict(list)
            for plane in "xy":
                for quad_type in ["normal", "skew"]:

                    if len(self.quad_col2names[quad_type]) == 0:
                        continue

                    src_q_names = quad_names_in_file[quad_type]
                    src_M = f["M_eta"][quad_type][plane][()]

                    actual_M_cols = []
                    for name_set in self.quad_col2names[quad_type]:
                        temp_col = np.zeros(src_M.shape[0])
                        for name in name_set:
                            i = src_q_names.index(name)
                            temp_col += src_M[:, i]
                        actual_M_cols.append(temp_col)
                    actual_M = np.vstack(actual_M_cols).T
                    eta[plane].append(actual_M)
            self.RM["etax"] = np.hstack(eta["x"])
            self.RM["etay"] = np.hstack(eta["y"])

            nu = defaultdict(list)
            for plane in "xy":
                for quad_type in ["normal", "skew"]:

                    if len(self.quad_col2names[quad_type]) == 0:
                        continue

                    src_q_names = quad_names_in_file[quad_type]
                    src_M = f["M_tune_tbt"][quad_type][plane][()]

                    actual_vec = []
                    for name_set in self.quad_col2names[quad_type]:
                        temp_val = 0.0
                        for name in name_set:
                            i = src_q_names.index(name)
                            temp_val += src_M[i]
                        actual_vec.append(temp_val)
                    nu[plane].extend(actual_vec)
            self.RM["nux"] = np.array(nu["x"])
            self.RM["nuy"] = np.array(nu["y"])

        if True:

            M = np.vstack([self.RM[k] * self.obs_weights[k] for k in self.obs_keys])
            _, sv, _ = calcSVD(M)

            plt.figure()
            plt.semilogy(sv / sv[0], ".-")
            plt.grid(True)

        if False:
            sv_last_list = []
            sv_last2_list = []
            for w in np.linspace(1.0, 50.0, 21):
                self.obs_weights["etax"] = self.obs_weights["etay"] = w

                M = np.vstack([self.RM[k] * self.obs_weights[k] for k in self.obs_keys])

                _, sv, _ = calcSVD(M)

                sv_last_list.append(sv[-1] / sv[0])
                sv_last2_list.append(sv[-2] / sv[0])
            plt.figure()
            plt.subplot(211)
            plt.plot(sv_last_list, "b.-")
            plt.subplot(212)
            plt.plot(sv_last2_list, "b.-")

            self.obs_weights["etax"] = self.obs_weights["etay"] = 10.0

        self._M = np.vstack([self.RM[k] * self.obs_weights[k] for k in self.obs_keys])

        self._U, self._sv, self._VT = calcSVD(self._M)
        if False:
            plt.figure()
            plt.semilogy(self._sv / self._sv[0], "b.-")

        Sinv_trunc = calcTruncSVMatrix(self._sv, rcond=rcond, nsv=None, disp=0)

        self._M_pinv = self._VT.T @ Sinv_trunc @ self._U.T

    def correct(self, cor_frac: float = 0.7):
        if "design" not in self.tbt_avg_nu:
            self.calc_design_lin_comp()

        obs_diff_list = []
        for k in self.obs_keys:
            w = self.obs_weights[k]

            if k in ("etax", "etay"):
                plane = k[-1]
                v_actual = self.twiss["actual"]["eta"]["bpms"][plane]
                v_design = self.twiss["design"]["eta"]["bpms"][plane]
            elif k in ("nux", "nuy"):
                plane = k[-1]
                v_actual = self.tbt_avg_nu["actual"][plane]
                v_design = self.tbt_avg_nu["design"][plane]
            else:
                if self.use_x1_y1_re_im:
                    xy12, re_or_im = k.split("_")
                    actual_C = self.norm_lin_comp["actual"][xy12]
                    design_C = self.norm_lin_comp["design"][xy12]
                    if re_or_im == "re":
                        v_actual = actual_C.real
                        v_design = design_C.real
                    else:
                        v_actual = actual_C.imag
                        v_design = design_C.imag
                else:
                    if k in self.norm_lin_comp["actual"]:
                        v_actual = self.norm_lin_comp["actual"][k]
                        v_design = self.norm_lin_comp["design"][k]
                    else:
                        xy12, re_or_im = k.split("_")
                        actual_C = self.norm_lin_comp["actual"][xy12]
                        design_C = self.norm_lin_comp["design"][xy12]
                        if re_or_im == "re":
                            v_actual = actual_C.real
                            v_design = design_C.real
                        else:
                            v_actual = actual_C.imag
                            v_design = design_C.imag

            obs_diff_list.append((v_design - v_actual) * w)

        dv = np.hstack(obs_diff_list)

        dK1s = self._M_pinv @ dv

        dK1s *= cor_frac

        self._dK1s_history.append(dK1s)

        quad_names_list = self.quad_col2names["normal"] + self.quad_col2names["skew"]
        assert len(quad_names_list) == len(dK1s)
        for quad_names, dK1 in zip(quad_names_list, dK1s):
            self.change_K1_setpoint_by(quad_names, dK1)

    def observe(self):
        if "design" not in self.tbt_avg_nu:
            self.calc_design_lin_comp()

        self.update_actual_twiss()
        self.calc_actual_lin_comp()

        hist = self._actual_design_diff_history

        dnu = []
        d = self.tbt_avg_nu
        for plane in "xy":
            dnu.append(d["actual"][plane] - d["design"][plane])
            hist[f"nu{plane}"].append(dnu[-1])
        # print(f'Tune diff (x,y) = ({dnu[0]:+.6f}, {dnu[1]:+.6f})')

        actual = self.twiss["actual"]["eta"]["bpms"]
        design = self.twiss["design"]["eta"]["bpms"]
        if False:
            plt.figure()
            plt.subplot(211)
            plt.plot(design["x"] * 1e3, "b.-")
            plt.plot(actual["x"] * 1e3, "r.-")
            plt.subplot(212)
            plt.plot(design["y"] * 1e3, "b.-")
            plt.plot(actual["y"] * 1e3, "r.-")

        deta = {}
        for plane in "xy":
            deta[plane] = actual[plane] - design[plane]
            hist[f"eta{plane}"].append(deta[plane])

        actual = self.norm_lin_comp["actual"]
        design = self.norm_lin_comp["design"]

        if self.use_x1_y1_re_im:
            for plane, order in product("xy", "12"):
                comp = f"{plane}{order}"
                hist[f"{comp}_re"].append(actual[comp].real - design[comp].real)
                hist[f"{comp}_im"].append(actual[comp].imag - design[comp].imag)
        else:
            order = 1
            for plane in "xy":
                comp = f"{plane}{order}"
                obs_key = f"{comp}_bbeat"
                hist[obs_key].append(actual[obs_key] - design[obs_key])
                obs_key = f"{comp}_dphi"
                hist[obs_key].append(actual[obs_key] / design[obs_key] - 1)

            order = 2
            for plane in "xy":
                comp = f"{plane}{order}"
                hist[f"{comp}_re"].append(actual[comp].real - design[comp].real)
                hist[f"{comp}_im"].append(actual[comp].imag - design[comp].imag)

    def plot_actual_design_diff(self, history=True):
        hist = self._actual_design_diff_history

        if history:
            legends = ["Initial"]
            for i in range(len(hist["nux"]))[1:]:
                legends.append(f"Iter. {i}")

        plt.figure()
        plt.subplot(211)
        plt.plot(hist["nux"], "b.-")
        plt.ylabel(r"$\Delta \nu_x$", size="x-large")
        plt.subplot(212)
        plt.plot(hist["nuy"], "b.-")
        plt.ylabel(r"$\Delta \nu_y$", size="x-large")
        plt.xlabel(r"$\mathrm{Correction\; Index}$", size="x-large")
        plt.tight_layout()

        plt.figure()
        for plane, plot_ind in zip("xy", [211, 212]):
            plt.subplot(plot_ind)
            if history:
                plt.plot(np.array(hist[f"eta{plane}"]).T * 1e3, ".-")
                plt.legend(legends, loc="best")
            else:
                plt.plot(hist[f"eta{plane}"][-1] * 1e3, ".-")
            plt.ylabel(rf"$\Delta \eta_{plane}\; [\mathrm{{mm}}]$", size="x-large")
        plt.xlabel(r"$\mathrm{BPM\; Index}$", size="x-large")
        plt.tight_layout()

        if self.use_x1_y1_re_im:
            for plane, order in product("xy", "12"):
                comp = f"{plane}{order}"

                plt.figure()

                plt.subplot(211)
                if history:
                    plt.plot(np.array(hist[f"{comp}_re"]).T, ".-")
                    plt.legend(legends, loc="best")
                else:
                    plt.plot(hist[f"{comp}_re"][-1], ".-")
                plt.ylabel(rf"$\Re{{(\Delta {plane}_{order})}}$", size="x-large")

                plt.subplot(212)
                if history:
                    plt.plot(np.array(hist[f"{comp}_im"]).T, ".-")
                    plt.legend(legends, loc="best")
                else:
                    plt.plot(hist[f"{comp}_im"][-1], ".-")
                plt.ylabel(rf"$\Im{{(\Delta {plane}_{order})}}$", size="x-large")

                plt.xlabel(r"$\mathrm{BPM\; Index}$", size="x-large")
                plt.tight_layout()
        else:
            order = 1
            for plane in "xy":
                comp = f"{plane}{order}"

                plt.figure()

                plt.subplot(211)
                if history:
                    plt.plot(np.array(hist[f"{comp}_bbeat"]).T * 1e2, ".-")
                    plt.legend(legends, loc="best")
                else:
                    plt.plot(hist[f"{comp}_bbeat"][-1] * 1e2, ".-")
                plt.ylabel(rf"$|\Delta {plane}_{order}| [\%]$", size="x-large")

                plt.subplot(212)
                if history:
                    plt.plot(np.array(hist[f"{comp}_dphi"]).T * 1e2, ".-")
                    plt.legend(legends, loc="best")
                else:
                    plt.plot(hist[f"{comp}_dphi"][-1] * 1e2, ".-")
                plt.ylabel(rf"$\Delta \angle {plane}_{order} [\%]$", size="x-large")

                plt.xlabel(r"$\mathrm{BPM\; Index}$", size="x-large")
                plt.tight_layout()

            order = 2
            for plane in "xy":
                comp = f"{plane}{order}"

                plt.figure()

                plt.subplot(211)
                if history:
                    plt.plot(np.array(hist[f"{comp}_re"]).T, ".-")
                    plt.legend(legends, loc="best")
                else:
                    plt.plot(hist[f"{comp}_re"][-1], ".-")
                plt.ylabel(rf"$\Re{{(\Delta {plane}_{order})}}$", size="x-large")

                plt.subplot(212)
                if history:
                    plt.plot(np.array(hist[f"{comp}_im"]).T, ".-")
                    plt.legend(legends, loc="best")
                else:
                    plt.plot(hist[f"{comp}_im"][-1], ".-")
                plt.ylabel(rf"$\Im{{(\Delta {plane}_{order})}}$", size="x-large")

                plt.xlabel(r"$\mathrm{BPM\; Index}$", size="x-large")
                plt.tight_layout()

    def plot_quad_change(self, history=True):
        hist = np.array(self._dK1s_history).T

        dK1s_normal = hist[: self.nQUAD["normal"], :]
        dK1s_skew = hist[self.nQUAD["normal"] :, :]

        if history:
            legends = []
            for i in range(hist.shape[1]):
                legends.append(f"Iter. {i+1}")

        plt.figure()
        plt.subplot(211)
        if history:
            plt.plot(dK1s_normal, ".-")
            plt.legend(legends, loc="best")
        else:
            plt.plot(dK1s_normal[:, -1], ".-")
        plt.ylabel(r"$\Delta K_1\; [\mathrm{m}^{-2}]$", size="x-large")
        plt.xlabel(r"$\mathrm{Normal\; Quad.\; Index}$", size="x-large")
        plt.subplot(212)
        if history:
            plt.plot(dK1s_skew, ".-")
            plt.legend(legends, loc="best")
        else:
            plt.plot(dK1s_skew[:, -1], ".-")
        plt.ylabel(r"$\Delta K_1\; [\mathrm{m}^{-2}]$", size="x-large")
        plt.xlabel(r"$\mathrm{Skew\; Quad.\; Index}$", size="x-large")
        plt.tight_layout()
