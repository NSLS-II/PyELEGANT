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
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from . import elebuilder, eleutil, ltemanager, sdds, sigproc, std_print_enabled
from .local import disable_stdout, enable_stdout, run
from .ltemanager import Lattice
from .orbit import ClosedOrbitCalculatorViaTraj, ClosedOrbitThreader, generate_TRM_file
from .remote import remote
from .respmat import calcSVD, calcTruncSVMatrix
from .sigproc import unwrap_montonically_increasing
from .twiss import calc_ring_twiss
from .util import chunk_list, load_pgz_file, unchunk_list_of_lists


class TbTLinOptCorrector:
    def __init__(
        self,
        actual_LTE: Lattice,
        E_MeV: float,
        bpmx_names: List[str],
        bpmy_names: List[str],
        normal_quad_names: List[Union[str, List[str]]],
        skew_quad_names: List[Union[str, List[str]]],
        n_turns: int = 256,
        actual_inj_CO: Union[Dict, None] = None,
        tbt_ps_offset_wrt_CO: Union[Dict, None] = None,
        design_LTE: Union[Lattice, None] = None,
        RM_filepath: Union[Path, str] = "",
        RM_obs_weights: Union[Dict, None] = None,
        max_beta_beat_thresh_for_coup_cor: float = 5e-2,
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

        self._avail_obs_keys = [
            "x1_bbeat",
            "x1_phi",
            "y1_bbeat",
            "y1_phi",
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

        self.max_beta_beat_thresh_for_coup_cor = max_beta_beat_thresh_for_coup_cor

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

        self.tempdir = tempfile.TemporaryDirectory(
            prefix="tmpLinOptCor_", dir=tempdir_path
        )

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

    def cleanup_tempdirs(self):
        self.remove_tempdir()

        self._co_calculator.remove_tempdir()

        for LTE in self.LTE_d.values():
            LTE.remove_tempdir()

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
            used_beamline_name=self.LTE_d["actual"].used_beamline_name,
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
            dir=self.tempdir.name,
            delete=False,
            prefix="tmpLinOptDesign_",
            suffix=".pgz",
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
    ):
        """
        For a given m-by-n TbT data ("m" turns for "n" BPMs), the first BPM data
        from second turn to the last (Turn Index 1 thru "m-1") will be added as the
        virtual last BPM (BPM Index "n") TbT data with a total of "m-1" turns.
        To match the total number of turns for all the BPMs, the last turn data
        (Turn Index "m-1") for the BPMs that actually exist (Index 0 thru
        "n-1") will be all discarded such that all the real (BPM Index 0 thru "n-1"
        and virtual (BPM Index "n") BPM TbT data will have only "m-1" turns. In
        other words, the final BPM TbT data shape will be "m-1"-by-"n+1".
        """

        # X, Y shape: (n_turns x nBPM)
        X = xtbt.T
        Y = ytbt.T

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
    def normalize_and_phase_diff_lin_freq_components(
        lin_comp_d, model_bpm_beta, tune_above_half
    ):
        """The first and the last data in `lin_comp_d` are assumed to be both from
        the first BPM (the first data from the first turn while the last data from
        the second turn)"""

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

        beta_x1 = (lin_comp_d["x1amp"] ** 2) / base_est_twoJx
        beta_y1 = (lin_comp_d["y1amp"] ** 2) / base_est_twoJy

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

        if tune_above_half["x"]:
            norm_x1C = lin_comp_d["x1C"].conj() / np.sqrt(base_est_twoJx)
            norm_y2C = lin_comp_d["y2C"].conj() / np.sqrt(base_est_twoJx)
        else:
            norm_x1C = lin_comp_d["x1C"] / np.sqrt(base_est_twoJx)
            norm_y2C = lin_comp_d["y2C"] / np.sqrt(base_est_twoJx)

        if tune_above_half["y"]:
            norm_y1C = lin_comp_d["y1C"].conj() / np.sqrt(base_est_twoJy)
            norm_x2C = lin_comp_d["x2C"].conj() / np.sqrt(base_est_twoJy)
        else:
            norm_y1C = lin_comp_d["y1C"] / np.sqrt(base_est_twoJy)
            norm_x2C = lin_comp_d["x2C"] / np.sqrt(base_est_twoJy)

        # Shift the phases so that the primary components of the first BPM will
        # have always zero phase.
        phix_adj = np.exp(1j * (-x1phi_rad[0]))
        phiy_adj = np.exp(1j * (-y1phi_rad[0]))
        norm_x1C *= phix_adj
        norm_y1C *= phiy_adj
        norm_x2C *= phiy_adj
        norm_y2C *= phix_adj
        np.testing.assert_almost_equal(np.angle(norm_x1C[0]), 0.0, decimal=15)
        np.testing.assert_almost_equal(np.angle(norm_y1C[0]), 0.0, decimal=15)
        # np.angle(norm_x2C[0]) & np.angle(norm_y2C[0]) are unlikely to be zero.

        # Calculate phase diff.
        dphix1 = np.diff(x1phi_rad)
        dphiy1 = np.diff(y1phi_rad)
        assert np.all(dphix1 > 0.0)
        assert np.all(dphiy1 > 0.0)

        # The first and last data for the primary amplitude are both for the first
        # BPM (1st turn for the first one and 2nd turn for the last one).
        # Since they are redundant information and thus useless, removing the last
        # data here.
        beta_x1 = beta_x1[:-1]
        beta_y1 = beta_y1[:-1]
        beta_beat_x1 = beta_x1 / model_bpm_beta["x"] - 1
        beta_beat_y1 = beta_y1 / model_bpm_beta["y"] - 1

        assert beta_x1.size == beta_y1.size == nBPM
        assert x1phi_rad.size == y1phi_rad.size == nBPM + 1
        assert dphix1.size == dphiy1.size == nBPM
        assert norm_x2C.size == norm_y2C.size == nBPM + 1

        # `x1_bbeat`, `y1_bbeat` [fraction]
        # `x1_dphi`, `y1_dphi` [rad]
        # `x1_phi`, `y1_phi` [rad]
        return dict(
            x1_bbeat=beta_beat_x1,
            x1_dphi=dphix1,
            x1_phi=x1phi_rad,
            y1_bbeat=beta_beat_y1,
            y1_dphi=dphiy1,
            y1_phi=y1phi_rad,
            x2=norm_x2C,
            y2=norm_y2C,
        )

    def calc_design_lin_comp(self):
        tbt = self.calc_design_tbt()

        self.lin_comp["design"] = self.extract_lin_freq_components_from_multi_BPM_tbt(
            tbt["x"], tbt["y"]
        )

        model_twi = self.twiss["design"]
        model_bpm_beta = model_twi["beta"]["bpms"]
        normalizer = self.normalize_and_phase_diff_lin_freq_components
        self.norm_lin_comp["design"] = normalizer(
            self.lin_comp["design"], model_bpm_beta, self.tune_above_half["design"]
        )

        raw_tbt_nus = {plane: self.lin_comp["design"][f"nu{plane}"] for plane in "xy"}
        self.tbt_avg_nu["design"] = {
            plane: 1 - np.mean(raw_tbt_nus[plane])
            if self.tune_above_half["design"][plane]
            else np.mean(raw_tbt_nus[plane])
            for plane in "xy"
        }

    def calc_actual_lin_comp(self):
        tbt = self.calc_actual_tbt()

        expected_frac_nu = {}
        for plane in ("x", "y"):
            expected_frac_nu[plane] = self.twiss["actual"][f"nu{plane}"]
            expected_frac_nu[plane] -= np.floor(expected_frac_nu[plane])

            if self.tune_above_half["actual"][plane]:
                expected_frac_nu[plane] = 1 - expected_frac_nu[plane]

        exp_mid_frac_nu = (expected_frac_nu["x"] + expected_frac_nu["y"]) / 2
        if exp_mid_frac_nu > expected_frac_nu["x"]:
            nux0_range = [0.0, exp_mid_frac_nu]
            nuy0_range = [exp_mid_frac_nu + 1e-12, 0.5]
        else:
            nux0_range = [exp_mid_frac_nu + 1e-12, 0.5]
            nuy0_range = [0.0, exp_mid_frac_nu]

        self.lin_comp["actual"] = self.extract_lin_freq_components_from_multi_BPM_tbt(
            tbt["x"],
            tbt["y"],
            nux0_range=nux0_range,
            nuy0_range=nuy0_range,
        )

        model_twi = self.twiss["design"]
        model_bpm_beta = model_twi["beta"]["bpms"]
        normalizer = self.normalize_and_phase_diff_lin_freq_components
        self.norm_lin_comp["actual"] = normalizer(
            self.lin_comp["actual"], model_bpm_beta, self.tune_above_half["actual"]
        )

        raw_tbt_nus = {plane: self.lin_comp["actual"][f"nu{plane}"] for plane in "xy"}
        self.tbt_avg_nu["actual"] = {
            plane: 1 - np.mean(raw_tbt_nus[plane])
            if self.tune_above_half["actual"][plane]
            else np.mean(raw_tbt_nus[plane])
            for plane in "xy"
        }

    def calc_response(self, quad_name_or_names, dK1, verbose=0):
        resp = {}

        if "design" not in self.lin_comp:
            self.calc_design_lin_comp()

        model_tbt_avg_nu = self.tbt_avg_nu["design"]

        model_twi = self.twiss["design"]
        model_bpm_eta = model_twi["eta"]["bpms"]

        if verbose >= 1:
            print(f"Changing K1 for [{quad_name_or_names}] by {dK1:.6g}")
        self.change_K1_setpoint_by(quad_name_or_names, dK1)

        self.update_actual_twiss()
        twi = self.twiss["actual"]

        self.calc_actual_lin_comp()

        tbt_avg_nu = self.tbt_avg_nu["actual"]

        resp["nu_tbt"] = {}
        resp["nu_twi"] = {}
        resp["eta"] = {}
        for plane in "xy":
            resp["nu_tbt"][plane] = (tbt_avg_nu[plane] - model_tbt_avg_nu[plane]) / dK1
            resp["nu_twi"][plane] = (twi[f"nu{plane}"] - model_twi[f"nu{plane}"]) / dK1

            _base_eta = model_bpm_eta[plane]
            _eta = twi["eta"]["bpms"][plane]
            resp["eta"][plane] = (_eta - _base_eta) / dK1

        model_norm_lin_comp = self.norm_lin_comp["design"]
        norm_lin_comp = self.norm_lin_comp["actual"]
        resp["_extra"] = dict(norm_lin_comp=norm_lin_comp)

        resp["norm_lin_comp"] = {
            comp: (norm_lin_comp[comp] - model_norm_lin_comp[comp]) / dK1
            for comp in ["x1_bbeat", "y1_bbeat", "x1_dphi", "y1_dphi", "x2", "y2"]
        }

        for plane in "xy":
            comp = f"{plane}1_phi"
            diff_rad = norm_lin_comp[comp] - model_norm_lin_comp[comp]
            assert diff_rad.size == self.nBPM[plane] + 1
            diff_rad_wo_1st_bpm = (diff_rad - diff_rad[0])[1:]
            resp["norm_lin_comp"][comp] = diff_rad_wo_1st_bpm / dK1

        # Reset the quad strength change
        if verbose >= 1:
            print(f"Changing K1 for [{quad_name_or_names}] by {-dK1:.6g}")
        self.change_K1_setpoint_by(quad_name_or_names, -dK1)

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
        self.quad_Ls = {}
        self.quad_col2names = {}

        for kind, kind_str, name_list in [
            ("normal", "normal", normal_quad_names),
            ("skew", "skew", skew_quad_names),
        ]:
            self.quad_flat_names[kind] = []
            self.quad_props[kind] = {}
            self.quad_Ls[kind] = {}
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

                self.quad_Ls[kind][name] = elem_props[name]["properties"].get("L", 0.0)

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
        self, elem_name_s: Union[str, Tuple, List, np.ndarray], value: float
    ):
        if isinstance(elem_name_s, str):
            _name = elem_name_s
            if _name in self.quad_props["normal"]:
                kind = "normal"
            elif _name in self.quad_props["skew"]:
                kind = "skew"
            else:
                raise ValueError(f'Element "{_name}" not in `self.quad_props`.')

            self.quad_props[kind][_name]["value"] = value

        elif isinstance(elem_name_s, (tuple, list, np.ndarray)):
            for _name in elem_name_s:
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
        self, elem_name_s: Union[str, Tuple, List, np.ndarray], delta_value: float
    ):
        if isinstance(elem_name_s, str):
            _name = elem_name_s
            if _name in self.quad_props["normal"]:
                kind = "normal"
            elif _name in self.quad_props["skew"]:
                kind = "skew"
            else:
                raise ValueError(f'Element "{_name}" not in `self.quad_props`.')

            self.quad_props[kind][_name]["value"] += delta_value

        elif isinstance(elem_name_s, (tuple, list, np.ndarray)):
            for _name in elem_name_s:
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
        self.obs_weights_wo_skews = {
            k: v if k not in ("x2_re", "x2_im", "y2_re", "y2_im", "etay") else 0.0
            for k, v in obs_weights.items()
        }

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

            mat_list = defaultdict(list)
            for comp, bbeat_or_phi in product(["x1", "y1"], ["bbeat", "phi"]):
                plane = comp[0]
                obs_key = f"{comp}_{bbeat_or_phi}"
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
                            temp_col += src_M[:, i]
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

        if False:
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
        self._M_wo_skews = np.vstack(
            [self.RM[k] * self.obs_weights_wo_skews[k] for k in self.obs_keys]
        )

        self._U, self._sv, self._VT = calcSVD(self._M)
        self._U_wo_skews, self._sv_wo_skews, self._VT_wo_skews = calcSVD(
            self._M_wo_skews
        )
        if False:
            plt.figure()
            plt.semilogy(self._sv / self._sv[0], "b.-")

            plt.figure()
            plt.semilogy(self._sv_wo_skews / self._sv_wo_skews[0], "b.-")

        Sinv_trunc = calcTruncSVMatrix(self._sv, rcond=rcond, nsv=None, disp=0)
        self._M_pinv = self._VT.T @ Sinv_trunc @ self._U.T

        Sinv_trunc = calcTruncSVMatrix(self._sv_wo_skews, rcond=rcond, nsv=None, disp=0)
        self._M_pinv_wo_skews = self._VT_wo_skews.T @ Sinv_trunc @ self._U_wo_skews.T

    def correct(self, cor_frac: float = 0.7):
        if "design" not in self.tbt_avg_nu:
            self.calc_design_lin_comp()

        hist = self._actual_design_diff_history
        last_bbeat_x_rms = np.std(hist["x1_bbeat"][-1])
        last_bbeat_y_rms = np.std(hist["y1_bbeat"][-1])
        _bbeat_info = f"({last_bbeat_x_rms*1e2:.3g}, {last_bbeat_y_rms*1e2:.3g})"
        print(f"\n# Beta-beat [%]: {_bbeat_info}")

        bbeat_thresh = self.max_beta_beat_thresh_for_coup_cor

        if max([last_bbeat_x_rms, last_bbeat_y_rms]) > bbeat_thresh:
            use_only_normal_quads = True
            _bbeat_info += f" "
            print(f"Beta-beat too large (> {bbeat_thresh*1e2:.3g}%)")
            print("Will only correct beta-beat and etax in this iteration.")
            obs_weights = self.obs_weights_wo_skews
            M_pinv = self._M_pinv_wo_skews
            M = self._M_wo_skews
        else:
            use_only_normal_quads = False
            obs_weights = self.obs_weights
            M_pinv = self._M_pinv
            M = self._M

        obs_diff_list = []
        for k in self.obs_keys:
            w = obs_weights[k]

            if k in ("etax", "etay"):
                plane = k[-1]
                v_actual = self.twiss["actual"]["eta"]["bpms"][plane]
                v_design = self.twiss["design"]["eta"]["bpms"][plane]
            elif k in ("nux", "nuy"):
                plane = k[-1]
                v_actual = self.tbt_avg_nu["actual"][plane]
                v_design = self.tbt_avg_nu["design"][plane]
            else:
                if k in self.norm_lin_comp["actual"]:
                    v_actual = self.norm_lin_comp["actual"][k]
                    v_design = self.norm_lin_comp["design"][k]
                    if k in ("x1_phi", "y1_phi"):
                        plane = k[0]
                        diff_rad = v_design - v_actual
                        diff_rad_wo_1st_bpm = (diff_rad - diff_rad[0])[1:]
                        v_design = diff_rad_wo_1st_bpm
                        v_actual = np.zeros_like(v_design)
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

        dK1s = M_pinv @ dv

        if use_only_normal_quads:
            dK1s[self.nQUAD["normal"] :] = 0.0

        if False:
            plt.figure()
            plt.plot(dv, "b.-")
            plt.plot(M @ dK1s, "r.-")

            plt.figure()
            plt.plot(dK1s[: self.nQUAD["normal"]], "b.-")
            if self.nQUAD["skew"] != 0:
                plt.plot(dK1s[self.nQUAD["normal"] :], "r.-")

        dK1s *= cor_frac

        self._dK1s_history.append(dK1s)

        self._back_up_quad_setpoints()

        quad_names_list = self.quad_col2names["normal"] + self.quad_col2names["skew"]
        assert len(quad_names_list) == len(dK1s)
        for quad_names, dK1 in zip(quad_names_list, dK1s):
            self.change_K1_setpoint_by(quad_names, dK1)

    def _back_up_quad_setpoints(self):
        self._backup_quad_setpoints = pickle.dumps(self.quad_props)

    def restore_backed_up_quad_setpoints(self):

        self._dK1s_history.pop()

        self.quad_props = pickle.loads(self._backup_quad_setpoints)
        self._uncommited_quad_change = True

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

        order = 1
        for plane in "xy":
            comp = f"{plane}{order}"
            obs_key = f"{comp}_bbeat"
            hist[obs_key].append(actual[obs_key] - design[obs_key])

            obs_key = f"{comp}_dphi"  # not used for correction
            hist[obs_key].append(actual[obs_key] - design[obs_key])

            obs_key = f"{comp}_phi"
            diff_rad = actual[obs_key] - design[obs_key]
            diff_rad_wo_1st_bpm = (diff_rad - diff_rad[0])[1:]
            hist[obs_key].append(diff_rad_wo_1st_bpm)

        order = 2
        for plane in "xy":
            comp = f"{plane}{order}"
            hist[f"{comp}_re"].append(actual[comp].real - design[comp].real)
            hist[f"{comp}_im"].append(actual[comp].imag - design[comp].imag)

    def plot_actual_design_diff(self, history=True):
        hist = self._actual_design_diff_history

        if history:
            legends = ["$\mathrm{Initial}$"]
            for i in range(len(hist["nux"]))[1:]:
                legends.append(f"$\mathrm{{Iter.\, {i}}}$")

        plt.figure()
        plt.subplot(211)
        plt.plot(hist["nux"], "b.-")
        plt.axhline(0.0, color="k")
        plt.ylabel(r"$\Delta \nu_x$", size="x-large")
        plt.subplot(212)
        plt.plot(hist["nuy"], "b.-")
        plt.axhline(0.0, color="k")
        plt.ylabel(r"$\Delta \nu_y$", size="x-large")
        plt.xlabel(r"$\mathrm{Iteration}$", size="x-large")
        plt.tight_layout()

        plt.figure()
        for plane, plot_ind in zip("xy", [211, 212]):
            plt.subplot(plot_ind)
            if history:
                plt.plot(np.array(hist[f"eta{plane}"]).T * 1e3, ".-")

                if plane == "x":
                    _add_top_legend_adaptive_ncol(legends, fontsize="medium")
            else:
                plt.plot(hist[f"eta{plane}"][-1] * 1e3, ".-")
            plt.ylabel(rf"$\Delta \eta_{plane}\; [\mathrm{{mm}}]$", size="x-large")
        plt.xlabel(r"$\mathrm{BPM\; Index}$", size="x-large")
        plt.tight_layout()

        if history:
            plt.figure()
            for plane, plot_ind in zip("xy", [211, 212]):
                plt.subplot(plot_ind)
                _rms_mm = np.std(hist[f"eta{plane}"], axis=1) * 1e3
                _ini_val = _rms_mm[0]
                _fin_val = _rms_mm[-1]
                plt.plot(_rms_mm, ".-")
                plt.ylabel(
                    rf"$\mathrm{{RMS}}\; \Delta \eta_{plane}\; [\mathrm{{mm}}]$",
                    size="x-large",
                )
                # plt.title(rf"$\mathrm{{RMS}}\; (\Delta \eta_x, \Delta \eta_y) = ({_ini_val['x']:.3g}, {_ini_val['y']:.3g}) \Rightarrow ({_fin_val['x']:.3g}, {_fin_val['y']:.3g}) \; [\mathrm{{mm}}]$")
                plt.title(
                    rf"$\mathrm{{RMS}}\; \Delta \eta_{plane} = {_ini_val:.3g} \Rightarrow {_fin_val:.3g} \; [\mathrm{{mm}}]$"
                )
                plt.ylim([0.0, plt.ylim()[1]])
            plt.xlabel(r"$\mathrm{Iteration}$", size="x-large")
            plt.tight_layout()

        order = 1
        for plane in "xy":
            comp = f"{plane}{order}"

            plt.figure(figsize=(8, 8))

            plt.subplot(311)
            if history:
                plt.plot(np.array(hist[f"{comp}_bbeat"]).T * 1e2, ".-")
                _add_top_legend_adaptive_ncol(legends, fontsize="medium")
            else:
                plt.plot(hist[f"{comp}_bbeat"][-1] * 1e2, ".-")
            plt.axhline(0.0, color="k")
            beta_beat_label = rf"$\Delta \beta_{plane} / \beta_{plane} [\%]$"
            plt.ylabel(beta_beat_label, size="x-large")

            plt.subplot(312)
            if history:
                plt.plot(np.array(hist[f"{comp}_dphi"]).T / (2 * np.pi), ".-")
            else:
                plt.plot(hist[f"{comp}_dphi"][-1] / (2 * np.pi), ".-")
            plt.axhline(0.0, color="k")
            dphi_label = rf"$\Delta (\delta\phi_{plane}) [\nu]$"
            plt.ylabel(dphi_label, size="x-large")

            plt.subplot(313)
            if history:
                plt.plot(np.array(hist[f"{comp}_phi"]).T / (2 * np.pi), ".-")
            else:
                plt.plot(hist[f"{comp}_phi"][-1] / (2 * np.pi), ".-")
            plt.axhline(0.0, color="k")
            phi_label = rf"$\Delta \phi_{plane} [\nu]$"
            plt.ylabel(phi_label, size="x-large")

            plt.xlabel(r"$\mathrm{BPM\; Index}$", size="x-large")
            plt.tight_layout()

            if history:
                plt.figure()
                plt.subplot(311)
                _rms_bbeat = np.std(hist[f"{comp}_bbeat"], axis=1) * 1e2
                plt.plot(_rms_bbeat, ".-")
                plt.ylim([0.0, plt.ylim()[1]])
                plt.ylabel(beta_beat_label, size="x-large")
                plt.title(r"$\mathrm{RMS}$", size="x-large")
                _ini_val = _rms_bbeat[0]
                _fin_val = _rms_bbeat[-1]
                plt.title(
                    rf"$\mathrm{{RMS}}\; \Delta \beta_{plane} / \beta_{plane} = {_ini_val:.3g} \Rightarrow {_fin_val:.3g} [\%]$"
                )
                plt.subplot(312)
                plt.plot(np.std(hist[f"{comp}_dphi"], axis=1) / (2 * np.pi), ".-")
                plt.ylim([0.0, plt.ylim()[1]])
                plt.ylabel(dphi_label, size="x-large")
                plt.tight_layout()
                plt.subplot(313)
                plt.plot(np.std(hist[f"{comp}_phi"], axis=1) / (2 * np.pi), ".-")
                plt.ylim([0.0, plt.ylim()[1]])
                plt.ylabel(phi_label, size="x-large")
                plt.xlabel(r"$\mathrm{Iteration}$", size="x-large")
                plt.tight_layout()

        order = 2
        for plane in "xy":
            comp = f"{plane}{order}"

            plt.figure()

            plt.subplot(211)
            if history:
                plt.plot(np.array(hist[f"{comp}_re"]).T, ".-")
                _add_top_legend_adaptive_ncol(legends, fontsize="medium")
            else:
                plt.plot(hist[f"{comp}_re"][-1], ".-")
            plt.ylabel(rf"$\Re{{(\Delta {plane}_{order})}}$", size="x-large")

            plt.subplot(212)
            if history:
                plt.plot(np.array(hist[f"{comp}_im"]).T, ".-")
            else:
                plt.plot(hist[f"{comp}_im"][-1], ".-")
            plt.ylabel(rf"$\Im{{(\Delta {plane}_{order})}}$", size="x-large")

            plt.xlabel(r"$\mathrm{BPM\; Index}$", size="x-large")
            plt.tight_layout()

            if history:
                plt.figure()
                plt.subplot(211)
                plt.plot(np.std(hist[f"{comp}_re"], axis=1), ".-")
                plt.ylim([0.0, plt.ylim()[1]])
                plt.ylabel(
                    rf"$\mathrm{{RMS}}\; \Re{{(\Delta {plane}_{order})}}$",
                    size="x-large",
                )
                plt.subplot(212)
                plt.plot(np.std(hist[f"{comp}_im"], axis=1), ".-")
                plt.ylim([0.0, plt.ylim()[1]])
                plt.ylabel(
                    rf"$\mathrm{{RMS}}\; \Im{{(\Delta {plane}_{order})}}$",
                    size="x-large",
                )
                plt.xlabel(r"$\mathrm{Iteration}$", size="x-large")
                plt.tight_layout()

    def plot_quad_change(self, integ_strength=True, history=True):
        hist = np.array(self._dK1s_history).T

        dK1s_normal = hist[: self.nQUAD["normal"], :]
        dK1s_skew = hist[self.nQUAD["normal"] :, :]

        if integ_strength:
            if dK1s_normal.shape[0] != 0:
                dK1Ls_normal = []
                assert len(self.quad_col2names["normal"]) == dK1s_normal.shape[0]
                for i, names in enumerate(self.quad_col2names["normal"]):
                    _dK1s = dK1s_normal[i, :]
                    _Ls = np.array([self.quad_Ls["normal"][name] for name in names])
                    dK1Ls_normal.append(np.array([_dK1s * _L for _L in _Ls]))
                dK1s_normal = np.vstack(dK1Ls_normal)

            if dK1s_skew.shape[0] != 0:
                dK1Ls_skew = []
                assert len(self.quad_col2names["skew"]) == dK1s_skew.shape[0]
                for i, names in enumerate(self.quad_col2names["skew"]):
                    _dK1s = dK1s_skew[i, :]
                    _Ls = np.array([self.quad_Ls["skew"][name] for name in names])
                    dK1Ls_skew.append(np.array([_dK1s * _L for _L in _Ls]))
                dK1s_skew = np.vstack(dK1Ls_skew)

            strength_str = "K_1 L"
            strength_unit = r"\mathrm{m}^{-1}"
        else:
            strength_str = "K_1"
            strength_unit = r"\mathrm{m}^{-2}"

        if history:
            legends = []
            for i in range(hist.shape[1]):
                legends.append(f"$\mathrm{{Iter.\, {i}}}$")

        plt.figure()
        plt.subplot(211)
        if history:
            plt.plot(dK1s_normal, ".-")
            _add_top_legend_adaptive_ncol(legends, fontsize="medium")
        else:
            plt.plot(dK1s_normal[:, -1], ".-")
        plt.ylabel(rf"$\Delta {strength_str}\; [{strength_unit}]$", size="x-large")
        plt.xlabel(r"$\mathrm{Normal\; Quad.\; Index}$", size="x-large")
        plt.subplot(212)
        if history:
            plt.plot(dK1s_skew, ".-")
        else:
            plt.plot(dK1s_skew[:, -1], ".-")
        plt.ylabel(rf"$\Delta {strength_str}\; [{strength_unit}]$", size="x-large")
        plt.xlabel(r"$\mathrm{Skew\; Quad.\; Index}$", size="x-large")
        plt.tight_layout()

        cum_dK1s_normal = np.cumsum(dK1s_normal, axis=1)
        cum_dK1s_skew = np.cumsum(dK1s_skew, axis=1)

        plt.figure()
        plt.subplot(211)
        if history:
            plt.plot(cum_dK1s_normal, ".-")
            _add_top_legend_adaptive_ncol(legends, fontsize="medium")
        else:
            plt.plot(cum_dK1s_normal[:, -1], ".-")
        plt.ylabel(
            rf"$\mathrm{{Cum.}}\; \Delta {strength_str}\; [{strength_unit}]$",
            size="x-large",
        )
        plt.xlabel(r"$\mathrm{Normal\; Quad.\; Index}$", size="x-large")
        plt.subplot(212)
        if history:
            plt.plot(cum_dK1s_skew, ".-")
        else:
            plt.plot(cum_dK1s_skew[:, -1], ".-")
        plt.ylabel(
            rf"$\mathrm{{Cum.}}\; \Delta {strength_str}\; [{strength_unit}]$",
            size="x-large",
        )
        plt.xlabel(r"$\mathrm{Skew\; Quad.\; Index}$", size="x-large")
        plt.tight_layout()


class AbstractFacility:
    def __init__(
        self,
        design_LTE: Lattice,
        n_turns: int = 128,
        tbt_ps_offset_wrt_CO: Union[None, Dict] = None,
        parallel: bool = True,
        output_folder: Union[None, Path, str] = None,
    ):
        if output_folder is None:
            self.output_folder = Path.cwd()
        else:
            self.output_folder = Path(output_folder)

        self.design_LTE = design_LTE
        self.n_turns = n_turns

        temp_tbt_ps_offset_wrt_CO = dict(x0=1e-6, y0=1e-6)
        if tbt_ps_offset_wrt_CO:
            temp_tbt_ps_offset_wrt_CO.update(tbt_ps_offset_wrt_CO)
        self.tbt_ps_offset_wrt_CO = temp_tbt_ps_offset_wrt_CO

        self.parallel = parallel

        self.fsdb = None
        self.TRM_filepath = None
        self.linopt_RM_filepath = None

        self.config_descr = dict(quad={}, bpm={})

        self.optcor = None

    def show_avail_quad_configs(self):
        msg = (
            "All available quad (knob) configuration set keys for "
            "linear optics/coupling correction:"
        )
        print(msg)
        for k, descr in self.config_descr["quad"].items():
            if isinstance(k, int):
                print(f"{k}: {descr}")
            elif isinstance(k, str):
                print(f"'{k}': {descr}")
            else:
                raise TypeError("Knob config key must be an integer or a string.")

    def show_avail_BPM_configs(self):
        msg = (
            "All available BPM (observable) configuration set keys for "
            "linear optics/coupling correction:"
        )
        print(msg)
        for k, descr in self.config_descr["bpm"].items():
            if isinstance(k, int):
                print(f"{k}: {descr}")
            elif isinstance(k, str):
                print(f"'{k}': {descr}")
            else:
                raise TypeError("Observable config key must be an integer or a string.")

    def get_BPM_elem_inds_for_orbit_cor(self):
        raise NotImplementedError

    def get_corrector_elem_inds_for_orbit_cor(self):
        raise NotImplementedError

    def get_BPM_elem_inds_for_linopt_RM(self):
        raise NotImplementedError

    def get_quad_elem_inds_for_linopt_RM(self):
        raise NotImplementedError

    def get_BPM_elem_inds_for_linopt_cor(self, set_key: Union[int, str]):
        raise NotImplementedError

    def get_quad_elem_inds_for_linopt_cor(self, set_key: Union[int, str]):
        raise NotImplementedError

    def generate_TRM_file(self):
        """Generate trajectory response matrix for orbit correction."""

        fsdb = self.fsdb

        bpm_elem_inds = self.get_BPM_elem_inds_for_orbit_cor()
        cor_elem_inds = self.get_corrector_elem_inds_for_orbit_cor()

        output_h5_filepath = (
            self.output_folder / f"{self.design_LTE.LTEZIP_filepath.stem}_TRM.h5"
        )

        self.TRM_filepath = generate_TRM_file(
            self.design_LTE,
            bpm_elem_inds["x"],
            bpm_elem_inds["y"],
            cor_elem_inds["x"],
            cor_elem_inds["y"],
            n_turns=2,
            N_KICKS=fsdb.N_KICKS,
            output_h5_filepath=output_h5_filepath,
        )

    def generate_linopt_numRM_file(
        self,
        remote_opts: Union[None, Dict] = None,
        normal_quad_inds: Union[None, List] = None,
        skew_quad_inds: Union[None, List] = None,
    ):
        """Generate response matrices for linear optics / coupling correction.

        normal_quad_inds, skew_quad_inds:
           `None` means all normal/skew quads will be included.
        """

        fsdb = self.fsdb

        LTE = self.design_LTE

        bpm_elem_inds = self.get_BPM_elem_inds_for_linopt_RM()
        bpm_names = {}
        for plane, inds in bpm_elem_inds.items():
            names = LTE.get_names_from_elem_inds(inds)
            assert len(names) == len(
                np.unique(names)
            )  # check uniqueness of element names
            bpm_names[plane] = names

        quad_elem_inds = self.get_quad_elem_inds_for_linopt_RM()
        quad_names = {}
        for quad_type, inds in quad_elem_inds.items():
            names = LTE.get_names_from_elem_inds(inds)
            assert len(names) == len(
                np.unique(names)
            )  # check uniqueness of element names
            quad_names[quad_type] = names

        args_optcor = (
            LTE,
            fsdb.E_MeV,
            bpm_names["x"],
            bpm_names["y"],
            quad_names["normal"],
            quad_names["skew"],
        )
        kwargs_optcor = dict(
            n_turns=self.n_turns,
            tbt_ps_offset_wrt_CO=self.tbt_ps_offset_wrt_CO,
            design_LTE=pickle.loads(pickle.dumps(self.design_LTE)),
            N_KICKS=fsdb.N_KICKS,
            tempdir_path=None,
        )

        dK1 = 1e-4  # [m^(-2)]

        if fsdb.LTE.LTEZIP_filepath == "":
            input_dict = dict(
                design_LTE_filepath=fsdb.LTE.LTE_filepath,
                used_beamline_name=fsdb.LTE.used_beamline_name,
            )

            self.linopt_RM_filepath = (
                self.output_folder
                / f"{input_dict['design_LTE_filepath'].stem}_linopt_numRM.h5"
            )

        else:
            input_dict = dict(design_LTEZIP_filepath=fsdb.LTE.LTEZIP_filepath)
            self.linopt_RM_filepath = (
                self.output_folder
                / f"{input_dict['design_LTEZIP_filepath'].stem}_linopt_numRM.h5"
            )

        input_dict.update(
            dict(
                lattice_type=fsdb.lat_type,
                E_MeV=fsdb.E_MeV,
                N_KICKS=fsdb.N_KICKS,
                bpm_names=bpm_names,
                quad_names=quad_names,
                n_turns=self.n_turns,
                tbt_ps_offset_wrt_CO=self.tbt_ps_offset_wrt_CO,
                dK1=dK1,
            )
        )

        if not self.parallel:
            if normal_quad_inds is None:
                quad_names["normal"] = list(quad_names["normal"])
            else:
                quad_names["normal"] = [
                    quad_names["normal"][_i] for _i in normal_quad_inds
                ]

            if skew_quad_inds is None:
                quad_names["skew"] = list(quad_names["skew"])
            else:
                quad_names["skew"] = [quad_names["skew"][_i] for _i in skew_quad_inds]

            resp_list = _calc_linopt_resp(
                quad_names["normal"] + quad_names["skew"],
                dK1,
                args_optcor,
                kwargs_optcor,
            )

        else:
            if remote_opts is None:
                remote_opts = dict(ntasks=100)

            params = list(quad_names["normal"]) + list(quad_names["skew"])
            remote_opts["ntasks"] = min([len(params), remote_opts["ntasks"]])
            chunked_list, reverse_mapping = chunk_list(params, remote_opts["ntasks"])

            module_name = "pyelegant.linopt_correct"
            func_name = "_calc_linopt_resp"
            args = (dK1, args_optcor, kwargs_optcor)

            if False:  # DEBUG
                import importlib

                mod = importlib.import_module(module_name)
                func = getattr(mod, func_name)
                for chunk in chunked_list[:2]:
                    out = func(chunk, *args)

            err_log_check = dict(funcs=[remote.check_remote_err_log_exit_code])

            chunked_output, slurm_info = remote.run_mpi_python(
                remote_opts,
                module_name,
                func_name,
                chunked_list,
                args,
                err_log_check=err_log_check,
                paths_to_prepend=[str(Path.cwd())],
                ret_slurm_info=True,
            )
            # print(chunked_output)

            if "Traceback" in slurm_info.get("err_log", ""):
                print(slurm_info["err_log"])
                raise RuntimeError(
                    f"### An error occurred during parallel run of {module_name}.{func_name}() ###"
                )

            resp_list = unchunk_list_of_lists(chunked_output, reverse_mapping)

        dnu_tbt_list = dict(x=[], y=[])
        dnu_twi_list = dict(x=[], y=[])
        deta_list = dict(x=[], y=[])
        dC_list = dict(x1=[], y1=[], x2=[], y2=[])
        dC_list = dict(x1_bbeat=[], x1_phi=[], y1_bbeat=[], y1_phi=[], x2=[], y2=[])

        for resp in resp_list:
            for plane in "xy":
                dnu_tbt_list[plane].append(resp["nu_tbt"][plane])
                dnu_twi_list[plane].append(resp["nu_twi"][plane])
                deta_list[plane].append(resp["eta"][plane])

            for comp in dC_list.keys():
                dC_list[comp].append(resp["norm_lin_comp"][comp])

        nQnormal = len(quad_names["normal"])
        s_ = dict(normal=np.s_[:nQnormal], skew=np.s_[nQnormal:])

        # --keys: x, y
        # shape = (nQuad,)
        M_tune_tbt = {
            quad_type: {k: np.array(v)[s_[quad_type]] for k, v in dnu_tbt_list.items()}
            for quad_type in ["normal", "skew"]
        }
        M_tune_twiss = {
            quad_type: {k: np.array(v)[s_[quad_type]] for k, v in dnu_twi_list.items()}
            for quad_type in ["normal", "skew"]
        }

        # -- keys: x1_bbeat, y1_bbeat
        # shape = (nBPM x nQuad) [1st-turn 1st BPM to 1st-turn last BPM]
        # -- keys: x1_phi, y1_phi
        # shape = (nBPM x nQuad) [1st-turn 2nd BPM to 2nd-turn 1st BPM]
        # (Phase shifted such that 1st-turn 1st BPM phase is always zero)
        # -- keys: x2, y2
        # shape = (nBPM+1 x nQuad) [1st-turn 1st BPM to 2nd-turn 1st BPM]
        M_lin_comp = {
            quad_type: {
                k: (np.array(v).T)[:, s_[quad_type]] for k, v in dC_list.items()
            }
            for quad_type in ["normal", "skew"]
        }

        # -- keys: x, y
        # shape = (nBPM x nQuad)
        M_eta = {
            quad_type: {
                k: (np.array(v).T)[:, s_[quad_type]] for k, v in deta_list.items()
            }
            for quad_type in ["normal", "skew"]
        }

        h5_kwargs = dict(compression="gzip")
        with h5py.File(self.linopt_RM_filepath, "w") as f:
            binary_data = pickle.dumps(input_dict)
            base64_input_data = base64.b64encode(binary_data).decode("utf-8")
            f["input_dict"] = base64_input_data
            f["input_dict"].attrs[
                "how_to_load"
            ] = """To retrieve this data from the saved file:
        base64_data = f['input_dict'][()]
        binary_data = base64.b64decode(base64_data)
        original_data = pickle.loads(binary_data)"""

            g1 = f.create_group("M_tune_tbt")
            for quad_type, v in M_tune_tbt.items():
                g2 = g1.create_group(quad_type)
                for plane, m in v.items():
                    g2.create_dataset(plane, data=m, **h5_kwargs)

            g1 = f.create_group("M_tune_twiss")
            for quad_type, v in M_tune_twiss.items():
                g2 = g1.create_group(quad_type)
                for plane, m in v.items():
                    g2.create_dataset(plane, data=m, **h5_kwargs)

            g1 = f.create_group("M_lin_comp")
            for quad_type, v in M_lin_comp.items():
                g2 = g1.create_group(quad_type)
                for comp, m in v.items():
                    g2.create_dataset(comp, data=m, **h5_kwargs)

            g1 = f.create_group("M_eta")
            for quad_type, v in M_eta.items():
                g2 = g1.create_group(quad_type)
                for plane, m in v.items():
                    g2.create_dataset(plane, data=m, **h5_kwargs)

    def correct_orbit(
        self,
        err_LTEZIP_filepath: Union[Path, str],
        zero_orbit_type="BBA",
        BBA_elem_type="KQUAD",
        BBA_elem_names=None,
        iter_opts=None,
        rcond=1e-4,
        plot=False,
    ):
        fsdb = self.fsdb

        err_LTEZIP_filepath = Path(err_LTEZIP_filepath)

        LTE = Lattice(LTEZIP_filepath=err_LTEZIP_filepath, tempdir_path=None, verbose=0)

        if std_print_enabled["out"]:
            disable_stdout()
            restore_stdout = True
        else:
            restore_stdout = False

        old_LTE_filepath = new_LTE_filepath = LTE.LTE_filepath
        eleutil.save_lattice_after_alter_elements(
            old_LTE_filepath,
            new_LTE_filepath,
            alter_elements=[
                dict(
                    name="*",
                    type=k.upper(),
                    item="N_KICKS",
                    value=v,
                    allow_missing_elements=True,
                )
                for k, v in fsdb.N_KICKS.items()
            ],
        )

        if restore_stdout:
            enable_stdout()

        bpm_elem_inds = self.get_BPM_elem_inds_for_orbit_cor()
        bpm_names = {}
        for plane, inds in bpm_elem_inds.items():
            names = LTE.get_names_from_elem_inds(inds)
            assert len(names) == len(np.unique(names))
            bpm_names[plane] = names

        cor_elem_inds = self.get_corrector_elem_inds_for_orbit_cor()
        cor_names = {}
        for plane, inds in cor_elem_inds.items():
            names = LTE.get_names_from_elem_inds(inds)
            assert len(names) == len(np.unique(names))
            cor_names[plane] = names

        threader = ClosedOrbitThreader(
            LTE,
            fsdb.E_MeV,
            bpm_names["x"],
            bpm_names["y"],
            cor_names["x"],
            cor_names["y"],
            zero_orbit_type=zero_orbit_type,
            BBA_elem_type=BBA_elem_type,
            BBA_elem_names=BBA_elem_names,
            TRM_filepath=self.TRM_filepath,
            iter_opts=iter_opts,
        )

        out = threader.start_fixed_energy_orbit_correction(
            rcond=rcond, debug_print=True, debug_plot=False
        )

        try:
            assert out["success"]
        except AssertionError:
            print(f"ERROR: start_fixed_energy_orbit_correction() apparently failed.")
            raise

        inj_coords_list = out["inj_coords_list"]
        # hkicks_hist = out["hkicks_hist"]
        # vkicks_hist = out["vkicks_hist"]
        # traj_hist = out["traj_hist"]

        # Maintain the closed trajectory while the ring length is made closer to
        # the circumference by adjusting the beam energy.
        init_inj_coords = inj_coords_list[-1]
        out = threader.start_fixed_length_orbit_correction(
            init_inj_coords=init_inj_coords,
            rcond=rcond,
            debug_print=True,
            debug_plot=False,
            plot=plot,
        )

        parent = err_LTEZIP_filepath.parent
        stem = err_LTEZIP_filepath.stem

        CO_filepath = parent / f"{stem}_cOrb_CO.hdf5"
        with h5py.File(CO_filepath, "w") as f:
            f["x"], f["xp"], f["y"], f["yp"] = out["fin_inj_coords"]
            f["dp"] = out["fin_dp"]

        fin_LTEZIP_filepath = parent / f"{stem}_cOrb.ltezip"
        threader.save_current_lattice_to_LTEZIP_file(fin_LTEZIP_filepath)

        LTE.remove_tempdir()
        threader.remove_tempdir()

        return dict(LTEZIP=fin_LTEZIP_filepath, CO_inj_ps=CO_filepath)

    def configure(
        self,
        LTE_to_be_corrected: Lattice,
        quad_set_key: Union[int, str, None] = None,
        quad_names: Union[Dict, None] = None,
        bpm_set_key: Union[int, str, None] = None,
        bpm_names: Union[Dict, None] = None,
        obs_weights: Union[Dict, None] = None,
        max_beta_beat_thresh_for_coup_cor: float = 5e-2,
        rcond: float = 1e-3,
        inj_CO_ps_filepath: Union[Path, str] = "",
        linopt_RM_filepath: Union[Path, str] = "",
    ):
        """(`obs_weights` == None) means the correction will use all observables with default weights."""
        fsdb = self.fsdb
        LTE = fsdb.LTE

        lumped_quad_names = None
        if quad_set_key is None:
            if quad_names is None:
                lumped_quad_inds = self.get_quad_elem_inds_for_linopt_cor(
                    "all_independent"
                )
            else:
                lumped_quad_names = quad_names
        else:
            if quad_names is None:
                lumped_quad_inds = self.get_quad_elem_inds_for_linopt_cor(quad_set_key)
            else:
                raise ValueError(
                    "Either `quad_set_key` or `quad_names` should be defined, not both"
                )

        if lumped_quad_names is None:
            lumped_quad_names = {
                quad_type: [
                    LTE.get_names_from_elem_inds(_inds) for _inds in list_of_inds
                ]
                for quad_type, list_of_inds in lumped_quad_inds.items()
            }

        temp_bpm_names = None
        if bpm_set_key is None:
            if bpm_names is None:
                bpm_elem_inds = self.get_BPM_elem_inds_for_linopt_cor("all")
            else:
                temp_bpm_names = bpm_names
        else:
            if bpm_names is None:
                bpm_elem_inds = self.get_BPM_elem_inds_for_linopt_cor(bpm_set_key)
            else:
                raise ValueError(
                    "Either `bpm_set_key` or `bpm_names` should be defined, not both"
                )

        if temp_bpm_names is None:
            bpm_names = {}
            for plane, inds in bpm_elem_inds.items():
                names = self.design_LTE.get_names_from_elem_inds(inds)
                assert len(names) == len(
                    np.unique(names)
                )  # check uniqueness of element names
                bpm_names[plane] = names
        else:
            for plane, names in bpm_names.items():
                assert len(names) == len(
                    np.unique(names)
                )  # check uniqueness of element names

        default_obs_weights = dict(
            x1_bbeat=1.0,
            y1_bbeat=1.0,
            x1_phi=1.0,
            y1_phi=1.0,
            nux=10.0,  # optional. "x1_phi" will correct "nux", too.
            nuy=10.0,  # optional. "y1_phi" will correct "nuy", too.
            etax=10.0,
            x2_re=1.0,
            x2_im=1.0,
            y2_re=1.0,
            y2_im=1.0,
            etay=50.0,
        )
        if obs_weights is None:
            obs_weights = default_obs_weights.copy()
        else:
            assert all([k in default_obs_weights for k in obs_weights.keys()])

        need_quads = dict(normal=False, skew=False)
        for k in obs_weights.keys():
            if k in ("x1_bbeat", "x1_phi", "y1_bbeat", "y1_phi", "etax", "nux", "nuy"):
                need_quads["normal"] = True
            elif k in ("x2_re", "x2_im", "y2_re", "y2_im", "etay"):
                need_quads["skew"] = True
            else:
                raise ValueError(k)

        if inj_CO_ps_filepath == "":
            ini_inj_CO = {f"{coord}0": 0.0 for coord in ["x", "xp", "y", "yp", "dp"]}
        else:
            with h5py.File(inj_CO_ps_filepath, "r") as f:
                ini_inj_CO = {
                    f"{coord}0": f[coord][()] for coord in ["x", "xp", "y", "yp", "dp"]
                }

        self.optcor = TbTLinOptCorrector(
            LTE_to_be_corrected,
            fsdb.E_MeV,
            bpm_names["x"],
            bpm_names["y"],
            lumped_quad_names["normal"] if need_quads["normal"] else [],
            lumped_quad_names["skew"] if need_quads["skew"] else [],
            n_turns=self.n_turns,
            actual_inj_CO=ini_inj_CO,
            tbt_ps_offset_wrt_CO=self.tbt_ps_offset_wrt_CO,
            design_LTE=self.design_LTE,
            RM_filepath=self.linopt_RM_filepath
            if linopt_RM_filepath == ""
            else linopt_RM_filepath,
            RM_obs_weights=obs_weights,
            max_beta_beat_thresh_for_coup_cor=max_beta_beat_thresh_for_coup_cor,
            rcond=rcond,
            N_KICKS=fsdb.N_KICKS,
            tempdir_path=None,
        )

    def plot_sv(self):
        plt.figure()
        plt.semilogy(self.optcor._sv / self.optcor._sv[0], "b.-")
        plt.grid(True)
        plt.xlabel("Index")
        plt.ylabel("Normalized Singular Value")
        plt.tight_layout()

    def observe_linopt(self):
        self.optcor.observe()

    def correct_linopt(self, cor_frac=0.7):
        self.optcor.correct(cor_frac=cor_frac)

    def restore_backed_up_quad_setpoints(self):
        self.optcor.restore_backed_up_quad_setpoints()

    def plot_history(self, integ_strength=True, pdf_filepath=""):
        self.plot_sv()

        self.optcor.plot_actual_design_diff(history=True)
        self.optcor.plot_quad_change(integ_strength=integ_strength, history=True)

        if pdf_filepath != "":
            pp = PdfPages(pdf_filepath)
            for fignum in plt.get_fignums():
                pp.savefig(figure=fignum)
            pp.close()

    def save_current_lattice_to_LTEZIP_file(
        self, new_LTEZIP_filepath: Union[Path, str]
    ):
        self.optcor.save_current_lattice_to_LTEZIP_file(new_LTEZIP_filepath)

    def cleanup_tempdirs(self):
        self.design_LTE.remove_tempdir()

        if self.optcor is not None:
            self.optcor.cleanup_tempdirs()


class NSLS2(AbstractFacility):
    def __init__(
        self,
        design_LTE: Lattice,
        lattice_type: str,
        parallel: bool = True,
        output_folder: Union[None, Path, str] = None,
    ):
        super().__init__(design_LTE, parallel=parallel, output_folder=output_folder)

        self.fsdb = ltemanager.NSLS2(self.design_LTE, lattice_type=lattice_type)
        self.LTE = self.fsdb.LTE

        self.config_descr = dict(quad={}, bpm={})
        _descr = self.config_descr["quad"]
        _descr["all_independent"] = "Each normal/skew magnet as independent knobs"
        _descr = self.config_descr["bpm"]
        _descr["all"] = "Include all regular (arc) BPMs"

    def get_BPM_elem_inds_for_orbit_cor(self):
        return self.fsdb.get_regular_BPM_elem_inds()

    def get_corrector_elem_inds_for_orbit_cor(self):
        return self.fsdb.get_slow_corrector_elem_inds()

    def get_BPM_elem_inds_for_linopt_RM(self):
        return self.get_BPM_elem_inds_for_linopt_cor("all")

    def get_quad_elem_inds_for_linopt_RM(self):
        names_d = self.fsdb.get_quad_names(flat_skew_quad_names=True)

        LTE = self.LTE

        return dict(
            normal=LTE.get_elem_inds_from_names(names_d["normal"]),
            skew=LTE.get_elem_inds_from_names(names_d["skew"]),
        )

    def get_BPM_elem_inds_for_linopt_cor(self, config_key: Union[int, str] = "all"):
        assert config_key in self.config_descr["bpm"]

        if config_key == "all":
            return self.fsdb.get_regular_BPM_elem_inds()
        else:
            raise ValueError(f"Invalid `config_key` value: '{config_key}'")

    def get_quad_elem_inds_for_linopt_cor(self, config_key: Union[int, str]):
        assert config_key in self.config_descr["quad"]

        names_d = self.fsdb.get_quad_names(flat_skew_quad_names=False)

        LTE = self.LTE

        lumped_inds = {}
        if config_key == "all_independent":
            lumped_inds["normal"] = [
                [ind] for ind in LTE.get_elem_inds_from_names(names_d["normal"])
            ]
            lumped_inds["skew"] = [
                LTE.get_elem_inds_from_names(skew_names).tolist()
                for skew_names in names_d["skew"]
            ]
        else:
            raise ValueError(f"Invalid `config_key` value: '{config_key}'")

        return lumped_inds

    def generate_linopt_numRM_file(
        self,
        remote_opts: Union[None, Dict] = None,
    ):
        if self.parallel:
            default_remote_opts = dict(
                job_name="RM", partition="normal", ntasks=100, time="30:00", qos="long"
            )

            if remote_opts is not None:
                default_remote_opts.update(remote_opts)

                remote_opts = default_remote_opts

        super().generate_linopt_numRM_file(remote_opts=remote_opts)


class NSLS2U(AbstractFacility):
    def __init__(
        self,
        design_LTE: ltemanager.Lattice,
        lattice_type: str,
        parallel: bool = True,
        output_folder: Union[None, Path, str] = None,
    ):
        super().__init__(design_LTE, parallel=parallel, output_folder=output_folder)

        self.fsdb = ltemanager.NSLS2U(self.design_LTE, lattice_type=lattice_type)
        self.LTE = self.fsdb.LTE

        self.config_descr = dict(quad={}, bpm={})
        _descr = self.config_descr["quad"]
        _descr["all_independent"] = "Each normal/skew magnet as independent knobs"
        _descr = self.config_descr["bpm"]
        _descr["all"] = "Include all regular (arc) BPMs"

    def get_BPM_elem_inds_for_orbit_cor(self):
        return self.fsdb.get_regular_BPM_elem_inds()

    def get_corrector_elem_inds_for_orbit_cor(self):
        return self.fsdb.get_slow_corrector_elem_inds()

    def get_BPM_elem_inds_for_linopt_RM(self):
        return self.get_BPM_elem_inds_for_linopt_cor("all")

    def get_quad_elem_inds_for_linopt_RM(self):
        names_d = self.fsdb.get_em_quad_names(flat_skew_quad_names=True)

        LTE = self.LTE

        return dict(
            normal=LTE.get_elem_inds_from_names(names_d["normal"]),
            skew=LTE.get_elem_inds_from_names(names_d["skew"]),
        )

    def get_BPM_elem_inds_for_linopt_cor(self, config_key: Union[int, str] = "all"):
        assert config_key in self.config_descr["bpm"]

        if config_key == "all":
            return self.fsdb.get_regular_BPM_elem_inds()
        else:
            raise ValueError(f"Invalid `config_key` value: '{config_key}'")

    def get_quad_elem_inds_for_linopt_cor(self, config_key: Union[int, str]):
        assert config_key in self.config_descr["quad"]

        names_d = self.fsdb.get_em_quad_names(flat_skew_quad_names=False)

        LTE = self.LTE

        lumped_inds = {}
        if config_key == "all_independent":
            lumped_inds["normal"] = [
                [ind] for ind in LTE.get_elem_inds_from_names(names_d["normal"])
            ]
            lumped_inds["skew"] = [
                LTE.get_elem_inds_from_names(skew_names).tolist()
                for skew_names in names_d["skew"]
            ]
        else:
            raise ValueError(f"Invalid `config_key` value: '{config_key}'")

        return lumped_inds

    def generate_linopt_numRM_file(
        self,
        remote_opts: Union[None, Dict] = None,
        normal_quad_inds: Union[None, List] = None,
        skew_quad_inds: Union[None, List] = None,
    ):
        if self.parallel:
            default_remote_opts = dict(
                job_name="RM", partition="normal", ntasks=100, time="30:00", qos="long"
            )

            if remote_opts is not None:
                default_remote_opts.update(remote_opts)

                remote_opts = default_remote_opts

        super().generate_linopt_numRM_file(
            remote_opts=remote_opts,
            normal_quad_inds=normal_quad_inds,
            skew_quad_inds=skew_quad_inds,
        )


def _calc_linopt_resp(quad_names, dK1, args_optcor, kwargs_optcor):
    optcor = TbTLinOptCorrector(*args_optcor, **kwargs_optcor)

    resp_list = []

    for name in quad_names:
        resp_list.append(optcor.calc_response(name, dK1))

    return resp_list


def _add_top_legend_adaptive_ncol(legends, fontsize="medium"):

    loc = "lower left"

    fig = plt.gcf()
    norm_fac = dict(h=fig.dpi * fig.get_figheight(), w=fig.dpi * fig.get_figwidth())

    for ncol in np.arange(len(legends))[::-1]:
        ncol += 1
        leg = plt.legend(
            legends, loc=loc, ncol=ncol, fontsize=fontsize, bbox_to_anchor=(-0.05, 1.0)
        )
        fig.canvas.draw()
        win = leg.get_window_extent()
        norm_w = win.width / norm_fac["w"]
        if norm_w <= 0.9:
            leg.remove()
            break
        else:
            leg.remove()

    leg = plt.legend(
        legends, loc=loc, ncol=ncol, fontsize=fontsize, bbox_to_anchor=(-0.05, 1.0)
    )

    return leg
