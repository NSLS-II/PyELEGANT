import os
from pathlib import Path
import pickle
import shlex
from subprocess import PIPE, Popen
import sys
import tempfile
import time
import warnings

import PIL
import h5py
import matplotlib.patches as patches
import matplotlib.pylab as plt
import numpy as np
import scipy.constants as PHYSCONST
from scipy.integrate import romberg
from scipy.interpolate import PchipInterpolator
from scipy.optimize import fmin

from . import (
    __version__,
    elebuilder,
    ltemanager,
    sdds,
    sigproc,
    std_print_enabled,
    twiss,
    util,
)
from .local import run
from .remote import remote


def calc_cmap_xy(
    output_filepath,
    LTE_filepath,
    E_MeV,
    xmin,
    xmax,
    ymin,
    ymax,
    nx,
    ny,
    n_turns=1,
    delta_offset=0.0,
    forward_backward=1,
    use_beamline=None,
    N_KICKS=None,
    transmute_elements=None,
    ele_filepath=None,
    output_file_type=None,
    del_tmp_files=True,
    run_local=False,
    remote_opts=None,
    err_log_check=None,
    nMaxRemoteRetry=2,
):
    """"""

    return _calc_cmap(
        output_filepath,
        LTE_filepath,
        E_MeV,
        "xy",
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        nx=nx,
        ny=ny,
        n_turns=n_turns,
        delta_offset=delta_offset,
        forward_backward=forward_backward,
        use_beamline=use_beamline,
        N_KICKS=N_KICKS,
        transmute_elements=transmute_elements,
        ele_filepath=ele_filepath,
        output_file_type=output_file_type,
        del_tmp_files=del_tmp_files,
        run_local=run_local,
        remote_opts=remote_opts,
        err_log_check=err_log_check,
        nMaxRemoteRetry=nMaxRemoteRetry,
    )


def calc_cmap_px(
    output_filepath,
    LTE_filepath,
    E_MeV,
    delta_min,
    delta_max,
    xmin,
    xmax,
    ndelta,
    nx,
    n_turns=1,
    y_offset=0.0,
    forward_backward=1,
    use_beamline=None,
    N_KICKS=None,
    transmute_elements=None,
    ele_filepath=None,
    output_file_type=None,
    del_tmp_files=True,
    run_local=False,
    remote_opts=None,
    err_log_check=None,
    nMaxRemoteRetry=2,
):
    """"""

    return _calc_cmap(
        output_filepath,
        LTE_filepath,
        E_MeV,
        "px",
        xmin=xmin,
        xmax=xmax,
        delta_min=delta_min,
        delta_max=delta_max,
        nx=nx,
        ndelta=ndelta,
        n_turns=n_turns,
        y_offset=y_offset,
        forward_backward=forward_backward,
        use_beamline=use_beamline,
        N_KICKS=N_KICKS,
        transmute_elements=transmute_elements,
        ele_filepath=ele_filepath,
        output_file_type=output_file_type,
        del_tmp_files=del_tmp_files,
        run_local=run_local,
        remote_opts=remote_opts,
        err_log_check=err_log_check,
        nMaxRemoteRetry=nMaxRemoteRetry,
    )


def _calc_cmap(
    output_filepath,
    LTE_filepath,
    E_MeV,
    plane,
    xmin=-0.1,
    xmax=0.1,
    ymin=1e-6,
    ymax=0.1,
    delta_min=0.0,
    delta_max=0.0,
    nx=20,
    ny=21,
    ndelta=1,
    n_turns=1,
    delta_offset=0.0,
    y_offset=0.0,
    forward_backward=1,
    use_beamline=None,
    N_KICKS=None,
    transmute_elements=None,
    ele_filepath=None,
    output_file_type=None,
    del_tmp_files=True,
    run_local=False,
    remote_opts=None,
    err_log_check=None,
    nMaxRemoteRetry=2,
):
    """"""

    if plane == "xy":
        pass
    elif plane == "px":
        pass
    else:
        raise ValueError('"plane" must be either "xy" or "px".')

    if forward_backward < 1:
        raise ValueError('"forward_backward" must be an integer >= 1.')

    with open(LTE_filepath, "r") as f:
        file_contents = f.read()

    input_dict = dict(
        LTE_filepath=os.path.abspath(LTE_filepath),
        E_MeV=E_MeV,
        n_turns=n_turns,
        forward_backward=forward_backward,
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
    input_dict["cmap_plane"] = plane
    if plane == "xy":
        input_dict["xmin"] = xmin
        input_dict["xmax"] = xmax
        input_dict["ymin"] = ymin
        input_dict["ymax"] = ymax
        input_dict["nx"] = nx
        input_dict["ny"] = ny
        input_dict["delta_offset"] = delta_offset

        plane_specific_chaos_map_block_opts = dict(
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            delta_min=delta_offset,
            delta_max=delta_offset,
            nx=nx,
            ny=ny,
            ndelta=1,
        )
    else:
        input_dict["delta_min"] = delta_min
        input_dict["delta_max"] = delta_max
        input_dict["xmin"] = xmin
        input_dict["xmax"] = xmax
        input_dict["ndelta"] = ndelta
        input_dict["nx"] = nx
        input_dict["y_offset"] = y_offset

        plane_specific_chaos_map_block_opts = dict(
            xmin=xmin,
            xmax=xmax,
            ymin=y_offset,
            ymax=y_offset,
            delta_min=delta_min,
            delta_max=delta_max,
            nx=nx,
            ny=1,
            ndelta=ndelta,
        )

    output_file_type = util.auto_check_output_file_type(
        output_filepath, output_file_type
    )
    input_dict["output_file_type"] = output_file_type

    if output_file_type in ("hdf5", "h5"):
        util.save_input_to_hdf5(output_filepath, input_dict)

    if ele_filepath is None:
        tmp = tempfile.NamedTemporaryFile(
            dir=os.getcwd(), delete=False, prefix=f"tmpCMAP{plane}_", suffix=".ele"
        )
        ele_filepath = os.path.abspath(tmp.name)
        tmp.close()

    ed = elebuilder.EleDesigner(ele_filepath, double_format=".12g")

    # Not all types of elements are capable of back-tracking. So, convert
    # those elements here.
    _def_transmute_elements = dict(
        MULT="EDRIFT", KICKER="EKICKER", HKICK="EHKICK", VKICK="EVKICK"
    )
    if transmute_elements is None:
        transmute_elements = _def_transmute_elements
    else:
        _def_transmute_elements.update(transmute_elements)
        transmute_elements = _def_transmute_elements

    elebuilder.add_transmute_blocks(ed, transmute_elements)

    ed.add_newline()

    ed.add_block(
        "run_setup",
        lattice=LTE_filepath,
        p_central_mev=E_MeV,
        use_beamline=use_beamline,
        semaphore_file="%s.done",
    )

    ed.add_newline()

    ed.add_block("run_control", n_passes=n_turns)

    ed.add_newline()

    elebuilder.add_N_KICKS_alter_elements_blocks(ed, N_KICKS)

    ed.add_block("bunched_beam", n_particles_per_bunch=1)

    ed.add_newline()

    ed.add_block("twiss_output", filename="%s.twi")

    ed.add_newline()

    ed.add_block(
        "chaos_map",
        output="%s.cmap",
        forward_backward=forward_backward,
        verbosity=False,
        **plane_specific_chaos_map_block_opts,
    )

    ed.write()
    # print(ed.actual_output_filepath_list)

    for fp in ed.actual_output_filepath_list:
        if fp.endswith(".cmap"):
            cmap_output_filepath = fp
        elif fp.endswith(".twi"):
            twi_filepath = fp
        elif fp.endswith(".done"):
            done_filepath = fp
        else:
            raise ValueError("This line should not be reached.")

    # Run Elegant
    if run_local:
        run(
            ele_filepath,
            print_cmd=False,
            print_stdout=std_print_enabled["out"],
            print_stderr=std_print_enabled["err"],
        )

        sbatch_info = None
    else:
        if remote_opts is None:
            remote_opts = dict(
                sbatch={"use": True, "wait": True},
                pelegant=True,
                job_name="cmap",
                output="cmap.%J.out",
                error="cmap.%J.err",
                partition="normal",
                ntasks=50,
            )

        sbatch_info = _relaunchable_remote_run(
            remote_opts, ele_filepath, err_log_check, nMaxRemoteRetry
        )

    tmp_filepaths = dict(cmap=cmap_output_filepath)
    output, meta = {}, {}
    for k, v in tmp_filepaths.items():
        try:
            output[k], meta[k] = sdds.sdds2dicts(v)
        except:
            continue

    timestamp_fin = util.get_current_local_time_str()

    if output_file_type in ("hdf5", "h5"):
        util.robust_sdds_hdf5_write(
            output_filepath, [output, meta], nMaxTry=10, sleep=10.0, mode="a"
        )
        f = h5py.File(output_filepath, "a")
        f["timestamp_fin"] = timestamp_fin
        if sbatch_info is not None:
            f["dt_total"] = sbatch_info["total"]
            f["dt_running"] = sbatch_info["running"]
            f["sbatch_nodes"] = sbatch_info["nodes"]
            f["ncores"] = sbatch_info["ncores"]
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

        output_dict = dict(
            data=mod_output,
            meta=mod_meta,
            input=input_dict,
            timestamp_fin=timestamp_fin,
            _version_PyELEGANT=__version__["PyELEGANT"],
            _version_ELEGANT=__version__["ELEGANT"],
        )
        if sbatch_info is not None:
            output_dict["dt_total"] = sbatch_info["total"]
            output_dict["dt_running"] = sbatch_info["running"]
            output_dict["sbatch_nodes"] = sbatch_info["nodes"]
            output_dict["ncores"] = sbatch_info["ncores"]

        util.robust_pgz_file_write(output_filepath, output_dict, nMaxTry=10, sleep=10.0)
    else:
        raise ValueError()

    if del_tmp_files:
        for fp in ed.actual_output_filepath_list + [ele_filepath]:
            if fp.startswith("/dev"):
                continue
            else:
                try:
                    os.remove(fp)
                except:
                    print(f'Failed to delete "{fp}"')

    return output_filepath


def plot_cmap_xy(
    output_filepath,
    title="",
    xlim=None,
    ylim=None,
    scatter=True,
    is_log10=True,
    cmin=-24,
    cmax=-10,
):
    """"""

    _plot_cmap(
        output_filepath,
        title=title,
        xlim=xlim,
        ylim=ylim,
        scatter=scatter,
        is_log10=is_log10,
        cmin=cmin,
        cmax=cmax,
    )


def plot_cmap_px(
    output_filepath,
    title="",
    deltalim=None,
    xlim=None,
    scatter=True,
    is_log10=True,
    cmin=-24,
    cmax=-10,
):
    """"""

    _plot_cmap(
        output_filepath,
        title=title,
        deltalim=deltalim,
        xlim=xlim,
        scatter=scatter,
        is_log10=is_log10,
        cmin=cmin,
        cmax=cmax,
    )


def _plot_cmap(
    output_filepath,
    title="",
    xlim=None,
    ylim=None,
    deltalim=None,
    scatter=True,
    is_log10=True,
    cmin=-24,
    cmax=-10,
):
    """"""

    try:
        d = util.load_pgz_file(output_filepath)
        plane = d["input"]["cmap_plane"]
        g = d["data"]["cmap"]["arrays"]
        if plane == "xy":
            v1 = g["x"]
            v2 = g["y"]
        elif plane == "px":
            v1 = g["delta"]
            v2 = g["x"]
        else:
            raise ValueError(f'Unexpected "cmap_plane" value: {plane}')

        survived = g["Survived"].astype(bool)

        if is_log10:
            chaos = g["Log10dF"]
        else:
            chaos = g["dF"]

        if not scatter:
            g = d["input"]
            if plane == "xy":
                v1max = g["xmax"]
                v1min = g["xmin"]
                v2max = g["ymax"]
                v2min = g["ymin"]
                n1 = g["nx"]
                n2 = g["ny"]
            else:
                v1max = g["delta_max"]
                v1min = g["delta_min"]
                v2max = g["xmax"]
                v2min = g["xmin"]
                n1 = g["ndelta"]
                n2 = g["nx"]

    except:
        f = h5py.File(output_filepath, "r")
        plane = f["input"]["cmap_plane"][()]
        g = f["cmap"]["arrays"]
        if plane == "xy":
            v1 = g["x"][()]
            v2 = g["y"][()]
        elif plane == "px":
            v1 = g["delta"][()]
            v2 = g["x"][()]
        else:
            raise ValueError(f'Unexpected "cmap_plane" value: {plane}')

        survived = g["Survived"][()].astype(bool)

        if is_log10:
            chaos = g["Log10dF"][()]
        else:
            chaos = g["dF"][()]

        if not scatter:
            g = f["input"]
            if plane == "xy":
                v1max = g["xmax"][()]
                v1min = g["xmin"][()]
                v2max = g["ymax"][()]
                v2min = g["ymin"][()]
                n1 = g["nx"][()]
                n2 = g["ny"][()]
            else:
                v1max = g["delta_max"][()]
                v1min = g["delta_min"][()]
                v2max = g["xmax"][()]
                v2min = g["xmin"][()]
                n1 = g["ndelta"][()]
                n2 = g["nx"][()]

        f.close()

    chaos[~survived] = np.nan

    if plane == "xy":
        v1name, v2name = "x", "y"
        v1unitsymb, v2unitsymb = r"\mathrm{mm}", r"\mathrm{mm}"
        v1unitconv, v2unitconv = 1e3, 1e3
        v1lim, v2lim = xlim, ylim
    else:
        v1name, v2name = "\delta", "x"
        v1unitsymb, v2unitsymb = r"\%", r"\mathrm{mm}"
        v1unitconv, v2unitconv = 1e2, 1e3
        v1lim, v2lim = deltalim, xlim

    if is_log10:
        EQ_STR = r"$\rm{log}_{10}(\Delta)$"
        values = chaos
    else:
        EQ_STR = r"$\Delta$"
        values = chaos

    LB = cmin
    UB = cmax

    if scatter:

        font_sz = 18

        plt.figure()
        plt.scatter(
            v1 * v1unitconv,
            v2 * v2unitconv,
            s=14,
            c=values,
            cmap="jet",
            vmin=LB,
            vmax=UB,
        )
        plt.xlabel(rf"${v1name}\, [{v1unitsymb}]$", size=font_sz)
        plt.ylabel(rf"${v2name}\, [{v2unitsymb}]$", size=font_sz)
        if v1lim is not None:
            plt.xlim([v * v1unitconv for v in v1lim])
        if v2lim is not None:
            plt.ylim([v * v2unitconv for v in v2lim])
        if title != "":
            plt.title(title, size=font_sz)
        cb = plt.colorbar()
        try:
            cb.set_ticks(range(LB, UB + 1))
            cb.set_ticklabels([str(i) for i in range(LB, UB + 1)])
        except:
            pass
        cb.ax.set_title(EQ_STR)
        cb.ax.title.set_position((0.5, 1.02))
        plt.tight_layout()

    else:

        font_sz = 18

        v1array = np.linspace(v1min, v1max, n1)
        v2array = np.linspace(v2min, v2max, n2)

        V1, V2 = np.meshgrid(v1array, v2array)
        D = V1 * np.nan

        v1inds = np.argmin(
            np.abs(v1array.reshape((-1, 1)) @ np.ones((1, v1.size)) - v1), axis=0
        )
        v2inds = np.argmin(
            np.abs(v2array.reshape((-1, 1)) @ np.ones((1, v2.size)) - v2), axis=0
        )
        flatinds = np.ravel_multi_index((v1inds, v2inds), V1.T.shape, order="F")
        D_flat = D.flatten()
        D_flat[flatinds] = values
        D = D_flat.reshape(D.shape)

        D = np.ma.masked_array(D, np.isnan(D))

        plt.figure()
        ax = plt.subplot(111)
        plt.pcolor(
            V1 * v1unitconv,
            V2 * v2unitconv,
            D,
            cmap="jet",
            vmin=LB,
            vmax=UB,
            shading="auto",
        )
        plt.xlabel(rf"${v1name}\, [{v1unitsymb}]$", size=font_sz)
        plt.ylabel(rf"${v2name}\, [{v2unitsymb}]$", size=font_sz)
        if v1lim is not None:
            plt.xlim([v * v1unitconv for v in v1lim])
        if v2lim is not None:
            plt.ylim([v * v2unitconv for v in v2lim])
        if title != "":
            plt.title(title, size=font_sz)
        cb = plt.colorbar()
        try:
            cb.set_ticks(range(LB, UB + 1))
            cb.set_ticklabels([str(i) for i in range(LB, UB + 1)])
        except:
            pass
        cb.ax.set_title(EQ_STR)
        cb.ax.title.set_position((0.5, 1.02))
        plt.tight_layout()


def calc_fma_xy(
    output_filepath,
    LTE_filepath,
    E_MeV,
    xmin,
    xmax,
    ymin,
    ymax,
    nx,
    ny,
    n_turns=1024,
    delta_offset=0.0,
    quadratic_spacing=False,
    full_grid_output=False,
    use_beamline=None,
    N_KICKS=None,
    transmute_elements=None,
    ele_filepath=None,
    output_file_type=None,
    del_tmp_files=True,
    run_local=False,
    remote_opts=None,
    err_log_check=None,
    nMaxRemoteRetry=2,
):
    """"""

    return _calc_fma(
        output_filepath,
        LTE_filepath,
        E_MeV,
        "xy",
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        nx=nx,
        ny=ny,
        n_turns=n_turns,
        delta_offset=delta_offset,
        quadratic_spacing=quadratic_spacing,
        full_grid_output=full_grid_output,
        use_beamline=use_beamline,
        N_KICKS=N_KICKS,
        transmute_elements=transmute_elements,
        ele_filepath=ele_filepath,
        output_file_type=output_file_type,
        del_tmp_files=del_tmp_files,
        run_local=run_local,
        remote_opts=remote_opts,
        err_log_check=err_log_check,
        nMaxRemoteRetry=nMaxRemoteRetry,
    )


def calc_fma_px(
    output_filepath,
    LTE_filepath,
    E_MeV,
    delta_min,
    delta_max,
    xmin,
    xmax,
    ndelta,
    nx,
    n_turns=1024,
    y_offset=0.0,
    quadratic_spacing=False,
    full_grid_output=False,
    use_beamline=None,
    N_KICKS=None,
    transmute_elements=None,
    ele_filepath=None,
    output_file_type=None,
    del_tmp_files=True,
    run_local=False,
    remote_opts=None,
    err_log_check=None,
    nMaxRemoteRetry=2,
):
    """"""

    return _calc_fma(
        output_filepath,
        LTE_filepath,
        E_MeV,
        "px",
        xmin=xmin,
        xmax=xmax,
        delta_min=delta_min,
        delta_max=delta_max,
        nx=nx,
        ndelta=ndelta,
        n_turns=n_turns,
        y_offset=y_offset,
        quadratic_spacing=quadratic_spacing,
        full_grid_output=full_grid_output,
        use_beamline=use_beamline,
        N_KICKS=N_KICKS,
        transmute_elements=transmute_elements,
        ele_filepath=ele_filepath,
        output_file_type=output_file_type,
        del_tmp_files=del_tmp_files,
        run_local=run_local,
        remote_opts=remote_opts,
        err_log_check=err_log_check,
        nMaxRemoteRetry=nMaxRemoteRetry,
    )


def _calc_fma(
    output_filepath,
    LTE_filepath,
    E_MeV,
    plane,
    xmin=-0.1,
    xmax=0.1,
    ymin=1e-6,
    ymax=0.1,
    delta_min=0.0,
    delta_max=0.0,
    nx=21,
    ny=21,
    ndelta=1,
    n_turns=1024,
    delta_offset=0.0,
    y_offset=0.0,
    quadratic_spacing=False,
    full_grid_output=False,
    use_beamline=None,
    N_KICKS=None,
    transmute_elements=None,
    ele_filepath=None,
    output_file_type=None,
    del_tmp_files=True,
    run_local=False,
    remote_opts=None,
    err_log_check=None,
    nMaxRemoteRetry=2,
):
    """
    If "err_log_check" is None, then "nMaxRemoteRetry" is irrelevant.
    """

    if plane == "xy":
        pass
    elif plane == "px":
        pass
    else:
        raise ValueError('"plane" must be either "xy" or "px".')

    with open(LTE_filepath, "r") as f:
        file_contents = f.read()

    input_dict = dict(
        LTE_filepath=os.path.abspath(LTE_filepath),
        E_MeV=E_MeV,
        n_turns=n_turns,
        quadratic_spacing=quadratic_spacing,
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
    input_dict["fma_plane"] = plane
    if plane == "xy":
        input_dict["xmin"] = xmin
        input_dict["xmax"] = xmax
        input_dict["ymin"] = ymin
        input_dict["ymax"] = ymax
        input_dict["nx"] = nx
        input_dict["ny"] = ny
        input_dict["delta_offset"] = delta_offset

        plane_specific_freq_map_block_opts = dict(
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            delta_min=delta_offset,
            delta_max=delta_offset,
            nx=nx,
            ny=ny,
            ndelta=1,
        )
    else:
        input_dict["delta_min"] = delta_min
        input_dict["delta_max"] = delta_max
        input_dict["xmin"] = xmin
        input_dict["xmax"] = xmax
        input_dict["ndelta"] = ndelta
        input_dict["nx"] = nx
        input_dict["y_offset"] = y_offset

        plane_specific_freq_map_block_opts = dict(
            xmin=xmin,
            xmax=xmax,
            ymin=y_offset,
            ymax=y_offset,
            delta_min=delta_min,
            delta_max=delta_max,
            nx=nx,
            ny=1,
            ndelta=ndelta,
        )

    output_file_type = util.auto_check_output_file_type(
        output_filepath, output_file_type
    )
    input_dict["output_file_type"] = output_file_type

    if output_file_type in ("hdf5", "h5"):
        util.save_input_to_hdf5(output_filepath, input_dict)

    if ele_filepath is None:
        tmp = tempfile.NamedTemporaryFile(
            dir=os.getcwd(), delete=False, prefix=f"tmpFMA{plane}_", suffix=".ele"
        )
        ele_filepath = os.path.abspath(tmp.name)
        tmp.close()

    ed = elebuilder.EleDesigner(ele_filepath, double_format=".12g")

    elebuilder.add_transmute_blocks(ed, transmute_elements)

    ed.add_newline()

    ed.add_block(
        "run_setup",
        lattice=LTE_filepath,
        p_central_mev=E_MeV,
        use_beamline=use_beamline,
        semaphore_file="%s.done",
    )

    ed.add_newline()

    ed.add_block("run_control", n_passes=n_turns)

    ed.add_newline()

    elebuilder.add_N_KICKS_alter_elements_blocks(ed, N_KICKS)

    ed.add_block("bunched_beam", n_particles_per_bunch=1)

    ed.add_newline()

    ed.add_block(
        "frequency_map",
        output="%s.fma",
        include_changes=True,
        quadratic_spacing=quadratic_spacing,
        full_grid_output=full_grid_output,
        **plane_specific_freq_map_block_opts,
    )

    ed.write()
    # print(ed.actual_output_filepath_list)

    for fp in ed.actual_output_filepath_list:
        if fp.endswith(".fma"):
            fma_output_filepath = fp
        elif fp.endswith(".done"):
            done_filepath = fp
        else:
            raise ValueError("This line should not be reached.")

    # Run Elegant
    if run_local:
        run(
            ele_filepath,
            print_cmd=False,
            print_stdout=std_print_enabled["out"],
            print_stderr=std_print_enabled["err"],
        )

        sbatch_info = None
    else:
        if remote_opts is None:
            remote_opts = dict(
                sbatch={"use": True, "wait": True},
                pelegant=True,
                job_name="fma",
                output="fma.%J.out",
                error="fma.%J.err",
                partition="normal",
                ntasks=50,
            )

        sbatch_info = _relaunchable_remote_run(
            remote_opts, ele_filepath, err_log_check, nMaxRemoteRetry
        )

    tmp_filepaths = dict(fma=fma_output_filepath)
    output, meta = {}, {}
    for k, v in tmp_filepaths.items():
        try:
            output[k], meta[k] = sdds.sdds2dicts(v, str_format="%25.16e")
        except:
            continue

    timestamp_fin = util.get_current_local_time_str()

    if output_file_type in ("hdf5", "h5"):
        util.robust_sdds_hdf5_write(
            output_filepath, [output, meta], nMaxTry=10, sleep=10.0, mode="a"
        )
        f = h5py.File(output_filepath, "a")
        f["timestamp_fin"] = timestamp_fin
        if sbatch_info is not None:
            f["dt_total"] = sbatch_info["total"]
            f["dt_running"] = sbatch_info["running"]
            f["sbatch_nodes"] = sbatch_info["nodes"]
            f["ncores"] = sbatch_info["ncores"]
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

        output_dict = dict(
            data=mod_output,
            meta=mod_meta,
            input=input_dict,
            timestamp_fin=timestamp_fin,
            _version_PyELEGANT=__version__["PyELEGANT"],
            _version_ELEGANT=__version__["ELEGANT"],
        )
        if sbatch_info is not None:
            output_dict["dt_total"] = sbatch_info["total"]
            output_dict["dt_running"] = sbatch_info["running"]
            output_dict["sbatch_nodes"] = sbatch_info["nodes"]
            output_dict["ncores"] = sbatch_info["ncores"]

        util.robust_pgz_file_write(output_filepath, output_dict, nMaxTry=10, sleep=10.0)
    else:
        raise ValueError()

    if del_tmp_files:
        for fp in ed.actual_output_filepath_list + [ele_filepath]:
            if fp.startswith("/dev"):
                continue
            else:
                try:
                    os.remove(fp)
                except:
                    print(f'Failed to delete "{fp}"')

    return output_filepath


def plot_fma_xy(
    output_filepath,
    title="",
    xlim=None,
    ylim=None,
    scatter=True,
    is_diffusion=True,
    cmin=-10,
    cmax=-2,
    correct_diffusion_err=False,
):
    """"""

    _plot_fma(
        output_filepath,
        title=title,
        xlim=xlim,
        ylim=ylim,
        scatter=scatter,
        is_diffusion=is_diffusion,
        cmin=cmin,
        cmax=cmax,
        correct_diffusion_err=correct_diffusion_err,
    )


def plot_fma_px(
    output_filepath,
    title="",
    deltalim=None,
    xlim=None,
    scatter=True,
    is_diffusion=True,
    cmin=-10,
    cmax=-2,
    correct_diffusion_err=False,
):
    """"""

    _plot_fma(
        output_filepath,
        title=title,
        deltalim=deltalim,
        xlim=xlim,
        scatter=scatter,
        is_diffusion=is_diffusion,
        cmin=cmin,
        cmax=cmax,
        correct_diffusion_err=correct_diffusion_err,
    )


def _plot_fma(
    output_filepath,
    title="",
    xlim=None,
    ylim=None,
    deltalim=None,
    scatter=True,
    is_diffusion=True,
    cmin=-10,
    cmax=-2,
    correct_diffusion_err=False,
):
    """"""

    try:
        d = util.load_pgz_file(output_filepath)
        plane = d["input"]["fma_plane"]
        g = d["data"]["fma"]["arrays"]
        if plane == "xy":
            v1 = g["x"]
            v2 = g["y"]
        elif plane == "px":
            v1 = g["delta"]
            v2 = g["x"]
        else:
            raise ValueError(f'Unexpected "fma_plane" value: {plane}')

        if is_diffusion:
            if not correct_diffusion_err:
                diffusion = g["diffusion"]
            else:
                # Define "mux1" and "mux2" be the tunes determined by &frequency_map,
                # which could be correct or incorrect by 1-mux. The following covers
                # all the potential cases. By picking the minimum dnux, you'll get
                # the correct dnux. The same is true for the vertical plane.
                dnu_sq = {}
                for _plane in ["x", "y"]:
                    mu1 = g[f"nu{_plane}"]
                    mu2p = mu1 + g[f"dnu{_plane}"]
                    mu2m = mu1 - g[f"dnu{_plane}"]
                    dnu_sq[_plane] = np.min(
                        np.vstack(
                            (
                                (mu2p - mu1) ** 2,
                                (mu2m - mu1) ** 2,
                                (mu2p - (1 - mu1)) ** 2,
                                (mu2m - (1 - mu1)) ** 2,
                                ((1 - mu2p) - mu1) ** 2,
                                ((1 - mu2m) - mu1) ** 2,
                                ((1 - mu2p) - (1 - mu1)) ** 2,
                                ((1 - mu2m) - (1 - mu1)) ** 2,
                            )
                        ),
                        axis=0,
                    )
                diffusion = np.log10(dnu_sq["x"] + dnu_sq["y"])
        else:
            diffusionRate = g["diffusionRate"]

        if not scatter:
            g = d["input"]
            quadratic_spacing = g["quadratic_spacing"]
            if plane == "xy":
                v1max = g["xmax"]
                v1min = g["xmin"]
                v2max = g["ymax"]
                v2min = g["ymin"]
                n1 = g["nx"]
                n2 = g["ny"]
            else:
                v1max = g["delta_max"]
                v1min = g["delta_min"]
                v2max = g["xmax"]
                v2min = g["xmin"]
                n1 = g["ndelta"]
                n2 = g["nx"]

    except:
        f = h5py.File(output_filepath, "r")
        plane = f["input"]["fma_plane"][()]
        g = f["fma"]["arrays"]
        if plane == "xy":
            v1 = g["x"][()]
            v2 = g["y"][()]
        elif plane == "px":
            v1 = g["delta"][()]
            v2 = g["x"][()]
        else:
            raise ValueError(f'Unexpected "fma_plane" value: {plane}')

        if is_diffusion:
            if not correct_diffusion_err:
                diffusion = g["diffusion"][()]
            else:
                # Define "mux1" and "mux2" be the tunes determined by &frequency_map,
                # which could be correct or incorrect by 1-mux. The following covers
                # all the potential cases. By picking the minimum dnux, you'll get
                # the correct dnux. The same is true for the vertical plane.
                dnu_sq = {}
                for _plane in ["x", "y"]:
                    mu1 = g[f"nu{_plane}"][()]
                    mu2p = mu1 + g[f"dnu{_plane}"][()]
                    mu2m = mu1 - g[f"dnu{_plane}"][()]
                    dnu_sq[_plane] = np.min(
                        np.vstack(
                            (
                                (mu2p - mu1) ** 2,
                                (mu2m - mu1) ** 2,
                                (mu2p - (1 - mu1)) ** 2,
                                (mu2m - (1 - mu1)) ** 2,
                                ((1 - mu2p) - mu1) ** 2,
                                ((1 - mu2m) - mu1) ** 2,
                                ((1 - mu2p) - (1 - mu1)) ** 2,
                                ((1 - mu2m) - (1 - mu1)) ** 2,
                            )
                        ),
                        axis=0,
                    )
                diffusion = np.log10(dnu_sq["x"] + dnu_sq["y"])
        else:
            diffusionRate = g["diffusionRate"][()]

        if not scatter:
            g = f["input"]
            quadratic_spacing = g["quadratic_spacing"][()]
            if plane == "xy":
                v1max = g["xmax"][()]
                v1min = g["xmin"][()]
                v2max = g["ymax"][()]
                v2min = g["ymin"][()]
                n1 = g["nx"][()]
                n2 = g["ny"][()]
            else:
                v1max = g["delta_max"][()]
                v1min = g["delta_min"][()]
                v2max = g["xmax"][()]
                v2min = g["xmin"][()]
                n1 = g["ndelta"][()]
                n2 = g["nx"][()]

        f.close()

    if plane == "xy":
        v1name, v2name = "x", "y"
        v1unitsymb, v2unitsymb = r"\mathrm{mm}", r"\mathrm{mm}"
        v1unitconv, v2unitconv = 1e3, 1e3
        v1lim, v2lim = xlim, ylim
    else:
        v1name, v2name = "\delta", "x"
        v1unitsymb, v2unitsymb = r"\%", r"\mathrm{mm}"
        v1unitconv, v2unitconv = 1e2, 1e3
        v1lim, v2lim = deltalim, xlim

    if is_diffusion:
        DIFFUSION_EQ_STR = r"$\rm{log}_{10}(\Delta{\nu_x}^2+\Delta{\nu_y}^2)$"
        values = diffusion
    else:
        DIFFUSION_EQ_STR = (
            r"$\rm{log}_{10}(\frac{\sqrt{\Delta{\nu_x}^2+\Delta{\nu_y}^2}}{N})$"
        )
        values = diffusionRate

    LB = cmin
    UB = cmax

    if scatter:

        font_sz = 18

        plt.figure()
        plt.scatter(
            v1 * v1unitconv,
            v2 * v2unitconv,
            s=14,
            c=values,
            cmap="jet",
            vmin=LB,
            vmax=UB,
        )
        plt.xlabel(rf"${v1name}\, [{v1unitsymb}]$", size=font_sz)
        plt.ylabel(rf"${v2name}\, [{v2unitsymb}]$", size=font_sz)
        if v1lim is not None:
            plt.xlim([v * v1unitconv for v in v1lim])
        if v2lim is not None:
            plt.ylim([v * v2unitconv for v in v2lim])
        if title != "":
            plt.title(title, size=font_sz)
        cb = plt.colorbar()
        try:
            cb.set_ticks(range(LB, UB + 1))
            cb.set_ticklabels([str(i) for i in range(LB, UB + 1)])
        except:
            pass
        cb.ax.set_title(DIFFUSION_EQ_STR)
        cb.ax.title.set_position((0.5, 1.02))
        plt.tight_layout()

    else:

        font_sz = 18

        if not quadratic_spacing:
            v1array = np.linspace(v1min, v1max, n1)
            v2array = np.linspace(v2min, v2max, n2)
        else:

            dv1 = v1max - max([0.0, v1min])
            v1array = np.sqrt(np.linspace((dv1**2) / n1, dv1**2, n1))
            # v1array - np.unique(v1)
            # plt.figure()
            # plt.plot(np.unique(v1), 'b-', v1array, 'r-')

            dv2 = v2max - max([0.0, v2min])
            v2array = v2min + np.sqrt(np.linspace((dv2**2) / n2, dv2**2, n2))
            # v2array - np.unique(v2)
            # plt.figure()
            # plt.plot(np.unique(v2), 'b-', v2array, 'r-')

        V1, V2 = np.meshgrid(v1array, v2array)
        D = V1 * np.nan

        v1inds = np.argmin(
            np.abs(v1array.reshape((-1, 1)) @ np.ones((1, v1.size)) - v1), axis=0
        )
        v2inds = np.argmin(
            np.abs(v2array.reshape((-1, 1)) @ np.ones((1, v2.size)) - v2), axis=0
        )
        flatinds = np.ravel_multi_index((v1inds, v2inds), V1.T.shape, order="F")
        D_flat = D.flatten()
        D_flat[flatinds] = values
        D = D_flat.reshape(D.shape)

        D = np.ma.masked_array(D, np.isnan(D))

        plt.figure()
        ax = plt.subplot(111)
        plt.pcolor(
            V1 * v1unitconv,
            V2 * v2unitconv,
            D,
            cmap="jet",
            vmin=LB,
            vmax=UB,
            shading="auto",
        )
        plt.xlabel(rf"${v1name}\, [{v1unitsymb}]$", size=font_sz)
        plt.ylabel(rf"${v2name}\, [{v2unitsymb}]$", size=font_sz)
        if v1lim is not None:
            plt.xlim([v * v1unitconv for v in v1lim])
        if v2lim is not None:
            plt.ylim([v * v2unitconv for v in v2lim])
        if title != "":
            plt.title(title, size=font_sz)
        cb = plt.colorbar()
        try:
            cb.set_ticks(range(LB, UB + 1))
            cb.set_ticklabels([str(i) for i in range(LB, UB + 1)])
        except:
            pass
        cb.ax.set_title(DIFFUSION_EQ_STR)
        cb.ax.title.set_position((0.5, 1.02))
        plt.tight_layout()


def calc_find_aper_nlines(
    output_filepath,
    LTE_filepath,
    E_MeV,
    xmax=0.1,
    ymax=0.1,
    ini_ndiv=21,
    n_lines=11,
    neg_y_search=False,
    n_turns=1024,
    use_beamline=None,
    N_KICKS=None,
    transmute_elements=None,
    ele_filepath=None,
    output_file_type=None,
    del_tmp_files=True,
    run_local=False,
    remote_opts=None,
    err_log_check=None,
    nMaxRemoteRetry=2,
):
    """"""

    assert n_lines >= 3

    with open(LTE_filepath, "r") as f:
        file_contents = f.read()

    input_dict = dict(
        LTE_filepath=os.path.abspath(LTE_filepath),
        E_MeV=E_MeV,
        n_turns=n_turns,
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
    input_dict["xmax"] = xmax
    input_dict["ymax"] = ymax
    input_dict["ini_ndiv"] = ini_ndiv
    input_dict["n_lines"] = n_lines
    input_dict["neg_y_search"] = neg_y_search

    output_file_type = util.auto_check_output_file_type(
        output_filepath, output_file_type
    )
    input_dict["output_file_type"] = output_file_type

    if output_file_type in ("hdf5", "h5"):
        util.save_input_to_hdf5(output_filepath, input_dict)

    if ele_filepath is None:
        tmp = tempfile.NamedTemporaryFile(
            dir=os.getcwd(), delete=False, prefix=f"tmpFindAper_", suffix=".ele"
        )
        ele_filepath = os.path.abspath(tmp.name)
        tmp.close()

    ed = elebuilder.EleDesigner(ele_filepath, double_format=".12g")

    elebuilder.add_transmute_blocks(ed, transmute_elements)

    ed.add_newline()

    ed.add_block(
        "run_setup",
        lattice=LTE_filepath,
        p_central_mev=E_MeV,
        use_beamline=use_beamline,
        semaphore_file="%s.done",
    )

    ed.add_newline()

    ed.add_block("run_control", n_passes=n_turns)

    ed.add_newline()

    elebuilder.add_N_KICKS_alter_elements_blocks(ed, N_KICKS)

    ed.add_newline()

    ed.add_block(
        "find_aperture",
        output="%s.aper",
        mode="n-lines",
        xmax=xmax,
        ymax=ymax,
        nx=ini_ndiv,
        n_lines=n_lines,
        full_plane=neg_y_search,
        offset_by_orbit=True,  # recommended according to the manual
    )

    ed.write()
    # print(ed.actual_output_filepath_list)

    for fp in ed.actual_output_filepath_list:
        if fp.endswith(".aper"):
            aper_output_filepath = fp
        elif fp.endswith(".done"):
            done_filepath = fp
        else:
            raise ValueError("This line should not be reached.")

    # Run Elegant
    if run_local:
        run(
            ele_filepath,
            print_cmd=False,
            print_stdout=std_print_enabled["out"],
            print_stderr=std_print_enabled["err"],
        )

        sbatch_info = None
    else:
        if remote_opts is None:
            remote_opts = dict(
                sbatch={"use": True, "wait": True},
                pelegant=True,
                job_name="findaper",
                output="findaper.%J.out",
                error="findaper.%J.err",
                partition="normal",
                ntasks=np.min([50, n_lines]),
            )

        sbatch_info = _relaunchable_remote_run(
            remote_opts, ele_filepath, err_log_check, nMaxRemoteRetry
        )

    tmp_filepaths = dict(aper=aper_output_filepath)
    output, meta = {}, {}
    for k, v in tmp_filepaths.items():
        try:
            output[k], meta[k] = sdds.sdds2dicts(v)
        except:
            continue

    timestamp_fin = util.get_current_local_time_str()

    if output_file_type in ("hdf5", "h5"):
        util.robust_sdds_hdf5_write(
            output_filepath, [output, meta], nMaxTry=10, sleep=10.0, mode="a"
        )
        f = h5py.File(output_filepath, "a")
        f["timestamp_fin"] = timestamp_fin
        if sbatch_info is not None:
            f["dt_total"] = sbatch_info["total"]
            f["dt_running"] = sbatch_info["running"]
            f["sbatch_nodes"] = sbatch_info["nodes"]
            f["ncores"] = sbatch_info["ncores"]
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

        output_dict = dict(
            data=mod_output,
            meta=mod_meta,
            input=input_dict,
            timestamp_fin=timestamp_fin,
            _version_PyELEGANT=__version__["PyELEGANT"],
            _version_ELEGANT=__version__["ELEGANT"],
        )
        if sbatch_info is not None:
            output_dict["dt_total"] = sbatch_info["total"]
            output_dict["dt_running"] = sbatch_info["running"]
            output_dict["sbatch_nodes"] = sbatch_info["nodes"]
            output_dict["ncores"] = sbatch_info["ncores"]

        util.robust_pgz_file_write(output_filepath, output_dict, nMaxTry=10, sleep=10.0)
    else:
        raise ValueError()

    if del_tmp_files:
        for fp in ed.actual_output_filepath_list + [ele_filepath]:
            if fp.startswith("/dev"):
                continue
            else:
                try:
                    os.remove(fp)
                except:
                    print(f'Failed to delete "{fp}"')

    return output_filepath


def plot_find_aper_nlines(output_filepath, title="", xlim=None, ylim=None):
    """"""

    ret = {}  # variable to be returned

    try:
        d = util.load_pgz_file(output_filepath)
        g = d["data"]["aper"]["arrays"]
        x = g["x"]
        y = g["y"]
        g = d["data"]["aper"]["scalars"]
        area = g["Area"]
        neg_y_search = d["input"]["neg_y_search"]

    except:
        f = h5py.File(output_filepath, "r")
        g = f["aper"]["arrays"]
        x = g["x"][()]
        y = g["y"][()]
        g = f["aper"]["scalars"]
        area = g["Area"][()]
        neg_y_search = f["input"]["neg_y_search"][()]
        f.close()

    font_sz = 18

    plt.figure()
    plt.plot(x * 1e3, y * 1e3, "b.-")
    plt.xlabel(r"$x\, [\mathrm{mm}]$", size=font_sz)
    plt.ylabel(r"$y\, [\mathrm{mm}]$", size=font_sz)
    if xlim is not None:
        plt.xlim([v * 1e3 for v in xlim])
    if ylim is not None:
        plt.ylim([v * 1e3 for v in ylim])
    ret["x_min"] = np.min(x)
    ret["x_max"] = np.max(x)
    ret["y_min"] = np.min(y)
    ret["y_max"] = np.max(y)
    ret["neg_y_search"] = neg_y_search
    ret["area"] = area
    xy_bd_title = (
        rf'$(x_{{\mathrm{{min}}}}={ret["x_min"]*1e3:.1f}, '
        rf'x_{{\mathrm{{max}}}}={ret["x_max"]*1e3:.1f}, '
    )
    if neg_y_search:
        xy_bd_title += rf'y_{{\mathrm{{min}}}}={ret["y_min"]*1e3:.1f}, '
    xy_bd_title += rf'y_{{\mathrm{{max}}}}={ret["y_max"]*1e3:.1f})\, [\mathrm{{mm}}]$'
    area_title = rf"$\mathrm{{Area}}={area*1e6:.1f}\, [\mathrm{{mm}}^2]$"
    if title != "":
        plt.title("\n".join([title, xy_bd_title, area_title]), size=font_sz)
    else:
        plt.title("\n".join([xy_bd_title, area_title]), size=font_sz)
    plt.tight_layout()

    return ret


def calc_rf_bucket_heights(E_GeV, alphac, U0_eV, h, rf_volts):
    """"""

    m_e_eV = (
        PHYSCONST.physical_constants["electron mass energy equivalent in MeV"][0] * 1e6
    )

    # See Section 3.1.4.6 on p.212 of Chao & Tigner, "Handbook
    # of Accelerator Physics and Engineering" for analytical
    # formula of RF bucket height, which is "A_s" in Eq. (32),
    # which is equal to (epsilon_max/E_0) [fraction] in Eq. (33).
    #
    # Note that the slip factor (eta) is approximately equal
    # to momentum compaction in the case of NSLS-II.
    gamma = 1.0 + E_GeV * 1e9 / m_e_eV
    gamma_t = 1.0 / np.sqrt(alphac)
    slip_fac = 1.0 / (gamma_t**2) - 1.0 / (
        gamma**2
    )  # approx. equal to "mom_compac"
    try:
        len(rf_volts)

        # When "rf_volts" is a vector

        rf_bucket_heights_percents = np.zeros_like(rf_volts)

        valid = np.array(rf_volts) > U0_eV
        q = rf_volts[valid] / U0_eV  # overvoltage factor
        F_q = 2.0 * (np.sqrt(q**2 - 1) - np.arccos(1.0 / q))
        rf_bucket_heights_percents[valid] = 1e2 * np.sqrt(
            U0_eV / (np.pi * np.abs(slip_fac) * h * (E_GeV * 1e9)) * F_q
        )

    except TypeError:  # When "rf_volts" is a scalar
        if rf_volts > U0_eV:
            q = rf_volts / U0_eV  # overvoltage factor
            F_q = 2.0 * (np.sqrt(q**2 - 1) - np.arccos(1.0 / q))
            rf_bucket_heights_percents = 1e2 * np.sqrt(
                U0_eV / (np.pi * np.abs(slip_fac) * h * (E_GeV * 1e9)) * F_q
            )
        else:
            rf_bucket_heights_percents = 0.0

    return rf_bucket_heights_percents


def calc_ring_rf_params(
    harmonic_number,
    circumf,
    U0_eV,
    rf_bucket_percent=None,
    rf_volt=None,
    overvoltage_factor=None,
    E_GeV=None,
    alphac=None,
):
    """"""

    if rf_bucket_percent is not None:

        try:
            assert rf_bucket_percent > 0.0
        except:
            print('"rf_bucket_percent" must be larger than 0.0')
            raise

        if overvoltage_factor is not None:
            print(
                (
                    'WARNING: Since "rf_bucket_percent" is specified, '
                    '"overvoltage_factor" will be ignored.'
                )
            )
        if rf_volt is not None:
            print(
                (
                    'WARNING: Since "rf_bucket_percent" is specified, '
                    '"rf_volt" will be ignored.'
                )
            )

        if E_GeV is None:
            raise ValueError(
                (
                    'You must also specify "E_GeV" '
                    'when "rf_bucket_percent" is specified.'
                )
            )
        if alphac is None:
            raise ValueError(
                (
                    'You must also specify "alphac" '
                    'when "rf_bucket_percent" is specified.'
                )
            )

        def _goal(rf_volt, target_rf_bucket_percent, E_GeV, alphac, U0_eV, h):
            rf_percent = calc_rf_bucket_heights(E_GeV, alphac, U0_eV, h, rf_volt)
            return (rf_percent - target_rf_bucket_percent) ** 2

        ini_rf_volt = U0_eV * 2.0

        opt = fmin(
            _goal,
            ini_rf_volt,
            args=(rf_bucket_percent, E_GeV, alphac, U0_eV, harmonic_number),
            xtol=1e-6,
            ftol=1e-6,
            disp=0,
            retall=0,
        )
        rf_volt = opt[0]

    elif overvoltage_factor is not None:
        if rf_volt is not None:
            print(
                (
                    'WARNING: Since "overvoltage_factor" is specified, '
                    '"rf_volt" will be ignored.'
                )
            )

        try:
            assert overvoltage_factor > 1.0
        except:
            print('"overvoltage_factor" must be larger than 1.0')
            raise

        rf_volt = overvoltage_factor * U0_eV

    elif rf_volt is not None:
        try:
            assert rf_volt > U0_eV
        except:
            print(f'"rf_volt" must be larger than U0_eV ({U0_eV:.6g})')
            raise

    else:
        raise ValueError(
            (
                "You must specifiy one of the following:\n"
                '"rf_bucket_percent" (> 0.0), "overvoltage_factor" (> 1.0)., '
                'and "rf_volt" (> energy loss per turn)'
            )
        )

    # freq_Hz_rpn_expr = notation.convert_infix_to_rpn(
    # f'c_mks / {circumf:.6g} * {harmonic_number:d}')
    freq_Hz = PHYSCONST.c / circumf * harmonic_number

    # phase_deg_rpn_expr = notation.convert_infix_to_rpn(
    # f'180.0 - dasin({U0_eV:.9g} / {rf_volt:.9g})')
    ## ^ Built-in RPN function "dasin" == np.rad2deg(np.arcsin)
    phase_deg = 180 - np.rad2deg(np.arcsin(U0_eV / rf_volt))

    rf_params = dict(rf_volt=rf_volt, freq_Hz=freq_Hz, phase_deg=phase_deg)

    rf_params["overvoltage_factor"] = rf_volt / U0_eV

    return rf_params


def _get_nonexistent_elem_beamline_name(elem_defs, beamline_defs, base_name):
    """"""

    all_elem_names = [elem_name for elem_name, _, _ in elem_defs]
    all_beamline_names = [beamline_name for beamline_name, _ in beamline_defs]
    all_existing_names = all_elem_names + all_beamline_names

    new_elem_name = base_name
    while new_elem_name in all_existing_names:
        new_elem_name += "0"

    return new_elem_name


def calc_mom_aper(
    output_filepath,
    LTE_filepath,
    E_MeV,
    x_initial=1e-5,
    y_initial=1e-5,
    delta_negative_start=-1e-3,
    delta_negative_limit=-5e-2,
    delta_positive_start=+1e-3,
    delta_positive_limit=+5e-2,
    init_delta_step_size=5e-3,
    s_start=0.0,
    s_end=None,
    include_name_pattern=None,
    steps_back=1,
    splits=2,
    split_step_divisor=10,
    verbosity=1,
    forbid_resonance_crossing=False,
    soft_failure=False,
    process_elements=2147483647,
    rf_cavity_on=True,
    radiation_on=True,
    harmonic_number=None,
    rf_bucket_percent=None,
    overvoltage_factor=None,
    rf_volt=None,
    n_turns=1024,
    use_beamline=None,
    N_KICKS=None,
    transmute_elements=None,
    ele_filepath=None,
    output_file_type=None,
    del_tmp_files=True,
    run_local=False,
    remote_opts=None,
    err_log_check=None,
    nMaxRemoteRetry=2,
):
    """"""

    if rf_cavity_on:
        if harmonic_number is None:
            raise ValueError(
                'When "rf_cavity_on" is True, you must specifiy "harmonic_number".'
            )

        if rf_bucket_percent is not None:
            if overvoltage_factor is not None:
                print(
                    (
                        'WARNING: Since "rf_bucket_percent" is specified, '
                        '"overvoltage_factor" will be ignored.'
                    )
                )
            if rf_volt is not None:
                print(
                    (
                        'WARNING: Since "rf_bucket_percent" is specified, '
                        '"rf_volt" will be ignored.'
                    )
                )
        elif overvoltage_factor is not None:
            if rf_volt is not None:
                print(
                    (
                        'WARNING: Since "overvoltage_factor" is specified, '
                        '"rf_volt" will be ignored.'
                    )
                )
        elif rf_volt is not None:
            pass
        else:
            raise ValueError(
                (
                    'When "rf_cavity_on" is True, you must also specifiy one of the following:\n'
                    '"rf_bucket_percent" (> 0.0), "overvoltage_factor" (> 1.0)., '
                    'and "rf_volt" (> energy loss per turn)'
                )
            )

    file_contents = Path(LTE_filepath).read_text()

    input_dict = dict(
        LTE_filepath=os.path.abspath(LTE_filepath),
        E_MeV=E_MeV,
        rf_cavity_on=rf_cavity_on,
        radiation_on=radiation_on,
        harmonic_number=harmonic_number,
        rf_bucket_percent=rf_bucket_percent,
        overvoltage_factor=overvoltage_factor,
        rf_volt=rf_volt,
        n_turns=n_turns,
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
    input_dict["x_initial"] = x_initial
    input_dict["y_initial"] = y_initial
    input_dict["delta_negative_start"] = delta_negative_start
    input_dict["delta_negative_limit"] = delta_negative_limit
    input_dict["delta_positive_start"] = delta_positive_start
    input_dict["delta_positive_limit"] = delta_positive_limit
    input_dict["init_delta_step_size"] = init_delta_step_size
    input_dict["s_start"] = s_start
    input_dict["s_end"] = s_end
    input_dict["include_name_pattern"] = include_name_pattern
    input_dict["steps_back"] = steps_back
    input_dict["splits"] = splits
    input_dict["split_step_divisor"] = split_step_divisor
    input_dict["verbosity"] = verbosity
    input_dict["forbid_resonance_crossing"] = forbid_resonance_crossing
    input_dict["soft_failure"] = soft_failure
    input_dict["process_elements"] = process_elements

    output_file_type = util.auto_check_output_file_type(
        output_filepath, output_file_type
    )
    input_dict["output_file_type"] = output_file_type

    if output_file_type in ("hdf5", "h5"):
        util.save_input_to_hdf5(output_filepath, input_dict)

    tmp = tempfile.NamedTemporaryFile(prefix=f"tmpTwi_", suffix=".pgz")
    twi_pgz_filepstr = str(Path(tmp.name).resolve())
    tmp.close()
    #
    twiss.calc_ring_twiss(
        twi_pgz_filepstr,
        LTE_filepath,
        E_MeV,
        use_beamline=use_beamline,
        parameters="%s.param",
        radiation_integrals=True,
        run_local=True,
    )
    # ^ "%s.twi" & "%s.param" are needed only for the purpose of showing magnet
    #   profiles in the plotting function.

    tmp_files_to_be_deleted = [twi_pgz_filepstr]

    if ele_filepath is None:
        tmp = tempfile.NamedTemporaryFile(
            dir=os.getcwd(), delete=False, prefix=f"tmpMomAper_", suffix=".ele"
        )
        ele_filepath = os.path.abspath(tmp.name)
        tmp.close()

        tmp_files_to_be_deleted.append(ele_filepath)

    if transmute_elements is None:
        transmute_elements = dict(
            SBEN="CSBEND", RBEN="CSBEND", QUAD="KQUAD", SEXT="KSEXT", OCTU="KOCT"
        )
    #
    for elem_type in ["RFCA", "SREFFECTS"]:
        if elem_type in transmute_elements:
            raise ValueError(
                (
                    f'User-specified transmuation of the element type "{elem_type}" '
                    f"for calc_mom_aper() is not allowed"
                )
            )

    twi = util.load_pgz_file(twi_pgz_filepstr)

    if rf_cavity_on or radiation_on:
        LTE = ltemanager.Lattice(
            LTE_filepath=LTE_filepath, used_beamline_name=use_beamline
        )
        d_LTE = LTE.get_used_beamline_element_defs(used_beamline_name=use_beamline)

        new_beamline_name = _get_nonexistent_elem_beamline_name(
            d_LTE["elem_defs"], d_LTE["beamline_defs"], "RINGRF"
        )

        new_beamline_def = [(d_LTE["used_beamline_name"], 1)]

        tmp = tempfile.NamedTemporaryFile(
            prefix=f"tmpRF_", suffix=".lte", dir=Path.cwd().resolve()
        )
        # ^ CRITICAL: must create this temp LTE file in cwd. If you create this
        #   LTE in /tmp, this file cannot be accessible from other nodes
        #   when Pelegant is used.
        temp_RF_LTE_filepstr = str(Path(tmp.name).resolve())
        tmp.close()
    else:
        temp_RF_LTE_filepstr = ""

    if radiation_on:
        # First find existing SREFFECTS elements and convert to MARK
        if "SREFFECTS" in [v[1] for v in d_LTE["elem_defs"]]:
            for i, (elem_name, elem_type, prop_str) in d_LTE["elem_defs"]:
                if elem_type == "SREFFECTS":
                    d_LTE["elem_defs"][i] = (elem_name, "MARK", "")

        # Add SREFFECTS element to the list of element definitions
        new_sreffects_elem_name = _get_nonexistent_elem_beamline_name(
            d_LTE["elem_defs"], d_LTE["beamline_defs"], "SR"
        )
        d_LTE["elem_defs"].append(
            (new_sreffects_elem_name, "SREFFECTS", "QEXCITATION=0")
        )

        new_beamline_def.append((new_sreffects_elem_name, 1))
    else:
        transmute_elements["SREFFECTS"] = "MARK"

    if rf_cavity_on:
        # First find existing RFCA elements and convert to DRIF
        if "RFCA" in [v[1] for v in d_LTE["elem_defs"]]:
            for i, (elem_name, elem_type, prop_str) in enumerate(d_LTE["elem_defs"]):
                if elem_type == "RFCA":
                    L = LTE.parse_elem_properties(prop_str).get("L", 0.0)
                    d_LTE["elem_defs"][i] = (
                        elem_name,
                        "DRIF",
                        "" if L == 0.0 else f"L={L:.9g}",
                    )

        circumf = twi["data"]["twi"]["arrays"]["s"][-1]
        U0_eV = twi["data"]["twi"]["scalars"]["U0"] * 1e6
        alphac = twi["data"]["twi"]["scalars"]["alphac"]
        if rf_bucket_percent is not None:
            rf_params = calc_ring_rf_params(
                harmonic_number,
                circumf,
                U0_eV,
                rf_bucket_percent=rf_bucket_percent,
                E_GeV=E_MeV / 1e3,
                alphac=alphac,
            )
        elif overvoltage_factor is not None:
            rf_params = calc_ring_rf_params(
                harmonic_number, circumf, U0_eV, overvoltage_factor=overvoltage_factor
            )
        elif rf_volt is not None:
            rf_params = calc_ring_rf_params(
                harmonic_number, circumf, U0_eV, rf_volt=rf_volt
            )
        else:
            raise RuntimeError("This line should not be reachable.")

        # Add derived rf parameters into "input_dict"
        rf_params["bucket_percent"] = calc_rf_bucket_heights(
            E_MeV / 1e3, alphac, U0_eV, harmonic_number, rf_params["rf_volt"]
        )
        input_dict["derived_rf_params"] = rf_params

        # Add RFCA element to the list of element definitions
        new_rfca_elem_name = _get_nonexistent_elem_beamline_name(
            d_LTE["elem_defs"], d_LTE["beamline_defs"], "CAV"
        )
        _rf_cav_def = (
            f'VOLT={rf_params["rf_volt"]:.12g}, '
            f'PHASE={rf_params["phase_deg"]:.12g}, '
            f'FREQ={rf_params["freq_Hz"]:.12g}'
        )
        print(f"Adding RFCA element: {_rf_cav_def}")
        d_LTE["elem_defs"].append((new_rfca_elem_name, "RFCA", _rf_cav_def))

        new_beamline_def.append((new_rfca_elem_name, 1))
    else:
        transmute_elements["RFCA"] = "MARK"

    if rf_cavity_on or radiation_on:
        # Create a new beamline with RFCA and/or SREFFECTS elements added

        d_LTE["beamline_defs"].append((new_beamline_name, new_beamline_def))

        LTE.write_LTE(
            temp_RF_LTE_filepstr,
            new_beamline_name,
            d_LTE["elem_defs"],
            d_LTE["beamline_defs"],
        )

        input_dict["modified_lattice_file_contents"] = Path(
            temp_RF_LTE_filepstr
        ).read_text()

        tmp_files_to_be_deleted.append(temp_RF_LTE_filepstr)

        mom_aper_LTE_filepstr = temp_RF_LTE_filepstr
        mom_aper_beamline_name = new_beamline_name
    else:
        mom_aper_LTE_filepstr = LTE_filepath
        mom_aper_beamline_name = use_beamline

    ed = elebuilder.EleDesigner(ele_filepath, double_format=".12g")

    elebuilder.add_transmute_blocks(ed, transmute_elements)

    ed.add_newline()

    ed.add_block(
        "run_setup",
        lattice=mom_aper_LTE_filepstr,
        p_central_mev=E_MeV,
        use_beamline=mom_aper_beamline_name,
        semaphore_file="%s.done",
    )

    ed.add_newline()

    elebuilder.add_N_KICKS_alter_elements_blocks(ed, N_KICKS)

    ed.add_newline()

    ed.add_block(
        "twiss_output",
        concat_order=2,
        radiation_integrals=True,
        output_at_each_step=True,
    )

    ed.add_newline()

    ed.add_block("run_control", n_passes=n_turns)

    ed.add_newline()

    _block_opts = dict(
        output="%s.mmap",
        x_initial=x_initial,
        y_initial=y_initial,
        delta_negative_start=delta_negative_start,
        delta_negative_limit=delta_negative_limit,
        delta_positive_start=delta_positive_start,
        delta_positive_limit=delta_positive_limit,
        delta_step_size=init_delta_step_size,
        s_start=s_start,
        s_end=(s_end if s_end is not None else sys.float_info.max),
        steps_back=steps_back,
        splits=splits,
        split_step_divisor=split_step_divisor,
        fiducialize=True,
        verbosity=verbosity,
        forbid_resonance_crossing=forbid_resonance_crossing,
        soft_failure=soft_failure,
        process_elements=process_elements,
    )
    if include_name_pattern is not None:
        _block_opts["include_name_pattern"] = include_name_pattern
    ed.add_block("momentum_aperture", **_block_opts)

    ed.write()
    # print(ed.actual_output_filepath_list)

    for fp in ed.actual_output_filepath_list:
        if fp.endswith(".mmap"):
            mmap_output_filepath = fp
        elif fp.endswith(".done"):
            done_filepath = fp
        else:
            raise ValueError("This line should not be reached.")

    # Run Elegant
    if run_local:
        run(
            ele_filepath,
            print_cmd=False,
            print_stdout=std_print_enabled["out"],
            print_stderr=std_print_enabled["err"],
        )

        sbatch_info = None
    else:
        if remote_opts is None:
            remote_opts = dict(
                sbatch={"use": True, "wait": True},
                pelegant=True,
                job_name="momaper",
                output="momaper.%J.out",
                error="momaper.%J.err",
                partition="normal",
                ntasks=50,
            )

        sbatch_info = _relaunchable_remote_run(
            remote_opts, ele_filepath, err_log_check, nMaxRemoteRetry
        )

    tmp_filepaths = dict(mmap=mmap_output_filepath)
    output, meta = {}, {}
    for k, v in tmp_filepaths.items():
        try:
            output[k], meta[k] = sdds.sdds2dicts(v)
        except:
            continue

    timestamp_fin = util.get_current_local_time_str()

    if output_file_type in ("hdf5", "h5"):
        for _k in ["twi", "param"]:
            output[_k] = {}
            meta[_k] = {}
            for _k2 in ["scalars", "arrays"]:
                if _k2 in twi["data"][_k]:
                    output[_k][_k2] = twi["data"][_k][_k2]
                    meta[_k][_k2] = twi["meta"][_k][_k2]

        util.robust_sdds_hdf5_write(
            output_filepath, [output, meta], nMaxTry=10, sleep=10.0, mode="a"
        )
        f = h5py.File(output_filepath, "a")
        f["timestamp_fin"] = timestamp_fin
        if sbatch_info is not None:
            f["dt_total"] = sbatch_info["total"]
            f["dt_running"] = sbatch_info["running"]
            f["sbatch_nodes"] = sbatch_info["nodes"]
            f["ncores"] = sbatch_info["ncores"]
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

        for _k in ["twi", "param"]:
            mod_output[_k] = {}
            mod_meta[_k] = {}
            for _k2 in ["scalars", "arrays"]:
                if _k2 in twi["data"][_k]:
                    mod_output[_k][_k2] = twi["data"][_k][_k2]
                    mod_meta[_k][_k2] = twi["meta"][_k][_k2]

        output_dict = dict(
            data=mod_output,
            meta=mod_meta,
            input=input_dict,
            timestamp_fin=timestamp_fin,
            _version_PyELEGANT=__version__["PyELEGANT"],
            _version_ELEGANT=__version__["ELEGANT"],
        )
        if sbatch_info is not None:
            output_dict["dt_total"] = sbatch_info["total"]
            output_dict["dt_running"] = sbatch_info["running"]
            output_dict["sbatch_nodes"] = sbatch_info["nodes"]
            output_dict["ncores"] = sbatch_info["ncores"]

        util.robust_pgz_file_write(output_filepath, output_dict, nMaxTry=10, sleep=10.0)
    else:
        raise ValueError()

    if del_tmp_files:

        for fp in ed.actual_output_filepath_list + tmp_files_to_be_deleted:
            if fp.startswith("/dev"):
                continue
            else:
                try:
                    os.remove(fp)
                except:
                    print(f'Failed to delete "{fp}"')

    return output_filepath


def plot_mom_aper(
    output_filepath,
    title="",
    add_mmap_info_to_title=True,
    deltalim=None,
    slim=None,
    s_margin_m=0.5,
    show_mag_prof=True,
):
    """"""

    ret = {}  # variable to be returned

    try:
        d = util.load_pgz_file(output_filepath)
        g = d["data"]["mmap"]["arrays"]
        deltaNegative = g["deltaNegative"]
        deltaPositive = g["deltaPositive"]
        s = g["s"]
        try:
            twi_arrays = d["data"]["twi"]["arrays"]
            param_arrays = d["data"]["param"]["arrays"]
        except:
            show_mag_prof = False
    except:
        f = h5py.File(output_filepath, "r")
        g = f["mmap"]["arrays"]
        deltaNegative = g["deltaNegative"][()]
        deltaPositive = g["deltaPositive"][()]
        s = g["s"][()]
        try:
            g = f["twi"]["arrays"]
            twi_arrays = {}
            for k in list(g):
                twi_arrays[k] = g[k][()]
            g = f["param"]["arrays"]
            param_arrays = {}
            for k in list(g):
                param_arrays[k] = g[k][()]
        except:
            show_mag_prof = False

        f.close()

    if slim is None:
        slim = [np.min(s), np.max(s)]

    slim = np.array(slim)

    _vis = _get_visible_inds(s, slim, s_margin_m=s_margin_m)

    font_sz = 18

    sort_inds = np.argsort(s)
    s = s[sort_inds]
    deltaNegative = deltaNegative[sort_inds]
    deltaPositive = deltaPositive[sort_inds]

    delta_percent = {
        "+": dict(min=np.min(deltaPositive) * 1e2, max=np.max(deltaPositive) * 1e2),
        "-": dict(min=np.min(deltaNegative) * 1e2, max=np.max(deltaNegative) * 1e2),
    }
    ret["delta_percent"] = delta_percent

    plt.figure()
    if show_mag_prof:
        nrows = 6
        ax1 = plt.subplot2grid((nrows, 1), (0, 0), rowspan=nrows - 1)
        ax2 = plt.subplot2grid((nrows, 1), (nrows - 1, 0), rowspan=1, sharex=ax1)
    else:
        ax1 = plt.gca()
    plt.sca(ax1)
    plt.plot(s[_vis], deltaNegative[_vis] * 1e2, "b-")
    plt.plot(s[_vis], deltaPositive[_vis] * 1e2, "r-")
    plt.axhline(0, color="k")
    plt.ylabel(r"$\delta_{+}, \delta_{-}\, [\%]$", size=font_sz)
    plt.xlim(slim)
    if deltalim is not None:
        plt.ylim([v * 1e2 for v in deltalim])
    mmap_info_title = r"${},\, {}\, [\%]$".format(
        rf'{delta_percent["+"]["min"]:.2f} < \delta_{{+}} < {delta_percent["+"]["max"]:.2f}',
        rf'{delta_percent["-"]["min"]:.2f} < \delta_{{-}} < {delta_percent["-"]["max"]:.2f}',
    )
    if title != "":
        if add_mmap_info_to_title:
            plt.title("\n".join([title, mmap_info_title]), size=font_sz)
        else:
            plt.title(title, size=font_sz)
    else:
        if add_mmap_info_to_title:
            plt.title(mmap_info_title, size=font_sz)
    if show_mag_prof:
        add_magnet_profiles(ax2, twi_arrays, param_arrays, slim, s_margin_m=s_margin_m)
        plt.setp(ax1.get_xticklabels(), visible=False)
        ax2.set_xlabel(r"$s\, [\mathrm{m}]$", size=font_sz)
        ax1.spines["bottom"].set_visible(False)
        ax2.spines["top"].set_visible(False)
    else:
        ax1.set_xlabel(r"$s\, [\mathrm{m}]$", size=font_sz)
    plt.tight_layout()
    if show_mag_prof:
        plt.subplots_adjust(hspace=0.0, wspace=0.0)

    return ret


def calc_Touschek_lifetime(
    output_filepath,
    LTE_filepath,
    E_MeV,
    mmap_filepath,
    charge_C,
    emit_ratio,
    RFvolt,
    RFharm,
    max_mom_aper_percent=None,
    ignoreMismatch=True,
    use_beamline=None,
    output_file_type=None,
    del_tmp_files=True,
    print_cmd=False,
):
    """
    For this function to work properly, the following must be for the full ring,
    NOT for one period of the ring:
        mmap_filepath
        use_beamline
    """

    with open(LTE_filepath, "r") as f:
        file_contents = f.read()

    nElectrons = int(np.round(charge_C / PHYSCONST.e))

    input_dict = dict(
        LTE_filepath=os.path.abspath(LTE_filepath),
        E_MeV=E_MeV,
        mmap_filepath=mmap_filepath,
        charge_C=charge_C,
        nElectrons=nElectrons,
        emit_ratio=emit_ratio,
        RFvolt=RFvolt,
        RFharm=RFharm,
        max_mom_aper_percent=max_mom_aper_percent,
        ignoreMismatch=ignoreMismatch,
        use_beamline=use_beamline,
        del_tmp_files=del_tmp_files,
        lattice_file_contents=file_contents,
        timestamp_ini=util.get_current_local_time_str(),
    )

    output_file_type = util.auto_check_output_file_type(
        output_filepath, output_file_type
    )
    input_dict["output_file_type"] = output_file_type

    if output_file_type in ("hdf5", "h5"):
        util.save_input_to_hdf5(output_filepath, input_dict)

    tmp = tempfile.NamedTemporaryFile(
        dir=os.getcwd(), delete=False, prefix=f"tmpTau_", suffix=".pgz"
    )
    twi_pgz_filepath = os.path.abspath(tmp.name)
    life_filepath = ".".join(twi_pgz_filepath.split(".")[:-1] + ["life"])
    tmp.close()

    tmp_filepaths = twiss.calc_ring_twiss(
        twi_pgz_filepath,
        LTE_filepath,
        E_MeV,
        use_beamline=use_beamline,
        parameters="%s.param",
        radiation_integrals=True,
        run_local=True,
        del_tmp_files=False,
    )
    # ^ "%s.param" is needed only for the purpose of showing magnet profiles
    #   in the plotting function.
    tmp_filepaths["twi_pgz"] = twi_pgz_filepath
    # print(tmp_filepaths)

    # twi = util.load_pgz_file(twi_pgz_filepath)
    # try:
    # os.remove(twi_pgz_filepath)
    # except:
    # pass

    try:
        sdds.sdds2dicts(mmap_filepath)
        # "mmap_filepath" is a valid SDDS file
        mmap_sdds_filepath = mmap_filepath
    except:
        # "mmap_filepath" is NOT a valid SDDS file. Try to see if the file is
        # gzipped pickle file (.pgz) or HDF5 file generated from an SDDS file.
        # Then convert it back to a valid SDDS file.
        try:
            # First try ".pgz"
            d = util.load_pgz_file(mmap_filepath)
            mmap_d = d["data"]["mmap"]
        except:
            # Then try HDF5
            d = util.load_sdds_hdf5_file(mmap_filepath)
            mmap_d = d[0]["mmap"]
        mmap_sdds_filepath = ".".join(twi_pgz_filepath.split(".")[:-1] + ["mmap"])
        sdds.dicts2sdds(
            mmap_sdds_filepath,
            params=mmap_d["scalars"],
            columns=mmap_d["arrays"],
            outputMode="binary",
        )
        tmp_filepaths["mmap"] = mmap_sdds_filepath

    cmd_str = (
        f'touschekLifetime {life_filepath} -twiss={tmp_filepaths["twi"]} '
        f"-aperture={mmap_sdds_filepath} -particles={nElectrons:d} "
        f"-coupling={emit_ratio:.9g} "
        f"-RF=Voltage={RFvolt/1e6:.9g},harmonic={RFharm:d},limit "
    )

    if max_mom_aper_percent is not None:
        cmd_str += f"-deltaLimit={max_mom_aper_percent:.9g} "

    if ignoreMismatch:
        cmd_str += "-ignoreMismatch"

    cmd_list = shlex.split(cmd_str)
    if print_cmd:
        print("\n$ " + " ".join(cmd_list) + "\n")

    p = Popen(cmd_list, stdout=PIPE, stderr=PIPE, encoding="utf-8")
    out, err = p.communicate()
    if std_print_enabled["out"]:
        print(out)
    if std_print_enabled["err"] and err:
        print("ERROR:")
        print(err)

    output_tmp_filepaths = dict(
        life=life_filepath, twi=tmp_filepaths["twi"], param=tmp_filepaths["param"]
    )
    output, meta = {}, {}
    for k, v in output_tmp_filepaths.items():
        try:
            output[k], meta[k] = sdds.sdds2dicts(v)
        except:
            continue

    timestamp_fin = util.get_current_local_time_str()

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

    tmp_filepaths.update(output_tmp_filepaths)
    if del_tmp_files:
        for fp in tmp_filepaths.values():
            if fp.startswith("/dev"):
                continue
            else:
                try:
                    os.remove(fp)
                except:
                    print(f'Failed to delete "{fp}"')

    return output_filepath


def generate_Touschek_F_interpolator():
    """"""

    ndiv = 121
    xarray = np.logspace(-3, 2, ndiv)
    Farray = np.full(xarray.shape, np.nan)
    for _i, x in enumerate(xarray):
        func = (
            lambda u, x=x: (1.0 / u - np.log(1.0 / u) / 2.0 - 1.0) * np.exp(-x / u)
            if u > 0.0
            else 0.0
        )
        Farray[_i] = romberg(
            func,
            0.0,
            1.0,
            args=(),
            tol=1e-16,  # tol=1e-10,
            rtol=1e-10,
            show=False,
            divmax=20,
            vec_func=False,
        )
    xasymp_array = np.logspace(-3, -2, 11)
    #
    plt.figure()
    plt.loglog(xarray, Farray, "b-")
    Eulers_number = 0.5772
    plt.loglog(xasymp_array, np.log(Eulers_number / xasymp_array) - 3.0 / 2.0, "r-")
    plt.grid(True)
    plt.tight_layout()

    valid = Farray > 1e-10

    interp = PchipInterpolator(xarray[valid], Farray[valid], extrapolate=False)
    #
    plt.figure()
    _plot_func = plt.loglog
    # _plot_func = plt.plot
    _plot_func(xarray, Farray, "b-")
    Eulers_number = 0.5772
    _plot_func(xasymp_array, np.log(Eulers_number / xasymp_array) - 3.0 / 2.0, "r-")
    _plot_func(xarray, interp(xarray), "k:")
    if True:
        ext_xarray = np.logspace(-6, 3, 1000)
        F_interp = get_Touschek_F_interpolator()
        _plot_func(ext_xarray, F_interp(ext_xarray), "m:")
    plt.grid(True)
    plt.tight_layout()

    d = dict(
        pchip_interp=interp, xmin=np.min(xarray[valid]), xmax=np.max(xarray[valid])
    )
    with open("Touschek_F_interpolator.pkl", "wb") as f:
        pickle.dump(d, f)


def get_Touschek_F_interpolator():
    """"""

    with open(Path(__file__).parent.joinpath("Touschek_F_interpolator.pkl"), "rb") as f:
        d = pickle.load(f)

    def interpolator(xarray):

        xarray = np.array(xarray)

        Farray = d["pchip_interp"](xarray)

        Farray[xarray > d["xmax"]] = 0.0

        Eulers_number = 0.5772
        asymp_range = xarray < d["xmin"]
        Farray[asymp_range] = np.log(Eulers_number / xarray[asymp_range]) - 3 / 2

        return Farray

    return interpolator


def plot_Touschek_lifetime(
    output_filepath,
    title="",
    add_tau_info_to_title=True,
    slim=None,
    s_margin_m=0.5,
    show_mag_prof=True,
):
    """"""

    try:
        d = util.load_pgz_file(output_filepath)
        tau_hr = d["data"]["life"]["scalars"]["tLifetime"]
        g = d["data"]["life"]["arrays"]
        FN = g["FN"]
        FP = g["FP"]
        s = g["s"]
        twi_arrays = d["data"]["twi"]["arrays"]
        param_arrays = d["data"]["param"]["arrays"]

    except:
        f = h5py.File(output_filepath, "r")
        tau_hr = f["life"]["scalars"]["tLifetime"][()]
        g = f["life"]["arrays"]
        FN = g["FN"][()]
        FP = g["FP"][()]
        s = g["s"][()]
        g = f["twi"]["arrays"]
        twi_arrays = {}
        for k in list(g):
            twi_arrays[k] = g[k][()]
        g = f["param"]["arrays"]
        param_arrays = {}
        for k in list(g):
            param_arrays[k] = g[k][()]
        f.close()

    if slim is None:
        slim = [np.min(s), np.max(s)]

    slim = np.array(slim)

    _vis = _get_visible_inds(s, slim, s_margin_m=s_margin_m)

    font_sz = 18

    plt.figure()
    if show_mag_prof:
        nrows = 6
        ax1 = plt.subplot2grid((nrows, 1), (0, 0), rowspan=nrows - 1)
        ax2 = plt.subplot2grid((nrows, 1), (nrows - 1, 0), rowspan=1, sharex=ax1)
    else:
        ax1 = plt.gca()
    plt.sca(ax1)
    plt.plot(s[_vis], FN[_vis], "b-", label=r"$\delta < 0$")
    plt.plot(s[_vis], FP[_vis], "r-", label=r"$\delta > 0$")
    plt.axhline(0, color="k")
    plt.ylabel(r"$f_{+}, f_{-}\, [\mathrm{s}^{-1}]$", size=font_sz)
    plt.xlim(slim)
    tau_info_title = rf"$\tau_{{\mathrm{{Touschek}}}} = {tau_hr:.3g}\, [\mathrm{{hr}}]$"
    if title != "":
        if add_tau_info_to_title:
            plt.title("\n".join([title, tau_info_title]), size=font_sz)
        else:
            plt.title(title, size=font_sz)
    else:
        if add_tau_info_to_title:
            plt.title(tau_info_title, size=font_sz)
    if show_mag_prof:
        add_magnet_profiles(ax2, twi_arrays, param_arrays, slim, s_margin_m=s_margin_m)
        plt.setp(ax1.get_xticklabels(), visible=False)
        ax2.set_xlabel(r"$s\, [\mathrm{m}]$", size=font_sz)
        ax1.spines["bottom"].set_visible(False)
        ax2.spines["top"].set_visible(False)
    else:
        ax1.set_xlabel(r"$s\, [\mathrm{m}]$", size=font_sz)
    plt.legend(loc="upper right", ncol=2, prop=dict(size=font_sz - 4))
    plt.tight_layout()
    if show_mag_prof:
        plt.subplots_adjust(hspace=0.0, wspace=0.0)


def _get_visible_inds(all_s_array, slim, s_margin_m=0.1):
    """
    s_margin_m [m]
    """

    shifted_slim = slim - slim[0]

    _visible = np.logical_and(
        all_s_array - slim[0] >= shifted_slim[0] - s_margin_m,
        all_s_array - slim[0] <= shifted_slim[1] + s_margin_m,
    )

    return _visible


def _get_param_val(param_name, parameters_dict, elem_name, elem_occur):
    """"""

    parameters = parameters_dict

    matched_elem_names = parameters["ElementName"] == elem_name
    matched_elem_occurs = parameters["ElementOccurence"] == elem_occur
    m = np.logical_and(matched_elem_names, matched_elem_occurs)
    if np.sum(m) == 0:
        m = np.where(matched_elem_names)[0]
        u_elem_occurs_int = np.unique(parameters["ElementOccurence"][m])
        if np.all(u_elem_occurs_int > elem_occur):
            elem_occur = np.min(u_elem_occurs_int)
        elif np.all(u_elem_occurs_int < elem_occur):
            elem_occur = np.max(u_elem_occurs_int)
        else:
            elem_occur = np.min(u_elem_occurs_int[u_elem_occurs_int >= elem_occur])
        matched_elem_occurs = parameters["ElementOccurence"] == elem_occur
        m = np.logical_and(matched_elem_names, matched_elem_occurs)
    m = np.logical_and(m, parameters["ElementParameter"] == param_name)
    assert np.sum(m) == 1

    return parameters["ParameterValue"][m][0]


def add_magnet_profiles(ax, twi_arrays, parameters_arrays, slim, s_margin_m=0.1):
    """"""

    prof_center_y = 0.0
    quad_height = 0.5
    sext_height = quad_height * 1.5
    oct_height = quad_height * 1.75
    bend_half_height = quad_height / 3.0

    ax.set_yticks([])

    ax.set_xlim(slim)
    max_height = max([quad_height, sext_height, oct_height, bend_half_height])
    ax.set_ylim(np.array([-max_height, +max_height]))

    twi_ar = twi_arrays
    parameters = parameters_arrays

    s0_m = 0.0  # shift in s-coord.
    prev_s = 0.0 - s0_m
    assert (
        len(twi_ar["s"])
        == len(twi_ar["ElementType"])
        == len(twi_ar["ElementName"])
        == len(twi_ar["ElementOccurence"])
    )
    for ei, (s, elem_type, elem_name, elem_occur) in enumerate(
        zip(
            twi_ar["s"],
            twi_ar["ElementType"],
            twi_ar["ElementName"],
            twi_ar["ElementOccurence"],
        )
    ):

        cur_s = s - s0_m

        if (s < slim[0] - s_margin_m) or (s > slim[1] + s_margin_m):
            prev_s = cur_s
            continue

        elem_type = elem_type.upper()

        if elem_type in ("QUAD", "KQUAD"):

            K1 = _get_param_val("K1", parameters, elem_name, elem_occur)
            c = "r"
            if K1 >= 0.0:  # Focusing Quad
                bottom, top = 0.0, quad_height
            else:  # Defocusing Quad
                bottom, top = -quad_height, 0.0

            # Shift vertically
            bottom += prof_center_y
            top += prof_center_y

            width = cur_s - prev_s
            height = top - bottom

            p = patches.Rectangle((prev_s, bottom), width, height, fill=True, color=c)
            ax.add_patch(p)

        elif elem_type in ("SEXT", "KSEXT"):

            K2 = _get_param_val("K2", parameters, elem_name, elem_occur)
            c = "b"
            if K2 >= 0.0:  # Focusing Sext
                bottom, mid_h, top = 0.0, sext_height / 2, sext_height
            else:  # Defocusing Sext
                bottom, mid_h, top = -sext_height, -sext_height / 2, 0.0

            # Shift vertically
            bottom += prof_center_y
            mid_h += prof_center_y
            top += prof_center_y

            mid_s = (prev_s + cur_s) / 2

            if K2 >= 0.0:  # Focusing Sext
                xy = np.array(
                    [
                        [prev_s, bottom],
                        [prev_s, mid_h],
                        [mid_s, top],
                        [cur_s, mid_h],
                        [cur_s, bottom],
                    ]
                )
            else:
                xy = np.array(
                    [
                        [prev_s, top],
                        [prev_s, mid_h],
                        [mid_s, bottom],
                        [cur_s, mid_h],
                        [cur_s, top],
                    ]
                )
            p = patches.Polygon(xy, closed=True, fill=True, color=c)
            ax.add_patch(p)

        elif elem_type in ("OCTU", "KOCT"):

            K3 = _get_param_val("K3", parameters, elem_name, elem_occur)
            c = "g"
            if K3 >= 0.0:  # Focusing Octupole
                bottom, mid_h, top = 0.0, oct_height / 2, oct_height
            else:  # Defocusing Octupole
                bottom, mid_h, top = -oct_height, -oct_height / 2, 0.0

            # Shift vertically
            bottom += prof_center_y
            mid_h += prof_center_y
            top += prof_center_y

            mid_s = (prev_s + cur_s) / 2

            if K3 >= 0.0:  # Focusing Octupole
                xy = np.array(
                    [
                        [prev_s, bottom],
                        [prev_s, mid_h],
                        [mid_s, top],
                        [cur_s, mid_h],
                        [cur_s, bottom],
                    ]
                )
            else:
                xy = np.array(
                    [
                        [prev_s, top],
                        [prev_s, mid_h],
                        [mid_s, bottom],
                        [cur_s, mid_h],
                        [cur_s, top],
                    ]
                )
            p = patches.Polygon(xy, closed=True, fill=True, color=c)
            ax.add_patch(p)

        elif elem_type in ("RBEND", "SBEND", "SBEN", "CSBEND"):
            bottom, top = -bend_half_height, bend_half_height

            # Shift vertically
            bottom += prof_center_y
            top += prof_center_y

            width = cur_s - prev_s
            height = top - bottom

            p = patches.Rectangle((prev_s, bottom), width, height, fill=True, color="k")
            ax.add_patch(p)
        else:
            ax.plot([prev_s, cur_s], np.array([0.0, 0.0]) + prof_center_y, "k-")

        prev_s = cur_s


def calc_chrom_twiss(
    output_filepath,
    LTE_filepath,
    E_MeV,
    delta_min,
    delta_max,
    ndelta,
    use_beamline=None,
    transmute_elements=None,
    ele_filepath=None,
    output_file_type=None,
    del_tmp_files=True,
    print_cmd=False,
    run_local=True,
    remote_opts=None,
):
    """"""

    with open(LTE_filepath, "r") as f:
        file_contents = f.read()

    input_dict = dict(
        LTE_filepath=os.path.abspath(LTE_filepath),
        E_MeV=E_MeV,
        delta_min=delta_min,
        delta_max=delta_max,
        ndelta=ndelta,
        use_beamline=use_beamline,
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
            dir=os.getcwd(), delete=False, prefix=f"tmpChromTwiss_", suffix=".ele"
        )
        ele_filepath = os.path.abspath(tmp.name)
        tmp.close()

    ed = elebuilder.EleDesigner(ele_filepath, double_format=".12g")

    if transmute_elements is None:
        transmute_elements = dict(RFCA="MARK", SREFFECTS="MARK")

    elebuilder.add_transmute_blocks(ed, transmute_elements)

    ed.add_newline()

    ed.add_block(
        "run_setup",
        lattice=LTE_filepath,
        p_central_mev=E_MeV,
        use_beamline=use_beamline,
    )

    ed.add_newline()

    ed.add_block("run_control")

    ed.add_newline()

    temp_malign_elem_name = "ELEGANT_CHROM_TWISS_MAL"
    temp_malign_elem_def = f"{temp_malign_elem_name}: MALIGN"

    ed.add_block(
        "insert_elements",
        name="*",
        exclude="*",
        add_at_start=True,
        element_def=temp_malign_elem_def,
    )

    ed.add_newline()

    ed.add_block(
        "alter_elements",
        name=temp_malign_elem_name,
        type="MALIGN",
        item="DP",
        value="<delta>",
    )

    ed.add_newline()

    ed.add_block("twiss_output", filename="%s.twi")

    ed.write()
    # print(ed.actual_output_filepath_list)

    for fp in ed.actual_output_filepath_list:
        if fp.endswith(".twi"):
            twi_filepath = fp
        else:
            raise ValueError("This line should not be reached.")

    delta_array = np.linspace(delta_min, delta_max, ndelta)

    # Run Elegant
    if run_local:
        nuxs = np.full(ndelta, np.nan)
        nuys = np.full(ndelta, np.nan)
        for i, delta in enumerate(delta_array):
            run(
                ele_filepath,
                print_cmd=print_cmd,
                macros=dict(delta=f"{delta:.12g}"),
                print_stdout=std_print_enabled["out"],
                print_stderr=std_print_enabled["err"],
            )

            output, _ = sdds.sdds2dicts(twi_filepath)

            nuxs[i] = output["params"]["nux"]
            nuys[i] = output["params"]["nuy"]
    else:
        raise NotImplementedError

    survived = ~np.isnan(nuxs) & ~np.isnan(nuys)
    undefined_tunes = ~survived

    timestamp_fin = util.get_current_local_time_str()

    _save_chrom_data(
        output_filepath,
        output_file_type,
        delta_array,
        nuxs,
        nuys,
        survived,
        undefined_tunes,
        timestamp_fin,
        input_dict,
    )

    if del_tmp_files:
        for fp in ed.actual_output_filepath_list + [ele_filepath]:
            if fp.startswith("/dev"):
                continue
            else:
                try:
                    os.remove(fp)
                except:
                    print(f'Failed to delete "{fp}"')

    return output_filepath


def _deprecated_msg_use_sddsnaff(use_sddsnaff, courant_snyder, method, return_fft_spec):

    if use_sddsnaff is not None:
        warnings.warn(
            "'use_sddsnaff' is deprecated. Use method='sddsnaff' for use_sddsnaff=True.",
            DeprecationWarning,
        )

    if courant_snyder is not None:
        warnings.warn(
            "'courant_snyder' is deprecated. Use method='DFT_Courant_Snyder' if courant_snyder=True; If False, use method='DFT_phase_space'.",
            DeprecationWarning,
        )

    orig_method = method

    if use_sddsnaff is None:
        if courant_snyder is None:
            pass
        else:
            # Assume "use_sddsnaff=False"
            if courant_snyder:
                method = "DFT_Courant_Snyder"
            else:
                method = "DFT_phase_space"
    else:
        if courant_snyder is None:
            # Assume "courant_snyder=True"
            courant_snyder = True

        if use_sddsnaff:
            if courant_snyder:
                warnings.warn('"courant_snyder=True" will be ignored.')
            if return_fft_spec:
                warnings.warn('"return_fft_spec=True" will be ignored.')

            method = "sddsnaff"
        else:
            if courant_snyder:
                method = "DFT_Courant_Snyder"
            else:
                method = "DFT_phase_space"

    if orig_method != method:
        warnings.warn(
            f"Overwriting specified option 'method' from '{orig_method}' to '{method}'"
        )

    use_sddsnaff = courant_snyder = None

    return use_sddsnaff, courant_snyder, method


def calc_chrom_track(
    output_filepath,
    LTE_filepath,
    E_MeV,
    delta_min,
    delta_max,
    ndelta,
    use_sddsnaff=None,
    courant_snyder=None,
    method="sddsnaff",
    return_fft_spec=True,
    save_tbt=True,
    n_turns=256,
    x0_offset=1e-5,
    y0_offset=1e-5,
    use_beamline=None,
    N_KICKS=None,
    transmute_elements=None,
    ele_filepath=None,
    output_file_type=None,
    del_tmp_files=True,
    print_cmd=False,
    run_local=True,
    remote_opts=None,
    err_log_check=None,
    nMaxRemoteRetry=3,
):
    """
    If "err_log_check" is None, then "nMaxRemoteRetry" is irrelevant.
    """

    assert method in ("sddsnaff", "DFT_Courant_Snyder", "DFT_phase_space")

    use_sddsnaff, courant_snyder, method = _deprecated_msg_use_sddsnaff(
        use_sddsnaff, courant_snyder, method, return_fft_spec
    )

    LTE_file_pathobj = Path(LTE_filepath)

    file_contents = LTE_file_pathobj.read_text()

    input_dict = dict(
        LTE_filepath=str(LTE_file_pathobj.resolve()),
        E_MeV=E_MeV,
        delta_min=delta_min,
        delta_max=delta_max,
        ndelta=ndelta,
        use_sddsnaff=use_sddsnaff,
        courant_snyder=courant_snyder,
        method=method,
        return_fft_spec=return_fft_spec,
        save_tbt=save_tbt,
        n_turns=n_turns,
        x0_offset=x0_offset,
        y0_offset=y0_offset,
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
            dir=Path.cwd(), delete=False, prefix=f"tmpChromTrack_", suffix=".ele"
        )
        ele_pathobj = Path(tmp.name)
        ele_filepath = str(ele_pathobj.resolve())
        tmp.close()

    watch_pathobj = ele_pathobj.with_suffix(".wc")
    twi_pgz_pathobj = ele_pathobj.with_suffix(".twi.pgz")

    ed = elebuilder.EleDesigner(ele_filepath, double_format=".12g")

    elebuilder.add_transmute_blocks(ed, transmute_elements)

    ed.add_newline()

    ed.add_block(
        "run_setup",
        lattice=LTE_filepath,
        p_central_mev=E_MeV,
        use_beamline=use_beamline,
    )

    ed.add_newline()

    temp_watch_elem_name = "ELEGANT_CHROM_TRACK_WATCH"
    if run_local:
        watch_filepath = str(watch_pathobj.resolve())
    else:
        watch_filepath = watch_pathobj.name
    temp_watch_elem_def = (
        f'{temp_watch_elem_name}: WATCH, FILENAME="{watch_filepath}", '
        "MODE=coordinate"
    )

    ed.add_block(
        "insert_elements",
        name="*",
        exclude="*",
        add_at_start=True,
        element_def=temp_watch_elem_def,
    )

    ed.add_newline()

    elebuilder.add_N_KICKS_alter_elements_blocks(ed, N_KICKS)

    ed.add_newline()

    ed.add_block("run_control", n_passes=n_turns)

    ed.add_newline()

    centroid = {}
    centroid[0] = x0_offset
    centroid[2] = y0_offset
    centroid[5] = "<delta>"
    #
    ed.add_block("bunched_beam", n_particles_per_bunch=1, centroid=centroid)

    ed.add_newline()

    ed.add_block("track")

    ed.write()
    # print(ed.actual_output_filepath_list)

    twiss.calc_ring_twiss(
        str(twi_pgz_pathobj),
        LTE_filepath,
        E_MeV,
        use_beamline=use_beamline,
        parameters=None,
        run_local=True,
        del_tmp_files=True,
    )
    _d = util.load_pgz_file(str(twi_pgz_pathobj))
    nux0 = _d["data"]["twi"]["scalars"]["nux"]
    nuy0 = _d["data"]["twi"]["scalars"]["nuy"]
    # alpha & beta at watch element (at the start of the lattice)
    betax = _d["data"]["twi"]["arrays"]["betax"][0]
    betay = _d["data"]["twi"]["arrays"]["betay"][0]
    alphax = _d["data"]["twi"]["arrays"]["alphax"][0]
    alphay = _d["data"]["twi"]["arrays"]["alphay"][0]
    twi_pgz_pathobj.unlink()

    delta_array = np.linspace(delta_min, delta_max, ndelta)

    # Run Elegant
    if run_local:
        tbt = dict(
            x=np.full((n_turns, ndelta), np.nan),
            y=np.full((n_turns, ndelta), np.nan),
        )
        if method in ("sddsnaff", "DFT_Courant_Snyder"):
            tbt["xp"] = np.full((n_turns, ndelta), np.nan)
            tbt["yp"] = np.full((n_turns, ndelta), np.nan)

        # tElapsed = dict(run_ele=0.0, sdds2dicts=0.0, tbt_population=0.0)

        for i, delta in enumerate(delta_array):
            # t0 = time.time()
            run(
                ele_filepath,
                print_cmd=print_cmd,
                macros=dict(delta=f"{delta:.12g}"),
                print_stdout=std_print_enabled["out"],
                print_stderr=std_print_enabled["err"],
            )
            # tElapsed['run_ele'] += time.time() - t0

            # t0 = time.time()
            output, _ = sdds.sdds2dicts(watch_pathobj)
            # tElapsed['sdds2dicts'] += time.time() - t0

            # t0 = time.time()
            cols = output["columns"]
            for k in list(tbt):
                tbt[k][: len(cols[k]), i] = cols[k]
            # tElapsed['tbt_population'] += time.time() - t0
    else:

        if remote_opts is None:
            remote_opts = dict(ntasks=20)

        remote_opts["ntasks"] = min([len(delta_array), remote_opts["ntasks"]])

        delta_sub_array_list, reverse_mapping = util.chunk_list(
            delta_array, remote_opts["ntasks"]
        )

        coords_list = ["x", "y"]
        if method in ("sddsnaff", "DFT_Courant_Snyder"):
            coords_list += ["xp", "yp"]

        module_name = "pyelegant.nonlin"
        func_name = "_calc_chrom_track_get_tbt"

        iRemoteTry = 0
        while True:
            chunked_results = remote.run_mpi_python(
                remote_opts,
                module_name,
                func_name,
                delta_sub_array_list,
                (
                    ele_pathobj.read_text(),
                    ele_pathobj.name,
                    watch_pathobj.name,
                    print_cmd,
                    std_print_enabled["out"],
                    std_print_enabled["err"],
                    coords_list,
                ),
                err_log_check=err_log_check,
            )

            if (err_log_check is not None) and isinstance(chunked_results, str):

                err_log_text = chunked_results
                print("\n** Error Log check found the following problem:")
                print(err_log_text)

                iRemoteTry += 1

                if iRemoteTry >= nMaxRemoteRetry:
                    raise RuntimeError(
                        "Max number of remote tries exceeded. Check the error logs."
                    )
                else:
                    print("\n** Re-trying the remote run...\n")
                    sys.stdout.flush()
            else:
                break

        tbt_chunked_list = dict()
        tbt_flat_list = dict()
        for plane in coords_list:
            tbt_chunked_list[plane] = [_d[plane] for _d in chunked_results]
            tbt_flat_list[plane] = util.unchunk_list_of_lists(
                tbt_chunked_list[plane], reverse_mapping
            )

        tbt = dict(
            x=np.full((n_turns, ndelta), np.nan), y=np.full((n_turns, ndelta), np.nan)
        )
        if method in ("sddsnaff", "DFT_Courant_Snyder"):
            tbt["xp"] = np.full((n_turns, ndelta), np.nan)
            tbt["yp"] = np.full((n_turns, ndelta), np.nan)
        for plane in coords_list:
            for iDelta, array in enumerate(tbt_flat_list[plane]):
                tbt[plane][: len(array), iDelta] = array

    # print(tElapsed)

    survived = np.all(~np.isnan(tbt["x"]), axis=0)

    # t0 = time.time()
    # Estimate tunes from TbT data
    if method == "sddsnaff":
        tmp = tempfile.NamedTemporaryFile(
            dir=None, delete=False, prefix=f"tmpTbt_", suffix=".sdds"
        )
        tbt_sdds_path = Path(tmp.name)
        naff_sdds_path = tbt_sdds_path.parent.joinpath(f"{tbt_sdds_path.stem}.naff")
        tmp.close()

        pass_array = np.array(range(tbt["x"].shape[0]))
        nus = dict(
            x=np.full(tbt["x"].shape[1], np.nan),
            y=np.full(tbt["x"].shape[1], np.nan),
        )
        for iDelta, (x, xp, y, yp) in enumerate(
            zip(tbt["x"].T, tbt["xp"].T, tbt["y"].T, tbt["yp"].T)
        ):
            sdds.dicts2sdds(
                tbt_sdds_path,
                columns=dict(Pass=pass_array, x=x, xp=xp, y=y, yp=yp),
                outputMode="binary",
                tempdir_path=None,
                suppress_err_msg=True,
            )

            cmd = (
                f"sddsnaff {tbt_sdds_path} {naff_sdds_path} "
                "-column=Pass -pair=x,xp -pair=y,yp -terminate=frequencies=1"
            )
            p = Popen(shlex.split(cmd), stdout=PIPE, stderr=PIPE, encoding="utf-8")
            out, err = p.communicate()
            if out:
                print(f"stdout: {out}")
            if err:
                print(f"stderr: {err}")
            if False:
                cmd = f'sddsprintout {naff_sdds_path} -col="(xFrequency,yFrequency)"'
                p = Popen(shlex.split(cmd), stdout=PIPE, stderr=PIPE, encoding="utf-8")
                out, err = p.communicate()
                if out:
                    print(f"stdout: {out}")
                if err:
                    print(f"stderr: {err}")
            naff_d, naff_meta = sdds.sdds2dicts(naff_sdds_path)

            nus["x"][iDelta] = naff_d["columns"]["xFrequency"][0]
            nus["y"][iDelta] = naff_d["columns"]["yFrequency"][0]

        try:
            tbt_sdds_path.unlink()
        except:
            pass
        try:
            naff_sdds_path.unlink()
        except:
            pass

        undefined_tunes = (nus["x"] == -1.0) | (nus["y"] == -1.0)
        nus["x"][undefined_tunes] = np.nan
        nus["y"][undefined_tunes] = np.nan

        if False:
            other_nus = calc_chrom_from_tbt_cs(
                delta_array,
                tbt["x"],
                tbt["y"],
                nux0,
                nuy0,
                tbt["xp"],
                tbt["yp"],
                betax,
                alphax,
                betay,
                alphay,
                init_guess_from_prev_step=True,
                return_fft_spec=False,
            )

            plt.figure()
            plt.plot(
                delta_array * 1e2, other_nus["x"], "b.-", label="calc_chrom_from_tbt_cs"
            )
            plt.plot(delta_array * 1e2, nus["x"], "r.-", label="sddsnaff")
            plt.xlabel(r"$\delta\, [\%]$", size="large")
            plt.ylabel(r"$\nu_x$", size="large")
            leg = plt.legend(loc="best")
            plt.tight_layout()

            plt.figure()
            plt.plot(
                delta_array * 1e2, other_nus["y"], "b.-", label="calc_chrom_from_tbt_cs"
            )
            plt.plot(delta_array * 1e2, nus["y"], "r.-", label="sddsnaff")
            plt.xlabel(r"$\delta\, [\%]$", size="large")
            plt.ylabel(r"$\nu_y$", size="large")
            leg = plt.legend(loc="best")
            plt.tight_layout()

        extra_save_kwargs = dict(xptbt=tbt["xp"], yptbt=tbt["yp"])

        if return_fft_spec:
            *_, fft_nus, fft_hAxs, fft_hAys = calc_chrom_from_tbt_cs(
                delta_array,
                tbt["x"],
                tbt["y"],
                nux0,
                nuy0,
                tbt["xp"],
                tbt["yp"],
                betax,
                alphax,
                betay,
                alphay,
                init_guess_from_prev_step=True,
                return_fft_spec=True,
            )

            extra_save_kwargs["fft_nus"] = fft_nus
            extra_save_kwargs["fft_hAxs"] = fft_hAxs
            extra_save_kwargs["fft_hAys"] = fft_hAys

    elif method == "DFT_Courant_Snyder":
        if return_fft_spec:
            nus, fft_nus, fft_hAxs, fft_hAys = calc_chrom_from_tbt_cs(
                delta_array,
                tbt["x"],
                tbt["y"],
                nux0,
                nuy0,
                tbt["xp"],
                tbt["yp"],
                betax,
                alphax,
                betay,
                alphay,
                init_guess_from_prev_step=True,
                return_fft_spec=True,
            )
            extra_save_kwargs = dict(
                xptbt=tbt["xp"],
                yptbt=tbt["yp"],
                betax=betax,
                alphax=alphax,
                betay=betay,
                alphay=alphay,
                fft_nus=fft_nus,
                fft_hAxs=fft_hAxs,
                fft_hAys=fft_hAys,
            )
        else:
            nus = calc_chrom_from_tbt_cs(
                delta_array,
                tbt["x"],
                tbt["y"],
                nux0,
                nuy0,
                tbt["xp"],
                tbt["yp"],
                betax,
                alphax,
                betay,
                alphay,
                init_guess_from_prev_step=True,
                return_fft_spec=False,
            )
            extra_save_kwargs = dict(
                xptbt=tbt["xp"],
                yptbt=tbt["yp"],
                betax=betax,
                alphax=alphax,
                betay=betay,
                alphay=alphay,
            )

        undefined_tunes = np.isnan(nus["x"]) | np.isnan(nus["y"])
    elif method == "DFT_phase_space":
        nus = calc_chrom_from_tbt_ps(delta_array, tbt["x"], tbt["y"], nux0, nuy0)
        extra_save_kwargs = {}

        undefined_tunes = np.isnan(nus["x"]) | np.isnan(nus["y"])
    else:
        raise ValueError(f"method={method}")
    nuxs = nus["x"]
    nuys = nus["y"]
    # print('* Time elapsed for tune estimation: {:.3f}'.format(time.time() - t0))

    timestamp_fin = util.get_current_local_time_str()

    _save_chrom_data(
        output_filepath,
        output_file_type,
        delta_array,
        nuxs,
        nuys,
        survived,
        undefined_tunes,
        timestamp_fin,
        input_dict,
        xtbt=tbt["x"],
        ytbt=tbt["y"],
        nux0=nux0,
        nuy0=nuy0,
        save_tbt=save_tbt,
        **extra_save_kwargs,
    )

    if del_tmp_files:
        util.delete_temp_files(
            ed.actual_output_filepath_list + [ele_filepath, str(watch_pathobj)]
        )

    return output_filepath


def calc_chrom_from_tbt_ps(delta_array, xtbt, ytbt, nux0, nuy0):
    """
    Using phase-space (ps) variables "x" and "y", which can only
    determine tunes within the range of [0, 0.5].
    """

    frac_nux0 = nux0 - np.floor(nux0)
    frac_nuy0 = nuy0 - np.floor(nuy0)

    if frac_nux0 > 0.5:
        frac_nux0 = 1 - frac_nux0
        nux_above_half = True
    else:
        nux_above_half = False

    if frac_nuy0 > 0.5:
        frac_nuy0 = 1 - frac_nuy0
        nuy_above_half = True
    else:
        nuy_above_half = False

    neg_delta_array = delta_array[delta_array < 0.0]
    neg_sort_inds = np.argsort(np.abs(neg_delta_array))
    sorted_neg_delta_inds = np.where(delta_array < 0.0)[0][neg_sort_inds]
    pos_delta_array = delta_array[delta_array >= 0.0]
    pos_sort_inds = np.argsort(pos_delta_array)
    sorted_pos_delta_inds = np.where(delta_array >= 0.0)[0][pos_sort_inds]
    sorted_neg_delta_inds, sorted_pos_delta_inds

    nus = dict(
        x=np.full(delta_array.shape, np.nan), y=np.full(delta_array.shape, np.nan)
    )

    n_turns = xtbt.shape[0]
    nu_vec = np.fft.fftfreq(n_turns)

    opts = dict(window="sine", resolution=1e-8)
    for sorted_delta_inds in [sorted_neg_delta_inds, sorted_pos_delta_inds]:
        init_nux = frac_nux0
        init_nuy = frac_nuy0
        for i in sorted_delta_inds:
            xarray = xtbt[:, i]
            yarray = ytbt[:, i]

            if np.any(np.isnan(xarray)) or np.any(np.isnan(yarray)):
                # Particle lost at some point.
                continue

            if False:
                # This algorithm does NOT work too well if tune change
                # between neighboring delta points are too large.
                out = sigproc.getDftPeak(xarray, init_nux, **opts)
                nus["x"][i] = out["nu"]
                init_nux = out["nu"]

                out = sigproc.getDftPeak(yarray, init_nuy, **opts)
                nus["y"][i] = out["nu"]
                init_nuy = out["nu"]
            else:
                # Find the rough peak first
                ff_rect = np.fft.fft(xarray - np.mean(xarray))
                A_arb = np.abs(ff_rect)
                init_nux = nu_vec[np.argmax(A_arb[: (n_turns // 2)])]
                # Then fine-tune
                out = sigproc.getDftPeak(xarray, init_nux, **opts)
                nus["x"][i] = out["nu"]

                # Find the rough peak first
                ff_rect = np.fft.fft(yarray - np.mean(yarray))
                A_arb = np.abs(ff_rect)
                init_nuy = nu_vec[np.argmax(A_arb[: (n_turns // 2)])]
                # Then fine-tune
                out = sigproc.getDftPeak(yarray, init_nuy, **opts)
                nus["y"][i] = out["nu"]

    if nux_above_half:
        nus["x"] = 1 - nus["x"]
    if nuy_above_half:
        nus["y"] = 1 - nus["y"]

    return nus


def angle2p(xp, yp, delta):
    """
    Convert angles (xp, yp) into canonical momenta (px, py)
    """

    fac = (1 + delta) / np.sqrt(1 + xp**2 + yp**2)

    px = xp * fac
    py = yp * fac

    return px, py


def p2angle(px, py, delta):
    """
    Convert canonical momenta (px, py) into angles (xp, yp)
    """

    fac = np.sqrt((1 + delta) ** 2 - (px**2 + py**2))

    xp = px / fac
    yp = py / fac

    return xp, yp


def approx_angle2p(xp, yp, delta):
    """
    Convert angles (xp, yp) into canonical momenta (px, py)
    **approximately** with small-angle assumption.
    """

    fac = 1 + delta

    px = xp * fac
    py = yp * fac

    return px, py


def approx_p2angle(px, py, delta):
    """
    Convert canonical momenta (px, py) into angles (xp, yp)
    **approximately** with small-angle assumption.
    """

    fac = 1 + delta

    xp = px / fac
    yp = py / fac

    return xp, yp


def calc_chrom_from_tbt_cs(
    delta_array,
    xtbt,
    ytbt,
    nux0,
    nuy0,
    xptbt,
    yptbt,
    betax,
    alphax,
    betay,
    alphay,
    init_guess_from_prev_step=True,
    return_fft_spec=True,
):
    """
    Using Courant-Snyder (CS) coordinates "xhat" and "yhat", which enables
    determination of tunes within the range of [0, 1.0].

    If `init_guess_from_prev_step` is True (recommended), then the tune peak
    fine-tuning starts from the tune peak found from the previous momentum offset
    to avoid any sudden tune jump along an increasing momentum offset vector.
    If False, then a rough tune peak is found from a simple FFT spectrum, which
    can potentially lead the initial search point to another strong resonance
    peak, away from the fundamental tune peak.
    """

    frac_nux0 = nux0 - np.floor(nux0)
    frac_nuy0 = nuy0 - np.floor(nuy0)

    if frac_nux0 > 0.5:
        frac_nux0 = frac_nux0 - 1
    if frac_nuy0 > 0.5:
        frac_nuy0 = frac_nuy0 - 1

    neg_delta_array = delta_array[delta_array < 0.0]
    neg_sort_inds = np.argsort(np.abs(neg_delta_array))
    sorted_neg_delta_inds = np.where(delta_array < 0.0)[0][neg_sort_inds]
    pos_delta_array = delta_array[delta_array >= 0.0]
    pos_sort_inds = np.argsort(pos_delta_array)
    sorted_pos_delta_inds = np.where(delta_array >= 0.0)[0][pos_sort_inds]
    sorted_neg_delta_inds, sorted_pos_delta_inds

    nus = dict(
        x=np.full(delta_array.shape, np.nan), y=np.full(delta_array.shape, np.nan)
    )

    n_turns = xtbt.shape[0]
    nu_vec = np.fft.fftfreq(n_turns)

    opts = dict(window="sine", resolution=1e-8, return_fft_spec=return_fft_spec)
    if return_fft_spec:
        fft_nus = None
        fft_hAxs_list = []
        fft_hAys_list = []
    for sorted_delta_inds in [sorted_neg_delta_inds, sorted_pos_delta_inds]:
        init_nux = frac_nux0
        init_nuy = frac_nuy0
        if return_fft_spec:
            fft_hAxs = []
            fft_hAys = []
        for i in sorted_delta_inds:
            xarray = xtbt[:, i]
            yarray = ytbt[:, i]

            if np.any(np.isnan(xarray)) or np.any(np.isnan(yarray)):
                # Particle lost at some point.
                if return_fft_spec:
                    fft_hAxs.append(np.full((n_turns,), np.nan))
                    fft_hAys.append(np.full((n_turns,), np.nan))
                continue

            xparray = xptbt[:, i]
            yparray = yptbt[:, i]
            delta = delta_array[i]
            pxarray, pyarray = angle2p(xparray, yparray, delta)

            xarray -= np.mean(xarray)
            yarray -= np.mean(yarray)
            pxarray -= np.mean(pxarray)
            pyarray -= np.mean(pyarray)

            xhat = xarray / np.sqrt(betax)
            pxhat = alphax / np.sqrt(betax) * xarray + np.sqrt(betax) * pxarray
            yhat = yarray / np.sqrt(betay)
            pyhat = alphay / np.sqrt(betay) * yarray + np.sqrt(betay) * pyarray

            hx = xhat - 1j * pxhat
            hy = yhat - 1j * pyhat

            if init_guess_from_prev_step:
                rough_peak_nux = sigproc.findNearestFftPeak(
                    hx, init_nux, window=opts["window"]
                )["nu"]
                # print(f'init/rough nux: {init_nux:.6f}/{rough_peak_nux:.6f}')
                out = sigproc.getDftPeak(hx, rough_peak_nux, **opts)
                nus["x"][i] = out["nu"]
                init_nux = out["nu"]
            else:
                # Find the rough peak first
                ff_rect = np.fft.fft(hx)
                A_arb = np.abs(ff_rect)
                init_nux = nu_vec[np.argmax(A_arb)]
                # Then fine-tune
                out = sigproc.getDftPeak(hx, init_nux, **opts)
                nus["x"][i] = out["nu"]
            if return_fft_spec:
                if fft_nus is None:
                    fft_nus = out["fft_nus"]
                fft_hAxs.append(out["fft_As"])

            if init_guess_from_prev_step:
                rough_peak_nuy = sigproc.findNearestFftPeak(
                    hy, init_nuy, window=opts["window"]
                )["nu"]
                # print(f'init/rough nuy: {init_nuy:.6f}/{rough_peak_nuy:.6f}')
                out = sigproc.getDftPeak(hy, rough_peak_nuy, **opts)
                nus["y"][i] = out["nu"]
                init_nuy = out["nu"]
            else:
                # Find the rough peak first
                ff_rect = np.fft.fft(hy)
                A_arb = np.abs(ff_rect)
                init_nuy = nu_vec[np.argmax(A_arb)]
                # Then fine-tune
                out = sigproc.getDftPeak(hy, init_nuy, **opts)
                nus["y"][i] = out["nu"]
            if return_fft_spec:
                fft_hAys.append(out["fft_As"])

        if return_fft_spec:
            fft_hAxs_list.append(np.array(fft_hAxs).T)
            fft_hAys_list.append(np.array(fft_hAys).T)

    nonnan_inds = np.where(~np.isnan(nus["x"]))[0]
    neg_inds = nonnan_inds[nus["x"][nonnan_inds] < 0]
    nus["x"][neg_inds] += 1
    nonnan_inds = np.where(~np.isnan(nus["y"]))[0]
    neg_inds = nonnan_inds[nus["y"][nonnan_inds] < 0]
    nus["y"][neg_inds] += 1

    if not return_fft_spec:
        return nus
    else:
        fft_hAxs = np.hstack((fft_hAxs_list[0][:, ::-1], fft_hAxs_list[1]))
        fft_hAys = np.hstack((fft_hAys_list[0][:, ::-1], fft_hAys_list[1]))
        return nus, fft_nus, fft_hAxs, fft_hAys


def _calc_chrom_track_get_tbt(
    delta_sub_array,
    ele_contents,
    ele_filename,
    watch_filename,
    print_cmd,
    print_stdout,
    print_stderr,
    coords_list,
    tempdir_path="/tmp",
):
    """"""

    if not Path(tempdir_path).exists():
        tempdir_path = Path.cwd()

    sub_tbt = dict()
    for k in coords_list:
        sub_tbt[k] = []

    with tempfile.TemporaryDirectory(
        prefix="tmpCalcChrom_", dir=tempdir_path
    ) as tmpdirname:

        ele_pathobj = Path(tmpdirname).joinpath(ele_filename)
        watch_pathobj = Path(tmpdirname).joinpath(watch_filename)

        ele_contents = ele_contents.replace(
            watch_filename, str(watch_pathobj.resolve())
        )

        ele_pathobj.write_text(ele_contents)

        ele_filepath = str(ele_pathobj.resolve())

        for delta in delta_sub_array:

            run(
                ele_filepath,
                print_cmd=print_cmd,
                macros=dict(delta=f"{delta:.12g}"),
                print_stdout=print_stdout,
                print_stderr=print_stderr,
            )

            output, _ = sdds.sdds2dicts(watch_pathobj)

            cols = output["columns"]
            for k in list(sub_tbt):
                sub_tbt[k].append(cols[k])

    return sub_tbt


def _save_chrom_data(
    output_filepath,
    output_file_type,
    delta_array,
    nuxs,
    nuys,
    survived,
    undefined_tunes,
    timestamp_fin,
    input_dict,
    xtbt=None,
    ytbt=None,
    nux0=None,
    nuy0=None,
    xptbt=None,
    yptbt=None,
    betax=None,
    alphax=None,
    betay=None,
    alphay=None,
    fft_nus=None,
    fft_hAxs=None,
    fft_hAys=None,
    save_tbt=True,
):
    """
    nux0, nuy0: on-momentum tunes
    """

    if output_file_type in ("hdf5", "h5"):
        _kwargs = dict(compression="gzip")
        f = h5py.File(output_filepath, "a")
        f["_version_PyELEGANT"] = __version__["PyELEGANT"]
        f["_version_ELEGANT"] = __version__["ELEGANT"]
        f.create_dataset("deltas", data=delta_array, **_kwargs)
        f.create_dataset("nuxs", data=nuxs, **_kwargs)
        f.create_dataset("nuys", data=nuys, **_kwargs)
        if save_tbt and (xtbt is not None):
            f.create_dataset("xtbt", data=xtbt, **_kwargs)
        if save_tbt and (ytbt is not None):
            f.create_dataset("ytbt", data=ytbt, **_kwargs)
        if nux0:
            f["nux0"] = nux0
        if nuy0:
            f["nuy0"] = nuy0
        if save_tbt and (xptbt is not None):
            f.create_dataset("xptbt", data=xptbt, **_kwargs)
        if save_tbt and (yptbt is not None):
            f.create_dataset("yptbt", data=yptbt, **_kwargs)
        if fft_nus is not None:
            f.create_dataset("fft_nus", data=fft_nus, **_kwargs)
            f.create_dataset("fft_hAxs", data=fft_hAxs, **_kwargs)
            f.create_dataset("fft_hAys", data=fft_hAys, **_kwargs)
        if betax:
            f["betax"] = betax
        if alphax:
            f["alphax"] = alphax
        if betay:
            f["betay"] = betay
        if alphay:
            f["alphay"] = alphay
        f.create_dataset("survived", data=survived, **_kwargs)
        f.create_dataset("undefined_tunes", data=undefined_tunes, **_kwargs)
        f["timestamp_fin"] = timestamp_fin
        f.close()

    elif output_file_type == "pgz":
        d = dict(
            deltas=delta_array,
            nuxs=nuxs,
            nuys=nuys,
            survived=survived,
            undefined_tunes=undefined_tunes,
            input=input_dict,
            timestamp_fin=timestamp_fin,
            _version_PyELEGANT=__version__["PyELEGANT"],
            _version_ELEGANT=__version__["ELEGANT"],
        )
        if save_tbt and (xtbt is not None):
            d["xtbt"] = xtbt
        if save_tbt and (ytbt is not None):
            d["ytbt"] = ytbt
        if nux0:
            d["nux0"] = nux0
        if nuy0:
            d["nuy0"] = nuy0
        if save_tbt and (xptbt is not None):
            d["xptbt"] = xptbt
        if save_tbt and (yptbt is not None):
            d["yptbt"] = yptbt
        if fft_nus is not None:
            d["fft_nus"] = fft_nus
            d["fft_hAxs"] = fft_hAxs
            d["fft_hAys"] = fft_hAys
        if betax:
            d["betax"] = betax
        if alphax:
            d["alphax"] = alphax
        if betay:
            d["betay"] = betay
        if alphay:
            d["alphay"] = alphay
        util.robust_pgz_file_write(output_filepath, d, nMaxTry=10, sleep=10.0)
    else:
        raise ValueError()


def plot_chrom(
    output_filepath,
    max_chrom_order=3,
    title="",
    deltalim=None,
    fit_deltalim=None,
    nuxlim=None,
    nuylim=None,
    footprint_nuxlim=None,
    footprint_nuylim=None,
    max_resonance_line_order=5,
    fit_label_format="+.3g",
    ax_nu_vs_delta=None,
    ax_nuy_vs_nux=None,
    plot_fft=False,
    fft_plot_opts=None,
    ax_fft_hx=None,
    ax_fft_hy=None,
):
    """"""

    assert max_resonance_line_order <= 5

    ret = {}  # variable to be returned
    ret["aper"] = {}

    is_nuxlim_frac = False
    if nuxlim is not None:
        if (0.0 <= nuxlim[0] <= 1.0) and (0.0 <= nuxlim[1] <= 1.0):
            is_nuxlim_frac = True

    is_nuylim_frac = False
    if nuylim is not None:
        if (0.0 <= nuylim[0] <= 1.0) and (0.0 <= nuylim[1] <= 1.0):
            is_nuylim_frac = True

    if is_nuxlim_frac and is_nuylim_frac:
        _plot_nu_frac = True
    elif (not is_nuxlim_frac) and (not is_nuylim_frac):
        _plot_nu_frac = False
    else:
        raise ValueError(
            '"nuxlim" and "nuylim" must be either both fractional or both non-fractional'
        )

    try:
        d = util.load_pgz_file(output_filepath)
        deltas = d["deltas"]
        nuxs = d["nuxs"]
        nuys = d["nuys"]
        if "nux0" in d:
            nux0 = d["nux0"]
            nuy0 = d["nuy0"]

            on_mom_nu0 = dict(x=nux0, y=nuy0)

            nux0_int = np.floor(nux0)
            nuy0_int = np.floor(nuy0)

            on_mom_delta_index = np.argmin(np.abs(deltas))
            _kwargs = dict(jump_thresh=0.5, ref_index=on_mom_delta_index)
            nuxs = smooth_nu_int_jump(nuxs, **_kwargs)
            nuys = smooth_nu_int_jump(nuys, **_kwargs)

            nuxs += nux0_int
            nuys += nuy0_int
        else:  # This method may well not be so robust.
            nuxs = smooth_nu_int_jump(nuxs, jump_thresh=0.5)
            nuys = smooth_nu_int_jump(nuys, jump_thresh=0.5)

            nux0 = np.nanmedian(nuxs)
            nuy0 = np.nanmedian(nuys)

            on_mom_nu0 = None

            nux0_int = np.floor(nux0)
            nuy0_int = np.floor(nuy0)
        if "fft_nus" in d:
            fft_d = {k: d[k] for k in ["fft_nus", "fft_hAxs", "fft_hAys"]}
        else:
            fft_d = None

        if "undefined_tunes" in d:
            undefined_tunes = d["undefined_tunes"]
        else:
            undefined_tunes = np.isnan(nuxs) | np.isnan(nuys)
    except:
        f = h5py.File(output_filepath, "r")
        deltas = f["deltas"][()]
        nuxs = f["nuxs"][()]
        nuys = f["nuys"][()]
        if "nux0" in f:
            nux0 = f["nux0"][()]
            nuy0 = f["nuy0"][()]

            on_mom_nu0 = dict(x=nux0, y=nuy0)

            nux0_int = np.floor(nux0)
            nuy0_int = np.floor(nuy0)

            on_mom_delta_index = np.argmin(np.abs(deltas))
            _kwargs = dict(jump_thresh=0.5, ref_index=on_mom_delta_index)
            nuxs = smooth_nu_int_jump(nuxs, **_kwargs)
            nuys = smooth_nu_int_jump(nuys, **_kwargs)

            nuxs += nux0_int
            nuys += nuy0_int
        else:  # This method may well not be so robust.
            nuxs = smooth_nu_int_jump(nuxs, jump_thresh=0.5)
            nuys = smooth_nu_int_jump(nuys, jump_thresh=0.5)

            nux0 = np.nanmedian(nuxs)
            nuy0 = np.nanmedian(nuys)

            on_mom_nu0 = None

            nux0_int = np.floor(nux0)
            nuy0_int = np.floor(nuy0)
        if "fft_nus" in f:
            fft_d = {k: f[k][()] for k in ["fft_nus", "fft_hAxs", "fft_hAys"]}
        else:
            fft_d = None

        if "undefined_tunes" in f:
            undefined_tunes = f["undefined_tunes"][()]
        else:
            undefined_tunes = np.isnan(nuxs) | np.isnan(nuys)

        f.close()

    # Find undefined tune boundaries (i.e., either particle lost or spectrum too chaotic)
    undef_deltas = deltas[undefined_tunes]
    #
    pos_undef_deltas = undef_deltas[undef_deltas >= 0.0]
    if pos_undef_deltas.size != 0:
        min_pos_undef_delta = np.min(pos_undef_deltas)
    else:
        min_pos_undef_delta = np.nan
    #
    neg_undef_deltas = undef_deltas[undef_deltas <= 0.0]
    if neg_undef_deltas.size != 0:
        max_neg_undef_delta = np.max(neg_undef_deltas)
    else:
        max_neg_undef_delta = np.nan

    # Correct nuxs if smoothing shifted from nux0 by ~1
    for i in np.argsort(np.abs(deltas)):
        if np.isnan(nuxs[i]):
            continue
        else:
            if nuxs[i] > nux0 + 0.5:
                nuxs -= 1
            elif nuxs[i] < nux0 - 0.5:
                nuxs += 1
            break
    # Correct nuys if smoothing shifted from nuy0 by ~1
    for i in np.argsort(np.abs(deltas)):
        if np.isnan(nuys[i]):
            continue
        else:
            if nuys[i] > nuy0 + 0.5:
                nuys -= 1
            elif nuys[i] < nuy0 - 0.5:
                nuys += 1
            break

    # Find first off-momentum integer/half-integer resonance crossing tunes
    upper_res_xing_nu = {}
    lower_res_xing_nu = {}
    if on_mom_nu0 is not None:
        for plane in ["x", "y"]:
            on_mom_nu0_int = np.floor(on_mom_nu0[plane])
            on_mom_nu0_frac = on_mom_nu0[plane] - on_mom_nu0_int
            if on_mom_nu0_frac < 0.5:
                upper_res_xing_nu[plane] = on_mom_nu0_int + 0.5
                lower_res_xing_nu[plane] = on_mom_nu0_int
            elif on_mom_nu0_frac > 0.5:
                upper_res_xing_nu[plane] = on_mom_nu0_int + 1.0
                lower_res_xing_nu[plane] = on_mom_nu0_int + 0.5
            else:
                pass
    min_pos_res_xing_delta = np.nan
    max_neg_res_xing_delta = np.nan
    if upper_res_xing_nu != {}:
        def_deltas = deltas[~undefined_tunes]
        if def_deltas.size != 0:
            def_nuxs = nuxs[~undefined_tunes]
            def_nuys = nuys[~undefined_tunes]
            positive = def_deltas >= 0.0
            negative = def_deltas <= 0.0
            if def_deltas[positive].size != 0:
                pos_xing_deltas = def_deltas[positive][
                    (def_nuxs[positive] > upper_res_xing_nu["x"])
                    | (def_nuxs[positive] < lower_res_xing_nu["x"])
                    | (def_nuys[positive] > upper_res_xing_nu["y"])
                    | (def_nuys[positive] < lower_res_xing_nu["y"])
                ]
                if pos_xing_deltas.size != 0:
                    min_pos_res_xing_delta = np.min(pos_xing_deltas)
            if def_deltas[negative].size != 0:
                neg_xing_deltas = def_deltas[negative][
                    (def_nuxs[negative] > upper_res_xing_nu["x"])
                    | (def_nuxs[negative] < lower_res_xing_nu["x"])
                    | (def_nuys[negative] > upper_res_xing_nu["y"])
                    | (def_nuys[negative] < lower_res_xing_nu["y"])
                ]
                if neg_xing_deltas.size != 0:
                    max_neg_res_xing_delta = np.max(neg_xing_deltas)

    ret["aper"]["scanned"] = [np.min(deltas), np.max(deltas)]
    if deltalim is not None:
        delta_incl = np.logical_and(
            deltas >= np.min(deltalim), deltas <= np.max(deltalim)
        )
    else:
        delta_incl = np.ones(deltas.shape).astype(bool)
    deltas = deltas[delta_incl]
    ret["aper"]["plotted"] = [np.min(deltas), np.max(deltas)]
    nuxs = nuxs[delta_incl]
    nuys = nuys[delta_incl]

    fit_deltas_for_plot = np.linspace(np.min(deltas), np.max(deltas), 101)

    if fit_deltalim is not None:
        fit_delta_min, fit_delta_max = np.min(fit_deltalim), np.max(fit_deltalim)
        fit_delta_incl = np.logical_and(
            deltas >= fit_delta_min, deltas <= fit_delta_max
        )
        fit_deltas = deltas[fit_delta_incl]
        fit_nuxs = nuxs[fit_delta_incl]
        fit_nuys = nuys[fit_delta_incl]
    else:
        fit_delta_min, fit_delta_max = np.min(deltas), np.max(deltas)
        fit_delta_incl = np.ones(deltas.shape).astype(bool)
        fit_deltas = deltas
        fit_nuxs = nuxs
        fit_nuys = nuys
    nux_nan_inds = np.isnan(fit_nuxs)
    nuy_nan_inds = np.isnan(fit_nuys)
    coeffs = dict(
        x=np.polyfit(
            fit_deltas[~nux_nan_inds], fit_nuxs[~nux_nan_inds], max_chrom_order
        ),
        y=np.polyfit(
            fit_deltas[~nuy_nan_inds], fit_nuys[~nuy_nan_inds], max_chrom_order
        ),
    )
    ret["fit_coeffs"] = coeffs

    fit_label = {}
    for plane in ["x", "y"]:
        fit_label[plane] = rf"\nu_{plane} = "
        for i, c in zip(range(max_chrom_order + 1)[::-1], coeffs[plane]):

            if i != 0:
                fit_label[plane] += util.pprint_sci_notation(c, fit_label_format)
            else:
                fit_label[plane] += util.pprint_sci_notation(c, "+.3f")

            if i == 1:
                fit_label[plane] += r"\delta "
            elif i >= 2:
                fit_label[plane] += rf"\delta^{i:d} "

        fit_label[plane] = "${}$".format(fit_label[plane].strip())

    font_sz = 22

    if ax_nu_vs_delta:
        ax1 = ax_nu_vs_delta
    else:
        fig, ax1 = plt.subplots()
    if _plot_nu_frac:
        offset = np.floor(nuxs)
    else:
        offset = np.zeros(nuxs.shape)
    lines1 = ax1.plot(deltas * 1e2, nuxs - offset, "b.", label=r"$\nu_x$")
    if nuxlim is not None:
        ax1.set_ylim(nuxlim)
    else:
        nuxlim = list(ax1.get_ylim())
    # fit_lines1 = ax1.plot(
    # fit_deltas * 1e2, np.poly1d(coeffs['x'])(fit_deltas) - offset[fit_delta_incl], 'b-',
    # label=fit_label['x'])
    interp_roi = np.logical_and(
        fit_deltas_for_plot >= fit_delta_min, fit_deltas_for_plot <= fit_delta_max
    )
    if _plot_nu_frac:
        fit_offset = np.floor(np.poly1d(coeffs["x"])(fit_deltas_for_plot[interp_roi]))
    else:
        fit_offset = np.zeros(fit_deltas_for_plot[interp_roi].shape)
    fit_lines1 = ax1.plot(
        fit_deltas_for_plot[interp_roi] * 1e2,
        np.poly1d(coeffs["x"])(fit_deltas_for_plot[interp_roi]) - fit_offset,
        "b-",
        label=fit_label["x"],
    )
    for extrap_roi in [
        fit_deltas_for_plot < fit_delta_min,
        fit_deltas_for_plot > fit_delta_max,
    ]:
        if _plot_nu_frac:
            fit_offset = np.floor(
                np.poly1d(coeffs["x"])(fit_deltas_for_plot[extrap_roi])
            )
        else:
            fit_offset = np.zeros(fit_deltas_for_plot[extrap_roi].shape)
        ax1.plot(
            fit_deltas_for_plot[extrap_roi] * 1e2,
            np.poly1d(coeffs["x"])(fit_deltas_for_plot[extrap_roi]) - fit_offset,
            "b:",
        )
    ax2 = ax1.twinx()
    if _plot_nu_frac:
        offset = np.floor(nuys)
    else:
        offset = np.zeros(nuys.shape)
    lines2 = ax2.plot(deltas * 1e2, nuys - offset, "r.", label=r"$\nu_y$")
    if nuylim is not None:
        ax2.set_ylim(nuylim)
    else:
        nuylim = list(ax2.get_ylim())
    # fit_lines2 = ax2.plot(
    # fit_deltas * 1e2, np.poly1d(coeffs['y'])(fit_deltas) - offset[fit_delta_incl], 'r-',
    # label=fit_label['y'])
    if _plot_nu_frac:
        fit_offset = np.floor(np.poly1d(coeffs["y"])(fit_deltas_for_plot[interp_roi]))
    else:
        fit_offset = np.zeros(fit_deltas_for_plot[interp_roi].shape)
    fit_lines2 = ax2.plot(
        fit_deltas_for_plot[interp_roi] * 1e2,
        np.poly1d(coeffs["y"])(fit_deltas_for_plot[interp_roi]) - fit_offset,
        "r-",
        label=fit_label["y"],
    )
    for extrap_roi in [
        fit_deltas_for_plot < fit_delta_min,
        fit_deltas_for_plot > fit_delta_max,
    ]:
        if _plot_nu_frac:
            fit_offset = np.floor(
                np.poly1d(coeffs["y"])(fit_deltas_for_plot[extrap_roi])
            )
        else:
            fit_offset = np.zeros(fit_deltas_for_plot[extrap_roi].shape)
        ax2.plot(
            fit_deltas_for_plot[extrap_roi] * 1e2,
            np.poly1d(coeffs["y"])(fit_deltas_for_plot[extrap_roi]) - fit_offset,
            "r:",
        )
    ax1.set_xlabel(r"$\delta\, [\%]$", size=font_sz)
    ax1.set_ylabel(r"$\nu_x$", size=font_sz, color="b")
    ax2.set_ylabel(r"$\nu_y$", size=font_sz, color="r")
    if deltalim is not None:
        ax1.set_xlim([v * 1e2 for v in deltalim])
    # Reset nux/nuy limits, which may have been changed by adding fitted lines
    ax1.set_ylim(nuxlim)
    ax2.set_ylim(nuylim)
    # Add integer/half-integer tune lines, if within visible range
    for _nu in range(int(np.floor(nuxlim[0])), int(np.ceil(nuxlim[1])) + 1):
        if nuxlim[0] <= _nu <= nuxlim[1]:  # integer tune line
            ax1.axhline(_nu, linestyle="--", color="b")
        if nuxlim[0] <= _nu + 0.5 <= nuxlim[1]:  # half-integer tune line
            ax1.axhline(_nu + 0.5, linestyle=":", color="b")
    for _nu in range(int(np.floor(nuylim[0])), int(np.ceil(nuylim[1])) + 1):
        if nuylim[0] <= _nu <= nuylim[1]:  # integer tune line
            ax2.axhline(_nu, linestyle="--", color="r")
        if nuylim[0] <= _nu + 0.5 <= nuylim[1]:  # half-integer tune line
            ax2.axhline(_nu + 0.5, linestyle=":", color="r")
    # Add lines at max defined tune boundaries
    _deltalim = ax1.get_xlim()
    ret["aper"]["undefined_tunes"] = []
    for _delta in [max_neg_undef_delta, min_pos_undef_delta]:
        if _deltalim[0] <= (_delta * 1e2) <= _deltalim[1]:
            ax1.axvline(_delta * 1e2, linestyle="--", color="k")
            ret["aper"]["undefined_tunes"].append(_delta)
    # Add lines at max apertures w/o crossing integer/half-integer resonance
    ret["aper"]["resonance_xing"] = []
    for _delta in [max_neg_res_xing_delta, min_pos_res_xing_delta]:
        if _deltalim[0] <= (_delta * 1e2) <= _deltalim[1]:
            ax1.axvline(_delta * 1e2, linestyle=":", color="k")
            ret["aper"]["resonance_xing"].append(_delta)
    #
    if title != "":
        ax1.set_title(title, size=font_sz, pad=60)
    combined_lines = fit_lines1 + fit_lines2
    leg = ax2.legend(
        combined_lines,
        [L.get_label() for L in combined_lines],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.3),
        fancybox=True,
        shadow=True,
        prop=dict(size=12),
    )
    plt.sca(ax1)
    plt.tight_layout()

    if footprint_nuxlim is None:
        footprint_nuxlim = nuxlim.copy()
    is_fp_nuxlim_frac = False
    if (0.0 <= footprint_nuxlim[0] <= 1.0) and (0.0 <= footprint_nuxlim[1] <= 1.0):
        is_fp_nuxlim_frac = True
    #
    if footprint_nuylim is None:
        footprint_nuylim = nuylim.copy()
    is_fp_nuylim_frac = False
    if (0.0 <= footprint_nuylim[0] <= 1.0) and (0.0 <= footprint_nuylim[1] <= 1.0):
        is_fp_nuylim_frac = True
    #
    if is_fp_nuxlim_frac and is_fp_nuylim_frac:
        _plot_fp_nu_frac = True
    elif (not is_fp_nuxlim_frac) and (not is_fp_nuylim_frac):
        _plot_fp_nu_frac = False
    else:
        raise ValueError(
            (
                '"footprint_nuxlim" and "footprint_nuylim" must be either '
                "both fractional or both non-fractional"
            )
        )

    if _plot_fp_nu_frac:
        _nuxs = nuxs - nux0_int
        _nuys = nuys - nuy0_int

        frac_nuxlim = footprint_nuxlim
        frac_nuylim = footprint_nuylim
    else:
        _nuxs = nuxs
        _nuys = nuys

        frac_nuxlim = footprint_nuxlim - nux0_int
        frac_nuylim = footprint_nuylim - nuy0_int

    if ax_nuy_vs_nux:
        ax = ax_nuy_vs_nux
    else:
        _, ax = plt.subplots()
    sc_obj = ax.scatter(_nuxs, _nuys, s=10, c=deltas * 1e2, marker="o", cmap="jet")
    ax.set_xlim(footprint_nuxlim)
    ax.set_ylim(footprint_nuylim)
    ax.set_xlabel(r"$\nu_x$", size=font_sz)
    ax.set_ylabel(r"$\nu_y$", size=font_sz)
    #
    rd = util.ResonanceDiagram()
    lineprops = dict(
        color=["k", "k", "g", "m", "m"],
        linestyle=["-", "--", "-", "-", ":"],
        linewidth=[2, 2, 0.5, 0.5, 0.5],
    )
    for n in range(1, max_resonance_line_order):
        d = rd.getResonanceCoeffsAndLines(
            n, np.array(frac_nuxlim) + nux0_int, np.array(frac_nuylim) + nuy0_int
        )  # <= CRITICAL: Must pass tunes w/ integer parts
        prop = {k: lineprops[k][n - 1] for k in ["color", "linestyle", "linewidth"]}
        assert len(d["lines"]) == len(d["coeffs"])
        for ((nux1, nuy1), (nux2, nuy2)), (nx, ny, _) in zip(d["lines"], d["coeffs"]):
            if _plot_fp_nu_frac:
                _x = np.array([nux1 - nux0_int, nux2 - nux0_int])
                _y = np.array([nuy1 - nuy0_int, nuy2 - nuy0_int])
            else:
                _x = np.array([nux1, nux2])
                _y = np.array([nuy1, nuy2])
            ax.plot(_x, _y, label=rd.getResonanceCoeffLabelString(nx, ny), **prop)
            # print(n, nx, ny, _x, _y, prop)
    # leg = plt.legend(loc='best')
    # leg.set_draggable(True, use_blit=True)
    #
    cb = plt.colorbar(sc_obj, ax=ax)
    cb.ax.set_title(r"$\delta\, [\%]$", size=16)
    if title != "":
        ax.set_title(title, size=font_sz)
    plt.sca(ax)
    plt.tight_layout()

    if (fft_d is not None) and plot_fft:

        if fft_plot_opts is None:
            fft_plot_opts = {}

        use_log = fft_plot_opts.get("logscale", False)
        use_full_ylim = fft_plot_opts.get("full_ylim", True)
        plot_shifted_curves = fft_plot_opts.get("shifted_curves", False)
        nu_size = fft_plot_opts.get("nu_size", None)

        font_sz = 18
        if use_log:
            EQ_STR = r"$\rm{log}_{10}(A/\mathrm{max}A)$"
        else:
            EQ_STR = r"$A/\mathrm{max}A$"

        v1array = deltas

        for _nu_plane in ["x", "y"]:

            v2array = fft_d["fft_nus"].copy()
            v2array[v2array < 0.0] += 1
            if nu_size is not None:
                assert np.all(np.diff(v2array) > 0.0)
                v2array_resized = np.linspace(np.min(v2array), np.max(v2array), nu_size)
            else:
                v2array_resized = None

            if _nu_plane == "x":
                v2array += nux0_int
                if v2array_resized is not None:
                    v2array_resized += nux0_int

                if ax_fft_hx:
                    ax1 = ax_fft_hx
                else:
                    fig, ax1 = plt.subplots()

                norm_fft_hAs = fft_d["fft_hAxs"] / np.max(fft_d["fft_hAxs"], axis=0)

                ylim = nuxlim
            else:
                v2array += nuy0_int
                if v2array_resized is not None:
                    v2array_resized += nuy0_int

                if ax_fft_hy:
                    ax1 = ax_fft_hy
                else:
                    fig, ax1 = plt.subplots()

                norm_fft_hAs = fft_d["fft_hAys"] / np.max(fft_d["fft_hAys"], axis=0)

                ylim = nuylim

            if nu_size is not None:
                im = PIL.Image.fromarray((norm_fft_hAs * 255).astype(np.uint8), "L")
                w = norm_fft_hAs.shape[1]
                h = nu_size
                im = im.resize((w, h), resample=PIL.Image.Resampling.LANCZOS)
                im = np.array(list(im.getdata())).reshape((h, w))
                norm_fft_hAs_resized = im / np.max(im, axis=0)

            V1, V2 = np.meshgrid(v1array, v2array)
            if v2array_resized is not None:
                V1r, V2r = np.meshgrid(v1array, v2array_resized)

            if not use_log:
                if v2array_resized is not None:
                    plt.pcolor(
                        V1r * 1e2, V2r, norm_fft_hAs_resized, cmap="jet", shading="auto"
                    )
                else:
                    plt.pcolor(V1 * 1e2, V2, norm_fft_hAs, cmap="jet", shading="auto")
            else:
                if v2array_resized is not None:
                    plt.pcolor(
                        V1r * 1e2,
                        V2r,
                        np.log10(norm_fft_hAs_resized),
                        cmap="jet",
                        shading="auto",
                    )
                else:
                    plt.pcolor(
                        V1 * 1e2, V2, np.log10(norm_fft_hAs), cmap="jet", shading="auto"
                    )

            plt.xlabel(rf"$\delta\, [\%]$", size=font_sz)
            plt.ylabel(rf"$\nu_{_nu_plane}$", size=font_sz)
            if not use_full_ylim:
                ax1.set_ylim(ylim)
            cb = plt.colorbar()
            cb.ax.set_title(EQ_STR)
            cb.ax.title.set_position((0.5, 1.02))
            plt.tight_layout()

            if plot_shifted_curves:
                # The following section plots slightly shifted FFT curves for better
                # curve height visualization than the pcolor plot above.

                M = np.zeros((500, 500))
                xarray = np.linspace(np.min(V2[:, 0]), np.max(V2[:, 0]), M.shape[1])
                max_y_offset = 5.0
                offset_frac = max_y_offset / V1.shape[1]
                yarray = np.linspace(0.0, 1.0 + max_y_offset, M.shape[0])
                M[yarray <= max_y_offset, :] = 1e-6
                for j in range(V1.shape[1]):
                    if np.any(np.isnan(norm_fft_hAs[:, j])):
                        assert np.all(np.isnan(norm_fft_hAs[:, j]))
                        continue

                    yinds = np.argmin(
                        np.abs(
                            yarray.reshape((-1, 1))
                            - (norm_fft_hAs[:, j] + j * offset_frac)
                        ),
                        axis=0,
                    )
                    xinds = np.argmin(
                        np.abs(xarray.reshape((-1, 1)) - V2[:, j]), axis=0
                    )

                    for dix in [-1, 0, 1]:
                        for diy in [-1, 0, 1]:
                            M_new = np.zeros_like(M)
                            M_new[
                                np.clip(yinds + diy, 0, M.shape[0] - 1),
                                np.clip(xinds + dix, 0, M.shape[1] - 1),
                            ] = norm_fft_hAs[:, j]

                            M_comb = np.dstack((M, M_new))
                            M = np.max(M_comb, axis=-1)

                real_yarray = (
                    np.min(v1array)
                    + (np.max(v1array) - np.min(v1array)) / max_y_offset * yarray
                )
                X, Y = np.meshgrid(xarray, real_yarray)
                M[M == 0.0] = np.nan

                plt.figure()
                plt.pcolor(Y.T * 1e2, X.T, M.T, cmap="jet", shading="auto")
                plt.xlabel(rf"$\delta\, [\%]$", size=font_sz)
                plt.ylabel(rf"$\nu_{_nu_plane}$", size=font_sz)
                if not use_full_ylim:
                    ax1.set_ylim(ylim)
                cb = plt.colorbar()
                cb.ax.set_title(EQ_STR)
                cb.ax.title.set_position((0.5, 1.02))
                plt.tight_layout()

    return ret


def calc_tswa_x(
    output_filepath,
    LTE_filepath,
    E_MeV,
    abs_xmax,
    nx,
    xsign="+",
    use_sddsnaff=None,
    courant_snyder=None,
    method="sddsnaff",
    return_fft_spec=True,
    save_tbt=True,
    n_turns=256,
    y0_offset=1e-5,
    use_beamline=None,
    N_KICKS=None,
    transmute_elements=None,
    ele_filepath=None,
    output_file_type=None,
    del_tmp_files=True,
    print_cmd=False,
    run_local=True,
    remote_opts=None,
    err_log_check=None,
    nMaxRemoteRetry=3,
):
    """
    If "err_log_check" is None, then "nMaxRemoteRetry" is irrelevant.
    """

    assert method in ("sddsnaff", "DFT_Courant_Snyder", "DFT_phase_space")

    use_sddsnaff, courant_snyder, method = _deprecated_msg_use_sddsnaff(
        use_sddsnaff, courant_snyder, method, return_fft_spec
    )

    if xsign == "+":
        x0_array = np.linspace(0.0, abs_xmax, nx)[1:]  # exclude x == 0.0
    elif xsign == "-":
        x0_array = np.linspace(0.0, abs_xmax * (-1), nx)[1:]  # exclude x == 0.0
    else:
        raise ValueError('`xsign` must be either "+" or "-".')
    y0_array = np.full(x0_array.shape, y0_offset)

    plane_specific_input = dict(
        abs_xmax=abs_xmax, nx=nx, xsign=xsign, y0_offset=y0_offset
    )

    return _calc_tswa(
        "x",
        plane_specific_input,
        output_filepath,
        LTE_filepath,
        E_MeV,
        x0_array,
        y0_array,
        use_sddsnaff=use_sddsnaff,
        courant_snyder=courant_snyder,
        method=method,
        return_fft_spec=return_fft_spec,
        save_tbt=save_tbt,
        n_turns=n_turns,
        use_beamline=use_beamline,
        N_KICKS=N_KICKS,
        transmute_elements=transmute_elements,
        ele_filepath=ele_filepath,
        output_file_type=output_file_type,
        del_tmp_files=del_tmp_files,
        print_cmd=print_cmd,
        run_local=run_local,
        remote_opts=remote_opts,
        err_log_check=err_log_check,
        nMaxRemoteRetry=nMaxRemoteRetry,
    )


def calc_tswa_y(
    output_filepath,
    LTE_filepath,
    E_MeV,
    abs_ymax,
    ny,
    ysign="+",
    use_sddsnaff=None,
    courant_snyder=None,
    method="sddsnaff",
    return_fft_spec=True,
    save_tbt=True,
    n_turns=256,
    x0_offset=1e-5,
    use_beamline=None,
    N_KICKS=None,
    transmute_elements=None,
    ele_filepath=None,
    output_file_type=None,
    del_tmp_files=True,
    print_cmd=False,
    run_local=True,
    remote_opts=None,
    err_log_check=None,
    nMaxRemoteRetry=3,
):
    """
    If "err_log_check" is None, then "nMaxRemoteRetry" is irrelevant.
    """

    assert method in ("sddsnaff", "DFT_Courant_Snyder", "DFT_phase_space")

    use_sddsnaff, courant_snyder, method = _deprecated_msg_use_sddsnaff(
        use_sddsnaff, courant_snyder, method, return_fft_spec
    )

    if ysign == "+":
        y0_array = np.linspace(0.0, abs_ymax, ny)[1:]  # exclude y == 0.0
    elif ysign == "-":
        y0_array = np.linspace(0.0, abs_ymax * (-1), ny)[1:]  # exclude y == 0.0
    else:
        raise ValueError('`ysign` must be either "+" or "-".')
    x0_array = np.full(y0_array.shape, x0_offset)

    plane_specific_input = dict(
        abs_ymax=abs_ymax, ny=ny, ysign=ysign, x0_offset=x0_offset
    )

    return _calc_tswa(
        "y",
        plane_specific_input,
        output_filepath,
        LTE_filepath,
        E_MeV,
        x0_array,
        y0_array,
        use_sddsnaff=use_sddsnaff,
        courant_snyder=courant_snyder,
        method=method,
        return_fft_spec=return_fft_spec,
        save_tbt=save_tbt,
        n_turns=n_turns,
        use_beamline=use_beamline,
        N_KICKS=N_KICKS,
        transmute_elements=transmute_elements,
        ele_filepath=ele_filepath,
        output_file_type=output_file_type,
        del_tmp_files=del_tmp_files,
        print_cmd=print_cmd,
        run_local=run_local,
        remote_opts=remote_opts,
        err_log_check=err_log_check,
        nMaxRemoteRetry=nMaxRemoteRetry,
    )


def _calc_tswa(
    scan_plane,
    plane_specific_input,
    output_filepath,
    LTE_filepath,
    E_MeV,
    x0_array,
    y0_array,
    use_sddsnaff=None,
    courant_snyder=None,
    method="sddsnaff",
    return_fft_spec=True,
    save_tbt=True,
    n_turns=256,
    use_beamline=None,
    N_KICKS=None,
    transmute_elements=None,
    ele_filepath=None,
    output_file_type=None,
    del_tmp_files=True,
    print_cmd=False,
    run_local=True,
    remote_opts=None,
    err_log_check=None,
    nMaxRemoteRetry=3,
):
    """
    If "err_log_check" is None, then "nMaxRemoteRetry" is irrelevant.
    """

    assert method in ("sddsnaff", "DFT_Courant_Snyder", "DFT_phase_space")

    use_sddsnaff, courant_snyder, method = _deprecated_msg_use_sddsnaff(
        use_sddsnaff, courant_snyder, method, return_fft_spec
    )

    assert x0_array.size == y0_array.size
    nscan = x0_array.size

    LTE_file_pathobj = Path(LTE_filepath)

    file_contents = LTE_file_pathobj.read_text()

    input_dict = dict(
        scan_plane=scan_plane,
        plane_specific_input=plane_specific_input,
        LTE_filepath=str(LTE_file_pathobj.resolve()),
        E_MeV=E_MeV,
        x0_array=x0_array,
        y0_array=y0_array,
        use_sddsnaff=use_sddsnaff,
        courant_snyder=courant_snyder,
        return_fft_spec=return_fft_spec,
        save_tbt=save_tbt,
        n_turns=n_turns,
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
            dir=Path.cwd(), delete=False, prefix=f"tmpTSwA_", suffix=".ele"
        )
        ele_pathobj = Path(tmp.name)
        ele_filepath = str(ele_pathobj.resolve())
        tmp.close()

    watch_pathobj = ele_pathobj.with_suffix(".wc")
    twi_pgz_pathobj = ele_pathobj.with_suffix(".twi.pgz")

    ed = elebuilder.EleDesigner(ele_filepath, double_format=".12g")

    elebuilder.add_transmute_blocks(ed, transmute_elements)

    ed.add_newline()

    ed.add_block(
        "run_setup",
        lattice=LTE_filepath,
        p_central_mev=E_MeV,
        use_beamline=use_beamline,
    )

    ed.add_newline()

    temp_watch_elem_name = "ELEGANT_TSWA_WATCH"
    if run_local:
        watch_filepath = str(watch_pathobj.resolve())
    else:
        watch_filepath = watch_pathobj.name
    temp_watch_elem_def = (
        f'{temp_watch_elem_name}: WATCH, FILENAME="{watch_filepath}", '
        "MODE=coordinate"
    )

    ed.add_block(
        "insert_elements",
        name="*",
        exclude="*",
        add_at_start=True,
        element_def=temp_watch_elem_def,
    )

    ed.add_newline()

    elebuilder.add_N_KICKS_alter_elements_blocks(ed, N_KICKS)

    ed.add_newline()

    ed.add_block("run_control", n_passes=n_turns)

    ed.add_newline()

    centroid = {}
    centroid[0] = "<x0>"
    centroid[2] = "<y0>"
    centroid[5] = 0.0
    #
    ed.add_block("bunched_beam", n_particles_per_bunch=1, centroid=centroid)

    ed.add_newline()

    ed.add_block("track")

    ed.write()
    # print(ed.actual_output_filepath_list)

    twiss.calc_ring_twiss(
        str(twi_pgz_pathobj),
        LTE_filepath,
        E_MeV,
        use_beamline=use_beamline,
        parameters=None,
        run_local=True,
        del_tmp_files=True,
    )
    _d = util.load_pgz_file(str(twi_pgz_pathobj))
    nux0 = _d["data"]["twi"]["scalars"]["nux"]
    nuy0 = _d["data"]["twi"]["scalars"]["nuy"]
    # alpha & beta at watch element (at the start of the lattice)
    betax = _d["data"]["twi"]["arrays"]["betax"][0]
    betay = _d["data"]["twi"]["arrays"]["betay"][0]
    alphax = _d["data"]["twi"]["arrays"]["alphax"][0]
    alphay = _d["data"]["twi"]["arrays"]["alphay"][0]
    twi_pgz_pathobj.unlink()

    # Run Elegant
    if run_local:
        tbt = dict(
            x=np.full((n_turns, nscan), np.nan), y=np.full((n_turns, nscan), np.nan)
        )
        if method in ("sddsnaff", "DFT_Courant_Snyder"):
            tbt["xp"] = np.full((n_turns, nscan), np.nan)
            tbt["yp"] = np.full((n_turns, nscan), np.nan)

        # tElapsed = dict(run_ele=0.0, sdds2dicts=0.0, tbt_population=0.0)

        for i, (x0, y0) in enumerate(zip(x0_array, y0_array)):
            # t0 = time.time()
            run(
                ele_filepath,
                print_cmd=print_cmd,
                macros=dict(x0=f"{x0:.12g}", y0=f"{y0:.12g}"),
                print_stdout=std_print_enabled["out"],
                print_stderr=std_print_enabled["err"],
            )
            # tElapsed['run_ele'] += time.time() - t0

            # t0 = time.time()
            output, _ = sdds.sdds2dicts(watch_pathobj)
            # tElapsed['sdds2dicts'] += time.time() - t0

            # t0 = time.time()
            cols = output["columns"]
            for k in list(tbt):
                tbt[k][: len(cols[k]), i] = cols[k]
            # tElapsed['tbt_population'] += time.time() - t0
    else:

        if remote_opts is None:
            remote_opts = dict(ntasks=20)

        xy0_array = np.vstack((x0_array, y0_array)).T

        remote_opts["ntasks"] = min([len(xy0_array), remote_opts["ntasks"]])

        xy0_sub_array_list, reverse_mapping = util.chunk_list(
            xy0_array, remote_opts["ntasks"]
        )

        coords_list = ["x", "y"]
        if method in ("sddsnaff", "DFT_Courant_Snyder"):
            coords_list += ["xp", "yp"]

        module_name = "pyelegant.nonlin"
        func_name = "_calc_tswa_get_tbt"
        iRemoteTry = 0
        while True:
            chunked_results = remote.run_mpi_python(
                remote_opts,
                module_name,
                func_name,
                xy0_sub_array_list,
                (
                    ele_pathobj.read_text(),
                    ele_pathobj.name,
                    watch_pathobj.name,
                    print_cmd,
                    std_print_enabled["out"],
                    std_print_enabled["err"],
                    coords_list,
                ),
                err_log_check=err_log_check,
            )

            if (err_log_check is not None) and isinstance(chunked_results, str):

                err_log_text = chunked_results
                print("\n** Error Log check found the following problem:")
                print(err_log_text)

                iRemoteTry += 1

                if iRemoteTry >= nMaxRemoteRetry:
                    raise RuntimeError(
                        "Max number of remote tries exceeded. Check the error logs."
                    )
                else:
                    print("\n** Re-trying the remote run...\n")
                    sys.stdout.flush()
            else:
                break

        tbt_chunked_list = dict()
        tbt_flat_list = dict()
        for plane in coords_list:
            tbt_chunked_list[plane] = [_d[plane] for _d in chunked_results]
            tbt_flat_list[plane] = util.unchunk_list_of_lists(
                tbt_chunked_list[plane], reverse_mapping
            )

        tbt = dict(
            x=np.full((n_turns, nscan), np.nan), y=np.full((n_turns, nscan), np.nan)
        )
        if method in ("sddsnaff", "DFT_Courant_Snyder"):
            tbt["xp"] = np.full((n_turns, nscan), np.nan)
            tbt["yp"] = np.full((n_turns, nscan), np.nan)
        for plane in coords_list:
            for iXY, array in enumerate(tbt_flat_list[plane]):
                tbt[plane][: len(array), iXY] = array

    # print(tElapsed)

    survived = np.all(~np.isnan(tbt["x"]), axis=0)

    # t0 = time.time()
    # Estimate tunes and amplitudes from TbT data
    if method == "sddsnaff":
        tmp = tempfile.NamedTemporaryFile(
            dir=None, delete=False, prefix=f"tmpTbt_", suffix=".sdds"
        )
        tbt_sdds_path = Path(tmp.name)
        naff_sdds_path = tbt_sdds_path.parent.joinpath(f"{tbt_sdds_path.stem}.naff")
        tmp.close()

        pass_array = np.array(range(tbt["x"].shape[0]))
        nus = dict(
            x=np.full(tbt["x"].shape[1], np.nan),
            y=np.full(tbt["x"].shape[1], np.nan),
        )
        As = dict(
            x=np.full(tbt["x"].shape[1], np.nan),
            y=np.full(tbt["x"].shape[1], np.nan),
        )
        for iXY, (x, xp, y, yp) in enumerate(
            zip(tbt["x"].T, tbt["xp"].T, tbt["y"].T, tbt["yp"].T)
        ):
            sdds.dicts2sdds(
                tbt_sdds_path,
                columns=dict(Pass=pass_array, x=x, xp=xp, y=y, yp=yp),
                outputMode="binary",
                tempdir_path=None,
                suppress_err_msg=True,
            )

            cmd = (
                f"sddsnaff {tbt_sdds_path} {naff_sdds_path} "
                "-column=Pass -pair=x,xp -pair=y,yp -terminate=frequencies=1"
            )
            p = Popen(shlex.split(cmd), stdout=PIPE, stderr=PIPE, encoding="utf-8")
            out, err = p.communicate()
            if out:
                print(f"stdout: {out}")
            if err:
                print(f"stderr: {err}")
            if False:
                cmd = f'sddsprintout {naff_sdds_path} -col="(xFrequency,yFrequency)"'
                p = Popen(shlex.split(cmd), stdout=PIPE, stderr=PIPE, encoding="utf-8")
                out, err = p.communicate()
                if out:
                    print(f"stdout: {out}")
                if err:
                    print(f"stderr: {err}")
            naff_d, naff_meta = sdds.sdds2dicts(naff_sdds_path)

            nus["x"][iXY] = naff_d["columns"]["xFrequency"][0]
            nus["y"][iXY] = naff_d["columns"]["yFrequency"][0]

            As["x"][iXY] = naff_d["columns"]["xAmplitude"][0]
            As["y"][iXY] = naff_d["columns"]["yAmplitude"][0]

        try:
            tbt_sdds_path.unlink()
        except:
            pass
        try:
            naff_sdds_path.unlink()
        except:
            pass

        undefined_tunes = (nus["x"] == -1.0) | (nus["y"] == -1.0)
        nus["x"][undefined_tunes] = np.nan
        nus["y"][undefined_tunes] = np.nan
        As["x"][undefined_tunes] = np.nan
        As["y"][undefined_tunes] = np.nan

        if False:
            other_nus, other_As = calc_tswa_from_tbt_cs(
                scan_plane,
                x0_array,
                y0_array,
                tbt["x"],
                tbt["y"],
                nux0,
                nuy0,
                tbt["xp"],
                tbt["yp"],
                betax,
                alphax,
                betay,
                alphay,
                init_guess_from_prev_step=True,
                return_fft_spec=False,
            )

            iPlane = 0 if scan_plane == "x" else 1

            plt.figure()
            plt.plot(
                xy0_array[:, iPlane] * 1e3,
                other_nus["x"],
                "b.-",
                label="calc_tswa_from_tbt_cs",
            )
            plt.plot(xy0_array[:, iPlane] * 1e3, nus["x"], "r.-", label="sddsnaff")
            plt.xlabel(rf"${scan_plane}_0\, [\mathrm{{mm}}]$", size="large")
            plt.ylabel(r"$\nu_x$", size="large")
            leg = plt.legend(loc="best")
            plt.tight_layout()

            plt.figure()
            plt.plot(
                xy0_array[:, iPlane] * 1e3,
                other_nus["y"],
                "b.-",
                label="calc_tswa_from_tbt_cs",
            )
            plt.plot(xy0_array[:, iPlane] * 1e3, nus["y"], "r.-", label="sddsnaff")
            plt.xlabel(rf"${scan_plane}_0\, [\mathrm{{mm}}]$", size="large")
            plt.ylabel(r"$\nu_y$", size="large")
            leg = plt.legend(loc="best")
            plt.tight_layout()

            plt.figure()
            plt.plot(
                xy0_array[:, iPlane] * 1e3,
                other_As["x"] * 1e3,
                "b.-",
                label="calc_tswa_from_tbt_cs",
            )
            plt.plot(xy0_array[:, iPlane] * 1e3, As["x"] * 1e3, "r.-", label="sddsnaff")
            plt.xlabel(rf"${scan_plane}_0\, [\mathrm{{mm}}]$", size="large")
            plt.ylabel(r"$A_x\, [\mathrm{{mm}}]$", size="large")
            leg = plt.legend(loc="best")
            plt.tight_layout()

            plt.figure()
            plt.plot(
                xy0_array[:, iPlane] * 1e3,
                other_As["y"] * 1e3,
                "b.-",
                label="calc_tswa_from_tbt_cs",
            )
            plt.plot(xy0_array[:, iPlane] * 1e3, As["y"] * 1e3, "r.-", label="sddsnaff")
            plt.xlabel(rf"${scan_plane}_0\, [\mathrm{{mm}}]$", size="large")
            plt.ylabel(r"$A_y\, [\mathrm{{mm}}]$", size="large")
            leg = plt.legend(loc="best")
            plt.tight_layout()

        extra_save_kwargs = dict(xptbt=tbt["xp"], yptbt=tbt["yp"])

        if return_fft_spec:
            *_, fft_nus, fft_hAxs, fft_hAys = calc_tswa_from_tbt_cs(
                scan_plane,
                x0_array,
                y0_array,
                tbt["x"],
                tbt["y"],
                nux0,
                nuy0,
                tbt["xp"],
                tbt["yp"],
                betax,
                alphax,
                betay,
                alphay,
                init_guess_from_prev_step=True,
                return_fft_spec=True,
            )

            extra_save_kwargs["fft_nus"] = fft_nus
            extra_save_kwargs["fft_hAxs"] = fft_hAxs
            extra_save_kwargs["fft_hAys"] = fft_hAys

    elif method == "DFT_Courant_Snyder":
        if return_fft_spec:
            nus, As, fft_nus, fft_hAxs, fft_hAys = calc_tswa_from_tbt_cs(
                scan_plane,
                x0_array,
                y0_array,
                tbt["x"],
                tbt["y"],
                nux0,
                nuy0,
                tbt["xp"],
                tbt["yp"],
                betax,
                alphax,
                betay,
                alphay,
                init_guess_from_prev_step=True,
                return_fft_spec=True,
            )
            extra_save_kwargs = dict(
                xptbt=tbt["xp"],
                yptbt=tbt["yp"],
                fft_nus=fft_nus,
                fft_hAxs=fft_hAxs,
                fft_hAys=fft_hAys,
            )
        else:
            nus, As = calc_tswa_from_tbt_cs(
                scan_plane,
                x0_array,
                y0_array,
                tbt["x"],
                tbt["y"],
                nux0,
                nuy0,
                tbt["xp"],
                tbt["yp"],
                betax,
                alphax,
                betay,
                alphay,
                init_guess_from_prev_step=True,
                return_fft_spec=False,
            )
            extra_save_kwargs = dict(xptbt=tbt["xp"], yptbt=tbt["yp"])

        undefined_tunes = np.isnan(nus["x"]) | np.isnan(nus["y"])
    elif method == "DFT_phase_space":
        nus, As = calc_tswa_from_tbt_ps(
            scan_plane, x0_array, y0_array, tbt["x"], tbt["y"], nux0, nuy0
        )
        extra_save_kwargs = {}

        undefined_tunes = np.isnan(nus["x"]) | np.isnan(nus["y"])
    else:
        raise ValueError(f"method={method}")
    nuxs, nuys = nus["x"], nus["y"]
    Axs, Ays = As["x"], As["y"]
    time_domain_Axs = np.std(tbt["x"], axis=0, ddof=1) * np.sqrt(2)
    time_domain_Ays = np.std(tbt["y"], axis=0, ddof=1) * np.sqrt(2)
    # print('* Time elapsed for tune/amplitude estimation: {:.3f}'.format(time.time() - t0))

    timestamp_fin = util.get_current_local_time_str()

    _save_tswa_data(
        output_filepath,
        output_file_type,
        x0_array,
        y0_array,
        tbt["x"],
        tbt["y"],
        betax,
        alphax,
        betay,
        alphay,
        nux0,
        nuy0,
        nuxs,
        nuys,
        Axs,
        Ays,
        time_domain_Axs,
        time_domain_Ays,
        survived,
        undefined_tunes,
        timestamp_fin,
        input_dict,
        save_tbt=save_tbt,
        **extra_save_kwargs,
    )

    if del_tmp_files:
        util.delete_temp_files(
            ed.actual_output_filepath_list + [ele_filepath, str(watch_pathobj)]
        )

    return output_filepath


def _calc_tswa_get_tbt(
    xy0_sub_array_list,
    ele_contents,
    ele_filename,
    watch_filename,
    print_cmd,
    print_stdout,
    print_stderr,
    coords_list,
    tempdir_path="/tmp",
):
    """"""

    if not Path(tempdir_path).exists():
        tempdir_path = Path.cwd()

    sub_tbt = dict()
    for k in coords_list:
        sub_tbt[k] = []

    with tempfile.TemporaryDirectory(
        prefix="tmpCalcTSwA_", dir=tempdir_path
    ) as tmpdirname:

        ele_pathobj = Path(tmpdirname).joinpath(ele_filename)
        watch_pathobj = Path(tmpdirname).joinpath(watch_filename)

        ele_contents = ele_contents.replace(
            watch_filename, str(watch_pathobj.resolve())
        )

        ele_pathobj.write_text(ele_contents)

        ele_filepath = str(ele_pathobj.resolve())

        for x0, y0 in xy0_sub_array_list:

            run(
                ele_filepath,
                print_cmd=print_cmd,
                macros=dict(x0=f"{x0:.12g}", y0=f"{y0:.12g}"),
                print_stdout=print_stdout,
                print_stderr=print_stderr,
            )

            output, _ = sdds.sdds2dicts(watch_pathobj)

            cols = output["columns"]
            for k in list(sub_tbt):
                sub_tbt[k].append(cols[k])

    return sub_tbt


def calc_tswa_from_tbt_ps(scan_plane, x0_array, y0_array, xtbt, ytbt, nux0, nuy0):
    """
    Using phase-space (ps) variables "x" and "y", which can only
    determine tunes within the range of [0, 0.5].
    """

    assert x0_array.shape == y0_array.shape

    nus = dict(x=np.full(x0_array.shape, np.nan), y=np.full(x0_array.shape, np.nan))
    As = dict(x=np.full(x0_array.shape, np.nan), y=np.full(x0_array.shape, np.nan))

    frac_nux0 = nux0 - np.floor(nux0)
    frac_nuy0 = nuy0 - np.floor(nuy0)

    n_turns, nscans = xtbt.shape
    nu_vec = np.fft.fftfreq(n_turns)

    opts = dict(window="sine", resolution=1e-8)
    init_nux = frac_nux0
    init_nuy = frac_nuy0
    for i in range(nscans):
        xarray = xtbt[:, i]
        yarray = ytbt[:, i]

        if np.any(np.isnan(xarray)) or np.any(np.isnan(yarray)):
            # Particle lost at some point.
            continue

        if False:
            # This algorithm does NOT work too well if tune change
            # between neighboring delta points are too large.
            out = sigproc.getDftPeak(xarray, init_nux, **opts)
            nus["x"][i] = out["nu"]
            As["x"][i] = out["A"]
            init_nux = out["nu"]

            out = sigproc.getDftPeak(yarray, init_nuy, **opts)
            nus["y"][i] = out["nu"]
            As["y"][i] = out["A"]
            init_nuy = out["nu"]
        else:
            # Find the rough peak first
            ff_rect = np.fft.fft(xarray - np.mean(xarray))
            A_arb = np.abs(ff_rect)
            init_nux = nu_vec[np.argmax(A_arb[: (n_turns // 2)])]
            # Then fine-tune
            out = sigproc.getDftPeak(xarray, init_nux, **opts)
            nus["x"][i] = out["nu"]
            As["x"][i] = out["A"]

            # Find the rough peak first
            ff_rect = np.fft.fft(yarray - np.mean(yarray))
            A_arb = np.abs(ff_rect)
            init_nuy = nu_vec[np.argmax(A_arb[: (n_turns // 2)])]
            # Then fine-tune
            out = sigproc.getDftPeak(yarray, init_nuy, **opts)
            nus["y"][i] = out["nu"]
            As["y"][i] = out["A"]

    return nus, As


def calc_tswa_from_tbt_cs(
    scan_plane,
    x0_array,
    y0_array,
    xtbt,
    ytbt,
    nux0,
    nuy0,
    xptbt,
    yptbt,
    betax,
    alphax,
    betay,
    alphay,
    init_guess_from_prev_step=True,
    return_fft_spec=True,
):
    """
    Using Courant-Snyder (CS) coordinates "xhat" and "yhat", which enables
    determination of tunes within the range of [0, 1.0].

    If `init_guess_from_prev_step` is True (recommended), then the tune peak
    fine-tuning starts from the tune peak found from the previous amplitude to
    avoid any sudden tune jump along an increasing amplitude vector. If False,
    then a rough tune peak is found from a simple FFT spectrum, which can
    potentially lead the initial search point to another strong resonance peak,
    away from the fundamental tune peak.
    """

    assert x0_array.shape == y0_array.shape

    nus = dict(x=np.full(x0_array.shape, np.nan), y=np.full(x0_array.shape, np.nan))
    As = dict(x=np.full(x0_array.shape, np.nan), y=np.full(x0_array.shape, np.nan))

    frac_nux0 = nux0 - np.floor(nux0)
    frac_nuy0 = nuy0 - np.floor(nuy0)

    if frac_nux0 > 0.5:
        frac_nux0 = frac_nux0 - 1
    if frac_nuy0 > 0.5:
        frac_nuy0 = frac_nuy0 - 1

    n_turns, nscans = xtbt.shape
    nu_vec = np.fft.fftfreq(n_turns)

    opts = dict(window="sine", resolution=1e-8, return_fft_spec=return_fft_spec)
    init_nux = frac_nux0
    init_nuy = frac_nuy0
    if return_fft_spec:
        fft_nus = None
        fft_hAxs = []
        fft_hAys = []
    for i in range(nscans):
        xarray = xtbt[:, i]
        yarray = ytbt[:, i]

        if np.any(np.isnan(xarray)) or np.any(np.isnan(yarray)):
            # Particle lost at some point.
            if return_fft_spec:
                fft_hAxs.append(np.full((n_turns,), np.nan))
                fft_hAys.append(np.full((n_turns,), np.nan))
            continue

        xparray = xptbt[:, i]
        yparray = yptbt[:, i]
        delta = 0.0  # Only on-momentum case is assumed
        pxarray, pyarray = angle2p(xparray, yparray, delta)

        xarray -= np.mean(xarray)
        yarray -= np.mean(yarray)
        pxarray -= np.mean(pxarray)
        pyarray -= np.mean(pyarray)

        xhat = xarray / np.sqrt(betax)
        pxhat = alphax / np.sqrt(betax) * xarray + np.sqrt(betax) * pxarray
        yhat = yarray / np.sqrt(betay)
        pyhat = alphay / np.sqrt(betay) * yarray + np.sqrt(betay) * pyarray

        hx = xhat - 1j * pxhat
        hy = yhat - 1j * pyhat

        if init_guess_from_prev_step:
            rough_peak_nux = sigproc.findNearestFftPeak(
                hx, init_nux, window=opts["window"]
            )["nu"]
            # print(f'init/rough nux: {init_nux:.6f}/{rough_peak_nux:.6f}')
            out = sigproc.getDftPeak(hx, rough_peak_nux, **opts)
            nus["x"][i] = out["nu"]
            init_nux = out["nu"]
        else:
            # Find the rough peak first
            ff_rect = np.fft.fft(hx)
            A_arb = np.abs(ff_rect)
            init_nux = nu_vec[np.argmax(A_arb)]
            # Then fine-tune
            out = sigproc.getDftPeak(hx, init_nux, **opts)
            nus["x"][i] = out["nu"]
        sqrt_twoJx = out["A"]
        As["x"][i] = sqrt_twoJx * np.sqrt(
            betax
        )  # Convert CS amplitude to phase-space amplitude
        if return_fft_spec:
            if fft_nus is None:
                fft_nus = out["fft_nus"]
            fft_hAxs.append(out["fft_As"])

        if init_guess_from_prev_step:
            rough_peak_nuy = sigproc.findNearestFftPeak(
                hy, init_nuy, window=opts["window"]
            )["nu"]
            # print(f'init/rough nuy: {init_nuy:.6f}/{rough_peak_nuy:.6f}')
            out = sigproc.getDftPeak(hy, rough_peak_nuy, **opts)
            nus["y"][i] = out["nu"]
            init_nuy = out["nu"]
        else:
            # Find the rough peak first
            ff_rect = np.fft.fft(hy)
            A_arb = np.abs(ff_rect)
            init_nuy = nu_vec[np.argmax(A_arb)]
            # Then fine-tune
            out = sigproc.getDftPeak(hy, init_nuy, **opts)
            nus["y"][i] = out["nu"]
        sqrt_twoJy = out["A"]
        As["y"][i] = sqrt_twoJy * np.sqrt(
            betay
        )  # Convert CS amplitude to phase-space amplitude
        if return_fft_spec:
            fft_hAys.append(out["fft_As"])

    nonnan_inds = np.where(~np.isnan(nus["x"]))[0]
    neg_inds = nonnan_inds[nus["x"][nonnan_inds] < 0]
    nus["x"][neg_inds] += 1
    nonnan_inds = np.where(~np.isnan(nus["y"]))[0]
    neg_inds = nonnan_inds[nus["y"][nonnan_inds] < 0]
    nus["y"][neg_inds] += 1

    if not return_fft_spec:
        return nus, As
    else:
        return nus, As, fft_nus, np.array(fft_hAxs).T, np.array(fft_hAys).T


def _save_tswa_data(
    output_filepath,
    output_file_type,
    x0_array,
    y0_array,
    xtbt,
    ytbt,
    betax,
    alphax,
    betay,
    alphay,
    nux0,
    nuy0,
    nuxs,
    nuys,
    Axs,
    Ays,
    time_domain_Axs,
    time_domain_Ays,
    survived,
    undefined_tunes,
    timestamp_fin,
    input_dict,
    xptbt=None,
    yptbt=None,
    fft_nus=None,
    fft_hAxs=None,
    fft_hAys=None,
    save_tbt=True,
):
    """ """

    if output_file_type in ("hdf5", "h5"):
        _kwargs = dict(compression="gzip")
        f = h5py.File(output_filepath, "a")
        f["_version_PyELEGANT"] = __version__["PyELEGANT"]
        f["_version_ELEGANT"] = __version__["ELEGANT"]
        f.create_dataset("x0s", data=x0_array, **_kwargs)
        f.create_dataset("y0s", data=y0_array, **_kwargs)
        if save_tbt:
            f.create_dataset("xtbt", data=xtbt, **_kwargs)
            f.create_dataset("ytbt", data=ytbt, **_kwargs)
        f["nux0"] = nux0
        f["nuy0"] = nuy0
        if save_tbt and (xptbt is not None):
            f.create_dataset("xptbt", data=xptbt, **_kwargs)
        if save_tbt and (yptbt is not None):
            f.create_dataset("yptbt", data=xptbt, **_kwargs)
        if fft_nus is not None:
            f.create_dataset("fft_nus", data=fft_nus, **_kwargs)
            f.create_dataset("fft_hAxs", data=fft_hAxs, **_kwargs)
            f.create_dataset("fft_hAys", data=fft_hAys, **_kwargs)
        f["betax"] = betax
        f["betay"] = betay
        f["alphax"] = alphax
        f["alphay"] = alphay
        f.create_dataset("nuxs", data=nuxs, **_kwargs)
        f.create_dataset("nuys", data=nuys, **_kwargs)
        f.create_dataset("Axs", data=Axs, **_kwargs)
        f.create_dataset("Ays", data=Ays, **_kwargs)
        f.create_dataset("time_domain_Axs", data=time_domain_Axs, **_kwargs)
        f.create_dataset("time_domain_Ays", data=time_domain_Ays, **_kwargs)
        f.create_dataset("survived", data=survived, **_kwargs)
        f.create_dataset("undefined_tunes", data=undefined_tunes, **_kwargs)
        f["timestamp_fin"] = timestamp_fin
        f.close()

    elif output_file_type == "pgz":
        d = dict(
            x0s=x0_array,
            y0s=y0_array,
            nux0=nux0,
            nuy0=nuy0,
            betax=betax,
            betay=betay,
            alphax=alphax,
            alphay=alphay,
            nuxs=nuxs,
            nuys=nuys,
            Axs=Axs,
            Ays=Ays,
            time_domain_Axs=time_domain_Axs,
            time_domain_Ays=time_domain_Ays,
            survived=survived,
            undefined_tunes=undefined_tunes,
            input=input_dict,
            timestamp_fin=timestamp_fin,
            _version_PyELEGANT=__version__["PyELEGANT"],
            _version_ELEGANT=__version__["ELEGANT"],
        )
        if save_tbt:
            d["xtbt"] = xtbt
            d["ytbt"] = ytbt
            if xptbt is not None:
                d["xptbt"] = xptbt
            if yptbt is not None:
                d["yptbt"] = yptbt
        if fft_nus is not None:
            d["fft_nus"] = fft_nus
            d["fft_hAxs"] = fft_hAxs
            d["fft_hAys"] = fft_hAys
        d["betax"] = betax
        d["betay"] = betay
        d["alphax"] = alphax
        d["alphay"] = alphay
        util.robust_pgz_file_write(output_filepath, d, nMaxTry=10, sleep=10.0)
    else:
        raise ValueError()


def smooth_nu_int_jump(nu_array, jump_thresh=0.5, ref_index=None):
    """"""

    assert jump_thresh > 0.0

    nu_array = nu_array.copy()

    for _ in range(nu_array.size):
        nu_diffs = np.diff(nu_array)

        nan_inds = np.isnan(nu_diffs)
        nonnan_inds = ~nan_inds
        nonnan_numinds = np.where(nonnan_inds)[0]

        jumped_plus = np.zeros(nu_array.shape).astype(bool)
        jumped_minus = np.zeros(nu_array.shape).astype(bool)

        jumped_plus[nonnan_numinds + 1] = nu_diffs[nonnan_inds] > jump_thresh
        jumped_minus[nonnan_numinds + 1] = nu_diffs[nonnan_inds] < -jump_thresh

        jumped_plus = np.where(jumped_plus)[0]
        jumped_minus = np.where(jumped_minus)[0]
        if jumped_plus.size == 0 and jumped_minus.size == 0:
            break
        elif jumped_plus.size != 0:
            nu_array[jumped_plus[0] :] -= 1.0
        elif jumped_minus.size != 0:
            nu_array[jumped_minus[0] :] += 1.0
        else:
            if jumped_plus[0] < jumped_minus[0]:
                nu_array[jumped_plus[0] :] -= 1.0
            elif jumped_plus[0] > jumped_minus[0]:
                nu_array[jumped_minus[0] :] += 1.0
            else:
                raise RuntimeError("This should not happen. Check algorithm.")
    else:
        raise RuntimeError(
            "Max # of shifting exceeded. This should not happen. Check algorithm."
        )

    if ref_index is not None:
        ref_nu = nu_array[ref_index]
        if ref_nu >= 0.0:
            nu_array -= np.floor(ref_nu)
        else:
            nu_array -= np.ceil(ref_nu)

    return nu_array


def plot_tswa(
    output_filepath,
    title="",
    fit_abs_xmax=None,
    fit_abs_ymax=None,
    plot_xy0=True,
    x0lim=None,
    y0lim=None,
    plot_Axy=False,
    use_time_domain_amplitude=True,
    Axlim=None,
    Aylim=None,
    nuxlim=None,
    nuylim=None,
    footprint_nuxlim=None,
    footprint_nuylim=None,
    max_resonance_line_order=5,
    ax_nu_vs_xy0=None,
    ax_nu_vs_A=None,
    ax_nuy_vs_nux=None,
    plot_fft=False,
    fft_plot_opts=None,
    ax_fft_hx=None,
    ax_fft_hy=None,
):
    """"""

    assert max_resonance_line_order <= 5

    ret = {}  # variable to be returned

    is_nuxlim_frac = False
    if nuxlim is not None:
        if (0.0 <= nuxlim[0] <= 1.0) and (0.0 <= nuxlim[1] <= 1.0):
            is_nuxlim_frac = True

    is_nuylim_frac = False
    if nuylim is not None:
        if (0.0 <= nuylim[0] <= 1.0) and (0.0 <= nuylim[1] <= 1.0):
            is_nuylim_frac = True

    if is_nuxlim_frac and is_nuylim_frac:
        _plot_nu_frac = True
    elif (not is_nuxlim_frac) and (not is_nuylim_frac):
        _plot_nu_frac = False
    else:
        raise ValueError(
            '"nuxlim" and "nuylim" must be either both fractional or both non-fractional'
        )

    try:
        d = util.load_pgz_file(output_filepath)
        scan_plane = d["input"]["scan_plane"]
        x0s, y0s = d["x0s"], d["y0s"]
        nuxs, nuys = d["nuxs"], d["nuys"]
        Axs, Ays = d["Axs"], d["Ays"]
        time_domain_Axs, time_domain_Ays = d["time_domain_Axs"], d["time_domain_Ays"]
        nux0, nuy0 = d["nux0"], d["nuy0"]
        betax, betay = d["betax"], d["betay"]
        # alphax, alphay = d['alphax'], d['alphay']
        if "fft_nus" in d:
            fft_d = {k: d[k] for k in ["fft_nus", "fft_hAxs", "fft_hAys"]}
        else:
            fft_d = None
    except:
        f = h5py.File(output_filepath, "r")
        scan_plane = f["input"]["scan_plane"][()]
        x0s = f["x0s"][()]
        y0s = f["y0s"][()]
        nuxs = f["nuxs"][()]
        nuys = f["nuys"][()]
        Axs = f["Axs"][()]
        Ays = f["Ays"][()]
        time_domain_Axs = f["time_domain_Axs"][()]
        time_domain_Ays = f["time_domain_Ays"][()]
        nux0 = f["nux0"][()]
        nuy0 = f["nuy0"][()]
        betax = f["betax"][()]
        betay = f["betay"][()]
        # alphax = f['alphax'][()]
        # alphay = f['alphay'][()]
        if "fft_nus" in f:
            fft_d = {k: f[k][()] for k in ["fft_nus", "fft_hAxs", "fft_hAys"]}
        else:
            fft_d = None
        f.close()

    if use_time_domain_amplitude:
        Axs = time_domain_Axs
        Ays = time_domain_Ays

    nux0_int = np.floor(nux0)
    nuy0_int = np.floor(nuy0)

    if False:
        nuxs = smooth_nu_int_jump(nuxs, jump_thresh=0.5)
        nuys = smooth_nu_int_jump(nuys, jump_thresh=0.5)

        nuxs += nux0_int
        nuys += nuy0_int

        if scan_plane == "x":
            v0s = x0s
        elif scan_plane == "y":
            v0s = y0s
        else:
            raise ValueError
        # Correct nuxs if smoothing shifted from nux0 by ~1
        for i in np.argsort(np.abs(v0s)):
            if np.isnan(nuxs[i]):
                continue
            else:
                if nuxs[i] > nux0 + 0.5:
                    nuxs -= 1
                elif nuxs[i] < nux0 - 0.5:
                    nuxs += 1
                break
        # Correct nuys if smoothing shifted from nuy0 by ~1
        for i in np.argsort(np.abs(v0s)):
            if np.isnan(nuys[i]):
                continue
            else:
                if nuys[i] > nuy0 + 0.5:
                    nuys -= 1
                elif nuys[i] < nuy0 - 0.5:
                    nuys += 1
                break
    else:
        if scan_plane == "x":
            v0s = x0s
        elif scan_plane == "y":
            v0s = y0s
        else:
            raise ValueError

        on_axis_index = np.argmin(np.abs(v0s))
        _kwargs = dict(jump_thresh=0.5, ref_index=on_axis_index)
        nuxs = smooth_nu_int_jump(nuxs, **_kwargs)
        nuys = smooth_nu_int_jump(nuys, **_kwargs)

        nuxs += nux0_int
        nuys += nuy0_int

    twoJxs = Axs**2 / betax
    twoJys = Ays**2 / betay
    Jxs = twoJxs / 2
    Jys = twoJys / 2

    twoJx0s = x0s**2 / betax
    twoJy0s = y0s**2 / betay
    Jx0s = twoJx0s / 2
    Jy0s = twoJy0s / 2

    beta_str = rf"\{{(\beta_x, \beta_y) [\mathrm{{m}}] = ({betax:.2f}, {betay:.2f})\}}"

    if scan_plane == "x":
        if np.sign(x0s[-1]) > 0:
            fit_roi = x0s <= fit_abs_xmax
            side = "+"
            scan_sign = +1.0
            scan_sign_str = ""
            scan_sign_beta_str = (
                "(x_0 > 0)$"
                "\n"
                rf"$(\beta_x, \beta_y) [\mathrm{{m}}] = ({betax:.2f}, {betay:.2f})"
            )
        else:
            fit_roi = x0s >= fit_abs_xmax * (-1)
            side = "-"
            scan_sign = -1.0
            scan_sign_str = "-"
            scan_sign_beta_str = (
                "(x_0 < 0)$"
                "\n"
                rf"$(\beta_x, \beta_y) [\mathrm{{m}}] = ({betax:.2f}, {betay:.2f})"
            )

        ret[side] = {}

        if plot_xy0:
            coeffs = np.polyfit(Jx0s[fit_roi], nuxs[fit_roi], 1)
            dnux_dJx0 = coeffs[0]
            ret[side]["dnux_dJx0"] = dnux_dJx0
            nux_fit0 = np.poly1d(coeffs)

            coeffs = np.polyfit(Jx0s[fit_roi], nuys[fit_roi], 1)
            dnuy_dJx0 = coeffs[0]
            ret[side]["dnuy_dJx0"] = dnuy_dJx0
            nuy_fit0 = np.poly1d(coeffs)

            if False:  # Not being used and will generate warning about poor fit
                # because the elements of Jy0s are all the same
                dnux_dJy0 = np.polyfit(Jy0s[fit_roi], nuxs[fit_roi], 1)[0]
                dnuy_dJy0 = np.polyfit(Jy0s[fit_roi], nuys[fit_roi], 1)[0]

            x0_fit = np.linspace(np.min(np.abs(x0s)), np.max(np.abs(x0s)), 101)
            Jx0_fit = (x0_fit**2 / betax) / 2

            dnux_dJx0_str = util.pprint_sci_notation(dnux_dJx0, "+.3g")
            dnuy_dJx0_str = util.pprint_sci_notation(dnuy_dJx0, "+.3g")

            fit0_label = dict(
                nux=rf"$d\nu_x / d J_x = {dnux_dJx0_str}\, [\mathrm{{m}}^{{-1}}]$",
                nuy=rf"$d\nu_y / d J_x = {dnuy_dJx0_str}\, [\mathrm{{m}}^{{-1}}]$",
            )

        if plot_Axy:
            coeffs = np.polyfit(Jxs[fit_roi], nuxs[fit_roi], 1)
            dnux_dJx = coeffs[0]
            ret[side]["dnux_dJx"] = dnux_dJx
            nux_fit = np.poly1d(coeffs)

            coeffs = np.polyfit(Jxs[fit_roi], nuys[fit_roi], 1)
            dnuy_dJx = coeffs[0]
            ret[side]["dnuy_dJx"] = dnuy_dJx
            nuy_fit = np.poly1d(coeffs)

            if False:  # Not being used
                dnux_dJy = np.polyfit(Jys[fit_roi], nuxs[fit_roi], 1)[0]
                dnuy_dJy = np.polyfit(Jys[fit_roi], nuys[fit_roi], 1)[0]

            Ax_fit = np.linspace(np.min(Axs), np.max(Axs), 101)
            Jx_fit = (Ax_fit**2 / betax) / 2

            dnux_dJx_str = util.pprint_sci_notation(dnux_dJx, "+.3g")
            dnuy_dJx_str = util.pprint_sci_notation(dnuy_dJx, "+.3g")

            fit_label = dict(
                nux=rf"$d\nu_x / d J_x = {dnux_dJx_str}\, [\mathrm{{m}}^{{-1}}]$",
                nuy=rf"$d\nu_y / d J_x = {dnuy_dJx_str}\, [\mathrm{{m}}^{{-1}}]$",
            )

    elif scan_plane == "y":
        if np.sign(y0s[-1]) > 0:
            fit_roi = y0s <= fit_abs_ymax
            side = "+"
            scan_sign = +1.0
            scan_sign_str = ""
            scan_sign_beta_str = (
                "(y_0 > 0)$"
                "\n"
                rf"$(\beta_x, \beta_y) [\mathrm{{m}}] = ({betax:.2f}, {betay:.2f})"
            )
        else:
            fit_roi = y0s >= fit_abs_ymax * (-1)
            side = "-"
            scan_sign = -1.0
            scan_sign_str = "-"
            scan_sign_beta_str = (
                "(y_0 < 0)$"
                "\n"
                rf"$(\beta_x, \beta_y) [\mathrm{{m}}] = ({betax:.2f}, {betay:.2f})"
            )

        ret[side] = {}

        if plot_xy0:
            coeffs = np.polyfit(Jy0s[fit_roi], nuxs[fit_roi], 1)
            dnux_dJy0 = coeffs[0]
            ret[side]["dnux_dJy0"] = dnux_dJy0
            nux_fit0 = np.poly1d(coeffs)

            coeffs = np.polyfit(Jy0s[fit_roi], nuys[fit_roi], 1)
            dnuy_dJy0 = coeffs[0]
            ret[side]["dnuy_dJy0"] = dnuy_dJy0
            nuy_fit0 = np.poly1d(coeffs)

            if False:  # Not being used and will generate warning about poor fit
                # because the elements of Jy0s are all the same
                dnux_dJx0 = np.polyfit(Jx0s[fit_roi], nuxs[fit_roi], 1)[0]
                dnuy_dJx0 = np.polyfit(Jx0s[fit_roi], nuys[fit_roi], 1)[0]

            y0_fit = np.linspace(np.min(np.abs(y0s)), np.max(np.abs(y0s)), 101)
            Jy0_fit = (y0_fit**2 / betay) / 2

            dnux_dJy0_str = util.pprint_sci_notation(dnux_dJy0, "+.3g")
            dnuy_dJy0_str = util.pprint_sci_notation(dnuy_dJy0, "+.3g")

            fit0_label = dict(
                nux=rf"$d\nu_x / d J_y = {dnux_dJy0_str}\, [\mathrm{{m}}^{{-1}}]$",
                nuy=rf"$d\nu_y / d J_y = {dnuy_dJy0_str}\, [\mathrm{{m}}^{{-1}}]$",
            )

        if plot_Axy:
            coeffs = np.polyfit(Jys[fit_roi], nuxs[fit_roi], 1)
            dnux_dJy = coeffs[0]
            ret[side]["dnux_dJy"] = dnux_dJy
            nux_fit = np.poly1d(coeffs)

            coeffs = np.polyfit(Jys[fit_roi], nuys[fit_roi], 1)
            dnuy_dJy = coeffs[0]
            ret[side]["dnuy_dJy"] = dnuy_dJy
            nuy_fit = np.poly1d(coeffs)

            if False:  # Not being used
                dnux_dJx = np.polyfit(Jxs[fit_roi], nuxs[fit_roi], 1)[0]
                dnuy_dJx = np.polyfit(Jxs[fit_roi], nuys[fit_roi], 1)[0]

            Ay_fit = np.linspace(np.min(Ays), np.max(Ays), 101)
            Jy_fit = (Ay_fit**2 / betay) / 2

            dnux_dJy_str = util.pprint_sci_notation(dnux_dJy, "+.3g")
            dnuy_dJy_str = util.pprint_sci_notation(dnuy_dJy, "+.3g")

            fit_label = dict(
                nux=rf"$d\nu_x / d J_y = {dnux_dJy_str}\, [\mathrm{{m}}^{{-1}}]$",
                nuy=rf"$d\nu_y / d J_y = {dnuy_dJy_str}\, [\mathrm{{m}}^{{-1}}]$",
            )

    else:
        raise ValueError

    if _plot_nu_frac:
        if plot_xy0:
            offset0 = dict(nux=np.floor(nuxs), nuy=np.floor(nuys))
            if scan_plane == "x":
                offset0["fit_nux"] = np.floor(nux_fit0(Jx0_fit))
                offset0["fit_nuy"] = np.floor(nuy_fit0(Jx0_fit))
            elif scan_plane == "y":
                offset0["fit_nux"] = np.floor(nux_fit0(Jy0_fit))
                offset0["fit_nuy"] = np.floor(nuy_fit0(Jy0_fit))
        if plot_Axy:
            offset = dict(nux=np.floor(nuxs), nuy=np.floor(nuys))
            if scan_plane == "x":
                offset["fit_nux"] = np.floor(nux_fit(Jx_fit))
                offset["fit_nuy"] = np.floor(nuy_fit(Jx_fit))
            elif scan_plane == "y":
                offset["fit_nux"] = np.floor(nux_fit(Jy_fit))
                offset["fit_nuy"] = np.floor(nuy_fit(Jy_fit))
    else:
        if plot_xy0:
            offset0 = dict(nux=np.zeros(nuxs.shape), nuy=np.zeros(nuys.shape))
            if scan_plane == "x":
                offset0["fit_nux"] = offset0["fit_nuy"] = np.zeros(Jx0_fit.shape)
            elif scan_plane == "y":
                offset0["fit_nux"] = offset0["fit_nuy"] = np.zeros(Jy0_fit.shape)
        if plot_Axy:
            offset = dict(nux=np.zeros(nuxs.shape), nuy=np.zeros(nuys.shape))
            if scan_plane == "x":
                offset["fit_nux"] = offset["fit_nuy"] = np.zeros(Jx_fit.shape)
            elif scan_plane == "y":
                offset["fit_nux"] = offset["fit_nuy"] = np.zeros(Jy_fit.shape)

    A_font_sz = 18
    font_sz = 22
    fit_x_line_style = "b-"
    fit_y_line_style = "r-"
    fit_x_extrap_line_style = "b:"
    fit_y_extrap_line_style = "r:"

    if plot_xy0:

        if ax_nu_vs_xy0:
            ax1 = ax_nu_vs_xy0
        else:
            fig, ax1 = plt.subplots()
        #
        if scan_plane == "x":
            lines1 = ax1.plot(
                scan_sign * x0s * 1e3, nuxs - offset0["nux"], "b.", label=r"$\nu_x$"
            )
            if nuxlim is not None:
                ax1.set_ylim(nuxlim)
            else:
                nuxlim = np.array(ax1.get_ylim())
            interp_roi = x0_fit <= fit_abs_xmax
            fit_lines1 = ax1.plot(
                x0_fit[interp_roi] * 1e3,
                nux_fit0(Jx0_fit[interp_roi]) - offset0["fit_nux"][interp_roi],
                fit_x_line_style,
                label=fit0_label["nux"],
            )
            ax1.plot(
                x0_fit[~interp_roi] * 1e3,
                nux_fit0(Jx0_fit[~interp_roi]) - offset0["fit_nux"][~interp_roi],
                fit_x_extrap_line_style,
            )
            ax2 = ax1.twinx()
            lines2 = ax2.plot(
                scan_sign * x0s * 1e3, nuys - offset0["nuy"], "r.", label=r"$\nu_y$"
            )
            if nuylim is not None:
                ax2.set_ylim(nuylim)
            else:
                nuylim = np.array(ax2.get_ylim())
            fit_lines2 = ax2.plot(
                x0_fit[interp_roi] * 1e3,
                nuy_fit0(Jx0_fit[interp_roi]) - offset0["fit_nuy"][interp_roi],
                fit_y_line_style,
                label=fit0_label["nuy"],
            )
            ax2.plot(
                x0_fit[~interp_roi] * 1e3,
                nuy_fit0(Jx0_fit[~interp_roi]) - offset0["fit_nuy"][~interp_roi],
                fit_y_extrap_line_style,
            )
            ax1.set_xlabel(
                rf"${scan_sign_str}x_0\, [\mathrm{{mm}}]\, {beta_str}$", size=A_font_sz
            )
            if x0lim is not None:
                ax1.set_xlim([v * 1e3 for v in x0lim])
        elif scan_plane == "y":
            lines1 = ax1.plot(
                scan_sign * y0s * 1e3, nuxs - offset0["nux"], "b.", label=r"$\nu_x$"
            )
            if nuxlim is not None:
                ax1.set_ylim(nuxlim)
            else:
                nuxlim = np.array(ax1.get_ylim())
            interp_roi = y0_fit <= fit_abs_ymax
            fit_lines1 = ax1.plot(
                y0_fit[interp_roi] * 1e3,
                nux_fit0(Jy0_fit[interp_roi]) - offset0["fit_nux"][interp_roi],
                fit_x_line_style,
                label=fit0_label["nux"],
            )
            ax1.plot(
                y0_fit[~interp_roi] * 1e3,
                nux_fit0(Jy0_fit[~interp_roi]) - offset0["fit_nux"][~interp_roi],
                fit_x_extrap_line_style,
            )
            ax2 = ax1.twinx()
            lines2 = ax2.plot(
                scan_sign * y0s * 1e3, nuys - offset0["nuy"], "r.", label=r"$\nu_y$"
            )
            if nuylim is not None:
                ax2.set_ylim(nuylim)
            else:
                nuylim = np.array(ax2.get_ylim())
            fit_lines2 = ax2.plot(
                y0_fit[interp_roi] * 1e3,
                nuy_fit0(Jy0_fit[interp_roi]) - offset0["fit_nuy"][interp_roi],
                fit_y_line_style,
                label=fit0_label["nuy"],
            )
            ax2.plot(
                y0_fit[~interp_roi] * 1e3,
                nuy_fit0(Jy0_fit[~interp_roi]) - offset0["fit_nuy"][~interp_roi],
                fit_y_extrap_line_style,
            )
            ax1.set_xlabel(
                rf"${scan_sign_str}y_0\, [\mathrm{{mm}}]\, {beta_str}$", size=A_font_sz
            )
            if y0lim is not None:
                ax1.set_xlim([v * 1e3 for v in y0lim])
        ax1.set_ylabel(r"$\nu_x$", size=font_sz, color="b")
        ax2.set_ylabel(r"$\nu_y$", size=font_sz, color="r")
        # Reset nux/nuy limits, which may have been changed by adding fitted
        # lines. Also, the fitted lines for nux & nuy will often overlap each
        # other with default ylim. So, here nux and nuy ranges are slided up
        # and down, respectively.
        nuxlim[0] -= (nuxlim[1] - nuxlim[0]) * 0.1
        nuylim[1] += (nuylim[1] - nuylim[0]) * 0.1
        ax1.set_ylim(nuxlim)
        ax2.set_ylim(nuylim)
        #
        if title != "":
            ax1.set_title(title, size=font_sz, pad=60)
        combined_lines = fit_lines1 + fit_lines2
        leg = ax2.legend(
            combined_lines,
            [L.get_label() for L in combined_lines],
            loc="upper center",
            ncol=2,
            bbox_to_anchor=(0.5, 1.2),
            fancybox=True,
            shadow=True,
            prop=dict(size=12),
        )
        plt.sca(ax1)
        plt.tight_layout()

    if plot_Axy:

        if ax_nu_vs_A:
            ax1 = ax_nu_vs_A
        else:
            fig, ax1 = plt.subplots()
        #
        if scan_plane == "x":
            lines1 = ax1.plot(Axs * 1e3, nuxs - offset["nux"], "b.", label=r"$\nu_x$")
            if nuxlim is not None:
                ax1.set_ylim(nuxlim)
            else:
                nuxlim = np.array(ax1.get_ylim())
            interp_roi = Ax_fit <= fit_abs_xmax
            fit_lines1 = ax1.plot(
                Ax_fit[interp_roi] * 1e3,
                nux_fit(Jx_fit[interp_roi]) - offset["fit_nux"][interp_roi],
                fit_x_line_style,
                label=fit_label["nux"],
            )
            ax1.plot(
                Ax_fit[~interp_roi] * 1e3,
                nux_fit(Jx_fit[~interp_roi]) - offset["fit_nux"][~interp_roi],
                fit_x_extrap_line_style,
            )
            ax2 = ax1.twinx()
            lines2 = ax2.plot(Axs * 1e3, nuys - offset["nuy"], "r.", label=r"$\nu_y$")
            if nuylim is not None:
                ax2.set_ylim(nuylim)
            else:
                nuylim = np.array(ax2.get_ylim())
            fit_lines2 = ax2.plot(
                Ax_fit[interp_roi] * 1e3,
                nuy_fit(Jx_fit[interp_roi]) - offset["fit_nuy"][interp_roi],
                fit_y_line_style,
                label=fit_label["nuy"],
            )
            ax2.plot(
                Ax_fit[~interp_roi] * 1e3,
                nuy_fit(Jx_fit[~interp_roi]) - offset["fit_nuy"][~interp_roi],
                fit_y_extrap_line_style,
            )
            ax1.set_xlabel(
                rf"$A_x\, [\mathrm{{mm}}]\, {scan_sign_beta_str}$", size=A_font_sz
            )
            if Axlim is not None:
                ax1.set_xlim([v * 1e3 for v in Axlim])
        elif scan_plane == "y":
            lines1 = ax1.plot(Ays * 1e3, nuxs - offset["nux"], "b.", label=r"$\nu_x$")
            if nuxlim is not None:
                ax1.set_ylim(nuxlim)
            else:
                nuxlim = np.array(ax1.get_ylim())
            interp_roi = Ay_fit <= fit_abs_ymax
            fit_lines1 = ax1.plot(
                Ay_fit[interp_roi] * 1e3,
                nux_fit(Jy_fit[interp_roi]) - offset["fit_nux"][interp_roi],
                fit_x_line_style,
                label=fit_label["nux"],
            )
            ax1.plot(
                Ay_fit[~interp_roi] * 1e3,
                nux_fit(Jy_fit[~interp_roi]) - offset["fit_nux"][~interp_roi],
                fit_x_extrap_line_style,
            )
            ax2 = ax1.twinx()
            lines2 = ax2.plot(Ays * 1e3, nuys - offset["nuy"], "r.", label=r"$\nu_y$")
            if nuylim is not None:
                ax2.set_ylim(nuylim)
            else:
                nuylim = np.array(ax2.get_ylim())
            fit_lines2 = ax2.plot(
                Ay_fit[interp_roi] * 1e3,
                nuy_fit(Jy_fit[interp_roi]) - offset["fit_nuy"][interp_roi],
                fit_y_line_style,
                label=fit_label["nuy"],
            )
            ax2.plot(
                Ay_fit[~interp_roi] * 1e3,
                nuy_fit(Jy_fit[~interp_roi]) - offset["fit_nuy"][~interp_roi],
                fit_y_extrap_line_style,
            )
            ax1.set_xlabel(
                rf"$A_y\, [\mathrm{{mm}}]\, {scan_sign_beta_str}$", size=A_font_sz
            )
            if Aylim is not None:
                ax1.set_xlim([v * 1e3 for v in Aylim])
        ax1.set_ylabel(r"$\nu_x$", size=font_sz, color="b")
        ax2.set_ylabel(r"$\nu_y$", size=font_sz, color="r")
        # Reset nux/nuy limits, which may have been changed by adding fitted
        # lines. Also, the fitted lines for nux & nuy will often overlap each
        # other with default ylim. So, here nux and nuy ranges are slided up
        # and down, respectively.
        nuxlim[0] -= (nuxlim[1] - nuxlim[0]) * 0.1
        nuylim[1] += (nuylim[1] - nuylim[0]) * 0.1
        ax1.set_ylim(nuxlim)
        ax2.set_ylim(nuylim)
        if title != "":
            ax1.set_title(title, size=font_sz, pad=60)
        combined_lines = fit_lines1 + fit_lines2
        leg = ax2.legend(
            combined_lines,
            [L.get_label() for L in combined_lines],
            loc="upper center",
            ncol=2,
            bbox_to_anchor=(0.5, 1.2),
            fancybox=True,
            shadow=True,
            prop=dict(size=12),
        )
        plt.sca(ax1)
        plt.tight_layout()

    if scan_plane == "x":
        # As = Axs
        xy0s = x0s
    else:
        # As = Ays
        xy0s = y0s

    is_fp_nuxlim_frac = False
    if footprint_nuxlim is not None:
        if (0.0 <= footprint_nuxlim[0] <= 1.0) and (0.0 <= footprint_nuxlim[1] <= 1.0):
            is_fp_nuxlim_frac = True
    else:
        footprint_nuxlim = nuxlim.copy()
    #
    is_fp_nuylim_frac = False
    if footprint_nuylim is not None:
        if (0.0 <= footprint_nuylim[0] <= 1.0) and (0.0 <= footprint_nuylim[1] <= 1.0):
            is_fp_nuylim_frac = True
    else:
        footprint_nuylim = nuylim.copy()
    #
    if is_fp_nuxlim_frac and is_fp_nuylim_frac:
        _plot_fp_nu_frac = True
    elif (not is_fp_nuxlim_frac) and (not is_fp_nuylim_frac):
        _plot_fp_nu_frac = False
    else:
        raise ValueError(
            (
                '"footprint_nuxlim" and "footprint_nuylim" must be either '
                "both fractional or both non-fractional"
            )
        )

    if _plot_fp_nu_frac:
        _nuxs = nuxs - nux0_int
        _nuys = nuys - nuy0_int

        frac_nuxlim = footprint_nuxlim
        frac_nuylim = footprint_nuylim
    else:
        _nuxs = nuxs
        _nuys = nuys

        frac_nuxlim = footprint_nuxlim - nux0_int
        frac_nuylim = footprint_nuylim - nuy0_int

    if ax_nuy_vs_nux:
        ax = ax_nuy_vs_nux
    else:
        _, ax = plt.subplots()
    # sc_obj = ax.scatter(_nuxs, _nuys, s=10, c=As * 1e3, marker='o', cmap='jet')
    sc_obj = ax.scatter(_nuxs, _nuys, s=10, c=xy0s * 1e3, marker="o", cmap="jet")
    ax.set_xlim(footprint_nuxlim)
    ax.set_ylim(footprint_nuylim)
    ax.set_xlabel(r"$\nu_x$", size=font_sz)
    ax.set_ylabel(r"$\nu_y$", size=font_sz)
    #
    rd = util.ResonanceDiagram()
    lineprops = dict(
        color=["k", "k", "g", "m", "m"],
        linestyle=["-", "--", "-", "-", ":"],
        linewidth=[2, 2, 0.5, 0.5, 0.5],
    )
    for n in range(1, max_resonance_line_order):
        d = rd.getResonanceCoeffsAndLines(
            n, np.array(frac_nuxlim) + nux0_int, np.array(frac_nuylim) + nuy0_int
        )  # <= CRITICAL: Must pass tunes w/ integer parts
        prop = {k: lineprops[k][n - 1] for k in ["color", "linestyle", "linewidth"]}
        assert len(d["lines"]) == len(d["coeffs"])
        for ((nux1, nuy1), (nux2, nuy2)), (nx, ny, _) in zip(d["lines"], d["coeffs"]):
            if _plot_fp_nu_frac:
                _x = np.array([nux1 - nux0_int, nux2 - nux0_int])
                _y = np.array([nuy1 - nuy0_int, nuy2 - nuy0_int])
            else:
                _x = np.array([nux1, nux2])
                _y = np.array([nuy1, nuy2])
            ax.plot(_x, _y, label=rd.getResonanceCoeffLabelString(nx, ny), **prop)
    # leg = plt.legend(loc='best')
    # leg.set_draggable(True, use_blit=True)
    #
    cb = plt.colorbar(sc_obj, ax=ax)
    # cb.ax.set_title(fr'$A_{scan_plane}\, [\mathrm{{mm}}]$', size=16)
    cb.ax.set_title(rf"${scan_plane}_0\, [\mathrm{{mm}}]$", size=16)
    if title != "":
        ax.set_title(title, size=font_sz)
    plt.sca(ax)
    plt.tight_layout()

    if (fft_d is not None) and plot_fft:

        if fft_plot_opts is None:
            fft_plot_opts = {}

        use_log = fft_plot_opts.get("logscale", False)
        use_full_ylim = fft_plot_opts.get("full_ylim", True)
        plot_shifted_curves = fft_plot_opts.get("shifted_curves", False)
        nu_size = fft_plot_opts.get("nu_size", None)

        font_sz = 18
        if use_log:
            EQ_STR = r"$\rm{log}_{10}(A/\mathrm{max}A)$"
        else:
            EQ_STR = r"$A/\mathrm{max}A$"

        if scan_plane == "x":
            v1array = np.abs(x0s)
        else:
            v1array = np.abs(y0s)

        for _nu_plane in ["x", "y"]:

            v2array = fft_d["fft_nus"].copy()
            v2array[v2array < 0.0] += 1
            if nu_size is not None:
                assert np.all(np.diff(v2array) > 0.0)
                v2array_resized = np.linspace(np.min(v2array), np.max(v2array), nu_size)
            else:
                v2array_resized = None

            if _nu_plane == "x":
                v2array += nux0_int
                if v2array_resized is not None:
                    v2array_resized += nux0_int

                if ax_fft_hx:
                    ax1 = ax_fft_hx
                else:
                    fig, ax1 = plt.subplots()

                norm_fft_hAs = fft_d["fft_hAxs"] / np.max(fft_d["fft_hAxs"], axis=0)

                ylim = nuxlim
            else:
                v2array += nuy0_int
                if v2array_resized is not None:
                    v2array_resized += nuy0_int

                if ax_fft_hy:
                    ax1 = ax_fft_hy
                else:
                    fig, ax1 = plt.subplots()

                norm_fft_hAs = fft_d["fft_hAys"] / np.max(fft_d["fft_hAys"], axis=0)

                ylim = nuylim

            if nu_size is not None:
                im = PIL.Image.fromarray((norm_fft_hAs * 255).astype(np.uint8), "L")
                w = norm_fft_hAs.shape[1]
                h = nu_size
                im = im.resize((w, h), resample=PIL.Image.Resampling.LANCZOS)
                im = np.array(list(im.getdata())).reshape((h, w))
                norm_fft_hAs_resized = im / np.max(im, axis=0)

            V1, V2 = np.meshgrid(v1array, v2array)
            if v2array_resized is not None:
                V1r, V2r = np.meshgrid(v1array, v2array_resized)

            if not use_log:
                if v2array_resized is not None:
                    plt.pcolor(
                        V1r * 1e3, V2r, norm_fft_hAs_resized, cmap="jet", shading="auto"
                    )
                else:
                    plt.pcolor(V1 * 1e3, V2, norm_fft_hAs, cmap="jet", shading="auto")
            else:
                if v2array_resized is not None:
                    plt.pcolor(
                        V1r * 1e3,
                        V2r,
                        np.log10(norm_fft_hAs_resized),
                        cmap="jet",
                        shading="auto",
                    )
                else:
                    plt.pcolor(
                        V1 * 1e3, V2, np.log10(norm_fft_hAs), cmap="jet", shading="auto"
                    )
            plt.xlabel(
                rf"${scan_sign_str}{scan_plane}_0\, [\mathrm{{mm}}]$", size=font_sz
            )
            plt.ylabel(rf"$\nu_{_nu_plane}$", size=font_sz)
            if not use_full_ylim:
                ax1.set_ylim(ylim)
            cb = plt.colorbar()
            cb.ax.set_title(EQ_STR)
            cb.ax.title.set_position((0.5, 1.02))
            plt.tight_layout()

            if plot_shifted_curves:
                # The following section plots slightly shifted FFT curves for better
                # curve height visualization than the pcolor plot above.

                M = np.zeros((500, 500))
                xarray = np.linspace(np.min(V2[:, 0]), np.max(V2[:, 0]), M.shape[1])
                max_y_offset = 5.0
                offset_frac = max_y_offset / V1.shape[1]
                yarray = np.linspace(0.0, 1.0 + max_y_offset, M.shape[0])
                M[yarray <= max_y_offset, :] = 1e-6
                for j in range(V1.shape[1]):
                    if np.any(np.isnan(norm_fft_hAs[:, j])):
                        assert np.all(np.isnan(norm_fft_hAs[:, j]))
                        continue

                    yinds = np.argmin(
                        np.abs(
                            yarray.reshape((-1, 1))
                            - (norm_fft_hAs[:, j] + j * offset_frac)
                        ),
                        axis=0,
                    )
                    xinds = np.argmin(
                        np.abs(xarray.reshape((-1, 1)) - V2[:, j]), axis=0
                    )

                    for dix in [-1, 0, 1]:
                        for diy in [-1, 0, 1]:
                            M_new = np.zeros_like(M)
                            M_new[
                                np.clip(yinds + diy, 0, M.shape[0] - 1),
                                np.clip(xinds + dix, 0, M.shape[1] - 1),
                            ] = norm_fft_hAs[:, j]

                            M_comb = np.dstack((M, M_new))
                            M = np.max(M_comb, axis=-1)

                real_yarray = (
                    np.min(v1array)
                    + (np.max(v1array) - np.min(v1array)) / max_y_offset * yarray
                )
                X, Y = np.meshgrid(xarray, real_yarray)
                M[M == 0.0] = np.nan

                plt.figure()
                plt.pcolor(Y.T * 1e3, X.T, M.T, cmap="jet", shading="auto")
                plt.xlabel(
                    rf"${scan_sign_str}{scan_plane}_0\, [\mathrm{{mm}}]$", size=font_sz
                )
                plt.ylabel(rf"$\nu_{_nu_plane}$", size=font_sz)
                if not use_full_ylim:
                    ax1.set_ylim(ylim)
                cb = plt.colorbar()
                cb.ax.set_title(EQ_STR)
                cb.ax.title.set_position((0.5, 1.02))
                plt.tight_layout()

    return ret


def plot_tswa_both_sides(
    output_filepath_positive,
    output_filepath_negative,
    title="",
    fit_xmax=None,
    fit_xmin=None,
    fit_ymax=None,
    fit_ymin=None,
    plot_xy0=True,
    x0lim=None,
    y0lim=None,
    plot_Axy=False,
    use_time_domain_amplitude=True,
    Axlim=None,
    Aylim=None,
    nuxlim=None,
    nuylim=None,
    footprint_nuxlim=None,
    footprint_nuylim=None,
    max_resonance_line_order=5,
    ax_nu_vs_xy0=None,
    ax_nu_vs_A=None,
    ax_nuy_vs_nux=None,
    plot_fft=False,
    fft_plot_opts=None,
    ax_fft_hx=None,
    ax_fft_hy=None,
):
    """"""

    assert max_resonance_line_order <= 5

    ret = {}  # variable to be returned
    ret["aper"] = {}

    is_nuxlim_frac = False
    if nuxlim is not None:
        if (0.0 <= nuxlim[0] <= 1.0) and (0.0 <= nuxlim[1] <= 1.0):
            is_nuxlim_frac = True

    is_nuylim_frac = False
    if nuylim is not None:
        if (0.0 <= nuylim[0] <= 1.0) and (0.0 <= nuylim[1] <= 1.0):
            is_nuylim_frac = True

    if is_nuxlim_frac and is_nuylim_frac:
        _plot_nu_frac = True
    elif (not is_nuxlim_frac) and (not is_nuylim_frac):
        _plot_nu_frac = False
    else:
        raise ValueError(
            '"nuxlim" and "nuylim" must be either both fractional or both non-fractional'
        )

    (
        scan_plane,
        x0s,
        y0s,
        nuxs,
        nuys,
        Axs,
        Ays,
        time_domain_Axs,
        time_domain_Ays,
        nux0,
        nuy0,
        betax,
        betay,
        fft_d,
        undefined_tunes,
    ) = [{} for _ in range(15)]

    for side, output_filepath in [
        ("+", output_filepath_positive),
        ("-", output_filepath_negative),
    ]:
        try:
            d = util.load_pgz_file(output_filepath)
            scan_plane[side] = d["input"]["scan_plane"]
            x0s[side], y0s[side] = d["x0s"], d["y0s"]
            nuxs[side], nuys[side] = d["nuxs"], d["nuys"]
            Axs[side], Ays[side] = d["Axs"], d["Ays"]
            time_domain_Axs[side], time_domain_Ays[side] = (
                d["time_domain_Axs"],
                d["time_domain_Ays"],
            )
            nux0[side], nuy0[side] = d["nux0"], d["nuy0"]
            betax[side], betay[side] = d["betax"], d["betay"]
            if "fft_nus" in d:
                fft_d[side] = {k: d[k] for k in ["fft_nus", "fft_hAxs", "fft_hAys"]}

            if "undefined_tunes" in d:
                undefined_tunes[side] = d["undefined_tunes"]
            else:
                undefined_tunes[side] = np.isnan(nuxs[side]) | np.isnan(nuys[side])
        except:
            f = h5py.File(output_filepath, "r")
            scan_plane[side] = f["input"]["scan_plane"][()]
            x0s[side] = f["x0s"][()]
            y0s[side] = f["y0s"][()]
            nuxs[side] = f["nuxs"][()]
            nuys[side] = f["nuys"][()]
            Axs[side] = f["Axs"][()]
            Ays[side] = f["Ays"][()]
            time_domain_Axs[side] = f["time_domain_Axs"][()]
            time_domain_Ays[side] = f["time_domain_Ays"][()]
            nux0[side] = f["nux0"][()]
            nuy0[side] = f["nuy0"][()]
            betax[side] = f["betax"][()]
            betay[side] = f["betay"][()]
            if "fft_nus" in f:
                fft_d[side] = {k: f[k][()] for k in ["fft_nus", "fft_hAxs", "fft_hAys"]}

            if "undefined_tunes" in f:
                undefined_tunes[side] = f["undefined_tunes"][()]
            else:
                undefined_tunes[side] = np.isnan(nuxs[side]) | np.isnan(nuys[side])

            f.close()

    if fft_d == {}:
        fft_d = None

    # Check consistency between the positive and negative results (i.e.,
    # see if the results are likley from the same lattice)
    assert scan_plane["+"] == scan_plane["-"]
    scan_plane = scan_plane["+"]
    np.testing.assert_almost_equal(nux0["+"], nux0["-"], decimal=12)
    np.testing.assert_almost_equal(nuy0["+"], nuy0["-"], decimal=12)
    nux0, nuy0 = nux0["+"], nuy0["+"]
    np.testing.assert_almost_equal(betax["+"], betax["-"], decimal=12)
    np.testing.assert_almost_equal(betay["+"], betay["-"], decimal=12)
    betax, betay = betax["+"], betay["+"]

    if scan_plane == "x":
        assert np.all(x0s["+"] >= 0.0)
        assert np.all(x0s["-"] <= 0.0)
    elif scan_plane == "y":
        assert np.all(y0s["+"] >= 0.0)
        assert np.all(y0s["-"] <= 0.0)
    else:
        raise ValueError

    if use_time_domain_amplitude:
        Axs = time_domain_Axs
        Ays = time_domain_Ays

    nux0_int = np.floor(nux0)
    nuy0_int = np.floor(nuy0)

    # Find undefined tune boundaries (i.e., either particle lost or spectrum too chaotic)
    if scan_plane == "x":
        v0s = x0s
    elif scan_plane == "y":
        v0s = y0s
    else:
        raise ValueError
    pos_undef_v0s = v0s["+"][undefined_tunes["+"]]
    if pos_undef_v0s.size != 0:
        min_pos_undef_v0 = np.min(pos_undef_v0s)
    else:
        min_pos_undef_v0 = np.nan
    neg_undef_v0s = v0s["-"][undefined_tunes["-"]]
    if neg_undef_v0s.size != 0:
        max_neg_undef_v0 = np.max(neg_undef_v0s)
    else:
        max_neg_undef_v0 = np.nan

    ret["aper"]["scanned"] = [np.min(v0s["-"]), np.max(v0s["+"])]

    twoJxs, twoJys, Jxs, Jys, nux_fit, nuy_fit, fit_label = [{} for _ in range(7)]
    twoJx0s, twoJy0s, Jx0s, Jy0s, nux_fit0, nuy_fit0, fit0_label = [
        {} for _ in range(7)
    ]
    if scan_plane == "x":
        Ax_fit, Jx_fit = {}, {}
        x0_fit, Jx0_fit = {}, {}
    elif scan_plane == "y":
        Ay_fit, Jy_fit = {}, {}
        y0_fit, Jy0_fit = {}, {}

    # beta_str = fr'(\beta_x, \beta_y) [\mathrm{{m}}] = ({betax:.2f}, {betay:.2f})'
    beta_str = rf"\{{(\beta_x, \beta_y) [\mathrm{{m}}] = ({betax:.2f}, {betay:.2f})\}}"

    # Find first integer/half-integer resonance crossing tunes
    upper_res_xing_nu = {}
    lower_res_xing_nu = {}
    for plane, _nu0 in [("x", nux0), ("y", nuy0)]:
        _nu0_int = np.floor(_nu0)
        _nu0_frac = _nu0 - _nu0_int
        if _nu0_frac < 0.5:
            upper_res_xing_nu[plane] = _nu0_int + 0.5
            lower_res_xing_nu[plane] = _nu0_int
        elif _nu0_frac > 0.5:
            upper_res_xing_nu[plane] = _nu0_int + 1.0
            lower_res_xing_nu[plane] = _nu0_int + 0.5
        else:
            pass
    #
    min_pos_res_xing_v = np.nan
    max_neg_res_xing_v = np.nan

    for side in ["+", "-"]:

        ret[side] = {}

        if False:
            nuxs[side] = smooth_nu_int_jump(nuxs[side], jump_thresh=0.5)
            nuys[side] = smooth_nu_int_jump(nuys[side], jump_thresh=0.5)

            nuxs[side] += nux0_int
            nuys[side] += nuy0_int

            if scan_plane == "x":
                v0s = x0s[side]
            elif scan_plane == "y":
                v0s = y0s[side]
            else:
                raise ValueError
            # Correct nuxs if smoothing shifted from nux0 by ~1
            for i in np.argsort(np.abs(v0s)):
                if np.isnan(nuxs[side][i]):
                    continue
                else:
                    if nuxs[side][i] > nux0 + 0.5:
                        nuxs[side] -= 1
                    elif nuxs[side][i] < nux0 - 0.5:
                        nuxs[side] += 1
                    break
            # Correct nuys if smoothing shifted from nuy0 by ~1
            for i in np.argsort(np.abs(v0s)):
                if np.isnan(nuys[side][i]):
                    continue
                else:
                    if nuys[side][i] > nuy0 + 0.5:
                        nuys[side] -= 1
                    elif nuys[side][i] < nuy0 - 0.5:
                        nuys[side] += 1
                    break
        else:
            if scan_plane == "x":
                v0s = x0s[side]
            elif scan_plane == "y":
                v0s = y0s[side]
            else:
                raise ValueError

            on_axis_index = np.argmin(np.abs(v0s))
            _kwargs = dict(jump_thresh=0.5, ref_index=on_axis_index)
            nuxs[side] = smooth_nu_int_jump(nuxs[side], **_kwargs)
            nuys[side] = smooth_nu_int_jump(nuys[side], **_kwargs)

            nuxs[side] += nux0_int
            nuys[side] += nuy0_int

        # Find xy0s at which tunes cross integer/half-integer resonance
        if upper_res_xing_nu != {}:
            def_v0s = v0s[~undefined_tunes[side]]
            if def_v0s.size != 0:
                def_nuxs = nuxs[side][~undefined_tunes[side]]
                def_nuys = nuys[side][~undefined_tunes[side]]
                xing_vs = def_v0s[
                    (def_nuxs > upper_res_xing_nu["x"])
                    | (def_nuxs < lower_res_xing_nu["x"])
                    | (def_nuys > upper_res_xing_nu["y"])
                    | (def_nuys < lower_res_xing_nu["y"])
                ]
                if xing_vs.size != 0:
                    if side == "+":
                        min_pos_res_xing_v = np.min(xing_vs)
                    elif side == "-":
                        max_neg_res_xing_v = np.max(xing_vs)

        twoJxs[side] = Axs[side] ** 2 / betax
        twoJys[side] = Ays[side] ** 2 / betay
        Jxs[side] = twoJxs[side] / 2
        Jys[side] = twoJys[side] / 2

        twoJx0s[side] = x0s[side] ** 2 / betax
        twoJy0s[side] = y0s[side] ** 2 / betay
        Jx0s[side] = twoJx0s[side] / 2
        Jy0s[side] = twoJy0s[side] / 2

        if scan_plane == "x":
            if side == "+":
                fit_roi = x0s[side] <= fit_xmax
            else:
                fit_roi = x0s[side] >= fit_xmin

            if plot_xy0:
                coeffs = np.polyfit(Jx0s[side][fit_roi], nuxs[side][fit_roi], 1)
                dnux_dJx0 = coeffs[0]
                ret[side]["dnux_dJx0"] = dnux_dJx0
                nux_fit0[side] = np.poly1d(coeffs)

                coeffs = np.polyfit(Jx0s[side][fit_roi], nuys[side][fit_roi], 1)
                dnuy_dJx0 = coeffs[0]
                ret[side]["dnuy_dJx0"] = dnuy_dJx0
                nuy_fit0[side] = np.poly1d(coeffs)

                if False:  # Not being used and will generate warning about poor fit
                    # because the elements of Jy0s are all the same
                    dnux_dJy0 = np.polyfit(Jy0s[side][fit_roi], nuxs[side][fit_roi], 1)[
                        0
                    ]
                    dnuy_dJy0 = np.polyfit(Jy0s[side][fit_roi], nuys[side][fit_roi], 1)[
                        0
                    ]

                x0_fit[side] = np.linspace(
                    np.min(np.abs(x0s[side])), np.max(np.abs(x0s[side])), 101
                )
                if side == "-":
                    x0_fit[side] *= -1
                Jx0_fit[side] = (x0_fit[side] ** 2 / betax) / 2

                dnux_dJx0_str = util.pprint_sci_notation(dnux_dJx0, "+.3g")
                dnuy_dJx0_str = util.pprint_sci_notation(dnuy_dJx0, "+.3g")

                fit0_label[side] = dict(
                    nux=dnux_dJx0_str,
                    nuy=dnuy_dJx0_str,
                )

            if plot_Axy:
                coeffs = np.polyfit(Jxs[side][fit_roi], nuxs[side][fit_roi], 1)
                dnux_dJx = coeffs[0]
                ret[side]["dnux_dJx"] = dnux_dJx
                nux_fit[side] = np.poly1d(coeffs)

                coeffs = np.polyfit(Jxs[side][fit_roi], nuys[side][fit_roi], 1)
                dnuy_dJx = coeffs[0]
                ret[side]["dnuy_dJx"] = dnuy_dJx
                nuy_fit[side] = np.poly1d(coeffs)

                if False:  # Not being used
                    dnux_dJy = np.polyfit(Jys[side][fit_roi], nuxs[side][fit_roi], 1)[0]
                    dnuy_dJy = np.polyfit(Jys[side][fit_roi], nuys[side][fit_roi], 1)[0]

                Ax_fit[side] = np.linspace(np.min(Axs[side]), np.max(Axs[side]), 101)
                Jx_fit[side] = (Ax_fit[side] ** 2 / betax) / 2

                dnux_dJx_str = util.pprint_sci_notation(dnux_dJx, "+.3g")
                dnuy_dJx_str = util.pprint_sci_notation(dnuy_dJx, "+.3g")

                fit_label[side] = dict(
                    nux=dnux_dJx_str,
                    nuy=dnuy_dJx_str,
                )

        elif scan_plane == "y":
            if side == "+":
                fit_roi = y0s[side] <= fit_ymax
            else:
                fit_roi = y0s[side] >= fit_ymin

            if plot_xy0:
                coeffs = np.polyfit(Jy0s[side][fit_roi], nuxs[side][fit_roi], 1)
                dnux_dJy0 = coeffs[0]
                ret[side]["dnux_dJy0"] = dnux_dJy0
                nux_fit0[side] = np.poly1d(coeffs)

                coeffs = np.polyfit(Jy0s[side][fit_roi], nuys[side][fit_roi], 1)
                dnuy_dJy0 = coeffs[0]
                ret[side]["dnuy_dJy0"] = dnuy_dJy0
                nuy_fit0[side] = np.poly1d(coeffs)

                if False:  # Not being used and will generate warning about poor fit
                    # because the elements of Jy0s are all the same
                    dnux_dJx0 = np.polyfit(Jx0s[side][fit_roi], nuxs[side][fit_roi], 1)[
                        0
                    ]
                    dnuy_dJx0 = np.polyfit(Jx0s[side][fit_roi], nuys[side][fit_roi], 1)[
                        0
                    ]

                y0_fit[side] = np.linspace(
                    np.min(np.abs(y0s[side])), np.max(np.abs(y0s[side])), 101
                )
                if side == "-":
                    y0_fit[side] *= -1
                Jy0_fit[side] = (y0_fit[side] ** 2 / betay) / 2

                dnux_dJy0_str = util.pprint_sci_notation(dnux_dJy0, "+.3g")
                dnuy_dJy0_str = util.pprint_sci_notation(dnuy_dJy0, "+.3g")

                fit0_label[side] = dict(
                    nux=dnux_dJy0_str,
                    nuy=dnuy_dJy0_str,
                )

            if plot_Axy:
                coeffs = np.polyfit(Jys[side][fit_roi], nuxs[side][fit_roi], 1)
                dnux_dJy = coeffs[0]
                ret[side]["dnux_dJy"] = dnux_dJy
                nux_fit[side] = np.poly1d(coeffs)

                coeffs = np.polyfit(Jys[side][fit_roi], nuys[side][fit_roi], 1)
                dnuy_dJy = coeffs[0]
                ret[side]["dnuy_dJy"] = dnuy_dJy
                nuy_fit[side] = np.poly1d(coeffs)

                if False:  # Not being used
                    dnux_dJx = np.polyfit(Jxs[side][fit_roi], nuxs[side][fit_roi], 1)[0]
                    dnuy_dJx = np.polyfit(Jxs[side][fit_roi], nuys[side][fit_roi], 1)[0]

                Ay_fit[side] = np.linspace(np.min(Ays[side]), np.max(Ays[side]), 101)
                Jy_fit[side] = (Ay_fit[side] ** 2 / betay) / 2

                dnux_dJy_str = util.pprint_sci_notation(dnux_dJy, "+.3g")
                dnuy_dJy_str = util.pprint_sci_notation(dnuy_dJy, "+.3g")

                fit_label[side] = dict(
                    nux=dnux_dJy_str,
                    nuy=dnuy_dJy_str,
                )

    if scan_plane == "x":
        if plot_xy0:
            fit0_label_combo = dict(
                nux=r"$d\nu_x / d J_x [\mathrm{{m}}^{{-1}}] = {} (-); {} (+)$".format(
                    fit0_label["-"]["nux"], fit0_label["+"]["nux"]
                ),
                nuy=r"$d\nu_y / d J_x [\mathrm{{m}}^{{-1}}] = {} (-); {} (+)$".format(
                    fit0_label["-"]["nuy"], fit0_label["+"]["nuy"]
                ),
            )
        if plot_Axy:
            fit_label_combo = dict(
                nux=r"$d\nu_x / d J_x [\mathrm{{m}}^{{-1}}] = {} (-); {} (+)$".format(
                    fit_label["-"]["nux"], fit_label["+"]["nux"]
                ),
                nuy=r"$d\nu_y / d J_x [\mathrm{{m}}^{{-1}}] = {} (-); {} (+)$".format(
                    fit_label["-"]["nuy"], fit_label["+"]["nuy"]
                ),
            )
    elif scan_plane == "y":
        if plot_xy0:
            fit0_label_combo = dict(
                nux=r"$d\nu_x / d J_y [\mathrm{{m}}^{{-1}}] = {} (-); {} (+)$".format(
                    fit0_label["-"]["nux"], fit0_label["+"]["nux"]
                ),
                nuy=r"$d\nu_y / d J_y [\mathrm{{m}}^{{-1}}] = {} (-); {} (+)$".format(
                    fit0_label["-"]["nuy"], fit0_label["+"]["nuy"]
                ),
            )
        if plot_Axy:
            fit_label_combo = dict(
                nux=r"$d\nu_x / d J_y [\mathrm{{m}}^{{-1}}] = {} (-); {} (+)$".format(
                    fit_label["-"]["nux"], fit_label["+"]["nux"]
                ),
                nuy=r"$d\nu_y / d J_y [\mathrm{{m}}^{{-1}}] = {} (-); {} (+)$".format(
                    fit_label["-"]["nuy"], fit_label["+"]["nuy"]
                ),
            )

    A_font_sz = 18
    font_sz = 22
    fit_x_line_style = "b-"
    fit_y_line_style = "r-"
    fit_x_extrap_line_style = "b:"
    fit_y_extrap_line_style = "r:"

    if plot_xy0:

        if ax_nu_vs_xy0:
            ax1 = ax_nu_vs_xy0
        else:
            fig, ax1 = plt.subplots()
        #
        nuxs_combo = np.append(nuxs["-"][::-1], nuxs["+"])
        nuys_combo = np.append(nuys["-"][::-1], nuys["+"])
        #
        if scan_plane == "x":
            nux_fit0_combo = np.append(
                nux_fit0["-"](Jx0_fit["-"])[::-1], nux_fit0["+"](Jx0_fit["+"])
            )
            nuy_fit0_combo = np.append(
                nuy_fit0["-"](Jx0_fit["-"])[::-1], nuy_fit0["+"](Jx0_fit["+"])
            )
        elif scan_plane == "y":
            nux_fit0_combo = np.append(
                nux_fit0["-"](Jy0_fit["-"])[::-1], nux_fit0["+"](Jy0_fit["+"])
            )
            nuy_fit0_combo = np.append(
                nuy_fit0["-"](Jy0_fit["-"])[::-1], nuy_fit0["+"](Jy0_fit["+"])
            )
        #
        if _plot_nu_frac:
            offset0_combo = dict(nux=np.floor(nuxs_combo), nuy=np.floor(nuys_combo))
            if scan_plane == "x":
                offset0_combo["fit_nux"] = np.floor(nux_fit0_combo)
                offset0_combo["fit_nuy"] = np.floor(nuy_fit0_combo)
            elif scan_plane == "y":
                offset0_combo["fit_nux"] = np.floor(nux_fit0_combo)
                offset0_combo["fit_nuy"] = np.floor(nuy_fit0_combo)
        else:
            offset0_combo = dict(
                nux=np.zeros(nuxs_combo.shape), nuy=np.zeros(nuys_combo.shape)
            )
            if scan_plane == "x":
                offset0_combo["fit_nux"] = offset0_combo["fit_nuy"] = np.zeros(
                    np.append(Jx0_fit["-"][::-1], Jx0_fit["+"]).shape
                )
            elif scan_plane == "y":
                offset0_combo["fit_nux"] = offset0_combo["fit_nuy"] = np.zeros(
                    np.append(Jy0_fit["-"][::-1], Jy0_fit["+"]).shape
                )
        #
        if scan_plane == "x":
            x0s_combo = np.append(x0s["-"][::-1], x0s["+"])
            x0_fit_combo = np.append(x0_fit["-"][::-1], x0_fit["+"])
            lines1 = ax1.plot(
                x0s_combo * 1e3,
                nuxs_combo - offset0_combo["nux"],
                "b.",
                label=r"$\nu_x$",
            )
            if nuxlim is not None:
                ax1.set_ylim(nuxlim)
            else:
                nuxlim = np.array(ax1.get_ylim())
            interp_roi = np.logical_and(
                fit_xmin <= x0_fit_combo, x0_fit_combo <= fit_xmax
            )
            fit_lines1 = ax1.plot(
                x0_fit_combo[interp_roi] * 1e3,
                nux_fit0_combo[interp_roi] - offset0_combo["fit_nux"][interp_roi],
                fit_x_line_style,
                label=fit0_label_combo["nux"],
            )
            for extrap_roi in [x0_fit_combo < fit_xmin, x0_fit_combo > fit_xmax]:
                ax1.plot(
                    x0_fit_combo[extrap_roi] * 1e3,
                    nux_fit0_combo[extrap_roi] - offset0_combo["fit_nux"][extrap_roi],
                    fit_x_extrap_line_style,
                )
            ax2 = ax1.twinx()
            lines2 = ax2.plot(
                x0s_combo * 1e3,
                nuys_combo - offset0_combo["nuy"],
                "r.",
                label=r"$\nu_y$",
            )
            if nuylim is not None:
                ax2.set_ylim(nuylim)
            else:
                nuylim = np.array(ax2.get_ylim())
            fit_lines2 = ax2.plot(
                x0_fit_combo[interp_roi] * 1e3,
                nuy_fit0_combo[interp_roi] - offset0_combo["fit_nuy"][interp_roi],
                fit_y_line_style,
                label=fit0_label_combo["nuy"],
            )
            for extrap_roi in [x0_fit_combo < fit_xmin, x0_fit_combo > fit_xmax]:
                ax2.plot(
                    x0_fit_combo[extrap_roi] * 1e3,
                    nuy_fit0_combo[extrap_roi] - offset0_combo["fit_nuy"][extrap_roi],
                    fit_y_extrap_line_style,
                )
            ax1.set_xlabel(rf"$x_0\, [\mathrm{{mm}}]\, {beta_str}$", size=A_font_sz)
            if x0lim is not None:
                ax1.set_xlim([v * 1e3 for v in x0lim])
        elif scan_plane == "y":
            y0s_combo = np.append(y0s["-"][::-1], y0s["+"])
            y0_fit_combo = np.append(y0_fit["-"][::-1], y0_fit["+"])
            lines1 = ax1.plot(
                y0s_combo * 1e3,
                nuxs_combo - offset0_combo["nux"],
                "b.",
                label=r"$\nu_x$",
            )
            if nuxlim is not None:
                ax1.set_ylim(nuxlim)
            else:
                nuxlim = np.array(ax1.get_ylim())
            interp_roi = np.logical_and(
                fit_ymin <= y0_fit_combo, y0_fit_combo <= fit_ymax
            )
            fit_lines1 = ax1.plot(
                y0_fit_combo[interp_roi] * 1e3,
                nux_fit0_combo[interp_roi] - offset0_combo["fit_nux"][interp_roi],
                fit_x_line_style,
                label=fit0_label_combo["nux"],
            )
            for extrap_roi in [y0_fit_combo < fit_ymin, y0_fit_combo > fit_ymax]:
                ax1.plot(
                    y0_fit_combo[extrap_roi] * 1e3,
                    nux_fit0_combo[extrap_roi] - offset0_combo["fit_nux"][extrap_roi],
                    fit_x_extrap_line_style,
                )
            ax2 = ax1.twinx()
            lines2 = ax2.plot(
                y0s_combo * 1e3,
                nuys_combo - offset0_combo["nuy"],
                "r.",
                label=r"$\nu_y$",
            )
            if nuylim is not None:
                ax2.set_ylim(nuylim)
            else:
                nuylim = np.array(ax2.get_ylim())
            fit_lines2 = ax2.plot(
                y0_fit_combo[interp_roi] * 1e3,
                nuy_fit0_combo[interp_roi] - offset0_combo["fit_nuy"][interp_roi],
                fit_y_line_style,
                label=fit0_label_combo["nuy"],
            )
            for extrap_roi in [y0_fit_combo < fit_ymin, y0_fit_combo > fit_ymax]:
                ax2.plot(
                    y0_fit_combo[extrap_roi] * 1e3,
                    nuy_fit0_combo[extrap_roi] - offset0_combo["fit_nuy"][extrap_roi],
                    fit_y_extrap_line_style,
                )
            ax1.set_xlabel(rf"$y_0\, [\mathrm{{mm}}]\, {beta_str}$", size=A_font_sz)
            if y0lim is not None:
                ax1.set_xlim([v * 1e3 for v in y0lim])
        ax1.set_ylabel(r"$\nu_x$", size=font_sz, color="b")
        ax2.set_ylabel(r"$\nu_y$", size=font_sz, color="r")
        # Reset nux/nuy limits, which may have been changed by adding fitted
        # lines. Also, the fitted lines for nux & nuy will often overlap each
        # other with default ylim. So, here nux and nuy ranges are slided up
        # and down, respectively.
        nuxlim[0] -= (nuxlim[1] - nuxlim[0]) * 0.1
        nuylim[1] += (nuylim[1] - nuylim[0]) * 0.1
        ax1.set_ylim(nuxlim)
        ax2.set_ylim(nuylim)
        # Add integer/half-integer tune lines, if within visible range
        for _nu in range(int(np.floor(nuxlim[0])), int(np.ceil(nuxlim[1])) + 1):
            if nuxlim[0] <= _nu <= nuxlim[1]:  # integer tune line
                ax1.axhline(_nu, linestyle="--", color="b")
            if nuxlim[0] <= _nu + 0.5 <= nuxlim[1]:  # half-integer tune line
                ax1.axhline(_nu + 0.5, linestyle=":", color="b")
        for _nu in range(int(np.floor(nuylim[0])), int(np.ceil(nuylim[1])) + 1):
            if nuylim[0] <= _nu <= nuylim[1]:  # integer tune line
                ax2.axhline(_nu, linestyle="--", color="r")
            if nuylim[0] <= _nu + 0.5 <= nuylim[1]:  # half-integer tune line
                ax2.axhline(_nu + 0.5, linestyle=":", color="r")
        # Add lines at max defined tune boundaries
        ret["aper"]["undefined_tunes"] = []
        _vlim = ax1.get_xlim()
        for _v0 in [max_neg_undef_v0, min_pos_undef_v0]:
            if _vlim[0] <= (_v0 * 1e3) <= _vlim[1]:
                ax1.axvline(_v0 * 1e3, linestyle="--", color="k")
                ret["aper"]["undefined_tunes"].append(_v0)
        # Add lines at max apertures w/o crossing integer/half-integer resonance
        ret["aper"]["resonance_xing"] = []
        for _v0 in [max_neg_res_xing_v, min_pos_res_xing_v]:
            if _vlim[0] <= (_v0 * 1e3) <= _vlim[1]:
                ax1.axvline(_v0 * 1e3, linestyle=":", color="k")
                ret["aper"]["resonance_xing"].append(_v0)
        #
        if title != "":
            ax1.set_title(title, size=font_sz, pad=60)
        combined_lines = fit_lines1 + fit_lines2
        leg = ax2.legend(
            combined_lines,
            [L.get_label() for L in combined_lines],
            loc="upper center",
            ncol=1,
            bbox_to_anchor=(0.5, 1.3),
            fancybox=True,
            shadow=True,
            prop=dict(size=12),
        )
        plt.sca(ax1)
        plt.tight_layout()

    if plot_Axy:

        if ax_nu_vs_A:
            ax1 = ax_nu_vs_A
        else:
            fig, ax1 = plt.subplots()
        #
        nuxs_combo = np.append(nuxs["-"][::-1], nuxs["+"])
        nuys_combo = np.append(nuys["-"][::-1], nuys["+"])
        #
        if scan_plane == "x":
            nux_fit_combo = np.append(
                nux_fit["-"](Jx_fit["-"])[::-1], nux_fit["+"](Jx_fit["+"])
            )
            nuy_fit_combo = np.append(
                nuy_fit["-"](Jx_fit["-"])[::-1], nuy_fit["+"](Jx_fit["+"])
            )
        elif scan_plane == "y":
            nux_fit_combo = np.append(
                nux_fit["-"](Jy_fit["-"])[::-1], nux_fit["+"](Jy_fit["+"])
            )
            nuy_fit_combo = np.append(
                nuy_fit["-"](Jy_fit["-"])[::-1], nuy_fit["+"](Jy_fit["+"])
            )
        #
        if _plot_nu_frac:
            offset_combo = dict(nux=np.floor(nuxs_combo), nuy=np.floor(nuys_combo))
            if scan_plane == "x":
                offset_combo["fit_nux"] = np.floor(nux_fit_combo)
                offset_combo["fit_nuy"] = np.floor(nuy_fit_combo)
            elif scan_plane == "y":
                offset_combo["fit_nux"] = np.floor(nux_fit_combo)
                offset_combo["fit_nuy"] = np.floor(nuy_fit_combo)
        else:
            offset_combo = dict(
                nux=np.zeros(nuxs_combo.shape), nuy=np.zeros(nuys_combo.shape)
            )
            if scan_plane == "x":
                offset_combo["fit_nux"] = offset_combo["fit_nuy"] = np.zeros(
                    np.append(Jx_fit["-"][::-1], Jx_fit["+"]).shape
                )
            elif scan_plane == "y":
                offset_combo["fit_nux"] = offset_combo["fit_nuy"] = np.zeros(
                    np.append(Jy_fit["-"][::-1], Jy_fit["+"]).shape
                )
        #
        if scan_plane == "x":
            Axs_combo = np.append(Axs["-"][::-1] * (-1), Axs["+"])
            Ax_fit_combo = np.append(Ax_fit["-"][::-1] * (-1), Ax_fit["+"])
            lines1 = ax1.plot(
                Axs_combo * 1e3,
                nuxs_combo - offset_combo["nux"],
                "b.",
                label=r"$\nu_x$",
            )
            if nuxlim is not None:
                ax1.set_ylim(nuxlim)
            else:
                nuxlim = np.array(ax1.get_ylim())
            interp_roi = np.logical_and(
                fit_xmin <= Ax_fit_combo, Ax_fit_combo <= fit_xmax
            )
            fit_lines1 = ax1.plot(
                Ax_fit_combo[interp_roi] * 1e3,
                nux_fit_combo[interp_roi] - offset_combo["fit_nux"][interp_roi],
                fit_x_line_style,
                label=fit_label_combo["nux"],
            )
            for extrap_roi in [Ax_fit_combo < fit_xmin, Ax_fit_combo > fit_xmax]:
                ax1.plot(
                    Ax_fit_combo[extrap_roi] * 1e3,
                    nux_fit_combo[extrap_roi] - offset_combo["fit_nux"][extrap_roi],
                    fit_x_extrap_line_style,
                )
            ax2 = ax1.twinx()
            lines2 = ax2.plot(
                Axs_combo * 1e3,
                nuys_combo - offset_combo["nuy"],
                "r.",
                label=r"$\nu_y$",
            )
            if nuylim is not None:
                ax2.set_ylim(nuylim)
            else:
                nuylim = np.array(ax2.get_ylim())
            fit_lines2 = ax2.plot(
                Ax_fit_combo[interp_roi] * 1e3,
                nuy_fit_combo[interp_roi] - offset_combo["fit_nuy"][interp_roi],
                fit_y_line_style,
                label=fit_label_combo["nuy"],
            )
            for extrap_roi in [Ax_fit_combo < fit_xmin, Ax_fit_combo > fit_xmax]:
                ax2.plot(
                    Ax_fit_combo[extrap_roi] * 1e3,
                    nuy_fit_combo[extrap_roi] - offset_combo["fit_nuy"][extrap_roi],
                    fit_y_extrap_line_style,
                )
            ax1.set_xlabel(rf"$A_x\, [\mathrm{{mm}}]\, {beta_str}$", size=A_font_sz)
            if Axlim is not None:
                ax1.set_xlim([v * 1e3 for v in Axlim])
        elif scan_plane == "y":
            Ays_combo = np.append(Ays["-"][::-1] * (-1), Ays["+"])
            Ay_fit_combo = np.append(Ay_fit["-"][::-1] * (-1), Ay_fit["+"])
            lines1 = ax1.plot(
                Ays_combo * 1e3,
                nuxs_combo - offset_combo["nux"],
                "b.",
                label=r"$\nu_x$",
            )
            if nuxlim is not None:
                ax1.set_ylim(nuxlim)
            else:
                nuxlim = np.array(ax1.get_ylim())
            interp_roi = np.logical_and(
                fit_ymin <= Ay_fit_combo, Ay_fit_combo <= fit_ymax
            )
            fit_lines1 = ax1.plot(
                Ay_fit_combo[interp_roi] * 1e3,
                nux_fit_combo[interp_roi] - offset_combo["fit_nux"][interp_roi],
                fit_x_line_style,
                label=fit_label_combo["nux"],
            )
            for extrap_roi in [Ay_fit_combo < fit_ymin, Ay_fit_combo > fit_ymax]:
                ax1.plot(
                    Ay_fit_combo[extrap_roi] * 1e3,
                    nux_fit_combo[extrap_roi] - offset_combo["fit_nux"][extrap_roi],
                    fit_x_extrap_line_style,
                )
            ax2 = ax1.twinx()
            lines2 = ax2.plot(
                Ays_combo * 1e3,
                nuys_combo - offset_combo["nuy"],
                "r.",
                label=r"$\nu_y$",
            )
            if nuylim is not None:
                ax2.set_ylim(nuylim)
            else:
                nuylim = np.array(ax2.get_ylim())
            fit_lines2 = ax2.plot(
                Ay_fit_combo[interp_roi] * 1e3,
                nuy_fit_combo[interp_roi] - offset_combo["fit_nuy"][interp_roi],
                fit_y_line_style,
                label=fit_label_combo["nuy"],
            )
            for extrap_roi in [Ay_fit_combo < fit_ymin, Ay_fit_combo > fit_ymax]:
                ax2.plot(
                    Ay_fit_combo[extrap_roi] * 1e3,
                    nuy_fit_combo[extrap_roi] - offset_combo["fit_nuy"][extrap_roi],
                    fit_y_extrap_line_style,
                )
            ax1.set_xlabel(rf"$A_y\, [\mathrm{{mm}}]\, {beta_str}$", size=A_font_sz)
            if Aylim is not None:
                ax1.set_xlim([v * 1e3 for v in Aylim])
        ax1.set_ylabel(r"$\nu_x$", size=font_sz, color="b")
        ax2.set_ylabel(r"$\nu_y$", size=font_sz, color="r")
        # Reset nux/nuy limits, which may have been changed by adding fitted
        # lines. Also, the fitted lines for nux & nuy will often overlap each
        # other with default ylim. So, here nux and nuy ranges are slided up
        # and down, respectively.
        nuxlim[0] -= (nuxlim[1] - nuxlim[0]) * 0.1
        nuylim[1] += (nuylim[1] - nuylim[0]) * 0.1
        ax1.set_ylim(nuxlim)
        ax2.set_ylim(nuylim)
        #
        if title != "":
            ax1.set_title(title, size=font_sz, pad=60)
        combined_lines = fit_lines1 + fit_lines2
        leg = ax2.legend(
            combined_lines,
            [L.get_label() for L in combined_lines],
            loc="upper center",
            ncol=1,
            bbox_to_anchor=(0.5, 1.3),
            fancybox=True,
            shadow=True,
            prop=dict(size=12),
        )
        plt.sca(ax1)
        plt.tight_layout()

    if scan_plane == "x":
        # As = np.append(Axs['-'][::-1] * (-1), Axs['+'])
        x0s_combo = np.append(x0s["-"][::-1], x0s["+"])
        xy0s = x0s_combo
    else:
        # As = np.append(Ays['-'][::-1] * (-1), Ays['+'])
        y0s_combo = np.append(y0s["-"][::-1], y0s["+"])
        xy0s = y0s_combo

    is_fp_nuxlim_frac = False
    if footprint_nuxlim is not None:
        if (0.0 <= footprint_nuxlim[0] <= 1.0) and (0.0 <= footprint_nuxlim[1] <= 1.0):
            is_fp_nuxlim_frac = True
    else:
        footprint_nuxlim = nuxlim.copy()
    #
    is_fp_nuylim_frac = False
    if footprint_nuylim is not None:
        if (0.0 <= footprint_nuylim[0] <= 1.0) and (0.0 <= footprint_nuylim[1] <= 1.0):
            is_fp_nuylim_frac = True
    else:
        footprint_nuylim = nuylim.copy()
    #
    if is_fp_nuxlim_frac and is_fp_nuylim_frac:
        _plot_fp_nu_frac = True
    elif (not is_fp_nuxlim_frac) and (not is_fp_nuylim_frac):
        _plot_fp_nu_frac = False
    else:
        raise ValueError(
            (
                '"footprint_nuxlim" and "footprint_nuylim" must be either '
                "both fractional or both non-fractional"
            )
        )

    if _plot_fp_nu_frac:
        _nuxs = nuxs_combo - nux0_int
        _nuys = nuys_combo - nuy0_int

        frac_nuxlim = footprint_nuxlim
        frac_nuylim = footprint_nuylim
    else:
        _nuxs = nuxs_combo
        _nuys = nuys_combo

        frac_nuxlim = footprint_nuxlim - nux0_int
        frac_nuylim = footprint_nuylim - nuy0_int

    if ax_nuy_vs_nux:
        ax = ax_nuy_vs_nux
    else:
        _, ax = plt.subplots()
    # sc_obj = ax.scatter(_nuxs, _nuys, s=10, c=As * 1e3, marker='o', cmap='jet')
    sc_obj = ax.scatter(_nuxs, _nuys, s=10, c=xy0s * 1e3, marker="o", cmap="jet")
    ax.set_xlim(footprint_nuxlim)
    ax.set_ylim(footprint_nuylim)
    ax.set_xlabel(r"$\nu_x$", size=font_sz)
    ax.set_ylabel(r"$\nu_y$", size=font_sz)
    #
    rd = util.ResonanceDiagram()
    lineprops = dict(
        color=["k", "k", "g", "m", "m"],
        linestyle=["-", "--", "-", "-", ":"],
        linewidth=[2, 2, 0.5, 0.5, 0.5],
    )
    for n in range(1, max_resonance_line_order):
        d = rd.getResonanceCoeffsAndLines(
            n, np.array(frac_nuxlim) + nux0_int, np.array(frac_nuylim) + nuy0_int
        )  # <= CRITICAL: Must pass tunes w/ integer parts
        prop = {k: lineprops[k][n - 1] for k in ["color", "linestyle", "linewidth"]}
        assert len(d["lines"]) == len(d["coeffs"])
        for ((nux1, nuy1), (nux2, nuy2)), (nx, ny, _) in zip(d["lines"], d["coeffs"]):
            if _plot_fp_nu_frac:
                _x = np.array([nux1 - nux0_int, nux2 - nux0_int])
                _y = np.array([nuy1 - nuy0_int, nuy2 - nuy0_int])
            else:
                _x = np.array([nux1, nux2])
                _y = np.array([nuy1, nuy2])
            ax.plot(_x, _y, label=rd.getResonanceCoeffLabelString(nx, ny), **prop)
    # leg = plt.legend(loc='best')
    # leg.set_draggable(True, use_blit=True)
    #
    cb = plt.colorbar(sc_obj, ax=ax)
    # cb.ax.set_title(fr'$A_{scan_plane}\, [\mathrm{{mm}}]$', size=16)
    cb.ax.set_title(rf"${scan_plane}_0\, [\mathrm{{mm}}]$", size=16)
    if title != "":
        ax.set_title(title, size=font_sz)
    plt.sca(ax)
    plt.tight_layout()

    if (fft_d is not None) and plot_fft:

        if fft_plot_opts is None:
            fft_plot_opts = {}

        use_log = fft_plot_opts.get("logscale", False)
        use_full_ylim = fft_plot_opts.get("full_ylim", True)
        plot_shifted_curves = fft_plot_opts.get("shifted_curves", False)
        nu_size = fft_plot_opts.get("nu_size", None)

        font_sz = 18
        if use_log:
            EQ_STR = r"$\rm{log}_{10}(A/\mathrm{max}A)$"
        else:
            EQ_STR = r"$A/\mathrm{max}A$"

        if scan_plane == "x":
            x0s_combo = np.append(x0s["-"][::-1], x0s["+"])
            v1array = x0s_combo
        else:
            y0s_combo = np.append(y0s["-"][::-1], y0s["+"])
            v1array = y0s_combo

        for _nu_plane in ["x", "y"]:

            assert np.all(fft_d["+"]["fft_nus"] == fft_d["-"]["fft_nus"])
            v2array = fft_d["+"]["fft_nus"].copy()
            v2array[v2array < 0.0] += 1
            if nu_size is not None:
                assert np.all(np.diff(v2array) > 0.0)
                v2array_resized = np.linspace(np.min(v2array), np.max(v2array), nu_size)
            else:
                v2array_resized = None

            if _nu_plane == "x":
                v2array += nux0_int
                if v2array_resized is not None:
                    v2array_resized += nux0_int

                if ax_fft_hx:
                    ax1 = ax_fft_hx
                else:
                    fig, ax1 = plt.subplots()

                norm_fft_hAs = np.hstack(
                    (
                        (
                            fft_d["-"]["fft_hAxs"]
                            / np.max(fft_d["-"]["fft_hAxs"], axis=0)
                        )[:, ::-1],
                        fft_d["+"]["fft_hAxs"] / np.max(fft_d["+"]["fft_hAxs"], axis=0),
                    )
                )

                ylim = nuxlim
            else:
                v2array += nuy0_int
                if v2array_resized is not None:
                    v2array_resized += nuy0_int

                if ax_fft_hy:
                    ax1 = ax_fft_hy
                else:
                    fig, ax1 = plt.subplots()

                norm_fft_hAs = np.hstack(
                    (
                        (
                            fft_d["-"]["fft_hAys"]
                            / np.max(fft_d["-"]["fft_hAys"], axis=0)
                        )[:, ::-1],
                        fft_d["+"]["fft_hAys"] / np.max(fft_d["+"]["fft_hAys"], axis=0),
                    )
                )

                ylim = nuylim

            if nu_size is not None:
                im = PIL.Image.fromarray((norm_fft_hAs * 255).astype(np.uint8), "L")
                w = norm_fft_hAs.shape[1]
                h = nu_size
                im = im.resize((w, h), resample=PIL.Image.Resampling.LANCZOS)
                im = np.array(list(im.getdata())).reshape((h, w))
                norm_fft_hAs_resized = im / np.max(im, axis=0)

            V1, V2 = np.meshgrid(v1array, v2array)
            if v2array_resized is not None:
                V1r, V2r = np.meshgrid(v1array, v2array_resized)

            if not use_log:
                if v2array_resized is not None:
                    plt.pcolor(
                        V1r * 1e3, V2r, norm_fft_hAs_resized, cmap="jet", shading="auto"
                    )
                else:
                    plt.pcolor(V1 * 1e3, V2, norm_fft_hAs, cmap="jet", shading="auto")
            else:
                if v2array_resized is not None:
                    plt.pcolor(
                        V1r * 1e3,
                        V2r,
                        np.log10(norm_fft_hAs_resized),
                        cmap="jet",
                        shading="auto",
                    )
                else:
                    plt.pcolor(
                        V1 * 1e3, V2, np.log10(norm_fft_hAs), cmap="jet", shading="auto"
                    )
            plt.xlabel(rf"${scan_plane}_0\, [\mathrm{{mm}}]$", size=font_sz)
            plt.ylabel(rf"$\nu_{_nu_plane}$", size=font_sz)
            if not use_full_ylim:
                ax1.set_ylim(ylim)
            cb = plt.colorbar()
            cb.ax.set_title(EQ_STR)
            cb.ax.title.set_position((0.5, 1.02))
            plt.tight_layout()

            if plot_shifted_curves:
                # The following section plots slightly shifted FFT curves for better
                # curve height visualization than the pcolor plot above.

                M = np.zeros((500, 500))
                xarray = np.linspace(np.min(V2[:, 0]), np.max(V2[:, 0]), M.shape[1])
                max_y_offset = 5.0
                offset_frac = max_y_offset / V1.shape[1]
                yarray = np.linspace(0.0, 1.0 + max_y_offset, M.shape[0])
                M[yarray <= max_y_offset, :] = 1e-6
                for j in range(V1.shape[1]):
                    if np.any(np.isnan(norm_fft_hAs[:, j])):
                        assert np.all(np.isnan(norm_fft_hAs[:, j]))
                        continue

                    yinds = np.argmin(
                        np.abs(
                            yarray.reshape((-1, 1))
                            - (norm_fft_hAs[:, j] + j * offset_frac)
                        ),
                        axis=0,
                    )
                    xinds = np.argmin(
                        np.abs(xarray.reshape((-1, 1)) - V2[:, j]), axis=0
                    )

                    for dix in [-1, 0, 1]:
                        for diy in [-1, 0, 1]:
                            M_new = np.zeros_like(M)
                            M_new[
                                np.clip(yinds + diy, 0, M.shape[0] - 1),
                                np.clip(xinds + dix, 0, M.shape[1] - 1),
                            ] = norm_fft_hAs[:, j]

                            M_comb = np.dstack((M, M_new))
                            M = np.max(M_comb, axis=-1)

                real_yarray = (
                    np.min(v1array)
                    + (np.max(v1array) - np.min(v1array)) / max_y_offset * yarray
                )
                X, Y = np.meshgrid(xarray, real_yarray)
                M[M == 0.0] = np.nan

                plt.figure()
                plt.pcolor(Y.T * 1e3, X.T, M.T, cmap="jet", shading="auto")
                plt.xlabel(rf"${scan_plane}_0\, [\mathrm{{mm}}]$", size=font_sz)
                plt.ylabel(rf"$\nu_{_nu_plane}$", size=font_sz)
                if not use_full_ylim:
                    ax1.set_ylim(ylim)
                cb = plt.colorbar()
                cb.ax.set_title(EQ_STR)
                cb.ax.title.set_position((0.5, 1.02))
                plt.tight_layout()

    return ret


def track(
    output_filepath,
    LTE_filepath,
    E_MeV,
    n_turns,
    x0=0.0,
    xp0=0.0,
    y0=0.0,
    yp0=0.0,
    delta0=0.0,
    output_coordinates=("x", "xp", "y", "yp", "delta"),
    double_format="",
    use_beamline=None,
    N_KICKS=None,
    transmute_elements=None,
    ele_filepath=None,
    output_file_type=None,
    del_tmp_files=True,
    print_cmd=False,
    run_local=True,
    remote_opts=None,
    err_log_check=None,
    nMaxRemoteRetry=3,
):
    """
    An example of "double_format" is '%25.16e'.
    """

    LTE_file_pathobj = Path(LTE_filepath)

    file_contents = LTE_file_pathobj.read_text()

    input_dict = dict(
        LTE_filepath=str(LTE_file_pathobj.resolve()),
        E_MeV=E_MeV,
        n_turns=n_turns,
        x0=x0,
        xp0=xp0,
        y0=y0,
        yp0=yp0,
        delta0=delta0,
        output_coordinates=output_coordinates,
        double_format=double_format,
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
            dir=Path.cwd(), delete=False, prefix=f"tmpTrack_", suffix=".ele"
        )
        ele_pathobj = Path(tmp.name)
        ele_filepath = str(ele_pathobj.resolve())
        tmp.close()

    watch_pathobj = ele_pathobj.with_suffix(".wc")

    ed = elebuilder.EleDesigner(ele_filepath, double_format=".12g")

    elebuilder.add_transmute_blocks(ed, transmute_elements)

    ed.add_newline()

    ed.add_block(
        "run_setup",
        lattice=LTE_filepath,
        p_central_mev=E_MeV,
        use_beamline=use_beamline,
    )

    ed.add_newline()

    temp_watch_elem_name = "ELEGANT_TRACK_WATCH"
    if run_local:
        watch_filepath = str(watch_pathobj.resolve())
    else:
        watch_filepath = watch_pathobj.name
    temp_watch_elem_def = (
        f'{temp_watch_elem_name}: WATCH, FILENAME="{watch_filepath}", '
        "MODE=coordinate"
    )

    ed.add_block(
        "insert_elements",
        name="*",
        exclude="*",
        add_at_start=True,
        element_def=temp_watch_elem_def,
    )

    nWatch = 1

    ed.add_newline()

    elebuilder.add_N_KICKS_alter_elements_blocks(ed, N_KICKS)

    ed.add_newline()

    ed.add_block("run_control", n_passes=n_turns)

    ed.add_newline()

    centroid = {}
    centroid[0] = x0
    centroid[1] = xp0
    centroid[2] = y0
    centroid[3] = yp0
    centroid[5] = delta0
    #
    ed.add_block("bunched_beam", n_particles_per_bunch=1, centroid=centroid)

    ed.add_newline()

    ed.add_block("track")

    ed.write()
    # print(ed.actual_output_filepath_list)

    tbt = dict(
        x=np.full((n_turns, nWatch), np.nan),
        y=np.full((n_turns, nWatch), np.nan),
        xp=np.full((n_turns, nWatch), np.nan),
        yp=np.full((n_turns, nWatch), np.nan),
        delta=np.full((n_turns, nWatch), np.nan),
        t=np.full((n_turns, nWatch), np.nan),
        dt=np.full((n_turns, nWatch), np.nan),
    )

    # Run Elegant
    if run_local:
        run(
            ele_filepath,
            print_cmd=print_cmd,
            print_stdout=std_print_enabled["out"],
            print_stderr=std_print_enabled["err"],
        )

        sbatch_info = None
    else:

        if remote_opts is None:
            remote_opts = dict(sbatch={"use": True, "wait": True})

        if ("pelegant" in remote_opts) and (remote_opts["pelegant"] is not False):
            print('"pelegant" option in `remote_opts` must be False for nonlin.track()')
            remote_opts["pelegant"] = False
        else:
            remote_opts["pelegant"] = False

        remote_opts["ntasks"] = 1
        # ^ If this is more than 1, you will likely see an error like "Unable to
        #   access file /.../tmp*.twi--file is locked (SDDS_InitializeOutput)"

        sbatch_info = _relaunchable_remote_run(
            remote_opts, ele_filepath, err_log_check, nMaxRemoteRetry
        )
    #
    output, _ = sdds.sdds2dicts(watch_pathobj)
    #
    cols = output["columns"]
    #
    if double_format != "":
        coord_keys = [k if k != "delta" else "p" for k in list(tbt)]
        _, _tbt_cols = sdds.printout(
            watch_pathobj,
            column_name_list=coord_keys,
            param_name_list=[],
            str_format=double_format,
            suppress_err_msg=True,
        )
        for k, v in _tbt_cols.items():
            cols[k] = v
    #
    for k in list(tbt):
        if k == "delta":
            _delta = cols["p"] / output["params"]["pCentral"] - 1.0
            tbt[k][: len(cols["p"]), :] = _delta.reshape((-1, 1))
        else:
            tbt[k][: len(cols[k]), :] = cols[k].reshape((-1, 1))

    timestamp_fin = util.get_current_local_time_str()

    if output_file_type in ("hdf5", "h5"):
        _kwargs = dict(compression="gzip")
        f = h5py.File(output_filepath, "a")
        for coord in output_coordinates:
            f.create_dataset(coord, data=tbt[coord], **_kwargs)
        f["timestamp_fin"] = timestamp_fin
        if sbatch_info is not None:
            f["dt_total"] = sbatch_info["total"]
            f["dt_running"] = sbatch_info["running"]
            f["sbatch_nodes"] = sbatch_info["nodes"]
            f["ncores"] = sbatch_info["ncores"]
        f.close()

    elif output_file_type == "pgz":
        d = dict(
            input=input_dict,
            timestamp_fin=timestamp_fin,
            _version_PyELEGANT=__version__["PyELEGANT"],
            _version_ELEGANT=__version__["ELEGANT"],
        )
        if sbatch_info is not None:
            d["dt_total"] = sbatch_info["total"]
            d["dt_running"] = sbatch_info["running"]
            d["sbatch_nodes"] = sbatch_info["nodes"]
            d["ncores"] = sbatch_info["ncores"]

        for coord in output_coordinates:
            try:
                d[coord] = tbt[coord]
            except KeyError:
                print("Available keys are the following:")
                print("     " + ", ".join(list(tbt)))
                raise
        util.robust_pgz_file_write(output_filepath, d, nMaxTry=10, sleep=10.0)
    else:
        raise ValueError()

    if del_tmp_files:
        util.delete_temp_files(
            ed.actual_output_filepath_list + [ele_filepath, str(watch_pathobj)]
        )

    return output_filepath


def calc_offmom_closed_orbits(
    output_filepath,
    LTE_filepath,
    E_MeV,
    delta_array,
    iteration_fraction=0.1,
    closed_orbit_iterations=500,
    use_beamline=None,
    N_KICKS=None,
    transmute_elements=None,
    ele_filepath=None,
    output_file_type=None,
    del_tmp_files=True,
    print_cmd=False,
    run_local=True,
    remote_opts=None,
):
    """"""

    LTE_file_pathobj = Path(LTE_filepath)

    file_contents = LTE_file_pathobj.read_text()

    input_dict = dict(
        LTE_filepath=str(LTE_file_pathobj.resolve()),
        E_MeV=E_MeV,
        delta_array=delta_array,
        iteration_fraction=iteration_fraction,
        closed_orbit_iterations=closed_orbit_iterations,
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
            dir=Path.cwd(), delete=False, prefix=f"tmpOffMomCO_", suffix=".ele"
        )
        ele_pathobj = Path(tmp.name)
        ele_filepath = str(ele_pathobj.resolve())
        tmp.close()

    ed = elebuilder.EleDesigner(ele_filepath, double_format=".12g")

    elebuilder.add_transmute_blocks(ed, transmute_elements)

    ed.add_newline()

    ed.add_block(
        "run_setup",
        lattice=LTE_filepath,
        p_central_mev=E_MeV,
        use_beamline=use_beamline,
        final="%s.fin",
    )

    ed.add_newline()

    temp_malign_elem_name = "ELEGANT_OFFMOM_CO_MAL"
    temp_malign_elem_def = f"{temp_malign_elem_name}: MALIGN"

    ed.add_block(
        "insert_elements",
        name="*",
        exclude="*",
        add_at_start=True,
        element_def=temp_malign_elem_def,
    )

    ed.add_newline()

    ed.add_block("run_control")

    ed.add_newline()

    ed.add_block(
        "alter_elements",
        name=temp_malign_elem_name,
        type="MALIGN",
        item="DP",
        value="<delta>",
    )

    ed.add_newline()

    ed.add_block(
        "closed_orbit",
        output="%s.clo",
        iteration_fraction=iteration_fraction,
        closed_orbit_iterations=closed_orbit_iterations,
    )

    ed.add_newline()

    ed.add_block("bunched_beam")

    ed.add_newline()

    ed.add_block("track")

    ed.write()
    # print(ed.actual_output_filepath_list)

    for fp in ed.actual_output_filepath_list:
        if fp.endswith(".clo"):
            clo_filepath = fp
        elif fp.endswith(".fin"):
            fin_filepath = fp
        else:
            raise ValueError("This line should not be reached.")

    delta_array = np.sort(delta_array)
    negative = delta_array < 0.0
    positive = ~negative
    negative_inds = np.where(negative)[0]
    positive_inds = np.where(positive)[0]

    survived = np.zeros_like(delta_array).astype(bool)
    converged = np.zeros_like(delta_array).astype(bool)
    clos = {}
    for k in [
        "s",
        "x",
        "xp",
        "y",
        "yp",
        "ElementName",
        "ElementOccurence",
        "ElementType",
    ]:
        clos[k] = None
    clos["xerr"] = np.full(delta_array.size, np.nan)
    clos["yerr"] = np.full(delta_array.size, np.nan)

    # Run Elegant
    if run_local:
        for _is_positive, _one_side_delta_array in [
            (True, delta_array[positive]),
            (False, delta_array[negative][::-1]),
        ]:

            for i, delta in enumerate(_one_side_delta_array):
                run(
                    ele_filepath,
                    print_cmd=print_cmd,
                    macros=dict(delta=f"{delta:.12g}"),
                    print_stdout=std_print_enabled["out"],
                    print_stderr=std_print_enabled["err"],
                )

                clo_output, _ = sdds.sdds2dicts(clo_filepath)
                fin_output, _ = sdds.sdds2dicts(fin_filepath)

                if _is_positive:
                    j = positive_inds[i]
                else:
                    j = negative_inds[::-1][i]

                survived[j] = fin_output["params"]["Transmission"] == 1.0
                if not survived[j]:
                    break

                converged[j] = clo_output["params"]["failed"] == 0

                clos["xerr"][j] = clo_output["params"]["xError"]
                clos["yerr"][j] = clo_output["params"]["yError"]

                if clos["s"] is None:
                    for k in ["s", "ElementName", "ElementOccurence", "ElementType"]:
                        clos[k] = clo_output["columns"][k]

                    for k in ["x", "xp", "y", "yp"]:
                        clos[k] = np.full((clos["s"].size, delta_array.size), np.nan)

                for k in ["x", "xp", "y", "yp"]:
                    clos[k][:, j] = clo_output["columns"][k]

    else:
        raise NotImplementedError

    timestamp_fin = util.get_current_local_time_str()

    if output_file_type in ("hdf5", "h5"):
        _kwargs = dict(compression="gzip")
        f = h5py.File(output_filepath, "a")
        f.create_dataset("survived", data=survived, **_kwargs)
        f.create_dataset("converged", data=converged, **_kwargs)
        g = f.create_group("clos")
        for k, v in clos.items():
            g.create_dataset(k, data=v, **_kwargs)
        f["timestamp_fin"] = timestamp_fin
        f.close()
    elif output_file_type == "pgz":
        d = dict(
            input=input_dict,
            timestamp_fin=timestamp_fin,
            _version_PyELEGANT=__version__["PyELEGANT"],
            _version_ELEGANT=__version__["ELEGANT"],
        )
        d["survived"] = survived
        d["converged"] = converged
        d["clos"] = clos
        util.robust_pgz_file_write(output_filepath, d, nMaxTry=10, sleep=10.0)
    else:
        raise ValueError()

    if del_tmp_files:
        for fp in ed.actual_output_filepath_list + [ele_filepath]:
            if fp.startswith("/dev"):
                continue
            else:
                try:
                    os.remove(fp)
                except:
                    print(f'Failed to delete "{fp}"')

    return output_filepath


def _relaunchable_remote_run(remote_opts, ele_filepath, err_log_check, nMaxRemoteRetry):
    """"""

    iRemoteTry = 0
    while True:
        sbatch_info = remote.run(
            remote_opts,
            ele_filepath,
            print_cmd=True,
            print_stdout=std_print_enabled["out"],
            print_stderr=std_print_enabled["err"],
            output_filepaths=None,
            err_log_check=err_log_check,
        )

        if (
            (err_log_check is not None)
            and (sbatch_info is not None)
            and sbatch_info["err_found"]
        ):

            err_log_text = sbatch_info["err_log"]
            print("\n** Error Log check found the following problem:")
            print(err_log_text)

            iRemoteTry += 1

            if iRemoteTry >= nMaxRemoteRetry:
                raise RuntimeError(
                    "Max number of remote tries exceeded. Check the error logs."
                )
            else:
                print("\n** Re-trying the remote run...\n")
                sys.stdout.flush()
        else:
            break

    return sbatch_info
