import os
from pathlib import Path
import tempfile
from typing import Dict, List, Optional, Union

import numpy as np

from . import elebuilder, sdds, std_print_enabled
from .local import run


def save_lattice_after_load_parameters(
    input_LTE_filepath: Union[Path, str],
    new_LTE_filepath: Union[Path, str],
    load_parameters: Dict,
) -> None:
    """"""

    input_LTE_filepath = Path(input_LTE_filepath)
    new_LTE_filepath = Path(new_LTE_filepath)

    tmp = tempfile.NamedTemporaryFile(
        dir=os.getcwd(), delete=False, prefix=f"tmp_", suffix=".ele"
    )
    ele_filepath = os.path.abspath(tmp.name)
    tmp.close()

    ed = elebuilder.EleDesigner(
        ele_filepath, double_format=".12g", auto_print_on_add=False
    )

    ed.add_block("run_setup", lattice=str(input_LTE_filepath), p_central_mev=1.0)
    # ^ The value for "p_central_mev" being used here is just an arbitrary
    #   non-zero value. If it stays as the default value of 0, then ELEGANT will
    #   complain.

    ed.add_newline()

    if "change_defined_values" in load_parameters:
        if not load_parameters["change_defined_values"]:
            raise ValueError('load_parameters["change_defined_values"] cannot be False')
    else:
        load_parameters["change_defined_values"] = True
    load_parameters.setdefault("allow_missing_elements", True)
    load_parameters.setdefault("allow_missing_parameters", True)
    ed.add_block("load_parameters", **load_parameters)

    ed.add_newline()

    ed.add_block("save_lattice", filename=str(new_LTE_filepath))

    ed.write()

    run(
        ele_filepath,
        print_cmd=False,
        print_stdout=std_print_enabled["out"],
        print_stderr=std_print_enabled["err"],
    )

    try:
        os.remove(ele_filepath)
    except:
        print(f'Failed to delete "{ele_filepath}"')


def save_lattice_after_alter_elements(
    input_LTE_filepath: Union[Path, str],
    new_LTE_filepath: Union[Path, str],
    alter_elements: Union[Dict, List],
) -> None:
    """"""

    input_LTE_filepath = Path(input_LTE_filepath)
    new_LTE_filepath = Path(new_LTE_filepath)

    tmp = tempfile.NamedTemporaryFile(
        dir=os.getcwd(), delete=False, prefix=f"tmp_", suffix=".ele"
    )
    ele_filepath = os.path.abspath(tmp.name)
    tmp.close()

    ed = elebuilder.EleDesigner(
        ele_filepath, double_format=".12g", auto_print_on_add=False
    )

    ed.add_block("run_setup", lattice=str(input_LTE_filepath), p_central_mev=1.0)
    # ^ The value for "p_central_mev" being used here is just an arbitrary
    #   non-zero value. If it stays as the default value of 0, then ELEGANT will
    #   complain.

    ed.add_newline()

    if isinstance(alter_elements, dict):
        ed.add_block("alter_elements", **alter_elements)
    else:
        for block in alter_elements:
            ed.add_block("alter_elements", **block)

    ed.add_newline()

    ed.add_block("save_lattice", filename=str(new_LTE_filepath))

    ed.write()

    run(
        ele_filepath,
        print_cmd=False,
        print_stdout=std_print_enabled["out"],
        print_stderr=std_print_enabled["err"],
    )

    try:
        os.remove(ele_filepath)
    except:
        print(f'Failed to delete "{ele_filepath}"')


def save_lattice_after_transmute_elements(
    input_LTE_filepath: Union[Path, str],
    new_LTE_filepath: Union[Path, str],
    transmute_elements: Union[Dict, List],
) -> None:
    """"""

    input_LTE_filepath = Path(input_LTE_filepath)
    new_LTE_filepath = Path(new_LTE_filepath)

    tmp = tempfile.NamedTemporaryFile(
        dir=os.getcwd(), delete=False, prefix=f"tmp_", suffix=".ele"
    )
    ele_filepath = os.path.abspath(tmp.name)
    tmp.close()

    ed = elebuilder.EleDesigner(
        ele_filepath, double_format=".12g", auto_print_on_add=False
    )

    if isinstance(transmute_elements, dict):
        ed.add_block("transmute_elements", **transmute_elements)
    else:
        for block in transmute_elements:
            ed.add_block("transmute_elements", **block)

    ed.add_newline()

    ed.add_block("run_setup", lattice=str(input_LTE_filepath), p_central_mev=1.0)
    # ^ The value for "p_central_mev" being used here is just an arbitrary
    #   non-zero value. If it stays as the default value of 0, then ELEGANT will
    #   complain.

    ed.add_newline()

    ed.add_block("save_lattice", filename=str(new_LTE_filepath))

    ed.write()

    run(
        ele_filepath,
        print_cmd=False,
        print_stdout=std_print_enabled["out"],
        print_stderr=std_print_enabled["err"],
    )

    try:
        os.remove(ele_filepath)
    except:
        print(f'Failed to delete "{ele_filepath}"')


def get_transport_matrices(
    input_LTE_filepath: Union[Path, str],
    use_beamline: Optional[str] = None,
    individual_matrices: bool = False,
    del_tmp_files: bool = True,
) -> Dict:
    """"""

    input_LTE_filepath = Path(input_LTE_filepath)

    tmp = tempfile.NamedTemporaryFile(
        dir=os.getcwd(), delete=False, prefix=f"tmp_", suffix=".ele"
    )
    ele_filepath = os.path.abspath(tmp.name)
    tmp.close()

    ed = elebuilder.EleDesigner(
        ele_filepath, double_format=".12g", auto_print_on_add=False
    )

    ed.add_block(
        "run_setup",
        lattice=str(input_LTE_filepath),
        use_beamline=use_beamline,
        p_central_mev=1.0,
    )
    # ^ The value for "p_central_mev" being used here is just an arbitrary
    #   non-zero value. If it stays as the default value of 0, then ELEGANT will
    #   complain.

    ed.add_newline()

    ed.add_block(
        "matrix_output",
        printout=None,
        SDDS_output="%s.mat",
        individual_matrices=individual_matrices,
    )

    ed.write()

    run(
        ele_filepath,
        print_cmd=False,
        print_stdout=std_print_enabled["out"],
        print_stderr=std_print_enabled["err"],
    )

    for fp in ed.actual_output_filepath_list:
        if fp.endswith(".mat"):
            data, meta = sdds.sdds2dicts(fp, str_format="%25.16e")

    if del_tmp_files:
        for fp in ed.actual_output_filepath_list + [ele_filepath]:
            if fp.startswith("/dev"):
                continue
            else:
                try:
                    os.remove(fp)
                except:
                    print(f'Failed to delete "{fp}"')

    return data["columns"]


def get_M66(
    input_LTE_filepath: Union[Path, str],
    use_beamline: Optional[str] = None,
    ini_elem_name: Optional[str] = None,
    ini_elem_occur: int = 1,
    fin_elem_name: Optional[str] = None,
    fin_elem_occur: int = 1,
) -> np.array:
    """
    Returns the transport matrix (6x6 numpy array) from the beginning to the
    end of the beamline specified by "use_beamline". You can also crop
    the matrix to start from the beginning of the first element
    specified by "ini_elem_name" and "ini_elem_occur" to the end of the last
    element specified by "fin_elem_name" and "fin_elem_occur".
    """

    d = get_transport_matrices(
        input_LTE_filepath,
        use_beamline=use_beamline,
        individual_matrices=False,
        del_tmp_files=True,
    )

    if ini_elem_name is None:
        ini_ind = 0
    else:
        ini_ind = np.where(
            np.logical_and(
                d["ElementName"] == ini_elem_name,
                d["ElementOccurence"] == ini_elem_occur,
            )
        )[0][0]

    if fin_elem_name is None:
        fin_ind = -1
    else:
        fin_ind = np.where(
            np.logical_and(
                d["ElementName"] == fin_elem_name,
                d["ElementOccurence"] == fin_elem_occur,
            )
        )[0][0]

    M_fin = np.full((6, 6), np.nan)
    for i in range(1, 6 + 1):
        for j in range(1, 6 + 1):
            M_fin[i - 1, j - 1] = d[f"R{i:d}{j:d}"][fin_ind]

    if ini_ind == 0:
        M = M_fin
    else:
        M_ini = np.full((6, 6), np.nan)
        for i in range(1, 6 + 1):
            for j in range(1, 6 + 1):
                M_ini[i - 1, j - 1] = d[f"R{i:d}{j:d}"][ini_ind - 1]

        M = M_fin @ np.linalg.inv(M_ini)

    return M


def get_one_turn_M_via_tracking(
    LTE_filepath: Union[Path, str] = "",
    LTEZIP_filepath: Union[Path, str] = "",
    used_beamline_name: str = "",
    N_KICKS: Union[None, Dict] = None,
):

    import scipy

    from . import disable_stdout, ltemanager, nonlin, util

    LTE = ltemanager.Lattice(
        LTE_filepath=LTE_filepath,
        LTEZIP_filepath=LTEZIP_filepath,
        used_beamline_name=used_beamline_name,
    )

    actual_LTE_filepath = LTE.LTE_filepath

    output_filepath = LTE_filepath.with_suffix(".pgz")

    speed_of_light = scipy.constants.c
    E_MeV = 3e3
    n_turns = 3
    output_coordinates = ("x", "xp", "y", "yp", "dt", "delta")

    delta_vals = [1e-6, 1e-6, 1e-6, 1e-6, 0.0, 1e-6]
    M = np.zeros((6, 6))

    disable_stdout()

    for iCoord, dv in enumerate(delta_vals):
        v_in = np.zeros(6)
        v_in[iCoord] = dv

        nonlin.track(
            str(output_filepath),
            str(LTE_filepath),
            E_MeV,
            n_turns,
            x0=v_in[0],
            xp0=v_in[1],
            y0=v_in[2],
            yp0=v_in[3],
            t0=v_in[4],
            delta0=v_in[5],
            output_coordinates=output_coordinates,
            use_beamline=LTE.used_beamline_name,
            N_KICKS=N_KICKS,
        )

        d = util.load_pgz_file(output_filepath)

        v_out_mult_turns = np.array(
            [d[coord].flatten() for coord in output_coordinates]
        )
        v_out = v_out_mult_turns[:, 1]

        if iCoord in (0, 1, 2, 3):
            assert v_out[5] == 0.0
            v_out[4] *= speed_of_light
            M[:, iCoord] = v_out / dv
        elif iCoord == 4:
            assert v_out[5] == 0.0

            diff = v_out_mult_turns[:, 1] - v_out_mult_turns[:, 0]
            if False:
                assert np.all(diff[:4] == 0.0)

            dt = v_out_mult_turns[iCoord, :]
            assert dt[0] == 0.0
            assert np.abs(dt[1]) < 1e-16
            v_out[4] = 1.0
            M[:, iCoord] = v_out
        elif iCoord == 5:
            v_out[4] *= speed_of_light
            M[:, iCoord] = v_out / dv

            if False:
                np.testing.assert_almost_equal(M[5, 5], 1.0, decimal=12)
                M[5, 5] = 1.0
        else:
            raise ValueError

    try:
        output_filepath.unlink()
    except:
        pass

    return M
