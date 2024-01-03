"""
Based on SC (https://github.com/ThorstenHellert/SC)
"""

from collections import defaultdict
import dataclasses as dc
from dataclasses import dataclass

# IMPORTANT: The name of the variable to which IntEnum() is assigned must be the
# same as the first string argument given to IntEnum(), if this enum needs to
# be pickled!
from enum import IntEnum
from functools import partial
from pathlib import Path
import pickle
import tempfile
from typing import Dict, List, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import truncnorm

if False:
    from . import ltemanager, sdds
else:
    import pyelegant as pe

    ltemanager = pe.ltemanager
    sdds = pe.sdds

SupportType = IntEnum("SupportType", ["section", "plinth", "girder"], start=0)

AVAIL_ELEM_PROPS = {}


def _update_AVAIL_ELEM_PROPS():
    if AVAIL_ELEM_PROPS:
        return

    d = ltemanager.get_ELEGANT_element_dictionary()

    sel_prop_names = [
        "DX",
        "DY",
        "DZ",
        "TILT",
        "PITCH",
        "YAW",
        "ETILT",
        "EPITCH",
        "EYAW",
    ]

    for p_name in sel_prop_names:
        AVAIL_ELEM_PROPS[p_name] = [
            elem_type
            for elem_type, sub_d in d["elements"].items()
            if any([L[0] == p_name for L in sub_d["table"]])
        ]


@dataclass
class TGES:
    """TruncatedGaussianErrorSpec(TGES)"""

    rms: float
    rms_unit: str = ""
    cutoff: float = 2.0
    mean: float = 0.0

    def __post_init__(self):
        if self.rms < 0.0:
            raise ValueError("`rms` must be >= 0")
        if not isinstance(self.rms_unit, str):
            raise ValueError("`rms_unit` must be str")
        if self.cutoff <= 0.0:
            raise ValueError("`cutoff` must be > 0")


@dataclass
class OffsetSpec3D:
    _def_fac = partial(TGES, rms=0.0, rms_unit="m")
    x: TGES = dc.field(default_factory=_def_fac)
    y: TGES = dc.field(default_factory=_def_fac)
    z: TGES = dc.field(default_factory=_def_fac)


@dataclass
class Offset3D:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class OffsetSpec2D:
    _def_fac = partial(TGES, rms=0.0, rms_unit="m")
    x: TGES = dc.field(default_factory=_def_fac)
    y: TGES = dc.field(default_factory=_def_fac)


@dataclass
class Offset2D:
    x: float = 0.0
    y: float = 0.0


@dataclass
class GainSpec:
    _def_fac = partial(TGES, rms=0.0, rms_unit="")
    x: TGES = dc.field(default_factory=_def_fac)
    y: TGES = dc.field(default_factory=_def_fac)


@dataclass
class Gain:
    x: float = 0.0
    y: float = 0.0


@dataclass
class RotationSpec1D:
    _def_fac = partial(TGES, rms=0.0, rms_unit="rad")

    roll: TGES = dc.field(default_factory=_def_fac)  # roll around z-axis [rad]


@dataclass
class Rotation1D:
    roll: float = 0.0  # roll around z-axis [rad]


@dataclass
class RotationSpec3D:
    _def_fac = partial(TGES, rms=0.0, rms_unit="rad")

    roll: TGES = dc.field(default_factory=_def_fac)  # roll around z-axis [rad]
    pitch: TGES = dc.field(default_factory=_def_fac)  # roll around x-axis [rad]
    yaw: TGES = dc.field(default_factory=_def_fac)  # roll around y-axis [rad]


@dataclass
class Rotation3D:
    roll: float = 0.0  # roll around z-axis [rad]
    pitch: float = 0.0  # roll around x-axis [rad]
    yaw: float = 0.0  # roll around y-axis [rad]


@dataclass
class NoiseSpec:
    _def_fac = partial(TGES, rms=0.0, rms_unit="m")
    x: TGES = dc.field(default_factory=_def_fac)
    y: TGES = dc.field(default_factory=_def_fac)


class BPMErrorSpec:
    def __init__(
        self,
        offset: Union[OffsetSpec2D, None] = None,
        gain: Union[GainSpec, None] = None,
        rot: Union[RotationSpec1D, None] = None,
        tbt_noise: Union[NoiseSpec, None] = None,
        co_noise: Union[NoiseSpec, None] = None,
    ) -> None:
        self.offset = OffsetSpec2D() if offset is None else offset
        self.gain = GainSpec() if gain is None else gain
        self.rot = RotationSpec1D() if rot is None else rot

        self.tbt_noise = NoiseSpec() if tbt_noise is None else tbt_noise
        self.closed_orbit_noise = NoiseSpec() if co_noise is None else co_noise


class BPMError:
    def __init__(
        self,
        elem_name: str,
        offset: Union[Offset2D, None] = None,
        gain: Union[Gain, None] = None,
        rot: Union[Rotation1D, None] = None,
    ) -> None:
        self.elem_name = elem_name
        self.offset = Offset2D() if offset is None else offset
        self.gain = Gain() if gain is None else gain
        self.rot = Rotation1D() if rot is None else rot

        assert isinstance(self.elem_name, str)
        assert isinstance(self.offset, Offset2D)
        assert isinstance(self.gain, Gain)
        assert isinstance(self.rot, Rotation1D)


class SupportErrorSpec3DRoll:
    def __init__(
        self,
        offset: Union[OffsetSpec3D, None] = None,
        rot: Union[RotationSpec3D, None] = None,
    ) -> None:
        self.offset = OffsetSpec3D() if offset is None else offset
        self.rot = RotationSpec3D() if rot is None else rot

        assert isinstance(self.offset, OffsetSpec3D)
        assert isinstance(self.rot, RotationSpec3D)


@dataclass
class SupportErrorSpec1DRoll:
    def __init__(
        self,
        us_offset: Union[OffsetSpec3D, None] = None,
        ds_offset: Union[OffsetSpec3D, None] = None,
        rot: Union[RotationSpec1D, None] = None,
    ) -> None:
        self.us_offset = OffsetSpec3D() if us_offset is None else us_offset
        self.ds_offset = OffsetSpec3D() if ds_offset is None else ds_offset
        self.rot = RotationSpec1D() if rot is None else rot

        assert isinstance(self.us_offset, OffsetSpec3D)
        assert isinstance(self.ds_offset, OffsetSpec3D)
        assert isinstance(self.rot, RotationSpec1D)


@dataclass
class SupportError:
    def __init__(
        self,
        elem_name: str,
        offset: Union[Offset3D, None] = None,
        rot: Union[Rotation3D, None] = None,
    ) -> None:
        self.elem_name = elem_name
        self.offset = Offset3D() if offset is None else offset
        self.rot = Rotation3D() if rot is None else rot

        assert isinstance(self.offset, Offset3D)
        assert isinstance(self.rot, Rotation3D)


@dataclass
class MainMultipoleErrorSpec:
    _def_fac = partial(TGES, rms=0.0, rms_unit="")
    # fse := Fractional Strength Error
    fse: TGES = dc.field(default_factory=_def_fac)


@dataclass
class MainMultipoleError:
    # fse := Fractional Strength Error
    fse: float = 0.0


class MultipoleErrorSpec:
    def __init__(
        self,
        n_main_poles: int,
        main_normal: bool,
        secondary_ref_radius: Union[float, None] = None,
        secondary_cutoff: float = 2.0,
        main_error: Union[MainMultipoleErrorSpec, None] = None,
    ) -> None:
        self._fields = []

        assert isinstance(n_main_poles, int)
        assert n_main_poles % 2 == 0
        self.n_main_poles = n_main_poles
        self._fields.append("n_main_poles")

        self.main_normal = main_normal
        self._fields.append("main_normal")

        # Must not be `None` if secondary multipole errors will be specified
        if secondary_ref_radius is not None:
            assert secondary_ref_radius > 0.0
        self.secondary_ref_radius = secondary_ref_radius
        self._fields.append("secondary_ref_radius")

        assert secondary_cutoff > 0.0
        self.secondary_cutoff = secondary_cutoff
        self._fields.append("secondary_cutoff")

        self.set_main_error_spec(main_error)
        self._fields.append("main_error")

        self.secondary_normal = {}
        self.secondary_skew = {}
        self._fields.append("secondary_normal")
        self._fields.append("secondary_skew")

    def set_main_error_spec(self, main_error_spec: Union[MainMultipoleErrorSpec, None]):
        if main_error_spec is None:
            self.main_error = MainMultipoleErrorSpec()
        else:
            assert isinstance(main_error_spec, MainMultipoleErrorSpec)
            self.main_error = main_error_spec

    def set_secondary_norm(self, n_poles, rms, cutoff=None, systematic=0.0):
        assert n_poles % 2 == 0
        if cutoff is None:
            cutoff = self.secondary_cutoff
        assert cutoff > 0.0
        self.secondary_normal[n_poles] = TGES(rms=rms, cutoff=cutoff, mean=systematic)

    def set_secondary_skew(self, n_poles, rms, cutoff=None, systematic=0.0):
        assert n_poles % 2 == 0
        if cutoff is None:
            cutoff = self.secondary_cutoff
        assert cutoff > 0.0
        self.secondary_skew[n_poles] = TGES(rms=rms, cutoff=cutoff, mean=systematic)

    def fields(self):
        return self._fields


class MultipoleError:
    def __init__(
        self,
        n_main_poles: Union[int, None] = None,
        main_normal: Union[bool, None] = None,
        secondary_ref_radius: Union[float, None] = None,
        secondary_cutoff: Union[float, None] = 2.0,
        main_error: Union[MainMultipoleError, None] = None,
        secondary_normal_error: Union[Dict, None] = None,
        secondary_skew_error: Union[Dict, None] = None,
    ) -> None:
        self._fields = []

        if n_main_poles is not None:
            self.set_n_main_poles(n_main_poles)
        else:
            self.n_main_poles = n_main_poles
        self._fields.append("n_main_poles")

        if main_normal is not None:
            self.set_main_normal(main_normal)
        else:
            self.main_normal = main_normal
        self._fields.append("main_normal")

        if secondary_ref_radius is not None:
            self.set_secondary_ref_radius(secondary_ref_radius)
        else:
            self.secondary_ref_radius = secondary_ref_radius
        self._fields.append("secondary_ref_radius")

        if secondary_cutoff is not None:
            self.set_secondary_cutoff(secondary_cutoff)
        else:
            self.secondary_cutoff = secondary_cutoff
        self._fields.append("secondary_cutoff")

        if main_error is not None:
            self.set_main_error(main_error)
        else:
            self.main_error = MainMultipoleError()
        self._fields.append("main_error")

        if secondary_normal_error is not None:
            self.set_secondary_normal_error(secondary_normal_error)
        else:
            self.secondary_normal_error = {}
        self._fields.append("secondary_normal_error")

        if secondary_skew_error is not None:
            self.set_secondary_skew_error(secondary_skew_error)
        else:
            self.secondary_skew_error = {}
        self._fields.append("secondary_skew_error")

    def fields(self):
        return self._fields

    def set_n_main_poles(self, n: int):
        assert isinstance(n, int)
        assert n % 2 == 0
        self.n_main_poles = n

    def set_main_normal(self, normal: bool):
        assert isinstance(normal, bool)
        self.main_normal = normal

    def set_secondary_ref_radius(self, radius: Union[float, None]):
        if radius is not None:
            assert isinstance(radius, float)
            assert radius > 0.0
        self.secondary_ref_radius = radius

    def set_secondary_cutoff(self, cutoff: float):
        assert isinstance(cutoff, float)
        assert cutoff > 0.0
        self.secondary_cutoff = cutoff

    def set_main_error(self, main_error: MainMultipoleError):
        assert isinstance(main_error, MainMultipoleError)
        self.main_error = main_error

    def set_secondary_normal_error_dict(self, normal_error: Dict):
        assert isinstance(normal_error, Dict)
        assert all([isinstance(v, int) for v in list(normal_error)])
        assert all([isinstance(v, float) for v in normal_error.values()])
        self.secondary_normal_error = normal_error

    def set_secondary_skew_error_dict(self, skew_error: Dict):
        assert isinstance(skew_error, Dict)
        assert all([isinstance(v, int) for v in list(skew_error)])
        assert all([isinstance(v, float) for v in skew_error.values()])
        self.secondary_skew_error = skew_error

    def set_secondary_normal_error(self, n_poles: int, error: float):
        assert isinstance(n_poles, int)
        assert n_poles % 2 == 0
        assert isinstance(error, float)
        self.secondary_normal_error[n_poles] = error

    def set_secondary_skew_error(self, n_poles: int, error: float):
        assert isinstance(n_poles, int)
        assert n_poles % 2 == 0
        assert isinstance(error, float)
        self.secondary_skew_error[n_poles] = error


class MagnetErrorSpec:
    def __init__(
        self,
        multipole: Union[MultipoleErrorSpec, None] = None,
        bending_angle: None = None,
        offset: Union[OffsetSpec2D, OffsetSpec3D, None] = None,
        rot: Union[RotationSpec1D, RotationSpec3D, None] = None,
    ) -> None:
        """
        Magnet - Multipole -- Main Multipole Error ------- FSE (Fractional Strength)
               |            |
               - Offset     - Secondary Multipole Errors - Normal
               |                                         |
               - Rotation                                - Skew
               |
               - Support
        """

        self._fields = []

        if multipole is None:
            self.multipole = None  # MultipoleErrorSpec()
        else:
            assert isinstance(multipole, MultipoleErrorSpec)
            self.multipole = multipole
        self._fields.append("multipole")

        if offset is None:
            self.offset = OffsetSpec2D()
        else:
            assert isinstance(offset, (OffsetSpec2D, OffsetSpec3D))
            self.offset = offset
        self._fields.append("offset")

        if rot is None:
            self.rot = RotationSpec1D()
        else:
            assert isinstance(rot, (RotationSpec1D, RotationSpec3D))
            self.rot = rot
        self._fields.append("rot")

    def fields(self):
        return self._fields


class MagnetError:
    def __init__(
        self,
        elem_name: str,
        offset: Union[Offset3D, None] = None,
        rot: Union[Rotation3D, None] = None,
        multipole: Union[MultipoleError, None] = None,
    ) -> None:
        self.elem_name = elem_name
        self.offset = Offset3D() if offset is None else offset
        self.rot = Rotation3D() if rot is None else rot
        self.multipole = multipole

        assert isinstance(self.elem_name, str)
        assert isinstance(self.offset, Offset3D)
        assert isinstance(self.rot, Rotation3D)
        assert (self.multipole is None) or isinstance(self.multipole, MultipoleError)

    def set_multipole_error(self, multipole_error: MultipoleError):
        assert isinstance(multipole_error, MultipoleError)
        self.multipole = multipole_error


class Errors:
    def __init__(
        self,
        design_LTE: ltemanager.Lattice,
        rng: Union[int, np.random.Generator] = 42,
        modified_name_prefix="",
        modified_name_suffix="",
    ) -> None:

        assert isinstance(design_LTE, ltemanager.Lattice)
        self.design_LTE = design_LTE

        self.used_beamline_name = self.design_LTE.used_beamline_name

        d = self.design_LTE.get_used_beamline_element_defs(
            used_beamline_name=self.used_beamline_name
        )
        self.beamline_defs = d["beamline_defs"]
        self.elem_defs = d["elem_defs"]
        self.flat_used_elem_names = d["flat_used_elem_names"]

        self.flat_LTE = self._individualize_families(
            modified_name_prefix, modified_name_suffix
        )

        self.lengths, self.s_ends, self.C = self.calc_spos()
        self._init_support_offsets()
        self._init_support_rotations()
        self.n_elems = len(self.flat_LTE.flat_used_elem_names) + 1  # +1 for __BEG__

        _flat_elem_names = [
            self.flat_LTE.flat_used_elem_names[i - 1] if i != 0 else "__BEG__"
            for i in range(self.n_elems)
        ]
        self.ring = [
            dict(name=name, elem_type=self.flat_LTE.get_elem_type_from_name(name))
            for name in _flat_elem_names
        ]

        self._supports_defined = {
            _type: np.zeros(self.n_elems).astype(bool) for _type in SupportType
        }

        self._dists = defaultdict(list)

        self.bpms = {}
        self._bpms_dist = defaultdict(list)

        self.supports = {}
        self._supports_dist = {}
        for _type in SupportType:
            self.supports[_type] = {}
            self._supports_dist[_type] = defaultdict(list)

        self.magnets = {}
        self._magnets_dist = defaultdict(list)

        if isinstance(rng, int):
            self.rng = np.random.default_rng(seed=rng)
        elif isinstance(rng, np.random.Generator):
            self.rng = rng
        else:
            raise TypeError("`rng` must be an integer or np.random.Generator")

        self.mod_prop_dict_list = []

    def _init_support_offsets(self):
        coord_strs = [f.name for f in dc.fields(Offset3D)]
        self.support_offsets = {
            coord: np.zeros_like(self.s_ends) for coord in coord_strs
        }

    def _init_support_rotations(self):
        coord_strs = [f.name for f in dc.fields(Rotation3D)]
        self.support_rots = {coord: np.zeros_like(self.s_ends) for coord in coord_strs}

    def _individualize_families(
        self, modified_name_prefix, modified_name_suffix, index_digits=3
    ):
        for s in [modified_name_prefix, modified_name_suffix]:
            assert all([c not in ":#*!()" + chr(34) + chr(39) + chr(44) for c in s])

        elem_names = [v[0] for v in self.elem_defs]

        kid_counts = defaultdict(int)
        elem_defs_inds = {}
        elem_defs_d = {}
        for orig_name in self.flat_used_elem_names:
            kid_counts[orig_name] += 1
            if orig_name not in elem_defs_d:
                index = elem_names.index(orig_name)
                elem_defs_inds[orig_name] = index
                elem_defs_d[orig_name] = self.elem_defs[index]

        max_counts = max(list(kid_counts.values()))
        index_digits = max([index_digits, int(np.ceil(np.log10(max_counts)))])

        sort_inds = np.argsort(
            [elem_def_ind for _, elem_def_ind in elem_defs_inds.items()]
        )[::-1]
        sorted_orig_names = np.array(list(elem_defs_inds))[sort_inds]
        assert np.unique(
            np.diff([elem_defs_inds[name] for name in sorted_orig_names])
        ) == np.array([-1])

        # Here `np.array(..., dtype=object)` is CRITICAL! Otherwise, if you try to
        # assign a new element name longer than the max length of the originial names,
        # the new name will be truncated, which is not what you want to happen.
        new_flat_used_elem_names = np.array(self.flat_used_elem_names[:], dtype=object)

        for orig_name in sorted_orig_names:
            counts = kid_counts[orig_name]
            if counts >= 2:
                elem_def = elem_defs_d[orig_name]
                assert elem_def[0] == orig_name
                matched_elem_inds = np.where(new_flat_used_elem_names == orig_name)[0]
                new_elem_defs = []
                for kid_index in range(counts):
                    index_str = f"{{:0{index_digits:d}d}}".format(kid_index)
                    new_indiv_name = f"{modified_name_prefix}{orig_name}{modified_name_suffix}_{index_str}"
                    new_elem_defs.append((new_indiv_name, elem_def[1], elem_def[2]))

                    ei = matched_elem_inds[kid_index]
                    assert new_flat_used_elem_names[ei] == orig_name
                    new_flat_used_elem_names[ei] = new_indiv_name

                _i = elem_defs_inds[orig_name]
                insert_slice = np.s_[_i : _i + 1]
                self.elem_defs[insert_slice] = new_elem_defs

        self.beamline_defs.clear()
        self.beamline_defs.append(
            (self.used_beamline_name, [(name, 1) for name in new_flat_used_elem_names])
        )
        self.flat_used_elem_names.clear()
        self.flat_used_elem_names.extend(new_flat_used_elem_names)

        tmp = tempfile.NamedTemporaryFile(
            prefix=f"tmpLteFlat_", suffix=".lte", dir=None, delete=True
        )
        new_LTE_filepath = Path(tmp.name)

        self.design_LTE.write_LTE(
            new_LTE_filepath,
            self.used_beamline_name,
            self.elem_defs,
            self.beamline_defs,
        )

        flat_LTE = ltemanager.Lattice(
            new_LTE_filepath, used_beamline_name=self.used_beamline_name
        )

        tmp.close()

        return flat_LTE

    def calc_spos(self):
        elem_defs = self.flat_LTE.get_used_beamline_element_defs()["elem_defs"]
        all_elem_names = [name for name, *_ in elem_defs]
        Ls = [0.0]  # Add zero-length for __BEG__
        for elem_name in self.flat_LTE.flat_used_elem_names:
            i = all_elem_names.index(elem_name)
            prop_str = elem_defs[i][2]
            L = self.flat_LTE.parse_elem_properties(prop_str).get("L", 0.0)
            Ls.append(L)
        Ls = np.array(Ls)

        s_ends = np.cumsum(Ls)
        # s_mids = s_ends - Ls / 2.0
        circumference = s_ends[-1]

        return Ls, s_ends, circumference

    def register_BPMs(
        self,
        elem_inds: List,
        err_spec: Union[BPMErrorSpec, None] = None,
        overwrite: bool = False,
    ) -> None:
        """Based on SCregisterBPMs.m"""

        if err_spec is None:
            err_spec = BPMErrorSpec()
        assert isinstance(err_spec, BPMErrorSpec)

        for ei in elem_inds:
            if ei in self.bpms:
                if not overwrite:
                    msg = f"BPM error already specified for Element Index {ei}. Set `overwrite=True` to ignore this."
                    raise RuntimeError(msg)
            self.bpms[ei] = err_spec

    def register_supports(
        self,
        support_type: SupportType,
        us_elem_inds: np.ndarray,
        ds_elem_inds: np.ndarray,
        err_spec: Union[SupportErrorSpec3DRoll, SupportErrorSpec1DRoll, None] = None,
        overwrite: bool = False,
    ) -> None:
        """Based on SCregisterSupport.m"""

        assert isinstance(support_type, SupportType)
        supports = self.supports[support_type]

        assert len(us_elem_inds) == len(ds_elem_inds)

        # Support edge elements must be zero-length elements
        assert np.all(self.lengths[us_elem_inds] == 0.0)
        assert np.all(self.lengths[ds_elem_inds] == 0.0)

        if err_spec is None:
            err_spec = SupportErrorSpec1DRoll()
        assert isinstance(err_spec, (SupportErrorSpec3DRoll, SupportErrorSpec1DRoll))

        for us_ei, ds_ei in zip(us_elem_inds, ds_elem_inds):
            if (us_ei, ds_ei) in supports:
                if not overwrite:
                    msg = f"Support error already specified for Element Index Pair {(us_ei, ds_ei)}. Set `overwrite=True` to ignore this."
                    raise RuntimeError(msg)

            if us_ei < ds_ei:
                s_ = slice(us_ei, ds_ei + 1)
                if np.any(self._supports_defined[support_type][s_]):
                    raise ValueError(
                        f"Support ({support_type.name}) error is already defined between Element Index {us_ei} and {ds_ei}."
                    )
                wrap_around = False
            elif us_ei == ds_ei:
                raise ValueError(
                    "Upstream and downstream support edge element index cannot be the same"
                )
            else:
                wrap_around = True
                for s_ in [slice(us_ei, None), slice(0, ds_ei + 1)]:
                    if np.any(self._supports_defined[support_type][s_]):
                        raise ValueError(
                            f"Support ({support_type.name}) error is already defined between Element Index {us_ei} and {ds_ei} (wrapped)."
                        )

            supports[(us_ei, ds_ei)] = err_spec

            if not wrap_around:
                self._supports_defined[support_type][us_ei : ds_ei + 1] = True
            else:
                self._supports_defined[support_type][us_ei:] = True
                self._supports_defined[support_type][:ds_ei] = True

    def register_magnets(
        self,
        elem_inds: List,
        err_spec: Union[MagnetErrorSpec, None] = None,
        overwrite: bool = False,
    ) -> None:
        """Based on SCregisterMagnets.m"""

        if err_spec is None:
            err_spec = MagnetErrorSpec()
        assert isinstance(err_spec, MagnetErrorSpec)

        for ei in elem_inds:
            if ei in self.magnets:
                if not overwrite:
                    msg = f"Magnet error already specified for Element Index {ei}. Set `overwrite=True` to ignore this."
                    raise RuntimeError(msg)
            self.magnets[ei] = err_spec

    def get_dist(self, rms, cutoff=2.0, mean=0.0):
        k = (rms, cutoff, mean)
        if k not in self._dists:
            self._dists[k] = truncnorm(-cutoff, +cutoff, loc=mean, scale=rms)

        return self._dists[k]

    def apply_errors(self):
        """Based on SCapplyErrors.m

        Apply errors to lattice and diagnostic devices.
        """

        # self._apply_cavity_errors()
        # self._apply_injection_errors()
        self._apply_BPM_errors()
        # self._apply_circumference_error()
        self._apply_support_alignment_errors()
        self._apply_magnet_errors()
        self._update_support()
        # self._update_magnetic_fields()
        # self._update_cavities()

    def _apply_cavity_errors(self):
        raise NotImplementedError

    def _apply_injection_errors(self):
        raise NotImplementedError

    def _apply_BPM_errors(self):
        self._bpms_dist.clear()

        for ei in np.sort(list(self.bpms)):
            elem_name = self.flat_LTE.get_names_from_elem_inds(ei)

            spec = self.bpms[ei]

            for v, prop_path, expected_unit in [
                (spec.offset.x, ["offset", "x"], "m"),
                (spec.offset.y, ["offset", "y"], "m"),
                (spec.rot.roll, ["rot", "roll"], "rad"),
                (spec.gain.x, ["gain", "x"], ""),
                (spec.gain.y, ["gain", "y"], ""),
            ]:
                rms = v.rms
                cutoff = v.cutoff
                mean = v.mean
                assert v.rms_unit == expected_unit

                self._bpms_dist[(rms, cutoff, mean)].append((ei, elem_name, prop_path))

        for (rms, cutoff, mean), elem_ind_elem_prop_paths in self._bpms_dist.items():
            if rms == 0.0:
                continue
            dist = self.get_dist(rms, cutoff, mean)
            prop_vals = dist.rvs(
                size=len(elem_ind_elem_prop_paths), random_state=self.rng
            )

            for (ei, elem_name, prop_path), prop_val in zip(
                elem_ind_elem_prop_paths, prop_vals
            ):
                assert self.ring[ei]["name"] == elem_name
                bpm_err = self.ring[ei].get("bpm", BPMError(elem_name))
                obj = bpm_err
                for prop in prop_path[:-1]:
                    obj = getattr(obj, prop)
                setattr(obj, prop_path[-1], prop_val)
                self.ring[ei]["bpm"] = bpm_err

    def _apply_circumference_error(self):
        raise NotImplementedError

    def _apply_support_alignment_errors(self):
        """Based on applySupportAlignmentError() in SCapplyErrors.m"""

        for _type in [SupportType.section, SupportType.plinth, SupportType.girder]:
            self._supports_dist[_type].clear()

            copy_errs = {}

            for (us_ei, ds_ei), spec in self.supports[_type].items():
                us_elem_name, ds_elem_name = self.flat_LTE.get_names_from_elem_inds(
                    [us_ei, ds_ei]
                )

                if us_ei < ds_ei:
                    struct_len = self.s_ends[ds_ei] - self.s_ends[us_ei - 1]
                elif us_ei > ds_ei:
                    struct_len = (self.C - self.s_ends[us_ei - 1]) + self.s_ends[ds_ei]
                else:
                    struct_len = 0.0

                if struct_len == 0.0:
                    print(
                        f"WARNING: zero-length support structure detected ({_type}, {us_ei} [{us_elem_name}] - {ds_ei} [{ds_elem_name}])"
                    )
                    continue

                if isinstance(spec, SupportErrorSpec1DRoll):
                    for v, ei, name, prop_path in [
                        (spec.us_offset.x, us_ei, us_elem_name, ["offset", "x"]),
                        (spec.us_offset.y, us_ei, us_elem_name, ["offset", "y"]),
                        (spec.ds_offset.x, ds_ei, ds_elem_name, ["offset", "x"]),
                        (spec.ds_offset.y, ds_ei, ds_elem_name, ["offset", "y"]),
                        (spec.rot.roll, us_ei, us_elem_name, ["rot", "roll"]),
                    ]:
                        rms = v.rms
                        cutoff = v.cutoff
                        if prop_path[0] == "offset":
                            assert v.rms_unit == "m"
                        elif prop_path[0] == "rot":
                            assert v.rms_unit == "rad"
                        else:
                            raise ValueError(prop_path[0])
                        mean = v.mean

                        self._supports_dist[_type][(rms, cutoff, mean)].append(
                            (ei, name, prop_path)
                        )

                elif isinstance(spec, SupportErrorSpec3DRoll):
                    copy_errs[ds_ei] = dict(src_ei=us_ei, elem_name=ds_elem_name)

                    for v, ei, name, prop_path in [
                        (spec.offset.x, us_ei, us_elem_name, ["offset", "x"]),
                        (spec.offset.y, us_ei, us_elem_name, ["offset", "y"]),
                        (spec.rot.roll, us_ei, us_elem_name, ["rot", "roll"]),
                        (spec.rot.pitch, us_ei, us_elem_name, ["rot", "pitch"]),
                        (spec.rot.yaw, us_ei, us_elem_name, ["rot", "yaw"]),
                    ]:
                        rms = v.rms
                        cutoff = v.cutoff
                        assert v.rms_unit == "m"

                        self._supports_dist[_type][(rms, cutoff, mean)].append(
                            (ei, name, prop_path)
                        )

                else:
                    raise TypeError(spec)

            for (rms, cutoff, mean), elem_ind_elem_prop_paths in self._supports_dist[
                _type
            ].items():
                if rms != 0.0:
                    dist = self.get_dist(rms, cutoff, mean)
                    prop_vals = dist.rvs(
                        size=len(elem_ind_elem_prop_paths), random_state=self.rng
                    )
                else:
                    prop_vals = np.zeros(len(elem_ind_elem_prop_paths))

                for (ei, elem_name, prop_path), prop_val in zip(
                    elem_ind_elem_prop_paths, prop_vals
                ):
                    assert self.ring[ei]["name"] == elem_name
                    sup_err = self.ring[ei].get(_type.name, SupportError(elem_name))
                    obj = sup_err
                    for prop in prop_path[:-1]:
                        obj = getattr(obj, prop)
                    setattr(obj, prop_path[-1], prop_val)
                    self.ring[ei][_type.name] = sup_err

            for ds_ei, d in copy_errs.items():
                us_ei = d["src_ei"]
                ds_elem_name = d["elem_name"]
                assert self.ring[ds_ei]["name"] == ds_elem_name
                src_sup_err = self.ring[us_ei][_type.name]
                self.ring[ds_ei][_type.name] = SupportError(**dc.asdict(src_sup_err))

            for us_ei, ds_ei in list(self.supports[_type]):
                us_sup_err = self.ring[us_ei][_type.name]
                ds_sup_err = self.ring[ds_ei][_type.name]

                if us_sup_err.rot.pitch != 0.0:
                    dy = us_sup_err.rot.pitch * struct_len / 2
                    # Adjust vertical offset for the support-structure start-element
                    us_sup_err.offset.y -= dy
                    # Adjust vertical offset for the support-structure end-element
                    ds_sup_err.offset.y += dy
                else:
                    us_sup_err.rot.pitch = (
                        ds_sup_err.offset.y - us_sup_err.offset.y
                    ) / struct_len

                if us_sup_err.rot.yaw != 0.0:
                    dx = us_sup_err.rot.yaw * struct_len / 2
                    # Adjust horiz. offset for the support-structure start-element
                    us_sup_err.offset.x -= dx
                    # Adjust horiz. offset for the support-structure end-element
                    ds_sup_err.offset.x += dx
                else:
                    us_sup_err.rot.yaw = (
                        ds_sup_err.offset.x - us_sup_err.offset.x
                    ) / struct_len

    def _apply_magnet_errors(self):
        self._magnets_dist.clear()

        for ei in np.sort(list(self.magnets)):
            elem_name = self.flat_LTE.get_names_from_elem_inds(ei)

            magnet_err = MagnetError(elem_name)
            mpole_err = MultipoleError()
            magnet_err.set_multipole_error(mpole_err)
            self.ring[ei]["magnet"] = magnet_err

            spec = self.magnets[ei]

            for prop_name in spec.fields():
                # print(prop_name)
                spec2 = getattr(spec, prop_name)
                if prop_name == "multipole":
                    if spec2 is None:  # No multipole error specified
                        continue
                    for prop2_name in spec2.fields():
                        if prop2_name == "main_error":
                            main_err_spec = getattr(spec2, prop2_name)
                            assert isinstance(main_err_spec, MainMultipoleErrorSpec)
                            for fld in dc.fields(main_err_spec):
                                prop3_name = fld.name
                                spec4 = getattr(main_err_spec, prop3_name)
                                prop_path = [prop_name, prop2_name, prop3_name]
                                assert isinstance(spec4, TGES)
                                unit = spec4.rms_unit
                                assert unit == ""
                                self._magnets_dist[
                                    (spec4.rms, spec4.cutoff, spec4.mean)
                                ].append((ei, elem_name, prop_path))
                        elif prop2_name in (
                            "n_main_poles",
                            "main_normal",
                            "secondary_ref_radius",
                            "secondary_cutoff",
                        ):
                            prop_val = getattr(spec2, prop2_name)
                            set_method = getattr(mpole_err, f"set_{prop2_name}")
                            set_method(prop_val)
                        elif prop2_name in ("secondary_normal", "secondary_skew"):
                            spec3_d = getattr(spec2, prop2_name)
                            for n_poles, sec_spec in spec3_d.items():
                                assert isinstance(sec_spec, TGES)
                                unit = sec_spec.rms_unit
                                assert unit == ""
                                prop_path = [prop_name, prop2_name, n_poles]
                                self._magnets_dist[
                                    (sec_spec.rms, sec_spec.cutoff, sec_spec.mean)
                                ].append((ei, elem_name, prop_path))
                        else:
                            raise ValueError(prop2_name)
                elif hasattr(spec2, "fields"):
                    # print(spec2.fields())
                    for prop2_name in spec2.fields():
                        spec3 = getattr(spec2, prop2_name)

                        if isinstance(spec3, TGES):
                            raise RuntimeError("This shouldn't be reachable")
                        elif isinstance(spec3, MainMultipoleErrorSpec):
                            for fld in dc.fields(spec3):
                                prop3_name = fld.name
                                spec4 = getattr(spec3, prop3_name)
                                prop_path = [prop_name, prop2_name, prop3_name]
                                assert isinstance(spec4, TGES)
                                unit = spec4.rms_unit
                                assert unit == ""
                                # print(f"{prop_path}, {spec4}, TGES")
                                self._magnets_dist[
                                    (spec4.rms, spec4.cutoff, spec4.mean)
                                ].append((ei, elem_name, prop_path))
                        elif isinstance(spec3, dict):
                            for n_poles, spec4 in spec3.items():
                                prop_path = [prop_name, prop2_name, n_poles]
                                assert isinstance(spec4, TGES)
                                unit = spec4.rms_unit
                                assert unit == ""
                                # print(f"{prop_path}, {spec4}, TGES")
                                self._magnets_dist[
                                    (spec4.rms, spec4.cutoff, spec4.mean)
                                ].append((ei, elem_name, prop_path))

                        else:
                            raise RuntimeError("This shouldn't be reachable")

                else:
                    # print([fld.name for fld in dc.fields(spec2)])
                    for fld in dc.fields(spec2):
                        prop2_name = fld.name
                        spec3 = getattr(spec2, prop2_name)
                        # print(f"{prop2_name}, {spec3}, {isinstance(spec3, TGES)}")

                        if isinstance(spec3, TGES):
                            prop_path = [prop_name, prop2_name]
                            unit = spec3.rms_unit
                            if prop2_name in ("x", "y", "z"):
                                assert unit == "m"
                            elif prop2_name in ("roll", "pitch", "yaw"):
                                assert unit == "rad"

                            self._magnets_dist[
                                (spec3.rms, spec3.cutoff, spec3.mean)
                            ].append((ei, elem_name, prop_path))

        for (rms, cutoff, mean), elem_ind_elem_prop_paths in self._magnets_dist.items():
            if rms == 0.0:
                if mean == 0.0:
                    continue

                prop_vals = np.ones(len(elem_ind_elem_prop_paths)) * mean
            else:
                dist = self.get_dist(rms, cutoff, mean)
                prop_vals = dist.rvs(
                    size=len(elem_ind_elem_prop_paths), random_state=self.rng
                )

            assert len(elem_ind_elem_prop_paths) == len(prop_vals)
            for (ei, elem_name, prop_path), prop_val in zip(
                elem_ind_elem_prop_paths, prop_vals
            ):
                assert self.ring[ei]["name"] == elem_name
                magnet_err = self.ring[ei]["magnet"]
                obj = magnet_err
                sec_err_setter = None
                for prop in prop_path[:-1]:
                    if prop == "secondary_normal":
                        sec_err_setter = obj.set_secondary_normal_error
                    elif prop == "secondary_skew":
                        sec_err_setter = obj.set_secondary_skew_error
                    else:
                        obj = getattr(obj, prop)

                if sec_err_setter:
                    sec_err_setter(prop_path[-1], prop_val)
                else:
                    setattr(obj, prop_path[-1], prop_val)

    def _update_support(self):
        """Based on SCupdateSupport.m"""

        # Update support offsets/rotations at all elements
        self._calc_support_offsets()
        self._calc_support_rotations()

    def _update_magnetic_fields(self):
        raise NotImplementedError

    def _update_cavities(self):
        raise NotImplementedError

    def _calc_support_offsets(self):
        """Calculates the combined support structure offset
        (Based on SCgetSupportOffset.m)
        """

        self._init_support_offsets()
        coord_strs = list(self.support_offsets)

        sup_elem_inds = {}

        for _type in [SupportType.section, SupportType.plinth, SupportType.girder]:
            if not self.supports[_type]:
                continue

            sup_elem_inds["us"], sup_elem_inds["ds"] = [
                np.array(tup) for tup in zip(*list(self.supports[_type]))
            ]

            offsets_sup_edges = {side: {} for side in list(sup_elem_inds)}

            for side, eis in sup_elem_inds.items():
                offset_objs = [self.ring[ei][_type.name].offset for ei in eis]
                for coord in coord_strs:
                    extra_offset_vals = np.array(
                        [getattr(obj, coord) for obj in offset_objs]
                    )
                    offsets_sup_edges[side][coord] = (
                        self.support_offsets[coord][eis] + extra_offset_vals
                    )

            # Interpolate between US/DS support edges
            for iSup, (us_ei, ds_ei) in enumerate(
                zip(sup_elem_inds["us"], sup_elem_inds["ds"])
            ):
                if us_ei < ds_ei:
                    xp = np.array([self.s_ends[us_ei], self.s_ends[ds_ei]])
                    assert np.all(np.diff(xp) > 0.0)  # make sure monotonic increase
                    roi = np.s_[us_ei : ds_ei + 1]  # (ROI) Region of Interpolation
                    x = self.s_ends[roi]
                    for coord in coord_strs:
                        fp = np.array(
                            [
                                offsets_sup_edges["us"][coord][iSup],
                                offsets_sup_edges["ds"][coord][iSup],
                            ]
                        )
                        self.support_offsets[coord][roi] = np.interp(
                            x, xp, fp, left=np.nan, right=np.nan
                        )
                elif us_ei > ds_ei:
                    xp = np.array([self.s_ends[us_ei], self.C + self.s_ends[ds_ei]])
                    assert np.all(np.diff(xp) > 0.0)  # make sure monotonic increase
                    # (ROI) Region of Interpolation
                    roi_1 = np.s_[us_ei:]
                    roi_2 = np.s_[: ds_ei + 1]
                    x = np.append(self.s_ends[roi_1], self.C + self.s_ends[roi_2])
                    n_wrap = len(self.s_ends[roi_1])
                    for coord in coord_strs:
                        fp = np.array(
                            [
                                offsets_sup_edges["us"][coord][iSup],
                                offsets_sup_edges["ds"][coord][iSup],
                            ]
                        )
                        f_interp = np.interp(x, xp, fp, left=np.nan, right=np.nan)
                        self.support_offsets[coord][roi_1] = f_interp[:n_wrap]
                        self.support_offsets[coord][roi_2] = f_interp[n_wrap:]
                else:
                    pass  # There is nothing to update.

        if False:
            rough_last_support_len = 10.0

            for coord in ["x", "y", "z"]:
                plt.figure()
                plt.subplot(211)
                plt.plot(self.s_ends, self.support_offsets[coord], ".-")
                plt.ylabel(coord)
                plt.subplot(212)
                sl_end = self.s_ends > (self.C - rough_last_support_len / 2)
                sl_beg = self.s_ends < (rough_last_support_len / 2)
                plt.plot(
                    np.append(self.s_ends[sl_end], self.C + self.s_ends[sl_beg]),
                    np.append(
                        self.support_offsets[coord][sl_end],
                        self.support_offsets[coord][sl_beg],
                    ),
                    ".-",
                )
                plt.ylabel(coord)
                plt.axvline(self.C, color="k")
                plt.xlabel("s [m]")
                plt.tight_layout()

    def _calc_support_rotations(self):
        """Calculates the combined support structure rotation (roll, pitch, & yaw angles)
        (Based on SCgetSupportRoll.m)

        The roll angle is a sum of the roll angles of all underlying support structures.
        """

        self._init_support_rotations()

        sup_off = self.support_offsets
        s_ends = self.s_ends

        sup_elem_inds = {}

        for _type in [SupportType.section, SupportType.plinth, SupportType.girder]:
            if not self.supports[_type]:
                continue

            sup_elem_inds["us"], sup_elem_inds["ds"] = [
                np.array(tup) for tup in zip(*list(self.supports[_type]))
            ]

            for iSup, (us_ei, ds_ei) in enumerate(
                zip(sup_elem_inds["us"], sup_elem_inds["ds"])
            ):
                roll = self.ring[us_ei][_type.name].rot.roll
                dx = sup_off["x"][ds_ei] - sup_off["x"][us_ei]
                dy = sup_off["y"][ds_ei] - sup_off["y"][us_ei]

                if us_ei < ds_ei:
                    s_ = np.s_[us_ei : ds_ei + 1]
                    # Simply add roll error from current support structure
                    self.support_rots["roll"][s_] += roll
                    # Overwite pitch and yaw angles from current support structure x & y offsets
                    distance = s_ends[ds_ei] - s_ends[us_ei]
                    self.support_rots["pitch"][s_] = dy / distance
                    self.support_rots["yaw"][s_] = dx / distance

                elif us_ei > ds_ei:
                    # Update the US support edge to the ring end
                    s_ = np.s_[us_ei:]
                    # -- Simply add roll error from current support structure
                    self.support_rots["roll"][s_] += roll
                    # -- Overwrite pitch angle from current support structure y offsets
                    distance = s_ends[ds_ei] + (self.C - s_ends[us_ei])
                    pitch = dy / distance
                    self.support_rots["pitch"][s_] = pitch
                    # -- Overwrite yaw angle from current support structure x offsets
                    yaw = dx / distance
                    self.support_rots["yaw"][s_] = yaw

                    # Update the ring beginning to the DS support edge
                    s_ = np.s_[: ds_ei + 1]
                    # -- Simply add roll error from current support structure
                    self.support_rots["roll"][s_] += roll
                    # -- Overwrite pitch angle from current support structure y offsets
                    self.support_rots["pitch"][s_] = pitch
                    # -- Overwrite yaw angle from current support structure x offsets
                    self.support_rots["yaw"][s_] = yaw
                else:
                    pass  # There is nothing to update.

        if False:
            plt.figure()
            plt.plot(self.s_ends, self.support_rots["roll"], ".-")

            plt.figure()
            plt.plot(self.s_ends, self.support_rots["pitch"], ".-")

            plt.figure()
            plt.plot(self.s_ends, self.support_rots["yaw"], ".-")

    def generate_LTE_file_wo_errors(
        self, output_LTE_filepath="", output_LTEZIP_filepath=""
    ):
        """This method generates an LTE/LTEZIP file with all the families with multiple
        occurrences individualized just as error instance LTE/LTEZIP files would be.

        This will be convenient when computing the design response matrices and their
        subsequent uses for optics corrections for an LTE with errors, for which the
        families are always individualized to avoid duplicate element names."""

        mods = {}

        if output_LTE_filepath != "":
            ltemanager.write_modified_LTE(
                output_LTE_filepath,
                mods,
                LTE_obj=self.flat_LTE,
            )
        elif output_LTEZIP_filepath != "":
            temp_LTE_filepath = ltemanager.write_temp_modified_LTE(
                mods, LTE_obj=self.flat_LTE
            )
            new_LTE = ltemanager.Lattice(
                temp_LTE_filepath, used_beamline_name=self.flat_LTE.used_beamline_name
            )

            new_LTE.zip_lte(output_LTEZIP_filepath)
            try:
                temp_LTE_filepath.unlink()
            except:
                pass
        else:
            raise ValueError(
                "Both `output_LTE_filepath` and `output_LTEZIP_filepath` cannot be empty strings."
            )

    def generate_LTE_file(self, output_LTEZIP_filepath):
        self.mod_prop_dict_list.clear()
        mods = self.mod_prop_dict_list

        MALIGN_METHOD = 2  # 0: old, 1: entrance-centered, 2: body-centered
        REFERENCE_CORRECTION = 1  # If nonzero, reference trajectory is subtracted from particle trajectories to compensate for inaccuracy in integration.

        support_type_names = [enum.name for enum in SupportType]

        temp_suppl_filepaths = []

        for ei, err_d in enumerate(self.ring):
            # For an element, both "bpm" and "magnet" errors shoud not have been specified.
            assert not (("bpm" in err_d) and ("magnet" in err_d))

            err_type_list = list(err_d)
            if "bpm" in err_type_list:
                err_type = "bpm"
                v = err_d["bpm"]
                elem_name = v.elem_name
            elif "magnet" in err_type_list:
                err_type = "magnet"
                v = err_d["magnet"]
                elem_name = v.elem_name
            else:
                for support_type in support_type_names:
                    if support_type in err_type_list:
                        err_type = "support"
                        v = err_d[support_type]
                        elem_name = v.elem_name
                        break
                else:
                    err_type = None

            if err_type is None:
                pass  # No error specified. No modification needed.

            elif err_type == "support":

                _update_AVAIL_ELEM_PROPS()
                elem_type = self.ring[ei]["elem_type"]

                for coord in "xyz":
                    prop_val = self.support_offsets[coord][ei]
                    if prop_val != 0.0:
                        prop_name = f"d{coord}".upper()
                        if elem_type not in AVAIL_ELEM_PROPS[prop_name]:
                            continue
                        mods.append(
                            dict(
                                elem_name=elem_name,
                                prop_name=prop_name,
                                prop_val=prop_val,
                            )
                        )

                prop_val = self.support_rots["roll"][ei]
                if prop_val != 0.0:
                    if elem_type in AVAIL_ELEM_PROPS["ETILT"]:
                        prop_name = "ETILT"
                    elif elem_type in AVAIL_ELEM_PROPS["TILT"]:
                        prop_name = "TILT"
                    else:
                        continue
                    mods.append(
                        dict(
                            elem_name=elem_name,
                            prop_name=prop_name,
                            prop_val=prop_val,
                        )
                    )

                for k in ["pitch", "yaw"]:
                    prop_val = self.support_rots[k][ei]
                    if prop_val != 0.0:
                        if elem_type in AVAIL_ELEM_PROPS[f"E{k.upper()}"]:
                            prop_name = f"E{k.upper()}"
                        elif elem_type in AVAIL_ELEM_PROPS[k.upper()]:
                            prop_name = k.upper()
                        else:
                            continue
                        mods.append(
                            dict(
                                elem_name=elem_name,
                                prop_name=prop_name,
                                prop_val=prop_val,
                            )
                        )

            elif err_type == "bpm":
                for coord in "xy":
                    prop_val = self.support_offsets[coord][ei]
                    prop_val += getattr(v.offset, coord)
                    if prop_val != 0.0:
                        mods.append(
                            dict(
                                elem_name=elem_name,
                                prop_name=f"d{coord}".upper(),
                                prop_val=prop_val,
                            )
                        )

                prop_val = self.support_rots["roll"][ei]
                prop_val += v.rot.roll
                if prop_val != 0.0:
                    mods.append(
                        dict(
                            elem_name=elem_name,
                            prop_name="TILT",
                            prop_val=prop_val,
                        )
                    )

                for coord in "xy":
                    prop_val = getattr(v.gain, coord)
                    if prop_val != 0.0:
                        mods.append(
                            dict(
                                elem_name=elem_name,
                                prop_name=f"{coord}CALIBRATION".upper(),
                                prop_val=prop_val + 1.0,
                            )
                        )

            elif err_type == "magnet":

                mpole_err = v.multipole

                is_bend = mpole_err.n_main_poles == 2

                misaligned = False

                for coord in "xyz":
                    prop_val = self.support_offsets[coord][ei]
                    prop_val += getattr(v.offset, coord)
                    if prop_val != 0.0:
                        misaligned = True
                        mods.append(
                            dict(
                                elem_name=elem_name,
                                prop_name=f"d{coord}".upper(),
                                prop_val=prop_val,
                            )
                        )

                # Simply add total roll/pitch/yaw errors from support structures
                for rot_type, ELEGANT_prop_name in [
                    ("roll", "TILT"),
                    ("pitch", "PITCH"),
                    ("yaw", "YAW"),
                ]:
                    prop_val = self.support_rots[rot_type][ei]
                    prop_val += getattr(v.rot, rot_type)
                    if prop_val != 0.0:
                        misaligned = True
                        if is_bend:
                            prop_name = f"E{ELEGANT_prop_name}"
                        else:
                            prop_name = ELEGANT_prop_name
                        mods.append(
                            dict(
                                elem_name=elem_name,
                                prop_name=prop_name,
                                prop_val=prop_val,
                            )
                        )

                if is_bend and (REFERENCE_CORRECTION != 0):
                    mods.append(
                        dict(
                            elem_name=elem_name,
                            prop_name="REFERENCE_CORRECTION",
                            prop_val=REFERENCE_CORRECTION,
                        )
                    )

                if misaligned and (MALIGN_METHOD != 0):
                    mods.append(
                        dict(
                            elem_name=elem_name,
                            prop_name="MALIGN_METHOD",
                            prop_val=MALIGN_METHOD,
                        )
                    )

                fse = mpole_err.main_error.fse
                if fse != 0.0:
                    mods.append(
                        dict(
                            elem_name=elem_name,
                            prop_name="FSE",
                            prop_val=f'{fse:.9e}"',
                        )
                    )

                norm_err_d = mpole_err.secondary_normal_error
                skew_err_d = mpole_err.secondary_skew_error

                u_n_poles = np.unique(list(norm_err_d) + list(skew_err_d))
                orders = (u_n_poles - 2) // 2

                norm_ar = np.array(
                    [norm_err_d.get(_n_poles, 0.0) for _n_poles in u_n_poles]
                )
                skew_ar = np.array(
                    [skew_err_d.get(_n_poles, 0.0) for _n_poles in u_n_poles]
                )

                if np.all(norm_ar == 0.0) and np.all(skew_ar == 0.0):
                    # No error specified. Proceed without modification.
                    continue

                sdds_params = {"referenceRadius": mpole_err.secondary_ref_radius}
                sdds_columns = {"normal": norm_ar, "skew": skew_ar, "order": orders}
                sdds_filepath = Path.cwd() / f"{elem_name}.MULT"
                temp_suppl_filepaths.append(sdds_filepath)
                sdds.dicts2sdds(
                    sdds_filepath,
                    params=sdds_params,
                    params_units={
                        "referenceRadius": "m"
                    },  # CRUCIAL: Without this, ELEGANT will crash!
                    columns=sdds_columns,
                    outputMode="binary",
                )
                mods.append(
                    dict(
                        elem_name=elem_name,
                        prop_name="SYSTEMATIC_MULTIPOLES",
                        prop_val=f'"{sdds_filepath.name}"',
                    )
                )

            else:
                raise ValueError(err_type)

        temp_LTE_filepath = ltemanager.write_temp_modified_LTE(
            mods, LTE_obj=self.flat_LTE
        )
        new_LTE = ltemanager.Lattice(
            temp_LTE_filepath, used_beamline_name=self.flat_LTE.used_beamline_name
        )
        new_LTE.zip_lte(output_LTEZIP_filepath)
        try:
            temp_LTE_filepath.unlink()
        except:
            pass

        for fp in temp_suppl_filepaths:
            try:
                fp.unlink()
            except:
                pass


if __name__ == "__main__":

    import sys

    import pyelegant as pe

    if False:
        LTE = pe.ltemanager.Lattice(
            "/epics/aphla/apconf_v2/nsls2/models/SR/pyelegant/LTEs/20230915_aphla_19ids_w_xbpms_MAG_17ids.lte"
        )
        LTE.zip_lte("temp.ltezip")
        sys.exit(0)
    elif False:
        pe.ltemanager.Lattice.unzip_lte(
            "temp.ltezip",
            output_lte_filepath_str="20230915_aphla_19ids_w_xbpms_MAG_17ids.lte",
            suppl_files_folderpath_str="./temp_lte_suppl",
        )
        sys.exit(0)
