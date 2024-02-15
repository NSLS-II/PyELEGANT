import collections
from collections import defaultdict
from dataclasses import dataclass
from enum import IntEnum
import gzip
import json
from pathlib import Path
import re
import tempfile
from typing import Dict, List, Tuple, Type, Union

import numpy as np

OutputType = IntEnum("OutputType", ["List", "Dict", "NumPy"])

PY_TYPES = {"np_array_or_list": Union[np.ndarray, List]}

ELEGANT_ELEM_DICT = {}


@dataclass
class ElementNameOccurrencePair:
    name: str
    occurrence_num: int  # start from 1


@dataclass
class ElementNameInstancePair:
    name: str
    instance_index: int  # start from 0


########################################################################
class Lattice:
    """ """

    def __init__(
        self,
        LTE_filepath: Union[Path, str] = "",
        LTEZIP_filepath: Union[Path, str] = "",
        used_beamline_name: str = "",
        tempdir_path: Union[Path, str, None] = None,
        parallel: bool = False,
        del_tempdir_on_exit: bool = True,
        verbose: int = 0,
    ):
        if used_beamline_name is None:
            used_beamline_name = ""

        self.handled_element_types = [
            "DRIF",
            "EDRIFT",
            "RFCA",
            "CSBEND",
            "CSBEN",
            "SBEN",
            "SBEND",
            "CCBEND",
            "KQUAD",
            "QUAD",
            "KSEXT",
            "SEXT",
            "KOCT",
            "OCTU",
            "MULT",
            "UKICKMAP",
            "HKICK",
            "VKICK",
            "KICKER",
            "EHKICK",
            "EVKICK",
            "EKICKER",
            "MARK",
            "MONI",
            "SCRAPER",
            #'SOLE',
            "MALIGN",
            "WATCH",
        ]

        self.tempdir = None
        self._LTE_suppl_files_folderpath = None
        self.del_tempdir_on_exit = del_tempdir_on_exit

        self.verbose = verbose

        self.parallel = parallel  # ensure visibility of temp. directory, if set to True

        if (LTE_filepath != "") or (LTEZIP_filepath != ""):
            self.load_LTE(
                LTE_filepath=LTE_filepath,
                LTEZIP_filepath=LTEZIP_filepath,
                used_beamline_name=used_beamline_name,
                tempdir_path=tempdir_path,
                del_tempdir_on_exit=del_tempdir_on_exit,
            )

        self._persistent_LTE_d = None

    def _clean_up_LTE_text(self):
        """"""

        self.cleaned_LTE_text = "\n" + self.LTE_text
        # ^ adding "\n" at the beginning for easier search
        self.cleaned_LTE_text = self.remove_comments(self.cleaned_LTE_text)
        self.cleaned_LTE_text = self.delete_ampersands(self.cleaned_LTE_text)

    @staticmethod
    def temp_unzip_ltezip(
        LTEZIP_filepath,
        tempdir_path=None,
        del_tempdir_on_exit: bool = True,
        verbose: int = 0,
        parallel: bool = False,
    ):
        """If `parallel` is `True`, `tempdir_path` will be force-set to `Path.cwd()`
        if it is `None`. This is done such that the tempoary folder in which the LTEZIP
        file is extracted will be visible to parallel worker nodes.
        """
        LTEZIP_filepath = Path(LTEZIP_filepath)
        assert LTEZIP_filepath.exists()

        if parallel:
            if tempdir_path is None:
                tempdir_path = Path.cwd()
                msg = (
                    "Due to `parallel` being True, `tempdir_path` is changed into `Path.cwd()` "
                    "to ensure file visibility to parallel worker nodes."
                )
                print(msg)

        tempdir = tempfile.TemporaryDirectory(prefix="tmpLteZip_", dir=tempdir_path)
        if verbose >= 1:
            print(
                f'\nTemporary directory "{tempdir.name}" has been created to unzip the LTEZIP file.'
            )

        generated_temp_folderpath = Path(tempdir.name)

        # Until Python 3.12, there is no "delete" option for TemporaryDirectory().
        # So this is the workaround to keep the temp directory, if so desired.
        if not del_tempdir_on_exit:
            tempdir.cleanup()
            generated_temp_folderpath.mkdir(parents=True, exist_ok=True)
            tempdir = None

        temp_lte_file = tempfile.NamedTemporaryFile(
            dir=generated_temp_folderpath, delete=False, suffix=".lte"
        )
        suppl_files_folderpath = generated_temp_folderpath / "lte_suppl"

        temp_lte_filepath = Lattice.unzip_lte(
            LTEZIP_filepath,
            output_lte_filepath=Path(temp_lte_file.name),
            suppl_files_folderpath=suppl_files_folderpath,
            overwrite_lte=True,
            overwrite_suppl=True,
            verbose=verbose,
        )
        if verbose >= 1:
            print(f'\nTemporary LTE file "{temp_lte_filepath}" has been created.')

        return dict(
            tempdir=tempdir,
            LTE_filepath=temp_lte_filepath,
            suppl_files_folderpath=suppl_files_folderpath,
        )

    def remove_tempdir(self):
        if self.tempdir is None:
            return

        self.tempdir.cleanup()

    def __del__(self):
        if self.del_tempdir_on_exit:
            self.remove_tempdir()

    def load_LTE(
        self,
        LTE_filepath: Union[Path, str] = "",
        LTEZIP_filepath: Union[Path, str] = "",
        used_beamline_name: str = "",
        elem_files_root_folderpath=None,
        tempdir_path=None,
        del_tempdir_on_exit=True,
    ):
        """"""

        if used_beamline_name is None:
            used_beamline_name = ""

        if (LTE_filepath == "") and (LTEZIP_filepath == ""):
            raise ValueError(
                "Either LTE_filepath or LTEZIP_filepath must be specified."
            )
        if (LTE_filepath != "") and (LTEZIP_filepath != ""):
            raise ValueError(
                "Both LTE_filepath and LTEZIP_filepath cannot be specified."
            )

        if LTE_filepath != "":
            LTE_filepath = Path(LTE_filepath)
            assert LTE_filepath.exists()

            self.LTEZIP_filepath = ""
        else:
            if LTEZIP_filepath == "":
                raise ValueError(
                    "Either LTE_filepath or LTEZIP_filepath must be specified."
                )

            LTEZIP_filepath = Path(LTEZIP_filepath)
            assert LTEZIP_filepath.exists()

            self.LTEZIP_filepath = LTEZIP_filepath

            self.del_tempdir_on_exit = del_tempdir_on_exit

            temp_d = Lattice.temp_unzip_ltezip(
                LTEZIP_filepath,
                tempdir_path=tempdir_path,
                del_tempdir_on_exit=del_tempdir_on_exit,
                verbose=self.verbose,
                parallel=self.parallel,
            )
            self.tempdir = temp_d["tempdir"]
            LTE_filepath = temp_d["LTE_filepath"]
            self._LTE_suppl_files_folderpath = temp_d["suppl_files_folderpath"]

        if elem_files_root_folderpath is None:
            self.elem_files_root_folder = LTE_filepath.parent
        else:
            self.elem_files_root_folder = Path(elem_files_root_folderpath)

        self.LTE_text = LTE_filepath.read_text()
        self.LTE_filepath = LTE_filepath

        self._clean_up_LTE_text()

        d = self.get_used_beamline_element_defs(used_beamline_name=used_beamline_name)

        self.used_beamline_name = d["used_beamline_name"]
        self.beamline_defs = d["beamline_defs"]
        self._beamline_defs_d = {e[0]: e[1] for e in self.beamline_defs}
        self.elem_defs = d["elem_defs"]
        self.flat_used_elem_names = d["flat_used_elem_names"]

        self._all_used_elem_names = [name for name, *_ in self.elem_defs]

        abs_kickmap_filepaths = self.get_kickmap_filepaths()["abs"]
        for name, _fp in abs_kickmap_filepaths.items():
            abs_kickmap_f = Path(_fp)
            if not abs_kickmap_f.exists():
                print(
                    (
                        f'Kickmap elment "{name}": File "{abs_kickmap_f}" '
                        f"does not exist."
                    )
                )

        unhandled_types = self.get_unhandled_element_types(self.elem_defs)
        if unhandled_types != []:
            print("Element types that are not handled:")
            print(unhandled_types)

        self._elem_map = {}

        self._elem_names = np.array(["__BEG__"] + self.flat_used_elem_names)
        self.n_elements = len(self._elem_names)
        self._elem_counts = defaultdict(int)
        self._elem_props = {}
        self._elem_map[("name", "elem_inds")] = defaultdict(list)
        name2inds = self._elem_map[("name", "elem_inds")]
        self._elem_map[("type", "elem_inds")] = defaultdict(list)
        type2inds = self._elem_map[("type", "elem_inds")]
        self._elem_instance_indexes = []
        for i, name in enumerate(self._elem_names):
            self._elem_instance_indexes.append(self._elem_counts[name])
            self._elem_counts[name] += 1
            name2inds[name].append(i)

            sub_d = {"elem_name": name, "index": i}

            if name != "__BEG__":
                matched_index = self._all_used_elem_names.index(name)
                _, sub_d["elem_type"], prop_str = self.elem_defs[matched_index]
                sub_d["properties"] = self.parse_elem_properties(prop_str)
            else:
                sub_d["elem_type"] = "__BEG__"
                sub_d["properties"] = {}

            self._elem_props[name] = sub_d

            type2inds[sub_d["elem_type"]].append(i)

        self._elem_instance_indexes = np.array(self._elem_instance_indexes)

        self._duplicate_elem_counts = {
            name: counts for name, counts in self._elem_counts.items() if counts != 1
        }

        # The following variables will be filled in as requested.

        self._lengths = None
        self._spos_us = None
        self._spos_ds = None
        self._spos_mid = None
        self._circumf = None

    def get_LTE_suppl_files_folderpath(self):
        return self._LTE_suppl_files_folderpath

    def get_kickmap_filepaths(self):
        """"""

        kickmap_filepaths = {"raw": {}, "abs": {}}

        for name, elem_type, prop_str in self.elem_defs:
            name = name.upper()
            if elem_type == "UKICKMAP":
                kickmap_fp = self.parse_elem_properties(prop_str)["INPUT_FILE"]
                if (kickmap_fp.startswith('"') and kickmap_fp.endswith('"')) or (
                    kickmap_fp.startswith("'") and kickmap_fp.endswith("'")
                ):
                    kickmap_fp = kickmap_fp[1:-1]

                kickmap_filepaths["raw"][name] = kickmap_fp

                abs_kickmap_f = self.elem_files_root_folder.joinpath(
                    Path(kickmap_fp)
                ).resolve()

                kickmap_filepaths["abs"][name] = str(abs_kickmap_f)

        return kickmap_filepaths

    def extract_filepaths_from_elem_defs(self, prop_name: str, elem_type: str = ""):
        filepaths = {"raw": {}, "abs": {}}

        for name, elem_type_in_def, prop_str in self.elem_defs:
            name = name.upper()

            if elem_type:
                if elem_type_in_def != elem_type:
                    continue

            fp_str = self.parse_elem_properties(prop_str).get(prop_name, "")

            if (fp_str.startswith('"') and fp_str.endswith('"')) or (
                fp_str.startswith("'") and fp_str.endswith("'")
            ):
                fp_str = fp_str[1:-1]

            if fp_str == "":
                continue

            filepaths["raw"][name] = fp_str

            abs_fp = (self.elem_files_root_folder / fp_str).resolve()

            filepaths["abs"][name] = str(abs_fp)

        return filepaths

    def get_systematic_multipole_filepaths(self):
        """"""

        return self.extract_filepaths_from_elem_defs("SYSTEMATIC_MULTIPOLES")

    def get_random_multipole_filepaths(self):
        """"""

        return self.extract_filepaths_from_elem_defs("RANDOM_MULTIPOLES")

    def remove_comments(self, text):
        """"""

        comment_char = "!"

        # Check if the comment character is the first character for comment
        # lines, as ELEGANT will not correctly parse the LTE file, even though
        # it may not crash.
        possibly_commented_lines_and_linenums = [
            (i, line)
            for i, line in enumerate(text.splitlines())
            if comment_char in line
        ]
        for lineIndex, line in possibly_commented_lines_and_linenums:
            if not line.startswith(comment_char):
                print(
                    (
                        f'\n** CRITICAL WARNING ** The character "{comment_char}" '
                        f"must be the first character on the comment line at "
                        f"Line {lineIndex}:"
                    )
                )  # Line number here should not be
                # incremented by 1, as the passes "text" has an extra line added
                # to the top.
                print(line)

        pattern = comment_char + ".*"
        return re.sub(pattern, "", text)

    def delete_ampersands(self, text):
        """"""

        pattern = r"&.*[\n\r\s]+"
        return re.sub(pattern, "", text)

    def get_all_elem_defs(self, LTE_text) -> List[Tuple]:
        """
        "LTE_text" must not contain comments and ampersands.
        """

        # matches = re.findall('\s+"?([\w\$]+)"?[ \t]*:[ \t]*(\w+)[ \t]*,?(.*)',
        #' '+LTE_text)
        matches = re.findall(
            '\s+"?([\w\$:\.]+)"?[ \t]*:[ \t]*(\w+)[ \t]*,?(.*)', " " + LTE_text
        )
        # ^ Need to add the initial whitespace to pick up the first occurrence

        elem_def = [
            (name.upper(), type_name.upper(), rest.strip())
            for (name, type_name, rest) in matches
            if type_name.upper() != "LINE"
        ]

        return elem_def

    def get_all_beamline_defs(self, LTE_text) -> List[Tuple]:
        """
        "LTE_text" must not contain comments and ampersands.
        """

        # matches = re.findall(
        #'\s+("?[\w\$:\.]+"?)[ \t]*:[ \t]*("?\w+"?)[ \t]*,?(.*)', LTE_text)
        matches = re.findall(
            '\s+"?([\w\$:\.]+)"?[ \t]*:[ \t]*"?([\w\$:\.]+)"?[ \t]*,?(.*)', LTE_text
        )

        beamline_def = []
        for name, type_name, rest in matches:
            if type_name.upper() == "LINE":
                rest = rest.strip().replace("=", "").replace("(", "").replace(")", "")
                name_list = [
                    s.strip().upper() for s in rest.split(",") if s.strip() != ""
                ]
                name_list = [
                    s[1:-1] if s.startswith('"') and s.endswith('"') else s
                    for s in name_list
                ]
                if name[0] == '"' or name[-1] == '"':
                    assert name[0] == name[-1] == '"'
                    name = name[1:-1]
                beamline_def.append((name.upper(), name_list))

        return beamline_def

    def _get_used_beamline_name(self, LTE_text):
        """
        "LTE_text" must not contain comments and ampersands.
        """

        matches = re.findall(
            '\s+USE[ \t]*,[ \t"]*([\w\$]+)[ \t\r\n"]*', LTE_text, re.IGNORECASE
        )

        if len(matches) > 1:
            print('Multiple "USE" lines detected. Using the last "USE" line.')
            return matches[-1].upper()
        elif len(matches) == 0:
            print('No "USE" line detected.')
            return ""
        else:
            return matches[0].upper()

    def expand_beamline_name(
        self,
        beamline_name,
        all_beamline_defs,
        all_beamline_names,
        reverse=False,
        used_beamline_names=None,
    ):
        """
        If you want to obtain the list of used beamline names, then pass an empty
        list to "used_beamline_names".
        """

        if beamline_name in all_beamline_names:
            if used_beamline_names is not None:
                used_beamline_names.append(beamline_name)

            _, expanded_name_list = all_beamline_defs[
                all_beamline_names.index(beamline_name)
            ]

            if reverse:
                expanded_name_list = [
                    _exp_name[1:] if _exp_name.startswith("-") else f"-{_exp_name}"
                    for _exp_name in expanded_name_list[::-1]
                ]
                # print('Reversed:')
                # print(expanded_name_list)

            for name in expanded_name_list:
                if "*" in name:
                    star_ind = name.index("*")
                    multiplier = int(name[:star_ind].strip())
                    name = name[(star_ind + 1) :].strip()
                else:
                    multiplier = 1

                if name.startswith("-"):
                    name = name[1:].strip()
                    if name in all_beamline_names:
                        reverse_next = True
                    else:
                        reverse_next = False
                else:
                    reverse_next = False

                for i in range(multiplier):
                    for sub in self.expand_beamline_name(
                        name,
                        all_beamline_defs,
                        all_beamline_names,
                        reverse=reverse_next,
                        used_beamline_names=used_beamline_names,
                    ):
                        yield sub
        else:
            yield beamline_name

    def flatten_nested_list(self, L):
        """
        The input argument of any nested list will be flattened to a simple flat list.

        Based on Cristian's answer on
        http://stackoverflow.com/questions/2158395/flatten-an-irregular-list-of-lists-in-python
        """

        for el in L:
            if isinstance(el, collections.abc.Iterable) and not isinstance(el, str):
                for sub in self.flatten_nested_list(el):
                    yield sub
            else:
                yield el

    def get_used_beamline_element_defs(self, used_beamline_name=""):
        """
        This function returns a new dictionary of the beamline/element definitions
        constructed from scratch based on the LTE file contents.

        If you want to use modified element definitions, use instead:
            get_persistent_used_beamline_element_defs()
        """

        if used_beamline_name is None:
            used_beamline_name = ""

        all_elem_defs = self.get_all_elem_defs(self.cleaned_LTE_text)
        all_beamline_defs = self.get_all_beamline_defs(self.cleaned_LTE_text)

        all_beamline_names = [name for name, _ in all_beamline_defs]
        all_elem_names = [name for name, _, _ in all_elem_defs]

        if used_beamline_name == "":
            used_beamline_name = self._get_used_beamline_name(self.cleaned_LTE_text)

        if used_beamline_name == "":
            print("Using the last defined beamline.")
            used_beamline_name = all_beamline_names[-1]

        used_beamline_name = used_beamline_name.upper()

        assert used_beamline_name in all_beamline_names

        try:
            assert len(all_beamline_names) == len(np.unique(all_beamline_names))
        except AssertionError:
            duplicate_names = [
                name
                for name in np.unique(all_beamline_names)
                if all_beamline_names.count(name) != 1
            ]
            msg = f"Duplicate beamline defintions for {duplicate_names}"
            raise AssertionError(msg)

        try:
            assert len(all_elem_names) == len(np.unique(all_elem_names))
        except AssertionError:
            duplicate_names = [
                name
                for name in np.unique(all_elem_names)
                if all_elem_names.count(name) != 1
            ]
            msg = f"Duplicate element defintions for {duplicate_names}"
            raise AssertionError(msg)

        actually_used_beamline_names = []  # placeholder

        nested_used_elem_name_generator = self.expand_beamline_name(
            used_beamline_name,
            all_beamline_defs,
            all_beamline_names,
            used_beamline_names=actually_used_beamline_names,
        )
        used_elem_name_generator = self.flatten_nested_list(
            nested_used_elem_name_generator
        )

        flat_used_elem_name_list = list(used_elem_name_generator)

        used_elem_names = [
            name if not name.startswith("-") else name[1:]
            for name in flat_used_elem_name_list
        ]
        used_elem_names = [
            name if "*" not in name else name[(name.index("*") + 1) :]
            for name in used_elem_names
        ]
        u_used_elem_names = np.unique(used_elem_names)

        self._unique_used_elem_names = u_used_elem_names.tolist()

        # Re-order in the order of appearance in the LTE file
        used_elem_defs = [
            all_elem_defs[all_elem_names.index(elem_name)]
            for elem_name in all_elem_names
            if elem_name in u_used_elem_names
        ]

        _, u_inds = np.unique(actually_used_beamline_names, return_index=True)

        # Re-order in the required order of definitions
        used_beamline_defs = [
            all_beamline_defs[all_beamline_names.index(beamline_name)]
            for beamline_name in np.array(actually_used_beamline_names)[
                sorted(u_inds)[::-1]
            ]
            if beamline_name in all_beamline_names
        ]

        # Separate the multiplier/reverser from beamline names
        used_beamline_defs_w_mults = []
        for defined_BL_name, unsep_name_list in used_beamline_defs:
            sep_name_multiplier_list = []
            for elem_or_BL_name in unsep_name_list:
                if elem_or_BL_name.startswith("-"):
                    sep_name_multiplier_list.append((elem_or_BL_name[1:], -1))
                elif "*" in elem_or_BL_name:
                    star_ind = elem_or_BL_name.index("*")
                    multiplier = int(elem_or_BL_name[:star_ind].strip())
                    name_only = elem_or_BL_name[(star_ind + 1) :].strip()
                    sep_name_multiplier_list.append((name_only, multiplier))
                else:
                    sep_name_multiplier_list.append((elem_or_BL_name, +1))

            used_beamline_defs_w_mults.append(
                (defined_BL_name, sep_name_multiplier_list)
            )

        # Re-order used beamline definitions in the order of appearance in the LTE file
        # (Otherwise, when writing to a new LTE, parsing by ELEGANT may fail.)
        _beamline_names_w_mults = [v[0] for v in used_beamline_defs_w_mults]
        used_beamline_defs_w_mults = [
            used_beamline_defs_w_mults[_beamline_names_w_mults.index(beamline_name)]
            for beamline_name in all_beamline_names
            if beamline_name in actually_used_beamline_names
        ]

        return dict(
            used_beamline_name=used_beamline_name,
            beamline_defs=used_beamline_defs_w_mults,
            elem_defs=used_elem_defs,
            flat_used_elem_names=flat_used_elem_name_list,
        )

    def get_persistent_used_beamline_element_defs(self, used_beamline_name=""):
        """"""

        if used_beamline_name is None:
            used_beamline_name = ""

        if self._persistent_LTE_d is None:
            d = self.get_used_beamline_element_defs(
                used_beamline_name=used_beamline_name
            )

            self._persistent_LTE_d = {d["used_beamline_name"]: d}

            # Add the default beamline name case
            if used_beamline_name == "":
                self._persistent_LTE_d[""] = d["used_beamline_name"]

        else:
            if used_beamline_name in self._persistent_LTE_d:
                pass
            else:
                d = self.get_used_beamline_element_defs(
                    used_beamline_name=used_beamline_name
                )

                self._persistent_LTE_d[d["used_beamline_name"]] = d

        return self._persistent_LTE_d[used_beamline_name]

    def parse_elem_properties(self, prop_str: str):
        """"""

        pat = r"(\w+)\s*=\s*([^,]+)"

        prop = dict()
        for prop_name, val_str in re.findall(pat, prop_str):
            try:
                prop[prop_name.upper()] = int(val_str)
            except ValueError:
                try:
                    prop[prop_name.upper()] = float(val_str)
                except ValueError:
                    prop[prop_name.upper()] = val_str

        return prop

    def _modify_elem_def(self, original_elem_def, modified_prop_dict):
        """"""

        elem_name, elem_type, elem_prop_str = original_elem_def

        elem_prop_d = self.parse_elem_properties(elem_prop_str)
        elem_prop_d.update(modified_prop_dict)

        new_elem_prop_str_list = []
        for k, v in elem_prop_d.items():
            if isinstance(v, float):
                val_str = f"{v:.16g}"
            elif isinstance(v, int):
                val_str = f"{v:d}"
            elif isinstance(v, str):
                val_str = v
            else:
                raise NotImplementedError

            new_elem_prop_str_list.append(f"{k}={val_str}")

        new_elem_prop_str = ", ".join(new_elem_prop_str_list)

        return (elem_name, elem_type, new_elem_prop_str)

    def modify_elem_properties(self, mod_prop_dict_list):
        """
        A valid example of "mod_prop_dict_list":
            mod_prop_dict_list = [
                {"elem_name": "Qh1G2c30a", "prop_name": "K1", "prop_val": 1.5},
                {"elem_name": "sH1g2C30A", "prop_name": "K2", "prop_val": 0.0},
            ]

        Note that the values of "elem_name" are case-insensitive.
        """

        LTE_d = self.get_persistent_used_beamline_element_defs(
            used_beamline_name=self.used_beamline_name
        )

        elem_defs = LTE_d["elem_defs"]
        elem_names = [v[0] for v in elem_defs]
        for mod in mod_prop_dict_list:
            elem_ind = elem_names.index(mod["elem_name"].upper())
            orig_elem_def = LTE_d["elem_defs"][elem_ind]
            assert mod["elem_name"].upper() == orig_elem_def[0]
            new_elem_def = self._modify_elem_def(
                orig_elem_def, {mod["prop_name"]: mod["prop_val"]}
            )
            LTE_d["elem_defs"][elem_ind] = new_elem_def

    def get_unhandled_element_types(self, elem_def_list):
        """"""

        unhandled_list = [
            type_name.upper()
            for (_, type_name, _) in elem_def_list
            if type_name.upper() not in self.handled_element_types
        ]

        return list(set(unhandled_list))

    def zip_lte(self, output_ltezip_filepath: Union[Path, str], header_comment=""):
        """"""

        contents = dict(
            header_comment=header_comment,
            orig_LTE_filepath=str(Path(self.LTE_filepath).resolve()),
            raw_LTE_text=self.LTE_text,  # save raw LTE text
            suppl={},
        )

        for file_type, method in [
            ("km", self.get_kickmap_filepaths),
            ("sys_mpole", self.get_systematic_multipole_filepaths),
            ("rnd_mpole", self.get_random_multipole_filepaths),
        ]:
            files_d = method()
            if files_d["raw"]:
                u_folderpaths = np.unique(
                    [
                        str(Path(abs_path).parent)
                        for elem_name, abs_path in files_d["abs"].items()
                    ]
                ).tolist()

                suppl_contents = dict(
                    unique_parents=u_folderpaths, meta={}, file_contents={}
                )

                for elem_name, abs_path in files_d["abs"].items():
                    suppl_contents["file_contents"][abs_path] = (
                        Path(abs_path).read_bytes().decode("latin-1")
                    )
                    suppl_contents["meta"][elem_name] = dict(
                        abs_path=abs_path,
                        folder_index=u_folderpaths.index(str(Path(abs_path).parent)),
                    )

                contents["suppl"][file_type] = suppl_contents

        with gzip.GzipFile(output_ltezip_filepath, "wb") as f:
            f.write(json.dumps(contents).encode("utf-8"))

    @staticmethod
    def unzip_lte(
        ltezip_filepath,
        output_lte_filepath: Union[Path, str] = "",
        suppl_files_folderpath: Union[Path, str] = "./lte_suppl",
        use_abs_paths_for_suppl_files=True,
        overwrite_lte=False,
        overwrite_suppl=False,
        double_format="%.16g",
        verbose=0,
        throw: bool = True,
    ):
        """
        If "output_lte_filepath" is not specified, a new LTE file will be
        created in the current directory with the same file name as the original
        LTE file.

        If "use_abs_paths_for_suppl_files" is True (default & recommended), the
        absolute, instead of relative, paths to supplementary files in the newly
        created LTE file. Since ELEGANT assumes relative file paths specified in
        an LTE file with respect to the current directory, NOT the directory
        where the LTE file (specified in an ELE file) is located, using absolute
        paths in LTE avoids any problem of finding necessary supplementary files.

        If a file already exists where the new LTE file will be written,
        it will throw an error if "overwrite_lte" is False (default).

        If a file already exists where a file supplementary to the LTE file file
        will be written, it will throw an error if "overwrite_suppl" is False
        (default) and "throw" is True (default).
        """

        double_format = double_format.replace("%", ":")

        suppl_files_folderpath_d = {}

        suppl_files_folderpath = Path(suppl_files_folderpath)

        with gzip.GzipFile(ltezip_filepath, "rb") as f:
            contents = json.loads(f.read().decode("utf-8"))

        if output_lte_filepath == "":
            output_lte_filepath = Path.cwd().joinpath(
                Path(contents["orig_LTE_filepath"]).name
            )
        else:
            output_lte_filepath = Path(output_lte_filepath)

        if output_lte_filepath.exists() and (not overwrite_lte):
            raise FileExistsError(
                f'Cannot write a new LTE file to "{output_lte_filepath}"'
            )

        # Make sure the parent folder exists.
        output_lte_filepath.parent.mkdir(parents=True, exist_ok=True)

        temp_LTE = Lattice()
        temp_LTE.LTE_text = contents["raw_LTE_text"]
        temp_LTE.LTE_filepath = contents["orig_LTE_filepath"]
        #
        temp_LTE._clean_up_LTE_text()
        #
        d = temp_LTE.get_used_beamline_element_defs()

        # Start building new LTE contents
        lines = [
            "! " + line
            for line in contents["header_comment"].split("\n")
            if line.strip()
        ]
        #
        # Add element definition sections
        lines.append("\n")
        _newly_created_suppl_filepaths = defaultdict(list)
        suppl_contents = contents["suppl"]
        elem_names_w_suppl_filepaths = {
            file_type: list(_d["meta"]) for file_type, _d in suppl_contents.items()
        }
        for elem_name, elem_type, prop_str in d["elem_defs"]:
            prop_d = temp_LTE.parse_elem_properties(prop_str)
            for file_type, suppl_elem_names in elem_names_w_suppl_filepaths.items():
                if elem_name not in suppl_elem_names:
                    continue

                if file_type not in suppl_files_folderpath_d:
                    suppl_files_folderpath_d[file_type] = (
                        suppl_files_folderpath / file_type
                    )
                    suppl_files_folderpath_d[file_type].mkdir(
                        parents=True, exist_ok=True
                    )

                meta = suppl_contents[file_type]["meta"][elem_name]

                orig_abs_path = meta["abs_path"]
                orig_filename = Path(orig_abs_path).name

                new_rel_path = suppl_files_folderpath_d[file_type] / orig_filename
                new_abs_path = new_rel_path.resolve()

                if new_abs_path not in _newly_created_suppl_filepaths[file_type]:
                    if new_abs_path.exists():
                        if overwrite_suppl:
                            _write = True
                        else:
                            if throw:
                                raise FileExistsError(
                                    (
                                        f"Cannot write a new LTE supplementary file "
                                        f'to "{new_abs_path}"'
                                    )
                                )

                            _write = False
                    else:
                        _write = True

                    if _write:
                        new_abs_path.write_bytes(
                            suppl_contents[file_type]["file_contents"][
                                orig_abs_path
                            ].encode("latin-1")
                        )
                        if verbose >= 1:
                            print(f"* Created LTE supplementary file: {new_abs_path}")
                        _newly_created_suppl_filepaths[file_type].append(new_abs_path)

                if file_type == "km":
                    filepath_prop_name = "INPUT_FILE"
                elif file_type == "sys_mpole":
                    filepath_prop_name = "SYSTEMATIC_MULTIPOLES"
                elif file_type == "rnd_mpole":
                    filepath_prop_name = "RANDOM_MULTIPOLES"
                else:
                    raise ValueError(file_type)

                assert filepath_prop_name in prop_d
                if use_abs_paths_for_suppl_files:
                    prop_d[filepath_prop_name] = f'"{new_abs_path}"'
                else:
                    prop_d[filepath_prop_name] = f'"{new_rel_path}"'

            prop_str = ", ".join(
                [
                    ("{}={%s}" % double_format).format(_k, _v)
                    if isinstance(_v, float)
                    else f"{_k}={_v}"
                    for _k, _v in prop_d.items()
                ]
            )

            if prop_str == "":
                temp_line = f"{elem_name}: {elem_type}"
            else:
                temp_line = f"{elem_name}: {elem_type}, {prop_str}"
            temp_line = temp_LTE.get_wrapped_line(temp_line)
            lines.extend(temp_line.split("\n"))
        #
        # Add beamline definition sections
        lines.append("\n")
        if d["beamline_defs"] != []:
            for beamline_name, sub_name_multiplier_tups in d["beamline_defs"]:
                sub_lines = []
                for sub_name, multiplier in sub_name_multiplier_tups:
                    if multiplier == 1:
                        sub_lines.append(sub_name)
                    elif multiplier == -1:
                        sub_lines.append("-" + sub_name)
                    else:
                        sub_lines.append(f"{multiplier:d}*{sub_name}")

                temp_line = temp_LTE.get_wrapped_line(
                    f'{beamline_name}: LINE=({",".join(sub_lines)})'
                )
                lines.extend(temp_line.split("\n"))
        else:
            # Note that d["flat_used_elem_names"] does NOT start with "_BEG_"
            full_beamline_line = temp_LTE.get_wrapped_line(
                f'{d["used_beamline_name"]}: LINE=({",".join(d["flat_used_elem_names"])})'
            )
            lines.extend(full_beamline_line.split("\n"))
        #
        # Finally add "USE" line
        lines.append("\n")
        lines.append(f'USE, {d["used_beamline_name"]}')

        # Write the LTE file
        output_lte_filepath.write_text("\n".join(lines))

        return output_lte_filepath

    @staticmethod
    def get_wrapped_line(line, max_len=80, sep=",", indent=2):
        """"""

        if len(line) <= max_len:
            return line
        else:
            new_line_list = []
            split_line_list = line.split(sep)
            ntokens = len(split_line_list)
            new_line = split_line_list[0]
            offset = 1
            while ntokens > offset:
                for iSeg, sub_line in enumerate(split_line_list[offset:]):
                    if len(" ".join([new_line, sub_line])) > max_len:
                        new_line_list.append(new_line)
                        if offset + iSeg + 1 < ntokens:
                            new_line = sub_line
                        else:
                            new_line_list.append(sub_line)
                        offset += iSeg + 1
                        break
                    else:
                        new_line = sep.join([new_line, sub_line])
                else:
                    new_line_list.append(new_line)
                    break

            return f'{sep} &\n{" " * indent}'.join(new_line_list)

    def get_all_names_ordered_by_s(self):
        return self._elem_names.copy()

    def get_all_elem_def_dict(self):
        return self._elem_props

    def get_all_beamline_def_dict(self):
        return self._beamline_defs_d

    def get_elem_inds_from_name(
        self, elem_name: str, output_type: OutputType = OutputType.NumPy
    ) -> PY_TYPES["np_array_or_list"]:
        """"""

        assert output_type in (OutputType.NumPy, OutputType.List)

        elem_map = self._elem_map[("name", "elem_inds")]

        if elem_name not in elem_map:
            raise ValueError(f"Element name '{elem_name}' does not exist!")

        elem_inds_list = elem_map[elem_name]

        if output_type == OutputType.NumPy:
            return np.array(elem_inds_list)
        else:
            return elem_inds_list

    def get_elem_inds_from_names(
        self,
        elem_names: Union[List[str], np.ndarray],
        output_type: OutputType = OutputType.NumPy,
    ) -> PY_TYPES["np_array_or_list"]:
        """Returned element indexes are ordered by s-position"""

        assert output_type in (OutputType.NumPy, OutputType.List)

        elem_inds = []
        for name in elem_names:
            elem_inds += self.get_elem_inds_from_name(name, output_type=OutputType.List)

        elem_inds = np.sort(elem_inds)

        if output_type == OutputType.NumPy:
            return elem_inds
        else:
            return elem_inds.tolist()

    def get_names_from_elem_inds(
        self,
        elem_inds: Union[List[int], np.ndarray],
        output_type: OutputType = OutputType.NumPy,
    ) -> PY_TYPES["np_array_or_list"]:
        """"""

        if isinstance(elem_inds, list):
            elem_inds = np.array(elem_inds)

        if output_type == OutputType.NumPy:
            return self._elem_names[elem_inds]
        else:
            return self._elem_names[elem_inds].tolist()

    def get_elem_inds_from_regex(
        self, pattern: str, output_type: OutputType = OutputType.NumPy
    ) -> PY_TYPES["np_array_or_list"]:
        """Returned element indexes are ordered by s-position"""

        matched_elem_names = [
            elem_name
            for elem_name in self._unique_used_elem_names
            if re.match(pattern, elem_name) is not None
        ]

        return self.get_elem_inds_from_names(
            matched_elem_names, output_type=output_type
        )

    def get_elem_props_from_elem_inds(
        self, elem_inds: List[int], output_type: OutputType = OutputType.Dict
    ) -> Union[Dict[str, Dict], List[Dict]]:
        assert output_type in (OutputType.List, OutputType.Dict)

        if output_type == OutputType.List:
            return [
                self._elem_props[name]
                for name in self.get_names_from_elem_inds(elem_inds)
            ]
        else:
            return {
                name: self._elem_props[name]
                for name in self.get_names_from_elem_inds(elem_inds)
            }

    def get_elem_props_from_regex(
        self, pattern: str, output_type: OutputType = OutputType.Dict
    ) -> Union[Dict[str, Dict], List[Dict]]:
        matched_elem_names = [
            elem_name
            for elem_name in self._unique_used_elem_names
            if re.match(pattern, elem_name) is not None
        ]

        sorted_elem_inds = self.get_elem_inds_from_names(matched_elem_names)

        return self.get_elem_props_from_elem_inds(
            sorted_elem_inds, output_type=output_type
        )

    def get_elem_props_from_names(
        self,
        elem_names: Union[List[str], np.ndarray],
        output_type: OutputType = OutputType.Dict,
    ) -> Union[Dict[str, Dict], List[Dict]]:
        assert output_type in (OutputType.Dict, OutputType.List)

        if output_type == OutputType.Dict:
            return {name: self._elem_props[name] for name in elem_names}
        else:
            return [self._elem_props[name] for name in elem_names]

    def get_closest_us_ds_elem_inds_from_ref_elem_ind(
        self, ref_elem_ind: int, elem_type_to_search: str, n_us: int = 1, n_ds: int = 1
    ) -> Dict[str, np.ndarray]:
        assert n_us >= 0
        assert n_ds >= 0
        assert n_us + n_ds >= 1

        inds = self.get_elem_inds_from_elem_type(elem_type_to_search)

        output = dict(us=[], ds=[])

        if inds.size == 0:
            return output

        n_elems = self.n_elements
        inds = np.hstack((inds - n_elems, inds, inds + n_elems))

        if n_us >= 1:
            us_inds = inds[inds < ref_elem_ind]
            if us_inds.size != 0:
                sort_inds = np.argsort(np.abs(us_inds - ref_elem_ind))
                sel_inds = us_inds[sort_inds[:n_us]]
                sel_inds[sel_inds < 0] += n_elems
                assert 0 <= np.min(sel_inds) < n_elems
                assert 0 <= np.max(sel_inds) < n_elems
                output["us"] = sel_inds

        if n_ds >= 1:
            ds_inds = inds[inds > ref_elem_ind]
            if ds_inds.size != 0:
                sort_inds = np.argsort(np.abs(ds_inds - ref_elem_ind))
                sel_inds = ds_inds[sort_inds[:n_ds]]
                sel_inds[sel_inds >= n_elems] -= n_elems
                assert 0 <= np.min(sel_inds) < n_elems
                assert 0 <= np.max(sel_inds) < n_elems
                output["ds"] = sel_inds

        return output

    def get_closest_us_ds_elem_inds_from_ref_name(
        self, ref_elem_name: str, elem_type_to_search: str, n_us: int = 1, n_ds: int = 1
    ) -> List[Dict[str, np.ndarray]]:
        return [
            self.get_closest_us_ds_elem_inds_from_ref_elem_ind(
                ref_elem_ind, elem_type_to_search, n_us=n_us, n_ds=n_ds
            )
            for ref_elem_ind in self.get_elem_inds_from_name(ref_elem_name)
        ]

    def get_closest_elem_inds_from_ref_elem_ind(
        self,
        ref_elem_ind: int,
        elem_type_to_search: str,
        n: int = 1,
        output_type: OutputType = OutputType.NumPy,
    ) -> np.ndarray:
        assert output_type in (OutputType.NumPy, OutputType.List)

        s_mid = self.get_s_mid_array()
        C = self.get_circumference()

        s_ref = s_mid[ref_elem_ind]

        inds = self.get_elem_inds_from_elem_type(elem_type_to_search)

        if inds.size == 0:
            return None

        s_ext = np.hstack((s_mid[inds] - C, s_mid[inds], s_mid[inds] + C))
        inds_ext = np.hstack((inds, inds, inds))

        sort_inds = np.argsort(np.abs(s_ext - s_ref))

        if output_type == OutputType.NumPy:
            return inds_ext[sort_inds[:n]]
        else:
            return inds_ext[sort_inds[:n]].tolist()

    def get_closest_elem_inds_from_ref_name(
        self, ref_elem_name: str, elem_type_to_search: str, n: int = 1
    ) -> List[np.ndarray]:
        return [
            self.get_closest_elem_inds_from_ref_elem_ind(
                ref_elem_ind, elem_type_to_search, n=n
            )
            for ref_elem_ind in self.get_elem_inds_from_name(ref_elem_name)
        ]

    def get_closest_names_from_ref_elem_ind(
        self,
        ref_elem_ind: int,
        elem_type_to_search: str,
        n=1,
        output_type: OutputType = OutputType.NumPy,
    ) -> PY_TYPES["np_array_or_list"]:
        assert output_type in (OutputType.NumPy, OutputType.List)

        inds = self.get_closest_elem_inds_from_ref_elem_ind(
            ref_elem_ind, elem_type_to_search, n=n
        )

        return self.get_names_from_elem_inds(inds, output_type=output_type)

    def get_closest_names_from_ref_name(
        self,
        ref_elem_name: str,
        elem_type_to_search: str,
        n=1,
    ) -> List[np.ndarray]:
        return [
            self.get_names_from_elem_inds(
                self.get_closest_elem_inds_from_ref_elem_ind(
                    ref_elem_ind, elem_type_to_search, n=n
                )
            )
            for ref_elem_ind in self.get_elem_inds_from_name(ref_elem_name)
        ]

    def get_elem_inds_from_name_instance_pairs(
        self,
        pairs: List[ElementNameInstancePair],
        output_type: OutputType = OutputType.NumPy,
    ) -> PY_TYPES["np_array_or_list"]:
        """"""

        assert output_type in (OutputType.NumPy, OutputType.List)

        elem_inds = []
        for p in pairs:
            matched_indexes = np.where(
                (self._elem_names == p.name)
                & (self._elem_instance_indexes == p.instance_index)
            )[0]
            if matched_indexes.size == 0:
                raise ValueError(
                    f"Element Name '{p.name}' Instance Index {p.instance_index} does not exist."
                )
            elif matched_indexes.size == 1:
                elem_inds.append(matched_indexes[0])
            else:
                raise RuntimeError("This cannot happen. Must debug PyELEGANT.")

        if output_type == OutputType.NumPy:
            return np.array(elem_inds)
        else:
            return elem_inds

    def get_elem_inds_from_name_occur_pairs(
        self,
        pairs: List[ElementNameOccurrencePair],
        output_type: OutputType = OutputType.NumPy,
    ) -> PY_TYPES["np_array_or_list"]:
        """"""

        assert output_type in (OutputType.NumPy, OutputType.List)

        elem_inds = []
        for p in pairs:
            matched_indexes = np.where(
                (self._elem_names == p.name)
                & (self._elem_instance_indexes == p.occurrence_num - 1)
            )[0]
            if matched_indexes.size == 0:
                raise ValueError(
                    f"Element Name '{p.name}' Occurrence Number {p.occurrence_num} does not exist."
                )
            elif matched_indexes.size == 1:
                elem_inds.append(matched_indexes[0])
            else:
                raise RuntimeError("This cannot happen. Must debug PyELEGANT.")

        if output_type == OutputType.NumPy:
            return np.array(elem_inds)
        else:
            return elem_inds

    def get_name_occur_pairs_from_elem_inds(
        self, elem_inds: Union[List[int], np.ndarray]
    ) -> List[ElementNameOccurrencePair]:
        """"""

        return [
            ElementNameOccurrencePair(
                name=self._elem_names[i],
                occurrence_num=self._elem_instance_indexes[i] + 1,
            )
            for i in elem_inds
        ]

    def get_name_instance_pairs_from_elem_inds(
        self, elem_inds: Union[List[int], np.ndarray]
    ) -> List[ElementNameInstancePair]:
        """"""

        return [
            ElementNameInstancePair(
                name=self._elem_names[i], instance_index=self._elem_instance_indexes[i]
            )
            for i in elem_inds
        ]

    def get_elem_type_from_name(self, elem_name: str) -> str:
        """"""

        return self._elem_props[elem_name]["elem_type"]

    def get_elem_inds_from_elem_type(
        self, elem_type: str, output_type: OutputType = OutputType.NumPy
    ) -> PY_TYPES["np_array_or_list"]:
        assert output_type in (OutputType.NumPy, OutputType.List)

        elem_inds_list = self._elem_map[("type", "elem_inds")][elem_type]

        if output_type == OutputType.NumPy:
            return np.array(elem_inds_list)
        else:
            return elem_inds_list

    def _calc_spos_variables(self):
        all_elem_props = self.get_elem_props_from_names(self._elem_names)

        self._lengths = np.array(
            [
                all_elem_props[name]["properties"].get("L", 0.0)
                for name in self._elem_names
            ]
        )

        spos_ds = np.cumsum(self._lengths)
        spos_us = spos_ds - self._lengths
        spos_mid = (spos_us + spos_ds) / 2

        # 0.0 inserted at the beginning for "__BEG__"
        self._spos_ds = np.append(0.0, spos_ds)
        self._spos_us = np.append(0.0, spos_us)
        self._spos_mid = np.append(0.0, spos_mid)

        self._circumf = spos_ds[-1]

    def get_s_us_array(self) -> np.ndarray:
        if self._spos_us is None:
            self._calc_spos_variables()

        return self._spos_us

    def get_s_ds_array(self) -> np.ndarray:
        if self._spos_ds is None:
            self._calc_spos_variables()

        return self._spos_ds

    def get_s_mid_array(self) -> np.ndarray:
        if self._spos_mid is None:
            self._calc_spos_variables()

        return self._spos_mid

    def get_circumference(self) -> float:
        if self._circumf is None:
            self._calc_spos_variables()

        return self._circumf

    def is_unique_elem_name(self, elem_name: str) -> bool:
        if elem_name not in self._elem_names:
            raise ValueError(
                f'Element name "{elem_name}" does not exist in the lattice'
            )

        if elem_name not in self._duplicate_elem_counts:
            return True
        else:
            return False

    @staticmethod
    def write_LTE(
        new_LTE_filepath: Union[Path, str],
        used_beamline_name: str,
        elem_defs: Union[Dict, List],
        beamline_defs: Union[Dict, List],
        double_format: str = "%.16g",
    ):
        """"""

        if isinstance(beamline_defs, list):
            beamline_defs_d = {e[0]: e[1] for e in beamline_defs}
        elif isinstance(beamline_defs, dict):
            beamline_defs_d = beamline_defs
        else:
            raise TypeError()

        used_beamline_name = used_beamline_name.upper()
        assert used_beamline_name in beamline_defs_d

        lines = []

        if isinstance(elem_defs, list):  # for backward compatibility
            for elem_name, elem_type, prop_str in elem_defs:
                if ":" in elem_name:  # Must enclose element name with double quotes
                    # if the name contains ":".
                    elem_name = f'"{elem_name}"'
                def_line = f"{elem_name}: {elem_type}"
                if prop_str != "":
                    def_line += f", {prop_str}"
                lines.append(Lattice.get_wrapped_line(def_line))
        else:
            double_format = double_format.replace("%", ":")

            for elem_name, d in elem_defs.items():
                if ":" in elem_name:  # Must enclose element name with double quotes
                    # if the name contains ":".
                    elem_name = f'"{elem_name}"'
                elem_type = d["elem_type"]
                def_line = f"{elem_name}: {elem_type}"

                prop_str = ", ".join(
                    [
                        ("{}={%s}" % double_format).format(_k, _v)
                        if isinstance(_v, float)
                        else f"{_k}={_v}"
                        for _k, _v in d["properties"].items()
                    ]
                )
                if prop_str != "":
                    def_line += f", {prop_str}"

                lines.append(Lattice.get_wrapped_line(def_line))

        lines.append(" ")

        for beamline_name, bdef in beamline_defs_d.items():
            def_parts = []
            for name, multiplier in bdef:
                if multiplier == 1:
                    def_parts.append(name)
                elif multiplier == -1:
                    def_parts.append(f"-{name}")
                else:
                    def_parts.append(f"{multiplier:d}*{name}")

            def_line = ", ".join(def_parts)

            if ":" in beamline_name:  # Must enclose beamline name with double
                # quotes if the name contains ":".
                beamline_name = f'"{beamline_name}"'

            lines.append(
                Lattice.get_wrapped_line(f"{beamline_name}: LINE=({def_line})")
            )

        lines.append(" ")

        lines.append(f'USE, "{used_beamline_name}"')
        lines.append("RETURN")

        Path(new_LTE_filepath).write_text("\n".join(lines))


def write_temp_modified_LTE(
    mod_prop_dict_list, base_LTE_filepstr="", base_used_beamline_name="", LTE_obj=None
):
    """
    A valid example of "mod_prop_dict_list":
        [
        {'elem_name': 'Q1', 'prop_name': 'K1', 'prop_val': 1.5},
        {'elem_name': 'S1', 'prop_name': 'K2', 'prop_val': 2.0},
        ]
    """

    tmp = tempfile.NamedTemporaryFile(
        prefix=f"tmpLteMod_", suffix=".lte", dir=Path.cwd().resolve(), delete=False
    )
    # ^ CRITICAL: must create this temp LTE file in cwd. If you create this
    #   LTE in /tmp, this file cannot be accessible from other nodes
    #   when Pelegant is used.
    temp_LTE_filepstr = str(Path(tmp.name).resolve())
    tmp.close()

    write_modified_LTE(
        temp_LTE_filepstr,
        mod_prop_dict_list,
        base_LTE_filepstr=base_LTE_filepstr,
        base_used_beamline_name=base_used_beamline_name,
        LTE_obj=LTE_obj,
    )

    return Path(temp_LTE_filepstr)


def write_modified_LTE(
    new_LTE_filepstr,
    mod_prop_dict_list,
    base_LTE_filepstr="",
    base_used_beamline_name="",
    LTE_obj=None,
):
    """
    A valid example of "mod_prop_dict_list":
        mod_prop_dict_list = [
            {"elem_name": "Qh1G2c30a", "prop_name": "K1", "prop_val": 1.5},
            {"elem_name": "sH1g2C30A", "prop_name": "K2", "prop_val": 0.0},
        ]

    Note that the values of "elem_name" are case-insensitive.
    """

    if LTE_obj is None:
        LTE = Lattice(
            LTE_filepath=base_LTE_filepstr, used_beamline_name=base_used_beamline_name
        )
    else:
        if base_LTE_filepstr != "":
            print(f'WARNING: Ignored: base_LTE_filepstr = "{base_LTE_filepstr}"')
        if base_used_beamline_name != "":
            print(
                f'WARNING: Ignored: base_used_beamline_name = "{base_used_beamline_name}"'
            )
        LTE = LTE_obj

    LTE.modify_elem_properties(mod_prop_dict_list)
    LTE_d = LTE.get_persistent_used_beamline_element_defs(
        used_beamline_name=LTE.used_beamline_name
    )

    LTE.write_LTE(
        new_LTE_filepstr,
        LTE.used_beamline_name,
        LTE_d["elem_defs"],
        LTE_d["beamline_defs"],
    )


def get_ELEGANT_element_dictionary():
    if ELEGANT_ELEM_DICT == {}:
        from ruamel import yaml

        yaml_filepath = Path(__file__).parent / "elegant_elem_dict.yaml"

        y = yaml.YAML()
        y.preserve_quotes = True
        d = y.load(yaml_filepath.read_text())

        # Strip away YAML stuff
        d = json.loads(json.dumps(d))

        ELEGANT_ELEM_DICT.update(d)

    return ELEGANT_ELEM_DICT


class AbstractFacility:
    def __init__(self, LTE: Lattice, lattice_type: str):
        assert isinstance(LTE, Lattice)
        self.LTE = LTE

        self.lat_type = lattice_type

        self.N_KICKS = {}


class NSLS2(AbstractFacility):
    def __init__(self, LTE: Lattice, lattice_type: str = "day1"):
        super().__init__(LTE, lattice_type)

        assert self.lat_type in ("day1", "C26_double_mini_beta")

        self.E_MeV = 3e3
        self.N_KICKS = dict(CSBEND=40, KQUAD=40, KSEXT=20, KOCT=20)

    def get_regular_BPM_elem_inds(self):
        """Get the element indexes for regular (arc) BPMs"""

        LTE = self.LTE

        inds = LTE.get_elem_inds_from_regex("^P[HLM]\w+$")
        assert len(inds) == 180

        return dict(x=inds.copy(), y=inds.copy())

    def get_regular_BPM_names(self):
        """Get the names for regular (arc) BPMs"""

        inds_d = self.get_regular_BPM_elem_inds()
        assert np.all(inds_d["x"] == inds_d["y"])

        LTE = self.LTE

        names = LTE.get_names_from_elem_inds(inds_d["x"])
        assert len(names) == 180

        return dict(x=names.copy(), y=names.copy())

    def get_slow_corrector_elem_inds(self):
        """Get the element indexes for slow orbit correctors"""
        LTE = self.LTE

        inds_x = LTE.get_elem_inds_from_regex("^C[HLM][1-2]XG[2-6]\w+$")
        assert len(inds_x) == 180

        inds_y = LTE.get_elem_inds_from_regex("^C[HLM][1-2]YG[2-6]\w+$")
        assert len(inds_y) == 180

        return dict(x=inds_x, y=inds_y)

    def get_slow_corrector_names(self):
        """Get the names for slow orbit correctors"""

        inds_d = self.get_slow_corrector_elem_inds()

        LTE = self.LTE

        hcor_names = LTE.get_names_from_elem_inds(inds_d["x"])
        assert len(hcor_names) == 180

        vcor_names = LTE.get_names_from_elem_inds(inds_d["y"])
        assert len(vcor_names) == 180

        return dict(x=hcor_names, y=vcor_names)

    def get_bend_elem_inds(self):
        """Get the element indexes for bends"""

        LTE = self.LTE
        inds = LTE.get_elem_inds_from_regex("^B[12]\w+$")
        assert len(inds) == 30 * 2
        return inds

    def get_bend_names(self):
        """Get the names for bends"""

        inds = self.get_bend_elem_inds()

        LTE = self.LTE
        names = LTE.get_names_from_elem_inds(inds)
        assert len(names) == 30 * 2

        return names

    def get_quad_names(self, flat_skew_quad_names: bool = False):
        LTE = self.LTE

        inds = LTE.get_elem_inds_from_regex("^Q[HLM]\w+$")

        if self.lat_type == "C26_double_mini_beta":
            inds = np.append(inds, LTE.get_elem_inds_from_regex("^Q[FD]H*C26[AB]$"))
            inds = np.sort(inds)

        normal_quad_names = LTE.get_names_from_elem_inds(inds)
        assert len(normal_quad_names) == len(np.unique(normal_quad_names))

        if self.lat_type != "C26_double_mini_beta":
            assert len(normal_quad_names) == 300
        else:
            assert len(normal_quad_names) in (
                300 + 3,  # QF not split in half
                300 + 4,  # QF split in half
            )

        inds = LTE.get_elem_inds_from_regex("^SQ[HM]\w+$")
        skew_quad_names = LTE.get_names_from_elem_inds(inds)
        assert (
            len(skew_quad_names) == 30 * 2
        )  # Skew quad elements are all split into half

        if flat_skew_quad_names:
            if len(skew_quad_names) == len(np.unique(skew_quad_names)):
                # When skew quads are split in half, but each half piece is named differently
                return dict(
                    normal=normal_quad_names, skew=skew_quad_names, skew_lumped=False
                )
            else:
                # When skew quads are split in half, with each half piece with the same name
                flat_unique_skew_quad_names = []
                for i, name in enumerate(skew_quad_names):
                    if i % 2 == 0:
                        assert name not in flat_unique_skew_quad_names
                        flat_unique_skew_quad_names.append(name)
                    else:
                        assert name == flat_unique_skew_quad_names[-1]

                return dict(
                    normal=normal_quad_names,
                    skew=np.array(flat_unique_skew_quad_names),
                    skew_lumped=False,
                )
        else:
            lumped_skew_quad_names = [
                skew_quad_names[i * 2 : i * 2 + 2] for i in range(30)
            ]
            assert len(lumped_skew_quad_names) == 30

            return dict(
                normal=normal_quad_names, skew=lumped_skew_quad_names, skew_lumped=True
            )

    def get_sext_elem_inds(self):
        """Get the element indexes for sextupoles"""

        LTE = self.LTE

        inds = LTE.get_elem_inds_from_regex("^S[HLM]\w+$")
        assert len(inds) == 270

        return inds

    def get_sext_names(self):
        """Get the names for sextupoles"""

        inds = self.get_sext_elem_inds()

        LTE = self.LTE
        names = LTE.get_names_from_elem_inds(inds)
        assert len(names) == 270

        return names

    def get_girder_marker_pairs(self):
        LTE = self.LTE

        gs_inds = LTE.get_elem_inds_from_regex("^GS\w+")
        gs_names = LTE.get_names_from_elem_inds(gs_inds)
        assert len(gs_names) == 180

        ge_inds = LTE.get_elem_inds_from_regex("^GE\w+")
        ge_names = LTE.get_names_from_elem_inds(ge_inds)
        assert len(ge_names) == 180

        g_paired_names = list(zip(gs_names[:-1], ge_names[1:]))
        g_paired_inds = list(zip(gs_inds[:-1], ge_inds[1:]))
        assert len(g_paired_inds) == 179

        g_paired_names.append((gs_names[-1], ge_names[0]))
        g_paired_inds.append((gs_inds[-1], ge_inds[0]))
        assert len(g_paired_inds) == 180

        for gs_name, ge_name in g_paired_names:
            try:
                assert gs_name[:2] == "GS"
                assert ge_name[:2] == "GE"
                if not gs_name.startswith("GSG4C"):
                    assert gs_name[2:] == ge_name[2:]
                else:
                    assert gs_name[2:-1] == ge_name[2:-1]
                    assert gs_name[-1] == "A"
                    assert ge_name[-1] == "B"

                # Check uniqueness of the element names
                assert len(LTE.get_elem_inds_from_name(gs_name)) == 1
                assert len(LTE.get_elem_inds_from_name(ge_name)) == 1
            except AssertionError:
                print(gs_name, ge_name)

        gs_inds, ge_inds = [np.array(tup) for tup in zip(*g_paired_inds)]

        return gs_inds, ge_inds
