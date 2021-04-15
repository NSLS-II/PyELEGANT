import re
from pathlib import Path
import collections
import gzip
import json
import tempfile

import numpy as np

########################################################################
class Lattice():
    """
    """

    def __init__(self, LTE_filepath='', used_beamline_name=''):
        """Constructor"""

        if used_beamline_name is None:
            used_beamline_name = ''

        self.handled_element_types = [
            'DRIF','EDRIFT','RFCA',
            'CSBEND', 'CSBEN', 'SBEN','SBEND', 'CCBEND',
            'KQUAD', 'QUAD',
            'KSEXT', 'SEXT',
            'KOCT', 'OCTU',
            'MULT',
            'UKICKMAP',
            'HKICK','VKICK','KICKER',
            'EHKICK','EVKICK','EKICKER',
            'MARK', 'MONI',
            'SCRAPER',
            #'SOLE',
            'MALIGN', 'WATCH',
            ]

        if LTE_filepath != '':
            self.load_LTE(LTE_filepath, used_beamline_name=used_beamline_name)

        self._persistent_LTE_d = None

    def _clean_up_LTE_text(self):
        """"""

        self.cleaned_LTE_text = '\n' + self.LTE_text
        # ^ adding "\n" at the beginning for easier search
        self.cleaned_LTE_text = self.remove_comments(self.cleaned_LTE_text)
        self.cleaned_LTE_text = self.delete_ampersands(self.cleaned_LTE_text)

    def load_LTE(self, LTE_filepath, used_beamline_name='',
                 elem_files_root_folderpath=None):
        """"""

        if used_beamline_name is None:
            used_beamline_name = ''

        LTE_file = Path(LTE_filepath)

        if elem_files_root_folderpath is None:
            self.elem_files_root_folder = LTE_file.parent
        else:
            self.elem_files_root_folder = Path(elem_files_root_folderpath)

        self.LTE_text = LTE_file.read_text()
        self.LTE_filepath = LTE_filepath

        self._clean_up_LTE_text()

        d = self.get_used_beamline_element_defs(
            used_beamline_name=used_beamline_name)

        self.used_beamline_name = d['used_beamline_name']
        self.beamline_defs = d['beamline_defs']
        self.elem_defs = d['elem_defs']
        self.flat_used_elem_names = d['flat_used_elem_names']

        self._all_used_elem_names = [name for name, *_ in self.elem_defs]

        abs_kickmap_filepaths = self.get_kickmap_filepaths()['abs']
        for name, _fp in abs_kickmap_filepaths.items():
            abs_kickmap_f = Path(_fp)
            if not abs_kickmap_f.exists():
                print((f'Kickmap elment "{name}": File "{abs_kickmap_f}" '
                       f'does not exist.'))

        unhandled_types = self.get_unhandled_element_types(self.elem_defs)
        if unhandled_types != []:
            print('Element types that are not handled:')
            print(unhandled_types)

    def get_kickmap_filepaths(self):
        """"""

        kickmap_filepaths = {'raw': {}, 'abs': {}}

        for name, elem_type, prop_str in self.elem_defs:
            name = name.upper()
            if elem_type == 'UKICKMAP':
                kickmap_fp = self.parse_elem_properties(prop_str)['INPUT_FILE']
                if (kickmap_fp.startswith('"') and kickmap_fp.endswith('"')) or \
                   (kickmap_fp.startswith("'") and kickmap_fp.endswith("'")):
                    kickmap_fp = kickmap_fp[1:-1]

                kickmap_filepaths['raw'][name] = kickmap_fp

                abs_kickmap_f = self.elem_files_root_folder.joinpath(
                    Path(kickmap_fp)).resolve()

                kickmap_filepaths['abs'][name] = str(abs_kickmap_f)

        return kickmap_filepaths

    def remove_comments(self, text):
        """"""

        comment_char = '!'

        # Check if the comment character is the first character for comment
        # lines, as ELEGANT will not correctly parse the LTE file, even though
        # it may not crash.
        possibly_commented_lines_and_linenums = [
            (i, line) for i, line in enumerate(text.splitlines())
            if comment_char in line]
        for lineIndex, line in possibly_commented_lines_and_linenums:
            if not line.startswith(comment_char):
                print(
                    (f'\n** CRITICAL WARNING ** The character "{comment_char}" '
                     f'must be the first character on the comment line at '
                     f'Line {lineIndex}:')) # Line number here should not be
                # incremented by 1, as the passes "text" has an extra line added
                # to the top.
                print(line)

        pattern = comment_char + '.*'
        return re.sub(pattern, '', text)

    def delete_ampersands(self, text):
        """"""

        pattern = r'&.*[\n\r\s]+'
        return re.sub(pattern, '', text)

    def get_all_elem_defs(self, LTE_text):
        """
        "LTE_text" must not contain comments and ampersands.
        """

        #matches = re.findall('\s+"?([\w\$]+)"?[ \t]*:[ \t]*(\w+)[ \t]*,?(.*)',
                             #' '+LTE_text)
        matches = re.findall('\s+"?([\w\$:\.]+)"?[ \t]*:[ \t]*(\w+)[ \t]*,?(.*)',
                             ' '+LTE_text)
        # ^ Need to add the initial whitespace to pick up the first occurrence

        elem_def = [(name.upper(), type_name.upper(), rest.strip())
                    for (name, type_name, rest)
                    in matches if type_name.upper() != 'LINE']

        return elem_def

    def get_all_beamline_defs(self, LTE_text):
        """
        "LTE_text" must not contain comments and ampersands.
        """

        #matches = re.findall(
            #'\s+("?[\w\$:\.]+"?)[ \t]*:[ \t]*("?\w+"?)[ \t]*,?(.*)', LTE_text)
        matches = re.findall(
            '\s+"?([\w\$:\.]+)"?[ \t]*:[ \t]*"?([\w\$:\.]+)"?[ \t]*,?(.*)', LTE_text)

        beamline_def = []
        for (name, type_name, rest) in matches:
            if type_name.upper() == 'LINE':
                rest = rest.strip().replace('=','').replace('(','').replace(')','')
                name_list = [s.strip().upper() for s in rest.split(',') if s.strip() != '']
                name_list = [s[1:-1] if s.startswith('"') and s.endswith('"')
                             else s for s in name_list]
                if name[0] == '"' or name[-1] == '"':
                    assert name[0] == name[-1] == '"'
                    name = name[1:-1]
                beamline_def.append((name.upper(), name_list))

        return beamline_def

    def _get_used_beamline_name(self, LTE_text):
        """
        "LTE_text" must not contain comments and ampersands.
        """

        matches = re.findall('\s+USE[ \t]*,[ \t"]*([\w\$]+)[ \t\r\n"]*', LTE_text,
                             re.IGNORECASE)

        if len(matches) > 1:
            print('Multiple "USE" lines detected. Using the last "USE" line.')
            return matches[-1].upper()
        elif len(matches) == 0:
            print('No "USE" line detected.')
            return ''
        else:
            return matches[0].upper()

    def expand_beamline_name(
        self, beamline_name, all_beamline_defs, all_beamline_names,
        reverse=False, used_beamline_names=None):
        """
        If you want to obtain the list of used beamline names, then pass an empty
        list to "used_beamline_names".
        """

        if beamline_name in all_beamline_names:

            if used_beamline_names is not None:
                used_beamline_names.append(beamline_name)

            _, expanded_name_list = all_beamline_defs[all_beamline_names.index(beamline_name)]

            if reverse:
                expanded_name_list = expanded_name_list[::-1]

            for name in expanded_name_list:
                if '*' in name:
                    star_ind = name.index('*')
                    multiplier = int(name[:star_ind].strip())
                    name = name[(star_ind+1):].strip()
                else:
                    multiplier = 1

                if name.startswith('-'):
                    reverse_next = True
                    name = name[1:].strip()
                else:
                    reverse_next = False

                for i in range(multiplier):
                    for sub in self.expand_beamline_name(
                        name, all_beamline_defs, all_beamline_names,
                        reverse=reverse_next, used_beamline_names=used_beamline_names):
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
            if isinstance(el, collections.Iterable) and not isinstance(el, str):
                for sub in self.flatten_nested_list(el):
                    yield sub
            else:
                yield el

    def get_used_beamline_element_defs(self, used_beamline_name=''):
        """
        This function returns a new dictionary of the beamline/element definitions
        constructed from scratch based on the LTE file contents.

        If you want to use modified element definitions, use instead:
            get_persistent_used_beamline_element_defs()
        """

        if used_beamline_name is None:
            used_beamline_name = ''

        all_elem_defs = self.get_all_elem_defs(self.cleaned_LTE_text)
        all_beamline_defs = self.get_all_beamline_defs(self.cleaned_LTE_text)

        all_beamline_names = [name for name, _ in all_beamline_defs]
        all_elem_names     = [name for name, _, _ in all_elem_defs]

        if used_beamline_name == '':
            used_beamline_name = self._get_used_beamline_name(self.cleaned_LTE_text)

        if used_beamline_name == '':
            print('Using the last defined beamline.')
            used_beamline_name = all_beamline_names[-1]

        used_beamline_name = used_beamline_name.upper()

        assert used_beamline_name in all_beamline_names

        assert len(all_beamline_names) == len(np.unique(all_beamline_names))
        assert len(all_elem_names) == len(np.unique(all_elem_names))

        actually_used_beamline_names = [] # placeholder

        nested_used_elem_name_generator = self.expand_beamline_name(
            used_beamline_name, all_beamline_defs, all_beamline_names,
            used_beamline_names=actually_used_beamline_names)
        used_elem_name_generator = self.flatten_nested_list(
            nested_used_elem_name_generator)

        flat_used_elem_name_list = list(used_elem_name_generator)

        used_elem_names = [name if not name.startswith('-')
                           else name[1:] for name in flat_used_elem_name_list]
        used_elem_names = [name if '*' not in name else name[(name.index('*')+1):]
                           for name in used_elem_names]
        u_used_elem_names = np.unique(used_elem_names)

        self._unique_used_elem_names = u_used_elem_names.tolist()

        # Re-order in the order of appearance in the LTE file
        used_elem_defs = [all_elem_defs[all_elem_names.index(elem_name)]
                          for elem_name in all_elem_names
                          if elem_name in u_used_elem_names]

        _, u_inds = np.unique(actually_used_beamline_names, return_index=True)

        # Re-order in the required order of definitions
        used_beamline_defs = [
            all_beamline_defs[all_beamline_names.index(beamline_name)]
            for beamline_name in
            np.array(actually_used_beamline_names)[sorted(u_inds)[::-1]]
            if beamline_name in all_beamline_names
        ]

        # Separate the multiplier/reverser from beamline names
        used_beamline_defs_w_mults = []
        for defined_BL_name, unsep_name_list in used_beamline_defs:
            sep_name_multiplier_list = []
            for elem_or_BL_name in unsep_name_list:
                if elem_or_BL_name.startswith('-'):
                    sep_name_multiplier_list.append((elem_or_BL_name[1:], -1))
                elif '*' in elem_or_BL_name:
                    star_ind = elem_or_BL_name.index('*')
                    multiplier = int(elem_or_BL_name[:star_ind].strip())
                    name_only = elem_or_BL_name[(star_ind+1):].strip()
                    sep_name_multiplier_list.append((name_only, multiplier))
                else:
                    sep_name_multiplier_list.append((elem_or_BL_name, +1))

            used_beamline_defs_w_mults.append(
                (defined_BL_name, sep_name_multiplier_list))

        # Re-order used beamline definitions in the order of appearance in the LTE file
        # (Otherwise, when writing to a new LTE, parsing by ELEGANT may fail.)
        _beamline_names_w_mults = [v[0] for v in used_beamline_defs_w_mults]
        used_beamline_defs_w_mults = [
            used_beamline_defs_w_mults[_beamline_names_w_mults.index(beamline_name)]
            for beamline_name in all_beamline_names
            if beamline_name in actually_used_beamline_names
        ]

        return dict(used_beamline_name=used_beamline_name,
                    beamline_defs=used_beamline_defs_w_mults,
                    elem_defs=used_elem_defs,
                    flat_used_elem_names=flat_used_elem_name_list)

    def get_persistent_used_beamline_element_defs(self, used_beamline_name=''):
        """"""

        if used_beamline_name is None:
            used_beamline_name = ''

        if self._persistent_LTE_d is None:

            d = self.get_used_beamline_element_defs(
                used_beamline_name=used_beamline_name)

            self._persistent_LTE_d = {d['used_beamline_name']: d}

            # Add the default beamline name case
            if used_beamline_name == '':
                self._persistent_LTE_d[''] = d['used_beamline_name']

        else:

            if used_beamline_name in self._persistent_LTE_d:
                pass
            else:
                d = self.get_used_beamline_element_defs(
                    used_beamline_name=used_beamline_name)

                self._persistent_LTE_d[d['used_beamline_name']] = d

        return self._persistent_LTE_d[used_beamline_name]


    def parse_elem_properties(self, prop_str: str):
        """"""

        pat = r'(\w+)\s*=\s*([^,]+)'

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
                val_str = f'{v:.16g}'
            elif isinstance(v, int):
                val_str = f'{v:d}'
            elif isinstance(v, str):
                val_str = v
            else:
                raise NotImplementedError

            new_elem_prop_str_list.append(f'{k}={val_str}')

        new_elem_prop_str = ', '.join(new_elem_prop_str_list)

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
            used_beamline_name=self.used_beamline_name)

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
            type_name.upper() for (_,type_name,_) in elem_def_list
            if type_name.upper() not in self.handled_element_types]

        return list(set(unhandled_list))

    def zip_lte(self, output_ltezip_filepath, header_comment=''):
        """"""

        contents = dict(
            header_comment=header_comment,
            orig_LTE_filepath=str(Path(self.LTE_filepath).resolve()),
            raw_LTE_text=self.LTE_text, # save raw LTE text
        )

        km_files = self.get_kickmap_filepaths()

        u_folderpaths = np.unique([
            str(Path(abs_path).parent)
            for elem_name, abs_path in km_files['abs'].items()]).tolist()

        contents['km'] = dict(unique_parents=u_folderpaths,
                              meta={}, file_contents={})

        for elem_name, abs_path in km_files['abs'].items():
            contents['km']['file_contents'][abs_path] = Path(
                abs_path).read_bytes().decode('latin-1')
            contents['km']['meta'][elem_name] = dict(
                abs_path=abs_path,
                folder_index=u_folderpaths.index(str(Path(abs_path).parent)),
            )

        with gzip.GzipFile(output_ltezip_filepath, 'wb') as f:
            f.write(json.dumps(contents).encode('utf-8'))

    @staticmethod
    def unzip_lte(
        ltezip_filepath, output_lte_filepath_str='',
        suppl_files_folderpath_str='./lte_suppl', use_abs_paths_for_suppl_files=True,
        overwrite_lte=False, overwrite_suppl=False):
        """
        If "output_lte_filepath_str" is not specified, a new LTE file will be
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
        (default).
        """

        suppl_files_folderpath_d = dict(km=None)

        with gzip.GzipFile(ltezip_filepath, 'rb') as f:
            contents = json.loads(f.read().decode('utf-8'))

        if output_lte_filepath_str == '':
            output_lte_filepath = Path.cwd().joinpath(
                Path(contents['orig_LTE_filepath']).name)
        else:
            output_lte_filepath = Path(output_lte_filepath_str)

        if output_lte_filepath.exists() and (not overwrite_lte):
            raise FileExistsError(
                f'Cannot write a new LTE file to "{output_lte_filepath}"')

        # Make sure the parent folder exists.
        output_lte_filepath.parent.mkdir(parents=True, exist_ok=True)

        temp_LTE = Lattice()
        temp_LTE.LTE_text = contents['raw_LTE_text']
        temp_LTE.LTE_filepath = contents['orig_LTE_filepath']
        #
        temp_LTE._clean_up_LTE_text()
        #
        d = temp_LTE.get_used_beamline_element_defs()

        # Start building new LTE contents
        lines = ['! ' + line for line in contents['header_comment'].split('\n')
                 if line.strip()]
        #
        # Add element definition sections
        lines.append('\n')
        _newly_created_suppl_filepaths = collections.defaultdict(list)
        for elem_name, elem_type, prop_str in d['elem_defs']:
            if elem_type == 'UKICKMAP':
                temp_prop_str_list = []
                for token in prop_str.split(','):
                    if token.split('=')[0].strip() == 'INPUT_FILE':
                        orig_abs_path = contents['km']['meta'][elem_name]['abs_path']
                        orig_filename = str(Path(orig_abs_path).name)

                        if suppl_files_folderpath_d['km'] is None:
                            suppl_files_folderpath_d['km'] = Path(
                                suppl_files_folderpath_str).joinpath('km')
                            suppl_files_folderpath_d['km'].mkdir(
                                parents=True, exist_ok=True)

                        new_rel_path = suppl_files_folderpath_d['km'].joinpath(
                            orig_filename)
                        new_abs_path = new_rel_path.resolve()
                        if new_abs_path not in _newly_created_suppl_filepaths['km']:
                            if new_abs_path.exists() and (not overwrite_suppl):
                                raise FileExistsError(
                                    (f'Cannot write a new LTE supplementary file '
                                     f'to "{new_abs_path}"'))

                            new_abs_path.write_bytes(
                                contents['km']['file_contents'][orig_abs_path].encode('latin-1'))
                            print(f'* Created LTE supplementary file: {new_abs_path}')
                            _newly_created_suppl_filepaths['km'].append(new_abs_path)

                        if use_abs_paths_for_suppl_files:
                            token = f'INPUT_FILE="{new_abs_path}"'
                        else:
                            token = f'INPUT_FILE="{new_rel_path}"'

                    temp_prop_str_list.append(token.strip())

                prop_str = ', '.join(temp_prop_str_list)
            else:
                prop_str = ', '.join([token.strip() for token in prop_str.split(',')])

            prop_str = prop_str.strip()
            if prop_str == '':
                temp_line = f'{elem_name}: {elem_type}'
            else:
                temp_line = f'{elem_name}: {elem_type}, {prop_str}'
            temp_line = temp_LTE.get_wrapped_line(temp_line)
            lines.extend(temp_line.split('\n'))
        #
        # Add beamline definition sections
        lines.append('\n')
        if d['beamline_defs'] != []:
            for beamline_name, sub_name_multiplier_tups in d['beamline_defs']:

                sub_lines = []
                for sub_name, multiplier in sub_name_multiplier_tups:
                    if multiplier == 1:
                        sub_lines.append(sub_name)
                    elif multiplier == -1:
                        sub_lines.append('-' + sub_name)
                    else:
                        sub_lines.append(f'{multiplier:d}*{sub_name}')

                temp_line = temp_LTE.get_wrapped_line(
                    f'{beamline_name}: LINE=({",".join(sub_lines)})')
                lines.extend(temp_line.split('\n'))
        else:
            # Note that d["flat_used_elem_names"] does NOT start with "_BEG_"
            full_beamline_line = temp_LTE.get_wrapped_line(
                f'{d["used_beamline_name"]}: LINE=({",".join(d["flat_used_elem_names"])})')
            lines.extend(full_beamline_line.split('\n'))
        #
        # Finally add "USE" line
        lines.append('\n')
        lines.append(f'USE, {d["used_beamline_name"]}')

        # Write the LTE file
        output_lte_filepath.write_text('\n'.join(lines))

        return str(output_lte_filepath.resolve())

    @staticmethod
    def get_wrapped_line(line, max_len=80, sep=',', indent=2):
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
                    if len(' '.join([new_line, sub_line])) > max_len:
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

    def get_elem_inds_from_name(self, elem_name):
        """"""

        elem_inds = np.where(np.array(self.flat_used_elem_names) == elem_name)[0]

        if elem_inds.size != 0:
            elem_inds += 1
            # Increase indexes by 1 to account for ELEGANT's insertion of the
            # special element "_BEG_" at the beginning of the beamline defined
            # in "self.flat_used_elem_names".

        return elem_inds

    def get_elem_inds_from_names(self, elem_names):
        """"""

        _flat_used_elem_names = np.array(self.flat_used_elem_names)

        match = (_flat_used_elem_names == elem_names[0])
        for name in elem_names[1:]:
            match = match | (_flat_used_elem_names == name)

        elem_inds = np.where(match)[0]

        if elem_inds.size != 0:
            elem_inds += 1
            # Increase indexes by 1 to account for ELEGANT's insertion of the
            # special element "_BEG_" at the beginning of the beamline defined
            # in "self.flat_used_elem_names".

        return elem_inds

    def get_names_from_elem_inds(self, elem_inds):
        """"""

        # Must decrease indexes by 1 to account for ELEGANT's insertion of the
        # special element "_BEG_" at the beginning of the beamline defined
        # in "self.flat_used_elem_names".
        return np.array(self.flat_used_elem_names)[np.array(elem_inds)-1]

    def get_elem_inds_from_regex(self, pattern):
        """"""

        matched_elem_names = [
            elem_name for elem_name in self._unique_used_elem_names
            if re.match(pattern, elem_name) is not None]

        return self.get_elem_inds_from_names(matched_elem_names)

    def get_elem_inds_from_name_occur_tuples(self, name_occur_tuples):
        """"""

        u_elem_names = np.unique([name for name, _ in name_occur_tuples])
        elem_inds_d = {name: self.get_elem_inds_from_name(name)
                       for name in u_elem_names}

        return np.array([
            elem_inds_d[elem_name][occur-1]
            # ^ Index here must be decreased by 1 as "ElementOccurence" is 1-based index
            for elem_name, occur in name_occur_tuples])

    def get_name_occur_tuples_from_elem_inds(self, elem_inds):
        """"""

        elem_names = self.get_names_from_elem_inds(elem_inds)
        assert len(elem_names) == len(elem_inds)

        u_elem_names = np.unique(elem_names)
        elem_inds_d = {name: self.get_elem_inds_from_name(name).tolist()
                       for name in u_elem_names}

        return [
            (name, elem_inds_d[name].index(i) + 1)
            # ^ Index here must be increased by 1 as "ElementOccurence" is 1-based index
            for name, i in zip(elem_names, elem_inds)]

    def get_elem_type_from_name(self, elem_name):
        """"""

        if elem_name in self._all_used_elem_names:
            matched_index = self._all_used_elem_names.index(elem_name)
            _, elem_type, _ = self.elem_defs[matched_index]
        else:
            elem_type = None

        return elem_type

    @staticmethod
    def write_LTE(
        new_LTE_filepath, used_beamline_name, elem_defs, beamline_defs):
        """"""

        used_beamline_name = used_beamline_name.upper()
        assert used_beamline_name in [beamline_name for (beamline_name, _)
                                      in beamline_defs]

        lines = []

        for elem_name, elem_type, prop_str in elem_defs:
            if ':' in elem_name: # Must enclose element name with double quotes
                # if the name contains ":".
                elem_name = f'"{elem_name}"'
            def_line = f'{elem_name}: {elem_type}'
            if prop_str != '':
                def_line += f', {prop_str}'
            lines.append(Lattice.get_wrapped_line(def_line))

        lines.append(' ')

        for beamline_name, bdef in beamline_defs:
            def_parts = []
            for name, multiplier in bdef:
                if multiplier == 1:
                    def_parts.append(name)
                elif multiplier == -1:
                    def_parts.append(f'-{name}')
                else:
                    def_parts.append(f'{multiplier:d}*{name}')

            def_line = ', '.join(def_parts)

            if ':' in beamline_name: # Must enclose beamline name with double
                # quotes if the name contains ":".
                beamline_name = f'"{beamline_name}"'

            lines.append(
                Lattice.get_wrapped_line(f'{beamline_name}: LINE=({def_line})'))

        lines.append(' ')

        lines.append(f'USE, "{used_beamline_name}"')
        lines.append('RETURN')

        Path(new_LTE_filepath).write_text('\n'.join(lines))


def write_temp_modified_LTE(
    mod_prop_dict_list, base_LTE_filepstr='', base_used_beamline_name='', LTE_obj=None
):
    """
    A valid example of "mod_prop_dict_list":
        [
        {'elem_name': 'Q1', 'prop_name': 'K1', 'prop_val': 1.5},
        {'elem_name': 'S1', 'prop_name': 'K2', 'prop_val': 2.0},
        ]
    """

    tmp = tempfile.NamedTemporaryFile(
        prefix=f'tmpLteMod_', suffix='.lte', dir=Path.cwd().resolve(), delete=False
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

    return temp_LTE_filepstr


def write_modified_LTE(
    new_LTE_filepstr,
    mod_prop_dict_list,
    base_LTE_filepstr='',
    base_used_beamline_name='',
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
        if base_LTE_filepstr != '':
            print(f'WARNING: Ignored: base_LTE_filepstr = "{base_LTE_filepstr}"')
        if base_used_beamline_name != '':
            print(
                f'WARNING: Ignored: base_used_beamline_name = "{base_used_beamline_name}"'
            )
        LTE = LTE_obj

    LTE.modify_elem_properties(mod_prop_dict_list)
    LTE_d = LTE.get_persistent_used_beamline_element_defs(
        used_beamline_name=LTE.used_beamline_name)

    LTE.write_LTE(
        new_LTE_filepstr,
        LTE.used_beamline_name,
        LTE_d['elem_defs'],
        LTE_d['beamline_defs'],
    )
