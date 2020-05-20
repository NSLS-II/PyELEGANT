import re
from pathlib import Path
import numpy as np
import collections

########################################################################
class Lattice():
    """
    """

    def __init__(self, LTE_filepath='', used_beamline_name=''):
        """Constructor"""

        self.convertible_element_types = [
            'DRIF','EDRIFT','RFCA',
            'CSBEND', 'CSBEN', 'SBEN','SBEND',
            'KQUAD', 'QUAD',
            'KSEXT', 'SEXT',
            'KOCT', 'OCTU',
            'MULT',
            'UKICKMAP',
            #'HKICK','VKICK','KICKER',
            'EHKICK','EVKICK','EKICKER',
            'MARK', 'MONI',
            #'SCRAPER',
            #'SOLE',
            'MALIGN', 'WATCH',
            ]

        if LTE_filepath != '':
            self.load_LTE(LTE_filepath, used_beamline_name=used_beamline_name)

    def load_LTE(self, LTE_filepath, used_beamline_name='',
                 elem_files_root_folderpath=None):
        """"""

        LTE_file = Path(LTE_filepath)

        if elem_files_root_folderpath is None:
            self.elem_files_root_folder = LTE_file.parent
        else:
            self.elem_files_root_folder = Path(elem_files_root_folderpath)

        self.LTE_text = LTE_file.read_text()
        self.LTE_filepath = LTE_filepath

        self.cleaned_LTE_text = '\n' + self.LTE_text
        # ^ adding "\n" at the beginning for easier search
        self.cleaned_LTE_text = self.remove_comments(self.cleaned_LTE_text)
        self.cleaned_LTE_text = self.delete_ampersands(self.cleaned_LTE_text)

        d = self.get_used_beamline_element_defs(
            used_beamline_name=used_beamline_name)

        self.used_beamline_name = d['used_beamline_name']
        self.beamline_defs = d['beamline_defs']
        self.elem_defs = d['elem_defs']
        self.flat_used_elem_names = d['flat_used_elem_names']

        abs_kickmap_filepaths = self.get_kickmap_filepaths()['abs']
        for name, _fp in abs_kickmap_filepaths.items():
            abs_kickmap_f = Path(_fp)
            if not abs_kickmap_f.exists():
                print((f'Kickmap elment "{name}": File "{abs_kickmap_f}" '
                       f'does not exist.'))

        inconvertible_types = self.get_inconvertible_element_types(self.elem_defs)
        if inconvertible_types != []:
            print('Element types that cannot be converted:')
            print(inconvertible_types)

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

        matches = re.findall('\s+"?(\w+)"?[ \t]*:[ \t]*(\w+)[ \t]*,?(.*)',
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

        matches = re.findall(
            '\s+("?\w+"?)[ \t]*:[ \t]*("?\w+"?)[ \t]*,?(.*)', LTE_text)

        beamline_def = []
        for (name, type_name, rest) in matches:
            if type_name.upper() == 'LINE':
                rest = rest.strip().replace('=','').replace('(','').replace(')','')
                name_list = [s.strip().upper() for s in rest.split(',') if s.strip() != '']
                if name[0] == '"' or name[-1] == '"':
                    assert name[0] == name[-1] == '"'
                    name = name[1:-1]
                beamline_def.append((name.upper(), name_list))

        return beamline_def

    def get_used_beamline_name(self, LTE_text):
        """
        "LTE_text" must not contain comments and ampersands.
        """

        matches = re.findall('\s+USE[ \t]*,[ \t"]*(\w+)[ \t\r\n"]*', LTE_text,
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
                    name = name[(star_ind+1):]
                else:
                    multiplier = 1

                if name.startswith('-'):
                    reverse_next = True
                    name = name[1:]
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
        """"""

        all_elem_defs = self.get_all_elem_defs(self.cleaned_LTE_text)
        all_beamline_defs = self.get_all_beamline_defs(self.cleaned_LTE_text)

        all_beamline_names = [name for name, _ in all_beamline_defs]
        all_elem_names     = [name for name, _, _ in all_elem_defs]

        if used_beamline_name == '':
            used_beamline_name = self.get_used_beamline_name(self.cleaned_LTE_text)

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

        return dict(used_beamline_name=used_beamline_name,
                    beamline_defs=used_beamline_defs_w_mults,
                    elem_defs=used_elem_defs,
                    flat_used_elem_names=flat_used_elem_name_list)

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

    def get_inconvertible_element_types(self, elem_def_list):
        """"""

        inconv_list = [
            type_name.upper() for (_,type_name,_) in elem_def_list
            if type_name.upper() not in self.convertible_element_types]

        return list(set(inconv_list))

########################################################################
class KQUAD():
    """
    10.46 KQUAD - A canonical kick quadrupole
    """

    #----------------------------------------------------------------------
    def __init__(self, **kwargs):
        """Constructor"""

        self.L = kwargs.get('L', 0.0)
