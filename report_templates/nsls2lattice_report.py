# TODO
# *) Prompt whether to re-create input LTE if it already exists
# *) Check LTE contents with current and the one saved in nonlin/lin data files,
#    If not, recalc (w/ prompt whether proceeding for recalc)
import sys
import os
import numpy as np
import getpass
import matplotlib.pylab as plt
from types import SimpleNamespace
from matplotlib.backends.backend_pdf import PdfPages
import pickle
#import yaml
from ruamel import yaml
# ^ ruamel's "yaml" does NOT suffer from the PyYAML(v5.3, YAML v1.1) problem
#   that a float value in scientific notation without "." and the sign after e/E
#   is treated as a string.
from pathlib import Path

import pyelegant as pe
from pyelegant import nonlin
pe.disable_stdout()
pe.enable_stderr()

import pylatex as plx

GREEK = dict()
for k in ['nu', 'delta', 'nu_x', 'nu_y', 'A_x', 'A_y']:
    GREEK[k] = plx.NoEscape(fr'\{k}')
GMATH = dict()
for k, v in GREEK.items():
    GMATH[k] = plx.Math(inline=True, data=[v])
#
# Convert to SimpleNamespace
GMATH = SimpleNamespace(**GMATH)

class MathEnglishTextV1():
    """"""

    def __init__(self, doc):
        """Constructor"""

        self.doc = doc
        self.list = []

    def _conv_str_to_math(self, s):
        """"""

        math_str = ''.join([f'{{{v}}}' if v != ' ' else r'\ ' for v in s])
        math_str = math_str.replace('}{', '')
        math_str = math_str.replace('{', '\mathrm{')

        return plx.Math(inline=True, data=[plx.NoEscape(math_str)])

    def append(self, obj):
        """"""

        if isinstance(obj, str):
            obj = self._conv_str_to_math(obj)
        else:
            assert isinstance(obj, plx.Math)

        self.list.append(obj)

    def extend(self, obj_list):
        """"""

        for i, obj in enumerate(obj_list):
            if isinstance(obj, str):
                obj_list[i] = self._conv_str_to_math(obj)
            else:
                assert isinstance(obj, plx.Math)

        self.list.extend(obj_list)

    def clear(self):
        """"""

        self.list.clear()

    def apply(self):
        """"""

        doc.extend(self.list)

class MathEnglishTextV2(plx.Math):
    """"""

    def __init__(self, obj_list=None):
        """Constructor"""

        super().__init__(inline=True, data=[])

        if obj_list is not None:
            self.extend(obj_list)

    def _conv_str_to_math(self, s):
        """"""

        math_str = ''.join([f'{{{v}}}' if v != ' ' else r'\ ' for v in s])
        math_str = math_str.replace('}{', '')
        math_str = math_str.replace('{', '\mathrm{')

        return plx.NoEscape(math_str)

    def append(self, obj):
        """"""

        if isinstance(obj, str):
            self.data.append(self._conv_str_to_math(obj))
        else:
            assert isinstance(obj, plx.Math)
            self.data.extend([plx.NoEscape(s) for s in obj.data])

    def extend(self, obj_list):
        """"""

        for i, obj in enumerate(obj_list):
            if isinstance(obj, str):
                self.data.append(self._conv_str_to_math(obj))
            else:
                assert isinstance(obj, plx.Math)
                self.data.extend([plx.NoEscape(s) for s in obj.data])

    def clear(self):
        """"""

        self.data.clear()

    def dumps_for_caption(self):
        """"""

        return plx.NoEscape(self.dumps())


class CombinedMathNormalText(plx.Math):
    """"""
    def __init__(self):
        """Constructor"""

        super().__init__(inline=True, data=[])

    def _conv_str_to_math(self, s):
        """"""

        math_str = ''.join([f'{{{v}}}' if v != ' ' else r'\ ' for v in s])
        math_str = math_str.replace('_', r'\_')
        math_str = math_str.replace('&', r'\&')
        math_str = math_str.replace('}{', '')
        math_str = math_str.replace('{', r'\mathrm{')

        return plx.NoEscape(math_str)

    def copy(self):
        """"""

        return pickle.loads(pickle.dumps(self))

    def __add__(self, other):
        """"""

        copy = CombinedMathNormalText()
        copy.data.extend(self.data)

        copy.append(other)

        return copy

    def __radd__(self, left):
        """"""

        if isinstance(left, str):
            copy = CombinedMathNormalText()
            copy.data.append(self._conv_str_to_math(left))
            left = copy

        return left.__add__(self)

    def append(self, obj):
        """"""

        if isinstance(obj, str):
            self.data.append(self._conv_str_to_math(obj))
        else:
            assert isinstance(obj, plx.Math)
            self.data.extend([plx.NoEscape(s) for s in obj.data])

    def extend(self, obj_list):
        """"""

        for i, obj in enumerate(obj_list):
            if isinstance(obj, str):
                self.data.append(self._conv_str_to_math(obj))
            else:
                assert isinstance(obj, plx.Math)
                self.data.extend([plx.NoEscape(s) for s in obj.data])

    def clear(self):
        """"""

        self.data.clear()

    def dumps_NoEscape(self):
        """"""

        return plx.NoEscape(self.dumps())

    def dumps_for_caption(self):
        """"""

        return self.dumps_NoEscape()

class MathText(CombinedMathNormalText):
    """"""

    def __init__(self, r_str):
        """Constructor"""

        super().__init__()
        self.data.append(plx.NoEscape(r_str))

        self.r_str = r_str # without this, print() of this object will fail

class NormalText(CombinedMathNormalText):
    """"""

    def __init__(self, s):
        """Constructor"""

        super().__init__()
        self.data.append(self._conv_str_to_math(s))

        self.s = s # without this, print() of this object will fail



class FigureForMultiPagePDF(plx.Figure):
    """"""

    _latex_name = "figure"

    def add_image(self, filename, *, width=plx.NoEscape(r'0.8\textwidth'),
                  page=None, placement=plx.NoEscape(r'\centering')):
        """Add an image to the figure.

        Args
        ----
        filename: str
            Filename of the image.
        width: str
            The width of the image
        page: int
            The page number of the PDF file for the image
        placement: str
            Placement of the figure, `None` is also accepted.

        """

        if width is not None:
            if self.escape:
                width = plx.escape_latex(width)

            image_options = 'width=' + str(width)

        if page is not None:
            image_options = [image_options, f'page={page:d}']

        if placement is not None:
            self.append(placement)

        self.append(plx.StandAloneGraphic(
            image_options=image_options,
            filename=plx.utils.fix_filename(filename)))

class SubFigureForMultiPagePDF(FigureForMultiPagePDF):
    """A class that represents a subfigure from the subcaption package."""

    _latex_name = "subfigure"

    packages = [plx.Package('subcaption')]

    #: By default a subfigure is not on its own paragraph since that looks
    #: weird inside another figure.
    separate_paragraph = False

    _repr_attributes_mapping = {
        'width': 'arguments',
    }

    def __init__(self, width=plx.NoEscape(r'0.45\linewidth'), **kwargs):
        """
        Args
        ----
        width: str
            Width of the subfigure itself. It needs a width because it is
            inside another figure.

        """

        super().__init__(arguments=width, **kwargs)

    def add_image(self, filename, *, width=plx.NoEscape(r'\linewidth'), page=None,
                  placement=None):
        """Add an image to the subfigure.

        Args
        ----
        filename: str
            Filename of the image.
        width: str
            Width of the image in LaTeX terms.
        page: int
            The page number of the PDF file for the image
        placement: str
            Placement of the figure, `None` is also accepted.
        """

        super().add_image(filename, width=width, page=page, placement=placement)

def generate_pdf_w_reruns(
    doc, filepath=None, *, clean=True, clean_tex=True,
    compiler=None, compiler_args=None, silent=True, nMaxReRuns=10):
    """
    When the output of running LaTeX contains "Rerun LaTeX.", this function
    will re-run latex until this message disappears up to "nMaxReRuns" times.


    Generate a pdf file from the document.

    Args
    ----
    filepath: str
        The name of the file (without .pdf), if it is `None` the
        ``default_filepath`` attribute will be used.
    clean: bool
        Whether non-pdf files created that are created during compilation
        should be removed.
    clean_tex: bool
        Also remove the generated tex file.
    compiler: `str` or `None`
        The name of the LaTeX compiler to use. If it is None, PyLaTeX will
        choose a fitting one on its own. Starting with ``latexmk`` and then
        ``pdflatex``.
    compiler_args: `list` or `None`
        Extra arguments that should be passed to the LaTeX compiler. If
        this is None it defaults to an empty list.
    silent: bool
        Whether to hide compiler output
    """

    import subprocess
    import errno
    rm_temp_dir = plx.utils.rm_temp_dir
    CompilerError = plx.errors.CompilerError

    if compiler_args is None:
        compiler_args = []

    filepath = doc._select_filepath(filepath)
    filepath = os.path.join('.', filepath)

    cur_dir = os.getcwd()
    dest_dir = os.path.dirname(filepath)
    basename = os.path.basename(filepath)

    if basename == '':
        basename = 'default_basename'

    os.chdir(dest_dir)

    doc.generate_tex(basename)

    if compiler is not None:
        compilers = ((compiler, []),)
    else:
        latexmk_args = ['--pdf']

        compilers = (
            ('latexmk', latexmk_args),
            ('pdflatex', [])
        )

    main_arguments = ['--interaction=nonstopmode', basename + '.tex']

    os_error = None

    for compiler, arguments in compilers:
        command = [compiler] + arguments + compiler_args + main_arguments

        if compiler == 'latexmk':
            actual_nMaxReRuns = 1
        else:
            actual_nMaxReRuns = nMaxReRuns

        check_next_compiler = False

        for iLaTeXRun in range(actual_nMaxReRuns):

            try:
                output = subprocess.check_output(command,
                                                 stderr=subprocess.STDOUT)
            except (OSError, IOError) as e:
                # Use FileNotFoundError when python 2 is dropped
                os_error = e

                if os_error.errno == errno.ENOENT:
                    # If compiler does not exist, try next in the list
                    check_next_compiler = True
                    break
                    #continue
                raise
            except subprocess.CalledProcessError as e:
                # For all other errors print the output and raise the error
                print(e.output.decode())
                raise
            else:
                if not silent:
                    print(output.decode())

            if (compiler == 'pdflatex') and ('Rerun LaTeX.' in output.decode()):
                print(f'\n\n*** LaTeX rerun instruction detected. '
                      f'Re-running LaTeX (Attempt #{iLaTeXRun+2:d})\n\n')
                continue
            else:
                break

        if check_next_compiler:
            continue

        if clean:
            try:
                # Try latexmk cleaning first
                subprocess.check_output(['latexmk', '-c', basename],
                                        stderr=subprocess.STDOUT)
            except (OSError, IOError, subprocess.CalledProcessError) as e:
                # Otherwise just remove some file extensions.
                extensions = ['aux', 'log', 'out', 'fls',
                              'fdb_latexmk']

                for ext in extensions:
                    try:
                        os.remove(basename + '.' + ext)
                    except (OSError, IOError) as e:
                        # Use FileNotFoundError when python 2 is dropped
                        if e.errno != errno.ENOENT:
                            raise
            rm_temp_dir()

        if clean_tex:
            os.remove(basename + '.tex')  # Remove generated tex file

        # Compilation has finished, so no further compilers have to be
        # tried
        break

    else:
        # Notify user that none of the compilers worked.
        raise(CompilerError(
            'No LaTex compiler was found\n' +
            'Either specify a LaTex compiler ' +
            'or make sure you have latexmk or pdfLaTex installed.'
        ))

    os.chdir(cur_dir)

def summarize_lin(
    LTE_filepath, E_MeV=3e3, use_beamline_cell='CELL', use_beamline_ring='RING',
    zeroSexts_LTE_filepath='', element_divisions=10, twiss_plot_opts=None,
    beta_elem_names=None, phase_adv_elem_names_indexes=None, length_elem_names=None,
    floor_elem_names_indexes=None):
    """"""

    existing_fignums = plt.get_fignums()
    fignums = {}

    sel_data = {'E_GeV': E_MeV / 1e3}
    interm_array_data = {} # holds data that will be only used to derive some other quantities

    raw_keys = dict(
        one_period=dict(eps_x='ex0', Jx='Jx', Jdelta='Jdelta'),
        ring=dict(
            nux='nux', nuy='nuy', ksi_x_cor='dnux/dp', ksi_y_cor='dnuy/dp',
            alphac='alphac', U0_MeV='U0', dE_E='Sdelta0'),
        ring_natural=dict(ksi_x_nat='dnux/dp', ksi_y_nat='dnuy/dp'),
    )

    nsls2_data = {}
    nsls2_data['circumf'] = 791.958 # [m]

    interm_array_keys = dict(
        one_period=dict(
            s_one_period='s', betax_1p='betax', betay_1p='betay', etax_1p='etax',
            elem_names_1p='ElementName'),
        ring=dict(
            s_ring='s', elem_names_ring='ElementName',
            betax_ring='betax', betay_ring='betay',
            psix_ring='psix', psiy_ring='psiy'),
    )

    output_filepath = 'test.pgz'
    #output_filepath = 'test.hdf5'

    pe.calc_ring_twiss(
        output_filepath, LTE_filepath, E_MeV, radiation_integrals=True,
        element_divisions=element_divisions, use_beamline=use_beamline_cell)

    if twiss_plot_opts is not None:

        for opts in twiss_plot_opts:
            pe.plot_twiss(output_filepath, **opts)
        #pe.plot_twiss(output_filepath, print_scalars=[], right_margin_adj=0.85)
        #pe.plot_twiss(
            #output_filepath, print_scalars=[],
            #slim=[0, 9], right_margin_adj=0.85,
            #disp_elem_names=dict(bends=True, quads=True, sexts=True,
                                 #font_size=8, extra_dy_frac=0.05))
        #pe.plot_twiss(
            #output_filepath, print_scalars=[],
            #slim=[4, 16], right_margin_adj=0.85,
            #disp_elem_names=dict(bends=True, quads=True, sexts=True,
                                 #font_size=8, extra_dy_frac=0.05))
        #pe.plot_twiss(
            #output_filepath, print_scalars=[],
            #slim=[14, 23], right_margin_adj=0.85,
            #disp_elem_names=dict(bends=True, quads=True, sexts=True,
                                 #font_size=8, extra_dy_frac=0.05))

        fignums['twiss'] = []
        for fignum in plt.get_fignums():
            if fignum not in existing_fignums:
                fignums['twiss'].append(fignum)
                existing_fignums.append(fignum)


    if output_filepath.endswith('.pgz'):
        twi = pe.util.load_pgz_file(output_filepath)['data']['twi']
    elif output_filepath.endswith(('.h5', '.hdf5')):
        twi = pe.util.load_sdds_hdf5_file(output_filepath)[0]['twi']
    else:
        raise ValueError

    for k, ele_k in raw_keys['one_period'].items():
        sel_data[k] = twi['scalars'][ele_k]
    for k, ele_k in interm_array_keys['one_period'].items():
        interm_array_data[k] = twi['arrays'][ele_k]

    if zeroSexts_LTE_filepath != '':
        pe.calc_ring_twiss(
            output_filepath, zeroSexts_LTE_filepath, E_MeV, radiation_integrals=True,
            use_beamline=use_beamline_ring)

        if output_filepath.endswith('.pgz'):
            twi = pe.util.load_pgz_file(output_filepath)['data']['twi']
        elif output_filepath.endswith(('.h5', '.hdf5')):
            twi = pe.util.load_sdds_hdf5_file(output_filepath)[0]['twi']
        else:
            raise ValueError

        for k, ele_k in raw_keys['ring_natural'].items():
            sel_data[k] = twi['scalars'][ele_k]

    pe.calc_ring_twiss(
        output_filepath, LTE_filepath, E_MeV, radiation_integrals=True,
        use_beamline=use_beamline_ring)

    if output_filepath.endswith('.pgz'):
        twi = pe.util.load_pgz_file(output_filepath)['data']['twi']
    elif output_filepath.endswith(('.h5', '.hdf5')):
        twi = pe.util.load_sdds_hdf5_file(output_filepath)[0]['twi']
    else:
        raise ValueError

    for k, ele_k in raw_keys['ring'].items():
        sel_data[k] = twi['scalars'][ele_k]
    for k, ele_k in interm_array_keys['ring'].items():
        interm_array_data[k] = twi['arrays'][ele_k]

    sel_data['circumf'] = interm_array_data['s_ring'][-1]

    sel_data['circumf_change_%'] = (sel_data['circumf'] / nsls2_data['circumf'] - 1) * 1e2

    ring_mult = sel_data['circumf'] / interm_array_data['s_one_period'][-1]
    sel_data['nPeriodsInRing'] = int(np.round(ring_mult))
    assert np.abs(ring_mult - sel_data['nPeriodsInRing']) < 1e-3

    sel_data['max_betax'] = np.max(interm_array_data['betax_1p'])
    sel_data['min_betax'] = np.min(interm_array_data['betax_1p'])
    sel_data['max_betay'] = np.max(interm_array_data['betay_1p'])
    sel_data['min_betay'] = np.min(interm_array_data['betay_1p'])
    sel_data['max_etax'] = np.max(interm_array_data['etax_1p'])
    sel_data['min_etax'] = np.min(interm_array_data['etax_1p'])

    if beta_elem_names is not None:
        for elem_name in beta_elem_names:
            elem_name = elem_name.upper()
            index = np.where(interm_array_data['elem_names_ring'] == elem_name)[0][0]
            #print(elem_name, index)
            sel_data[f'betax@{elem_name}'] = interm_array_data['betax_ring'][index]
            sel_data[f'betay@{elem_name}'] = interm_array_data['betay_ring'][index]

    if phase_adv_elem_names_indexes is not None:
        for key, (elem_name_1, occur_1, elem_name_2, occur_2
                  ) in phase_adv_elem_names_indexes.items():
            index_1 = np.where(
                interm_array_data['elem_names_ring'] == elem_name_1.upper())[0][occur_1]
            index_2 = np.where(
                interm_array_data['elem_names_ring'] == elem_name_2.upper())[0][occur_2]
            sel_data[f'dnux@{key}'] = (
                interm_array_data['psix_ring'][index_2] -
                interm_array_data['psix_ring'][index_1]) / (2 * np.pi)
            sel_data[f'dnuy@{key}'] = (
                interm_array_data['psiy_ring'][index_2] -
                interm_array_data['psiy_ring'][index_1]) / (2 * np.pi)

        #try:
            #mdisp_inds = np.where(twi['arrays']['ElementName'] == 'MDISP')[0]
            #dphix = np.diff(twi['arrays']['psix'][mdisp_inds])[0]
            #dphiy = np.diff(twi['arrays']['psiy'][mdisp_inds])[0]

            #print_lines.append('Phase Adv. btw. Disp. Sec. Centers (x, y) = ({:.6f}, {:.6f})'.format(
                #dphix / (2 * np.pi), dphiy / (2 * np.pi)))
        #except:
            #pass

    ed = pe.elebuilder.EleDesigner()
    ed.add_block('run_setup',
        lattice = LTE_filepath, p_central_mev = E_MeV,
        use_beamline=use_beamline_cell, parameters='%s.param')
    ed.add_block('floor_coordinates', filename = '%s.flr')
    #
    # The following are required to generate "%s.param"
    ed.add_block('run_control')
    ed.add_block('bunched_beam')
    ed.add_block('track')
    #
    ed.write()
    pe.run(ed.ele_filepath, print_stdout=False, print_stderr=True)
    res = ed.load_sdds_output_files()['data']
    ed.delete_temp_files()
    ed.delete_ele_file()

    elem_defs = dict(bends={}, quads={}, sexts={})
    for elem_name, elem_type, elem_def in ed.get_LTE_all_elem_defs():
        elem_name = elem_name.upper()
        elem_type = elem_type.upper()

        if elem_type in ('CSBEND', 'RBEN', 'SBEN', 'RBEND', 'SBEND'):
            props = {}
            for k in ['L', 'ANGLE', 'E1', 'E2', 'K1']:
                temp = ed.get_LTE_elem_prop(elem_name, k)
                if temp is not None:
                    props[k] = temp
                else:
                    props[k] = 0.0
            elem_defs['bends'][elem_name] = props

        elif elem_type in ('QUAD', 'KQUAD'):
            props = {}
            for k in ['L', 'K1']:
                temp = ed.get_LTE_elem_prop(elem_name, k)
                if temp is not None:
                    props[k] = temp
                else:
                    props[k] = 0.0
            elem_defs['quads'][elem_name] = props

        elif elem_type in ('SEXT', 'KSEXT'):
            props = {}
            for k in ['L', 'K2']:
                temp = ed.get_LTE_elem_prop(elem_name, k)
                if temp is not None:
                    props[k] = temp
                else:
                    props[k] = 0.0
            elem_defs['sexts'][elem_name] = props
    sel_data['elem_defs'] = elem_defs

    titles = dict(s='s [m]', ElementName='Element Name', ElementType='Element Type')
    s_decimal = 3
    widths = {}
    widths['s'] = int(np.ceil(np.max(np.log10(
        res['flr']['columns']['s'][res['flr']['columns']['s'] > 0.0])))
                      ) + 1 + s_decimal + 2 # Last "2" for a period and potential negative sign
    widths['ElementName'] = np.max([len(name) for name in res['flr']['columns']['ElementName']])
    widths['ElementType'] = np.max([len(name) for name in res['flr']['columns']['ElementType']])
    for k, v in widths.items():
        widths[k] = max([v, len(titles[k])])
    header_template = (f'{{:>{widths["s"]}}} : '
                       f'{{:{widths["ElementName"]}}} : '
                       f'{{:{widths["ElementType"]}}}')
    value_template = (f'{{:>{widths["s"]}.{s_decimal:d}f}} : '
                      f'{{:{widths["ElementName"]}}} : '
                      f'{{:{widths["ElementType"]}}}')
    header = header_template.format(
        titles['s'], titles['ElementName'], titles['ElementType'])
    flat_elem_s_name_type_list = [header, '-' * len(header)]
    for s, elem_name, elem_type in zip(
        res['flr']['columns']['s'], res['flr']['columns']['ElementName'],
        res['flr']['columns']['ElementType']):
        flat_elem_s_name_type_list.append(value_template.format(s, elem_name, elem_type))
    #
    sel_data['flat_elem_s_name_type_list'] = flat_elem_s_name_type_list

    if length_elem_names is not None:
        _s = res['flr']['columns']['s']
        for len_name, elem_name_list in length_elem_names.items():
            if len_name in sel_data:
                raise ValueError(f'Key "{len_name}" cannot be used for "length_elem_names"')
            first_inds = np.where(res['flr']['columns']['ElementName']
                                  == elem_name_list[0])[0]
            for fi in first_inds:
                sel_data[len_name] = _s[fi] - _s[fi-1]
                for offset, elem_name in enumerate(elem_name_list[1:]):
                    if res['flr']['columns']['ElementName'][fi + offset + 1] == elem_name:
                        sel_data[len_name] += _s[fi + offset + 1] - _s[fi + offset]
                    else:
                        break
                else:
                    break
            else:
                raise ValueError(
                    'Invalid "length_elem_names" for key "{}": {}'.format(
                        len_name, ', '.join(elem_name_list)))


    nsls2_flr_filepath = '/GPFS/APC/yhidaka/git_repos/nsls2cb/nsls2.flr'
    flr_data = pe.sdds.sdds2dicts(nsls2_flr_filepath)[0]
    N2_X_all = flr_data['columns']['X']
    N2_Z_all = flr_data['columns']['Z']
    N2_ElemNames = flr_data['columns']['ElementName']
    N2_SS_index, N2_LS_index = np.where(N2_ElemNames == 'MID')[0][:2]
    N2_X = dict(SS=N2_X_all[N2_SS_index], LS=N2_X_all[N2_LS_index])
    N2_Z = dict(SS=N2_Z_all[N2_SS_index], LS=N2_Z_all[N2_LS_index])

    N2U_X, N2U_Z = {}, {}
    for straight_name, (marker_name, occur) in floor_elem_names_indexes.items():
        ind = np.where(
            res['flr']['columns']['ElementName'] == marker_name)[0][occur]

        N2U_X[straight_name] = res['flr']['columns']['X'][ind]
        N2U_Z[straight_name] = res['flr']['columns']['Z'][ind]

        sel_data[f'dx@{straight_name}'] = N2U_X[straight_name] - N2_X[straight_name]
        sel_data[f'dz@{straight_name}'] = N2U_Z[straight_name] - N2_Z[straight_name]
    #print(f'LS Source Diff. (dx, dz) [mm] = ({(N2U_X["LS"] - N2_X["LS"]) * 1e3:.3g}, {(N2U_Z["LS"] - N2_Z["LS"]) * 1e3:.3g})')
    #print(f'SS Source Diff. (dx, dz) [mm] = ({(N2U_X["SS"] - N2_X["SS"]) * 1e3:.3g}, {(N2U_Z["SS"] - N2_Z["SS"]) * 1e3:.3g})')

    return dict(versions=pe.__version__, fignums=fignums, sel_data=sel_data)

def summarize_nonlin(
    LTE_filepath, E_MeV, N_KICKS=None, use_beamline='RING',
    plot_fm_xy=True, plot_fm_px=True, plot_cmap_xy=True, plot_cmap_px=True,
    plot_tswa=True, plot_nonlin_chrom=True, chrom_deltalim=None,
    ntasks=100, title=None):

    existing_fignums = plt.get_fignums()
    fignums = {}

    if N_KICKS is None:
        N_KICKS = dict(KQUAD=40, KSEXT=40, CSBEND=40)

    if title is None:
        LTE_mod = LTE_filepath.replace('_', '\_')
        title = f'$\mathrm{{{LTE_mod}}}$'

    x_offset = y_offset = 1e-6
    xmin = -8e-3 + x_offset
    xmax = +8e-3 + x_offset
    ymin = 0.0 + y_offset
    ymax = 2e-3 + y_offset
    delta_offset = 0.0

    delta_min = -0.05
    delta_max = +0.05

    fm_xy_filepath = LTE_filepath.replace('.lte', '_fma_x201_y201_n1024_Q0_FG0.pgz')
    fm_px_filepath = LTE_filepath.replace('.lte', '_fma_p201_x201_n1024_Q0_FG0.pgz')

    if plot_fm_xy:

        existing_fignums = plt.get_fignums()

        if not os.path.exists(fm_xy_filepath):
            nx = ny = 201
            n_turns = 1024
            quadratic_spacing = False
            full_grid_output = False

            username = getpass.getuser()
            remote_opts = dict(
                use_sbatch=True, exit_right_after_sbatch=False, pelegant=True,
                job_name='fma_xy', partition='short', ntasks=ntasks, #nodelist=['apcpu-004'],
                #time='7:00',
                #mail_type_end=True, mail_user=f'{username}@bnl.gov',
            )

            output_filepath = fm_xy_filepath

            nonlin.calc_fma_xy(
                output_filepath, LTE_filepath, E_MeV, xmin, xmax, ymin, ymax, nx, ny,
                use_beamline=use_beamline, N_KICKS=N_KICKS,
                n_turns=n_turns, delta_offset=delta_offset,
                quadratic_spacing=quadratic_spacing, full_grid_output=full_grid_output,
                del_tmp_files=True, run_local=False, remote_opts=remote_opts)

        nonlin.plot_fma_xy(
            fm_xy_filepath, title=title, #xlim=[-10e-3, +10e-3],
            is_diffusion=True, scatter=False)

        fignums['fma_xy'] = [fignum for fignum in plt.get_fignums()
                             if fignum not in existing_fignums]

    if plot_fm_px:

        existing_fignums = plt.get_fignums()

        if not os.path.exists(fm_px_filepath):
            nx = ndelta = 201
            n_turns = 1024
            quadratic_spacing = False
            full_grid_output = False

            username = getpass.getuser()
            remote_opts = dict(
                use_sbatch=True, exit_right_after_sbatch=False, pelegant=True,
                job_name='fma_px', partition='short', ntasks=ntasks, #nodelist=['apcpu-004'],
                #time='7:00',
                #mail_type_end=True, mail_user=f'{username}@bnl.gov',
            )

            output_filepath = fm_px_filepath

            nonlin.calc_fma_px(
                output_filepath, LTE_filepath, E_MeV, delta_min, delta_max, xmin, xmax, ndelta, nx,
                use_beamline=use_beamline, N_KICKS=N_KICKS,
                n_turns=n_turns, y_offset=y_offset,
                quadratic_spacing=quadratic_spacing, full_grid_output=full_grid_output,
                del_tmp_files=True, run_local=False, remote_opts=remote_opts)

        nonlin.plot_fma_px(
            fm_px_filepath, title=title, #xlim=[-10e-3, +10e-3],
            is_diffusion=True, scatter=False)

        fignums['fma_px'] = [fignum for fignum in plt.get_fignums()
                             if fignum not in existing_fignums]

    cmap_xy_filepath = LTE_filepath.replace('.lte', '_cmap_x201_y201_n128_fb1.pgz')
    cmap_px_filepath = LTE_filepath.replace('.lte', '_cmap_p201_x201_n128_fb1.pgz')

    if plot_cmap_xy:

        existing_fignums = plt.get_fignums()

        if not os.path.exists(cmap_xy_filepath):
            nx = ny = 201
            n_turns = 128
            forward_backward = 1

            username = getpass.getuser()
            remote_opts = dict(
                use_sbatch=True, exit_right_after_sbatch=False, pelegant=True,
                job_name='cmap_xy', partition='short', ntasks=ntasks, #nodelist=['apcpu-004'],
                #time='7:00',
                #mail_type_end=True, mail_user=f'{username}@bnl.gov',
            )

            output_filepath = cmap_xy_filepath

            nonlin.calc_cmap_xy(
                output_filepath, LTE_filepath, E_MeV, xmin, xmax, ymin, ymax, nx, ny,
                use_beamline=use_beamline, N_KICKS=N_KICKS,
                n_turns=n_turns, delta_offset=delta_offset,
                forward_backward=forward_backward,
                del_tmp_files=True, run_local=False, remote_opts=remote_opts)

        nonlin.plot_cmap_xy(
            cmap_xy_filepath, title=title, cmin=-24, cmax=-10, #xlim=[-10e-3, +10e-3],
            is_log10=True, scatter=False)

        fignums['cmap_xy'] = [fignum for fignum in plt.get_fignums()
                              if fignum not in existing_fignums]

    if plot_cmap_px:

        existing_fignums = plt.get_fignums()

        if not os.path.exists(cmap_px_filepath):
            nx = ndelta = 201
            n_turns = 128
            forward_backward = 1

            username = getpass.getuser()
            remote_opts = dict(
                use_sbatch=True, exit_right_after_sbatch=False, pelegant=True,
                job_name='cmap_px', partition='short', ntasks=ntasks, #nodelist=['apcpu-004'],
                #time='7:00',
                #mail_type_end=True, mail_user=f'{username}@bnl.gov',
            )

            output_filepath = cmap_px_filepath

            nonlin.calc_cmap_px(
                output_filepath, LTE_filepath, E_MeV, delta_min, delta_max, xmin, xmax, ndelta, nx,
                use_beamline=use_beamline, N_KICKS=N_KICKS,
                n_turns=n_turns, y_offset=y_offset,
                forward_backward=forward_backward,
                del_tmp_files=True, run_local=False, remote_opts=remote_opts)

        nonlin.plot_cmap_px(
            cmap_px_filepath, title=title, cmin=-24, cmax=-10, #xlim=[-10e-3, +10e-3],
            is_log10=True, scatter=False)

        fignums['cmap_px'] = [fignum for fignum in plt.get_fignums()
                              if fignum not in existing_fignums]

    if plot_tswa:

        existing_fignums = plt.get_fignums()

        abs_xmax = 1e-3
        abs_ymax = 0.5e-3
        nx = ny = 50
        n_turns = 1024
        remote_opts = dict(job_name='tswa', partition='short', ntasks=min([ntasks, nx*ny]))

        tswa_fp = dict(x={}, y={})
        tswa_fp['x']['+'] = LTE_filepath.replace('.lte', f'_tswa_n{n_turns:d}_xplus.pgz')
        tswa_fp['x']['-'] = LTE_filepath.replace('.lte', f'_tswa_n{n_turns:d}_xminus.pgz')
        tswa_fp['y']['+'] = LTE_filepath.replace('.lte', f'_tswa_n{n_turns:d}_yplus.pgz')
        tswa_fp['y']['-'] = LTE_filepath.replace('.lte', f'_tswa_n{n_turns:d}_yminus.pgz')

        for plane in ['x', 'y']:
            for sign in ['+', '-']:
                output_filepath = tswa_fp[plane][sign]
                if not os.path.exists(output_filepath):
                    if plane == 'x':
                        nonlin.calc_tswa_x(output_filepath, LTE_filepath, E_MeV, abs_xmax, nx, sign,
                            n_turns=n_turns, N_KICKS=N_KICKS, run_local=False, remote_opts=remote_opts)
                    else:
                        nonlin.calc_tswa_y(output_filepath, LTE_filepath, E_MeV, abs_ymax, ny, sign,
                            n_turns=n_turns, N_KICKS=N_KICKS, run_local=False, remote_opts=remote_opts)

                if plane == 'x':
                    nonlin.plot_tswa(output_filepath, title=title, fit_abs_xmax=0.5e-3)
                else:
                    nonlin.plot_tswa(output_filepath, title=title, fit_abs_ymax=0.25e-3)

        fignums['tswa'] = [fignum for fignum in plt.get_fignums()
                           if fignum not in existing_fignums]

    if plot_nonlin_chrom:

        existing_fignums = plt.get_fignums()

        ndelta = 50
        n_turns = 1024
        remote_opts = dict(job_name='nonlin_chrom', partition='short', ntasks=min([ntasks, ndelta]))

        output_filepath = LTE_filepath.replace('.lte', f'_nonlin_chrom_n{n_turns:d}.pgz')

        if not os.path.exists(output_filepath):
            nonlin.calc_chrom_track(
                output_filepath, LTE_filepath, E_MeV, delta_min, delta_max, ndelta,
                n_turns=n_turns, N_KICKS=N_KICKS, run_local=False, remote_opts=remote_opts)

        nonlin.plot_chrom(
            output_filepath, max_chrom_order=3, title=title, deltalim=chrom_deltalim,)
            #nuxlim=[0, 0.3], nuylim=[0.1, 0.3])

        fignums['nonlin_chrom'] = [fignum for fignum in plt.get_fignums()
                                   if fignum not in existing_fignums]

    return fignums

def deepcopy_dict(d):
    """"""

    return pickle.loads(pickle.dumps(d))

def get_nonlin_data_filepaths(LTE_filepath, nonlin_config):
    """"""

    assert LTE_filepath.endswith('.lte')

    output_filetype = 'pgz'
    #output_filetype = 'hdf5'

    ncf = nonlin_config

    suffix_list = []
    data_file_key_list = []
    for calc_type in [
        'fmap_xy', 'fmap_px', 'cmap_xy', 'cmap_px', 'tswa', 'nonlin_chrom']:

        if not ncf['include'].get(calc_type, False):
            continue

        calc_opts = ncf[f'{calc_type}_calc_opts']
        grid_name = calc_opts['grid_name']
        n_turns = calc_opts['n_turns']

        if calc_type.startswith(('fmap', 'cmap')):
            suffix_list.append(f'_{calc_type[:4]}_{grid_name}_n{n_turns}.{output_filetype}')
            data_file_key_list.append(calc_type)
        elif calc_type == 'tswa':
            for plane in ['x', 'y']:
                for sign in ['plus', 'minus']:
                    suffix_list.append(f'_tswa_{grid_name}_n{n_turns}_{plane}{sign}.{output_filetype}')
                    data_file_key_list.append(f'tswa_{plane}{sign}')
        elif calc_type == 'nonlin_chrom':
            suffix_list.append(f'_nonlin_chrom_{grid_name}_n{n_turns}.{output_filetype}')
            data_file_key_list.append(calc_type)
        else:
            raise ValueError

    assert len(suffix_list) == len(data_file_key_list)
    nonlin_data_filepaths = {}
    for k, suffix in zip(data_file_key_list, suffix_list):
        nonlin_data_filepaths[k] = LTE_filepath.replace('.lte', suffix)

    return nonlin_data_filepaths


def calc_fmap_xy(
    LTE_filepath, E_MeV, nonlin_config, use_beamline, N_KICKS,
    nonlin_data_filepaths, common_remote_opts):
    """"""

    map_type = 'f'

    _calc_map_xy(
        map_type, LTE_filepath, E_MeV, nonlin_config, use_beamline, N_KICKS,
        nonlin_data_filepaths, common_remote_opts)

def calc_cmap_xy(
    LTE_filepath, E_MeV, nonlin_config, use_beamline, N_KICKS,
    nonlin_data_filepaths, common_remote_opts):
    """"""

    map_type = 'c'

    _calc_map_xy(
        map_type, LTE_filepath, E_MeV, nonlin_config, use_beamline, N_KICKS,
        nonlin_data_filepaths, common_remote_opts)

def _calc_map_xy(
    map_type, LTE_filepath, E_MeV, nonlin_config, use_beamline, N_KICKS,
    nonlin_data_filepaths, common_remote_opts):
    """"""

    if map_type not in ('c', 'f'):
        raise ValueError(f'Invalid "map_type": {map_type}')

    ncf = nonlin_config

    calc_type = f'{map_type}map_xy'

    output_filepath = nonlin_data_filepaths[calc_type]

    calc_opts = ncf[f'{calc_type}_calc_opts']

    n_turns = calc_opts['n_turns']

    g = ncf['xy_grids'][calc_opts['grid_name']]
    nx, ny = g['nx'], g['ny']
    x_offset = g.get('x_offset', 1e-6)
    y_offset = g.get('y_offset', 1e-6)
    delta_offset = g.get('delta_offset', 0.0)
    xmin = g['xmin'] + x_offset
    xmax = g['xmax'] + x_offset
    ymin = g['ymin'] + y_offset
    ymax = g['ymax'] + y_offset

    remote_opts = dict(
        use_sbatch=True, exit_right_after_sbatch=False, pelegant=True,
        job_name=calc_type)
    remote_opts.update(deepcopy_dict(common_remote_opts))
    remote_opts.update(deepcopy_dict(calc_opts.get('remote_opts', {})))

    if calc_type == 'fmap_xy':
        kwargs = dict(quadratic_spacing=False, full_grid_output=False)
        func = pe.nonlin.calc_fma_xy
    elif calc_type == 'cmap_xy':
        kwargs = dict(forward_backward=True)
        func = pe.nonlin.calc_cmap_xy
    else:
        raise ValueError

    func(output_filepath, LTE_filepath, E_MeV, xmin, xmax, ymin, ymax, nx, ny,
         use_beamline=use_beamline, N_KICKS=N_KICKS,
         n_turns=n_turns, delta_offset=delta_offset,
         del_tmp_files=True, run_local=False, remote_opts=remote_opts, **kwargs)

def calc_fmap_px(LTE_filepath, E_MeV, nonlin_config, use_beamline, N_KICKS,
                 nonlin_data_filepaths, common_remote_opts):
    """"""

    map_type = 'f'

    _calc_map_px(
        map_type, LTE_filepath, E_MeV, nonlin_config, use_beamline, N_KICKS,
        nonlin_data_filepaths, common_remote_opts)

def calc_cmap_px(LTE_filepath, E_MeV, nonlin_config, use_beamline, N_KICKS,
                 nonlin_data_filepaths, common_remote_opts):
    """"""

    map_type = 'c'

    _calc_map_px(
        map_type, LTE_filepath, E_MeV, nonlin_config, use_beamline, N_KICKS,
        nonlin_data_filepaths, common_remote_opts)

def _calc_map_px(
    map_type, LTE_filepath, E_MeV, nonlin_config, use_beamline, N_KICKS,
    nonlin_data_filepaths, common_remote_opts):
    """"""

    if map_type not in ('c', 'f'):
        raise ValueError(f'Invalid "map_type": {map_type}')

    ncf = nonlin_config

    calc_type = f'{map_type}map_px'

    output_filepath = nonlin_data_filepaths[calc_type]

    calc_opts = ncf[f'{calc_type}_calc_opts']

    n_turns = calc_opts['n_turns']

    g = ncf['px_grids'][calc_opts['grid_name']]
    ndelta, nx = g['ndelta'], g['nx']
    x_offset = g.get('x_offset', 1e-6)
    y_offset = g.get('y_offset', 1e-6)
    delta_offset = g.get('delta_offset', 0.0)
    delta_min = g['delta_min'] + delta_offset
    delta_max = g['delta_max'] + delta_offset
    xmin = g['xmin'] + x_offset
    xmax = g['xmax'] + x_offset

    remote_opts = dict(
        use_sbatch=True, exit_right_after_sbatch=False, pelegant=True,
        job_name=calc_type)
    remote_opts.update(deepcopy_dict(common_remote_opts))
    remote_opts.update(deepcopy_dict(calc_opts.get('remote_opts', {})))

    if calc_type == 'fmap_px':
        kwargs = dict(quadratic_spacing=False, full_grid_output=False)
        func = pe.nonlin.calc_fma_px
    elif calc_type == 'cmap_px':
        kwargs = dict(forward_backward=True)
        func = pe.nonlin.calc_cmap_px
    else:
        raise ValueError

    func(output_filepath, LTE_filepath, E_MeV, delta_min, delta_max,
         xmin, xmax, ndelta, nx, use_beamline=use_beamline, N_KICKS=N_KICKS,
         n_turns=n_turns, y_offset=y_offset,
         del_tmp_files=True, run_local=False, remote_opts=remote_opts, **kwargs)

def calc_tswa(
    LTE_filepath, E_MeV, nonlin_config, use_beamline, N_KICKS,
    nonlin_data_filepaths, common_remote_opts):
    """"""

    ncf = nonlin_config

    calc_type = 'tswa'

    calc_opts = ncf[f'{calc_type}_calc_opts']

    n_turns = calc_opts['n_turns']

    g = ncf['tswa_grids'][calc_opts['grid_name']]
    nx, ny = g['nx'], g['ny']
    x_offset = g.get('x_offset', 1e-6)
    y_offset = g.get('y_offset', 1e-6)
    abs_xmax = g['abs_xmax']
    abs_ymax = g['abs_ymax']

    remote_opts = dict(job_name=calc_type)
    remote_opts.update(deepcopy_dict(common_remote_opts))
    remote_opts.update(deepcopy_dict(calc_opts.get('remote_opts', {})))

    for plane in ['x', 'y']:

        if plane == 'x':
            func = pe.nonlin.calc_tswa_x
            kwargs = dict(y0_offset=y_offset)
            abs_max = abs_xmax
            n = nx
        else:
            func = pe.nonlin.calc_tswa_y
            kwargs = dict(x0_offset=x_offset)
            abs_max = abs_ymax
            n = ny

        plane_specific_remote_opts = deepcopy_dict(remote_opts)
        plane_specific_remote_opts['ntasks'] = min([remote_opts['ntasks'], n])

        for sign, sign_symbol in [('plus', '+'), ('minus', '-')]:

            output_filepath = nonlin_data_filepaths[f'{calc_type}_{plane}{sign}']

            mod_kwargs = deepcopy_dict(kwargs)
            mod_kwargs[f'{plane}sign'] = sign_symbol

            func(output_filepath, LTE_filepath, E_MeV, abs_max, n,
                 courant_snyder=True, return_fft_spec=True,
                 n_turns=n_turns, N_KICKS=N_KICKS,
                 del_tmp_files=True, run_local=False,
                 remote_opts=plane_specific_remote_opts, **mod_kwargs)

def calc_nonlin_chrom(
    LTE_filepath, E_MeV, nonlin_config, use_beamline, N_KICKS,
    nonlin_data_filepaths, common_remote_opts):
    """"""

    ncf = nonlin_config

    calc_type = 'nonlin_chrom'

    output_filepath = nonlin_data_filepaths[calc_type]

    calc_opts = ncf[f'{calc_type}_calc_opts']

    n_turns = calc_opts['n_turns']

    g = ncf['nonlin_chrom_grids'][calc_opts['grid_name']]
    ndelta = g['ndelta']
    x_offset = g.get('x_offset', 1e-6)
    y_offset = g.get('y_offset', 1e-6)
    delta_offset = g.get('delta_offset', 0.0)
    delta_min = g['delta_min'] + delta_offset
    delta_max = g['delta_max'] + delta_offset

    remote_opts = dict(job_name=calc_type)
    remote_opts.update(deepcopy_dict(common_remote_opts))
    remote_opts.update(deepcopy_dict(calc_opts.get('remote_opts', {})))
    #
    remote_opts['ntasks'] = min([remote_opts['ntasks'], ndelta])

    pe.nonlin.calc_chrom_track(
        output_filepath, LTE_filepath, E_MeV, delta_min, delta_max, ndelta,
        courant_snyder=True, return_fft_spec=True,
        use_beamline=use_beamline, N_KICKS=N_KICKS,
        n_turns=n_turns, x0_offset=x_offset, y0_offset=y_offset,
        del_tmp_files=True, run_local=False, remote_opts=remote_opts)

def calc_nonlin_props(LTE_filepath, E_MeV, nonlin_config):
    """"""
    ncf = nonlin_config

    nonlin_data_filepaths = get_nonlin_data_filepaths(LTE_filepath, ncf)
    use_beamline = ncf['use_beamline']
    N_KICKS = ncf.get('N_KICKS', dict(KQUAD=40, KSEXT=40, CSBEND=40))

    common_remote_opts = ncf['common_remote_opts']

    calc_type = 'fmap_xy'
    if (calc_type in nonlin_data_filepaths) and \
       (ncf['recalc'][calc_type] or
        (not os.path.exists(nonlin_data_filepaths[calc_type]))):

        calc_fmap_xy(LTE_filepath, E_MeV, ncf, use_beamline, N_KICKS,
                     nonlin_data_filepaths, common_remote_opts)

    calc_type = 'fmap_px'
    if (calc_type in nonlin_data_filepaths) and \
       (ncf['recalc'][calc_type] or
        (not os.path.exists(nonlin_data_filepaths[calc_type]))):

        calc_fmap_px(LTE_filepath, E_MeV, ncf, use_beamline, N_KICKS,
                     nonlin_data_filepaths, common_remote_opts)

    calc_type = 'cmap_xy'
    if (calc_type in nonlin_data_filepaths) and \
       (ncf['recalc'][calc_type] or
        (not os.path.exists(nonlin_data_filepaths[calc_type]))):

        calc_cmap_xy(LTE_filepath, E_MeV, ncf, use_beamline, N_KICKS,
                     nonlin_data_filepaths, common_remote_opts)

    calc_type = 'cmap_px'
    if (calc_type in nonlin_data_filepaths) and \
       (ncf['recalc'][calc_type] or
        (not os.path.exists(nonlin_data_filepaths[calc_type]))):

        calc_cmap_px(LTE_filepath, E_MeV, ncf, use_beamline, N_KICKS,
                     nonlin_data_filepaths, common_remote_opts)

    if ('tswa_xplus' in nonlin_data_filepaths) and \
       (ncf['recalc']['tswa'] or
        (not os.path.exists(nonlin_data_filepaths['tswa_xplus'])) or
        (not os.path.exists(nonlin_data_filepaths['tswa_xminus'])) or
        (not os.path.exists(nonlin_data_filepaths['tswa_yplus'])) or
        (not os.path.exists(nonlin_data_filepaths['tswa_yminus']))
        ):

        calc_tswa(LTE_filepath, E_MeV, ncf, use_beamline, N_KICKS,
                  nonlin_data_filepaths, common_remote_opts)

    calc_type = 'nonlin_chrom'
    if (calc_type in nonlin_data_filepaths) and \
       (ncf['recalc'][calc_type] or
        (not os.path.exists(nonlin_data_filepaths[calc_type]))):

        calc_nonlin_chrom(LTE_filepath, E_MeV, nonlin_config, use_beamline, N_KICKS,
                          nonlin_data_filepaths, common_remote_opts)

    return nonlin_data_filepaths

def plot_nonlin_props(LTE_filepath, nonlin_config, pdf_file_prefix):
    """"""

    ncf = nonlin_config

    nonlin_data_filepaths = get_nonlin_data_filepaths(LTE_filepath, ncf)

    existing_fignums = plt.get_fignums()

    calc_type = 'fmap_xy'
    if calc_type in nonlin_data_filepaths:
        pe.nonlin.plot_fma_xy(
            nonlin_data_filepaths[calc_type], title='',
            is_diffusion=True, scatter=False)

        pp = PdfPages(f'{pdf_file_prefix}.{calc_type}.pdf')
        for fignum in [fignum for fignum in plt.get_fignums()
                       if fignum not in existing_fignums]:
            pp.savefig(figure=fignum)
            plt.close(fignum)
        pp.close()


    calc_type = 'fmap_px'
    if calc_type in nonlin_data_filepaths:
        pe.nonlin.plot_fma_px(
            nonlin_data_filepaths[calc_type], title='',
            is_diffusion=True, scatter=False)

        pp = PdfPages(f'{pdf_file_prefix}.{calc_type}.pdf')
        for fignum in [fignum for fignum in plt.get_fignums()
                       if fignum not in existing_fignums]:
            pp.savefig(figure=fignum)
            plt.close(fignum)
        pp.close()

    calc_type = 'cmap_xy'
    if calc_type in nonlin_data_filepaths:
        _plot_kwargs = ncf.get(f'{calc_type}_plot_opts', {})
        pe.nonlin.plot_cmap_xy(
            nonlin_data_filepaths[calc_type], title='', is_log10=True,
            scatter=False, **_plot_kwargs)

        pp = PdfPages(f'{pdf_file_prefix}.{calc_type}.pdf')
        for fignum in [fignum for fignum in plt.get_fignums()
                       if fignum not in existing_fignums]:
            pp.savefig(figure=fignum)
            plt.close(fignum)
        pp.close()

    calc_type = 'cmap_px'
    if calc_type in nonlin_data_filepaths:
        _plot_kwargs = ncf.get(f'{calc_type}_plot_opts', {})
        pe.nonlin.plot_cmap_px(
            nonlin_data_filepaths[calc_type], title='', #xlim=[-10e-3, +10e-3],
            is_log10=True, scatter=False, **_plot_kwargs)

        pp = PdfPages(f'{pdf_file_prefix}.{calc_type}.pdf')
        for fignum in [fignum for fignum in plt.get_fignums()
                       if fignum not in existing_fignums]:
            pp.savefig(figure=fignum)
            plt.close(fignum)
        pp.close()

    calc_type = 'tswa'
    if f'tswa_xplus' in nonlin_data_filepaths:
        if False:
            #plot_plus_minus_combined = False
            plot_plus_minus_combined = True

            plot_xy0 = True
            #plot_xy0 = False

            #plot_Axy = True
            plot_Axy = False

            use_time_domain_amplitude = True # Only relevant when "plot_Axy=True"
            #use_time_domain_amplitude = False # Only relevant when "plot_Axy=True"

            #plot_fft = True
            plot_fft = False

            _plot_kwargs = dict(
                plot_plus_minus_combined=plot_plus_minus_combined,
                footprint_nuxlim=[0.0, 1.0], footprint_nuylim=[0.0, 1.0],
                plot_xy0=plot_xy0, plot_Axy=plot_Axy, plot_fft=plot_fft,
                use_time_domain_amplitude=use_time_domain_amplitude)

            _plot_kwargs['fit_xmin'] = -0.5e-3
            _plot_kwargs['fit_xmax'] = +0.5e-3
            _plot_kwargs['fit_ymin'] = -0.25e-3
            _plot_kwargs['fit_ymax'] = +0.25e-3

        else:
            _plot_kwargs = ncf.get(f'{calc_type}_plot_opts', {})

        plot_plus_minus_combined = _plot_kwargs.get(
            'plot_plus_minus_combined', True)
        del _plot_kwargs['plot_plus_minus_combined']

        if plot_plus_minus_combined:
            sel_tswa_caption_keys = [
                'nu_vs_x0', 'tunefootprint_vs_x0',
                'nu_vs_y0', 'tunefootprint_vs_y0']
        else:
            sel_tswa_caption_keys = [
                'nu_vs_x0plus', 'nu_vs_x0minus',
                'tunefootprint_vs_x0plus', 'tunefootprint_vs_x0minus',
                'nu_vs_y0plus', 'nu_vs_y0minus',
                'tunefootprint_vs_y0plus', 'tunefootprint_vs_y0minus',
            ]

        tswa_captions = []
        tswa_caption_keys = []
        if not plot_plus_minus_combined:

            fit_abs_xmax = dict(plus=_plot_kwargs['fit_xmax'],
                                minus=np.abs(_plot_kwargs['fit_xmin']))
            fit_abs_ymax = dict(plus=_plot_kwargs['fit_ymax'],
                                minus=np.abs(_plot_kwargs['fit_ymin']))

            for plane in ['x', 'y']:
                for sign in ['plus', 'minus']:
                    data_key = f'tswa_{plane}{sign}'
                    if plane == 'x':
                        pe.nonlin.plot_tswa(
                            nonlin_data_filepaths[data_key],
                            title='', fit_abs_xmax=fit_abs_xmax[sign], #0.5e-3,
                            **_plot_kwargs)
                    else:
                        pe.nonlin.plot_tswa(
                            nonlin_data_filepaths[data_key],
                            title='', fit_abs_ymax=fit_abs_ymax[sign], #0.25e-3,
                            **_plot_kwargs)

                    if sign == 'plus':
                        if plot_xy0:
                            tswa_captions.append(
                                MathText(r'\nu') + ' vs. ' + MathText(fr'{plane}_0'))
                            tswa_caption_keys.append(f'nu_vs_{plane}0{sign}')
                        if plot_Axy:
                            tswa_captions.append(
                                MathText(r'\nu') + ' vs. ' + MathText(fr'A_{plane} (> 0)'))
                            tswa_caption_keys.append(f'nu_vs_A{plane}{sign}')
                        tswa_captions.append(
                            'Tune footprint vs. ' + MathText(fr'{plane}_0'))
                        tswa_caption_keys.append(f'tunefootprint_vs_{plane}0{sign}')
                        if plot_fft:
                            tswa_captions.append(
                                'FFT ' + MathText(r'\nu_x') + ' vs. ' + MathText(fr'{plane}_0'))
                            tswa_caption_keys.append(f'fft_nux_vs_{plane}0{sign}')
                            tswa_captions.append(
                                'FFT ' + MathText(r'\nu_y') + ' vs. ' + MathText(fr'{plane}_0'))
                            tswa_caption_keys.append(f'fft_nuy_vs_{plane}0{sign}')
                    else:
                        if plot_xy0:
                            tswa_captions.append(
                                MathText(r'\nu') + ' vs. ' + MathText(fr'-{plane}_0'))
                            tswa_caption_keys.append(f'nu_vs_{plane}0{sign}')
                        if plot_Axy:
                            tswa_captions.append(
                                MathText(r'\nu') + ' vs. ' + MathText(fr'A_{plane} (< 0)'))
                            tswa_caption_keys.append(f'nu_vs_A{plane}{sign}')
                        tswa_captions.append(
                            'Tune footprint vs. ' + MathText(fr'-{plane}_0'))
                        tswa_caption_keys.append(f'tunefootprint_vs_{plane}0{sign}')
                        if plot_fft:
                            tswa_captions.append(
                                'FFT ' + MathText(r'\nu_x') + ' vs. ' + MathText(fr'-{plane}_0'))
                            tswa_caption_keys.append(f'fft_nux_vs_{plane}0{sign}')
                            tswa_captions.append(
                                'FFT ' + MathText(r'\nu_y') + ' vs. ' + MathText(fr'-{plane}_0'))
                            tswa_caption_keys.append(f'fft_nuy_vs_{plane}0{sign}')

        else:
            for plane in ['x', 'y']:
                pe.nonlin.plot_tswa_both_sides(
                    nonlin_data_filepaths[f'tswa_{plane}plus'],
                    nonlin_data_filepaths[f'tswa_{plane}minus'],
                    title='',
                    #fit_xmin=-0.5e-3, fit_xmax=+0.5e-3,
                    #fit_ymin=-0.25e-3, fit_ymax=+0.25e-3,
                    **_plot_kwargs
                )

                if _plot_kwargs['plot_xy0']:
                    tswa_captions.append(
                        MathText(r'\nu') + ' vs. ' + MathText(fr'{plane}_0'))
                    tswa_caption_keys.append(f'nu_vs_{plane}0')
                if _plot_kwargs['plot_Axy']:
                    tswa_captions.append(
                        MathText(r'\nu') + ' vs. ' + MathText(fr'A_{plane}'))
                    tswa_caption_keys.append(f'nu_vs_A{plane}')
                tswa_captions.append(
                    'Tune footprint vs. ' + MathText(fr'{plane}_0'))
                tswa_caption_keys.append(f'tunefootprint_vs_{plane}0')
                if _plot_kwargs['plot_fft']:
                    tswa_captions.append(
                        'FFT ' + MathText(r'\nu_x') + ' vs. ' + MathText(fr'{plane}_0'))
                    tswa_caption_keys.append(f'fft_nux_vs_{plane}0')
                    tswa_captions.append(
                        'FFT ' + MathText(r'\nu_y') + ' vs. ' + MathText(fr'{plane}_0'))
                    tswa_caption_keys.append(f'fft_nuy_vs_{plane}0')

        pp = PdfPages(f'{pdf_file_prefix}.{calc_type}.pdf')
        for fignum in [fignum for fignum in plt.get_fignums()
                       if fignum not in existing_fignums]:
            pp.savefig(figure=fignum)
            plt.close(fignum)
        pp.close()


    calc_type = 'nonlin_chrom'
    if calc_type in nonlin_data_filepaths:
        if False:
            #plot_fft = False
            plot_fft = True

            _plot_kwargs = dict(plot_fft=plot_fft, max_chrom_order=4,
                                fit_deltalim=[-2e-2, +2e-2])
        else:
            _plot_kwargs = ncf.get(f'{calc_type}_plot_opts', {})

        pe.nonlin.plot_chrom(
            nonlin_data_filepaths[calc_type], title='',
            **_plot_kwargs)
            #deltalim=chrom_deltalim,)
            #nuxlim=[0, 0.3], nuylim=[0.1, 0.3])

        pp = PdfPages(f'{pdf_file_prefix}.{calc_type}.pdf')
        for fignum in [fignum for fignum in plt.get_fignums()
                       if fignum not in existing_fignums]:
            pp.savefig(figure=fignum)
            plt.close(fignum)
        pp.close()

    tswa_page_caption_list = []
    for k in sel_tswa_caption_keys:
        i = tswa_caption_keys.index(k)
        page = i + 1
        tswa_page_caption_list.append((page, tswa_captions[i]))

    #return fignums, tswa_page_caption_list
    return tswa_page_caption_list


MT = MathText
NT = NormalText

def create_bend_elements_subsection(doc, elem_defs):
    """"""

    if elem_defs['bends']:
        d = elem_defs['bends']

        with doc.create(plx.Subsection('Bend Elements')):
            ncol = 6
            table_spec = ' '.join(['l'] * ncol)
            with doc.create(plx.LongTable(table_spec)) as table:
                table.add_hline()
                table.add_row([
                    'Name', MathText('L')+' [m]',
                    MathText(r'\theta_{\mathrm{bend}}') + ' [mrad]',
                    MathText(r'\theta_{\mathrm{in}}') + ' [mrad]',
                    MathText(r'\theta_{\mathrm{out}}') + ' [mrad]',
                    MathText('K_1\ [\mathrm{m}^{-2}]')])
                table.add_hline()
                table.end_table_header()
                table.add_hline()
                table.add_row((
                    plx.MultiColumn(ncol, align='r', data='Continued onto Next Page'),))
                table.add_hline()
                table.end_table_footer()
                table.add_hline()
                table.end_table_last_footer()

                for k in sorted(list(d)):
                    L, angle, e1, e2, K1 = (
                        d[k]['L'], d[k]['ANGLE'] * 1e3,
                        d[k]['E1'] * 1e3, d[k]['E2'] * 1e3, d[k]['K1'])
                    table.add_row([
                        k, MathText(f'{L:.3f}'), MathText(f'{angle:+.3f}'),
                        MathText(f'{e1:+.3f}'), MathText(f'{e2:+.3f}'),
                        MathText(f'{K1:+.4g}')])

def create_quad_elements_subsection(doc, elem_defs):
    """"""

    if elem_defs['quads']:
        d = elem_defs['quads']

        with doc.create(plx.Subsection('Quadrupole Elements')):
            ncol = 3
            table_spec = ' '.join(['l'] * ncol)
            with doc.create(plx.LongTable(table_spec)) as table:
                table.add_hline()
                table.add_row([
                    'Name', MathText('L')+' [m]',
                    MathText('K_1\ [\mathrm{m}^{-2}]')])
                table.add_hline()
                table.end_table_header()
                table.add_hline()
                table.add_row((
                    plx.MultiColumn(ncol, align='r', data='Continued onto Next Page'),))
                table.add_hline()
                table.end_table_footer()
                table.add_hline()
                table.end_table_last_footer()

                for k in sorted(list(d)):
                    L, K1 = d[k]['L'], d[k]['K1']
                    table.add_row([
                        k, MathText(f'{L:.3f}'), MathText(f'{K1:+.4g}')])

def create_sext_elements_subsection(doc, elem_defs):
    """"""

    if elem_defs['sexts']:
        d = elem_defs['sexts']

        with doc.create(plx.Subsection('Sextupole Elements')):
            ncol = 3
            table_spec = ' '.join(['l'] * ncol)
            with doc.create(plx.LongTable(table_spec)) as table:
                table.add_hline()
                table.add_row([
                    'Name', MathText('L')+' [m]',
                    MathText('K_2\ [\mathrm{m}^{-3}]')])
                table.add_hline()
                table.end_table_header()
                table.add_hline()
                table.add_row((
                    plx.MultiColumn(ncol, align='r', data='Continued onto Next Page'),))
                table.add_hline()
                table.end_table_footer()
                table.add_hline()
                table.end_table_last_footer()

                for k in sorted(list(d)):
                    L, K2 = d[k]['L'], d[k]['K2']
                    table.add_row([
                        k, MathText(f'{L:.3f}'), MathText(f'{K2:+.4g}')])

def add_cline(table, col_index_range):
    """
    `col_index_range` examples:
        '1-2'
        '4-4'
    """

    table.data.append(plx.Command('cline', plx.NoEscape(col_index_range)))

def create_beamline_elements_list_subsection(doc, flat_elem_s_name_type_list):
    """"""

    flat_elem_s_name_type_list = flat_elem_s_name_type_list[2:] # skipping first 2 header & divider lines
    nLines = len(flat_elem_s_name_type_list)
    line_width = len(flat_elem_s_name_type_list[0])
    max_page_char_width = 80
    nFolds = int(np.floor(max_page_char_width / line_width))
    nLinesInFold = int(np.ceil(len(flat_elem_s_name_type_list) / nFolds))

    folded_list = []
    for iLine in range(nLinesInFold):
        indexes = [i * nLinesInFold + iLine for i in range(nFolds)]
        line = ': # :'.join([
            flat_elem_s_name_type_list[i] if i < nLines
            else ':'.join([' '] * 3) for i in indexes])
        # ^ Add "#" to denote empty column
        folded_list.append(line)

    with doc.create(plx.Subsection('Beamline Elements List')):
        table_spec = ' c '.join(['r l l'] * nFolds)
        ncol = len(table_spec.split())

        hline_col_ind_ranges = []
        base_header = [MathText('s')+' [m]', 'Element Name', 'Element Type']
        header_list = []
        header_list.extend(base_header)
        start_ind = 1
        for i, _spec in enumerate(table_spec.split()):
            if _spec == 'c':
                end_ind = i
                hline_col_ind_ranges.append(f'{start_ind}-{end_ind}')
                start_ind = i + 2

                header_list.append('   ')
                header_list.extend(base_header)
        end_ind = ncol
        hline_col_ind_ranges.append(f'{start_ind}-{end_ind}')

        with doc.create(plx.LongTable(table_spec)) as table:
            for ind_range in hline_col_ind_ranges:
                add_cline(table, ind_range)
            table.add_row(header_list)
            for ind_range in hline_col_ind_ranges:
                add_cline(table, ind_range)
            table.end_table_header()
            for ind_range in hline_col_ind_ranges:
                add_cline(table, ind_range)
            table.add_row((
                plx.MultiColumn(ncol, align='r', data='Continued onto Next Page'),))
            for ind_range in hline_col_ind_ranges:
                add_cline(table, ind_range)
            table.end_table_footer()
            for ind_range in hline_col_ind_ranges:
                add_cline(table, ind_range)
            table.end_table_last_footer()

            for line in folded_list:
                table.add_row([_s.strip().replace('#', '') for _s in line.split(':')])

def create_lattice_prop_section(doc, lin_data, pdf_file_prefix, versions):
    """"""

    with doc.create(plx.Section('Lattice Properties')):
        keys = [k for k in list(lin_data)
                if k not in ('elem_defs', 'flat_elem_s_name_type_list')]
        val_strs = {}
        labels = {}
        units = {}
        for k in keys:
            labels[k] = k
            units[k] = ''
            if k.startswith('ksi_'):
                #val_strs[k] = '{:+.3f}'.format(lin_data[k])
                #_, _plane, _type = k.split('_')
                #labels[k] = MathText(fr'\xi_{_plane}^{{\mathrm{{{_type}}}}}')
                if k == 'ksi_x_nat':
                    val_strs[k] = '({:+.3f}, {:+.3f})'.format(
                        lin_data['ksi_x_nat'], lin_data['ksi_y_nat'])
                    labels[k] = 'Natural Chromaticities ' + MathText(
                        r'(\xi_x^{\mathrm{nat}}, \xi_y^{\mathrm{nat}})')
                elif k == 'ksi_x_cor':
                    val_strs[k] = '({:+.3f}, {:+.3f})'.format(
                        lin_data['ksi_x_cor'], lin_data['ksi_y_cor'])
                    labels[k] = 'Corrected Chromatiticities ' + MathText(
                        r'(\xi_x^{\mathrm{cor}}, \xi_y^{\mathrm{cor}})')
                elif k in ('ksi_y_nat', 'ksi_y_cor'):
                    del labels[k]
                    del units[k]
                else:
                    raise ValueError
                if k in val_strs:
                    val_strs[k] = MathText(val_strs[k])

            elif k.endswith(('_betax', '_betay')) or k.startswith(('betax@', 'betay@')):
                #val_strs[k] = '{:.2f}'.format(lin_data[k])
                #units[k] = ' [m]'
                #if k.endswith(('_betax', '_betay')):
                    #prefix, prop = k.split('_')
                    #plane = prop[-1]
                    #labels[k] = prefix + MathText(fr'\ \beta_{plane}')
                #elif k.startswith(('betax@', 'betay@')):
                    #prop, loc = k.split('@')
                    #plane = prop[-1]
                    #labels[k] = MathText(fr'\beta_{plane}') + ' @ ' + loc
                #else:
                    #raise ValueError
                units[k] = ' [m]'
                if k == 'max_betax':
                    val_strs[k] = '({:.2f}, {:.2f})'.format(
                        lin_data['max_betax'], lin_data['max_betay'])
                    labels[k] = 'max ' + MathText(r'(\beta_x, \beta_y)')
                elif k == 'min_betax':
                    val_strs[k] = '({:.2f}, {:.2f})'.format(
                        lin_data['min_betax'], lin_data['min_betay'])
                    labels[k] = 'min ' + MathText(r'(\beta_x, \beta_y)')
                elif k.startswith('betax@'):
                    val_strs[k] = '({:.2f}, {:.2f})'.format(
                        lin_data[k], lin_data[k.replace('betax@', 'betay@')])
                    prop, loc = k.split('@')
                    labels[k] = MathText(r'(\beta_x, \beta_y)') + ' @ ' + loc
                elif k in ('max_betay', 'min_betay') or k.startswith('betay@'):
                    del labels[k]
                    del units[k]
                else:
                    raise ValueError
                if k in val_strs:
                    val_strs[k] = MathText(val_strs[k])

            elif k.endswith(('_etax',)):
                #val_strs[k] = '{:+.1f}'.format(lin_data[k] * 1e3)
                #units[k] = ' [mm]'
                #prefix, prop = k.split('_')
                #plane = prop[-1]
                #labels[k] = prefix + MathText(fr'\ \eta_{plane}')
                units[k] = ' [mm]'
                if k == 'min_etax':
                    val_strs[k] = '({:+.1f}, {:+.1f})'.format(
                        lin_data['min_etax'] * 1e3, lin_data['max_etax'] * 1e3)
                    labels[k] = MathText(r'\eta_x') + ' (min, max)'
                elif k in ('max_etax',):
                    del labels[k]
                    del units[k]
                else:
                    raise ValueError
                if k in val_strs:
                    val_strs[k] = MathText(val_strs[k])

            elif k in ('Jx', 'Jdelta'):
                #val_strs[k] = '{:.2f}'.format(lin_data[k])
                #if k == 'Jx':
                    #labels[k] = MathText(r'J_x')
                #elif k == 'Jdelta':
                    #labels[k] = MathText(r'J_{\delta}')
                #else:
                    #raise ValueError()
                if k == 'Jx':
                    val_strs[k] = '({:.2f}, {:.2f}, {:.2f})'.format(
                        lin_data['Jx'], 1.0, lin_data['Jdelta'])
                    labels[k] = 'Damping Partitions ' + MathText(r'(J_x, J_y, J_{\delta})')
                elif k == 'Jdelta':
                    del labels[k]
                    del units[k]
                else:
                    raise ValueError()
                if k in val_strs:
                    val_strs[k] = MathText(val_strs[k])

            elif k in ('nux', 'nuy'):
                #val_strs[k] = '{:.3f}'.format(lin_data[k])
                #labels[k] = MathText(fr'\nu_{k[-1]}')
                if k == 'nux':
                    val_strs[k] = '({:.3f}, {:.3f})'.format(
                        lin_data['nux'], lin_data['nuy'])
                    labels[k] = 'Ring Tunes ' + MathText(r'(\nu_x, \nu_y)')
                elif k == 'nuy':
                    del labels[k]
                    del units[k]
                else:
                    raise ValueError
                if k in val_strs:
                    val_strs[k] = MathText(val_strs[k])

            elif k in ('alphac',):
                val_strs[k] = MathText(pe.util.pprint_sci_notation(lin_data[k], '.2e'))
                labels[k] = 'Momentum Compaction ' + MathText(r'\alpha_c')

            elif k in ('dE_E',):
                val_strs[k] = '{:.3f}'.format(lin_data[k] * 1e2)
                units[k] = r' [\%]'
                labels[k] = 'Energy Spread ' + MathText(r'\sigma_{\delta}')
                val_strs[k] = MathText(val_strs[k])

            elif k == 'eps_x':
                val_strs[k] = '{:.1f}'.format(lin_data[k] * 1e12)
                units[k] = ' [pm]'
                labels[k] = 'Natural Horizontal Emittance ' + MathText(r'\epsilon_x')
                val_strs[k] = MathText(val_strs[k])

            elif k == 'U0_MeV':
                val_strs[k] = '{:.0f}'.format(lin_data[k] * 1e3)
                units[k] = ' [keV]'
                labels[k] = 'Energy Loss per Turn ' + MathText(r'U_0')
                val_strs[k] = MathText(val_strs[k])

            elif (k == 'circumf') or k.startswith('L@'):
                val_strs[k] = '{:.3f}'.format(lin_data[k])
                units[k] = ' [m]'
                if k == 'circumf':
                    labels[k] = 'Circumference ' + MathText(r'C')
                else:
                    _, loc = k.split('@')
                    #labels[k] = MathText(fr'L \ @ \ \mathrm{{{loc}}}')
                    labels[k] = f'Length of {loc}'
                val_strs[k] = MathText(val_strs[k])

            elif k == 'circumf_change_%':
                val_strs[k] = '{:+.3f}'.format(lin_data[k])
                units[k] = r' [\%]'
                labels[k] = 'Circumference Change ' + MathText(r'\Delta C / C')
                val_strs[k] = MathText(val_strs[k])

            elif k.startswith(('dx@', 'dz@')):
                #val_strs[k] = '{:+.2f}'.format(lin_data[k] * 1e3)
                #units[k] = ' [mm]'
                #dplane, loc = k.split('@')
                #plane = dplane[-1]
                #labels[k] = MathText(fr'\Delta {plane}\ @ \ \mathrm{{{loc}}}')
                units[k] = ' [mm]'
                if k.startswith('dx@'):
                    val_strs[k] = '({:+.2f}, {:+.2f})'.format(
                        lin_data[k] * 1e3, lin_data[k.replace('dx@', 'dz@')] * 1e3)
                    dplane, loc = k.split('@')
                    labels[k] = f'Source Point Diff. @ {loc} ' + MathText(r'(\Delta x, \Delta z)')
                elif k.startswith('dz@'):
                    del labels[k]
                    del units[k]
                else:
                    raise ValueError
                if k in val_strs:
                    val_strs[k] = MathText(val_strs[k])

            elif k.startswith(('dnux@', 'dnuy@')):
                units[k] = MathText(' [2\pi]')
                if k.startswith('dnux@'):
                    val_strs[k] = '({:.6f}, {:.6f})'.format(
                        lin_data[k], lin_data[k.replace('dnux@', 'dnuy@')])
                    dplane, loc = k.split('@')
                    labels[k] = f'Phase Adv. btw. {loc} ' + MathText(r'(\Delta\nu_x, \Delta\nu_y)')
                elif k.startswith('dnuy@'):
                    del labels[k]
                    del units[k]
                else:
                    raise ValueError
                if k in val_strs:
                    val_strs[k] = MathText(val_strs[k])

            elif k in ('nPeriodsInRing'):
                val_strs[k] = '{:d}'.format(lin_data[k])
                labels[k] = '# of Super-periods'

            elif k == 'E_GeV':
                units[k] = ' [GeV]'
                val_strs[k] = f'{lin_data[k]:.0f}'
                labels[k] = f'Beam Energy'

                val_strs[k] = MathText(val_strs[k])

            else:
                raise RuntimeError(f'Unhandled key: {k}')


        with doc.create(plx.LongTable('l l')) as table:
            table.add_hline()
            table.add_row(['Property', 'Value'])
            table.add_hline()
            table.end_table_header()
            table.add_hline()
            table.add_row((
                plx.MultiColumn(2, align='r', data='Continued onto Next Page'),))
            table.add_hline()
            table.end_table_footer()
            table.add_hline()
            #table.add_row((
                #plx.MultiColumn(2, align='r', data='NOT Continued onto Next Page'),))
            #table.add_hline()
            table.end_table_last_footer()

            #for k in sorted(keys):
            for k in sorted(list(val_strs)):
                table.add_row([labels[k] + units[k], val_strs[k]])

    twiss_fig_pdf_filepath = meta_twiss['pdf_filepath']
    if os.path.exists(twiss_fig_pdf_filepath):
        doc.append('Description of Twiss and other lattice properties goes here.')
        ver_sentence = (
            f'ELEGANT version {versions["ELEGANT"]} was used '
            f'to compute the lattice properties.')
        doc.append(plx.Command('par'))
        doc.append(ver_sentence)
        #doc.append(plx.Command('vspace', plx.NoEscape('-10pt')))
        doc.append(plx.VerticalSpace(plx.NoEscape('-10pt')))
        with doc.create(plx.Figure(position='h!')) as fig:
            doc.append(plx.NoEscape(r'\centering'))

            for iPage, caption in enumerate(meta_twiss['captions']):
                if (np.mod(iPage, 2) == 0) and (iPage != 0):
                    doc.append(plx.LineBreak()) # This will move next 2 plots to next row
                with doc.create(SubFigureForMultiPagePDF(
                    position='b', width=plx.utils.NoEscape(r'0.5\linewidth'))) as subfig:
                    subfig.add_image(
                        twiss_fig_pdf_filepath, page=iPage+1,
                        width=plx.utils.NoEscape(r'\linewidth'))
                    #doc.append(plx.Command('vspace', plx.NoEscape('-10pt')))
                    doc.append(plx.VerticalSpace(plx.NoEscape('-10pt')))
                    subfig.add_caption(caption)
            #doc.append(plx.Command('vspace', plx.NoEscape('-10pt')))
            doc.append(plx.VerticalSpace(plx.NoEscape('-10pt')))
            fig.add_caption('Twiss functions.')

def create_fmap_section():
    """"""

def create_cmap_section():
    """"""

def create_twsa_section():
    """"""

def create_nonlin_chrom_section():
    """"""

def prep_multiline_paragraph_str(paragraph):
    """"""

    return (' '.join([line.strip() for line in paragraph.split('\n')])).strip()

if __name__ == '__main__':

    #use_yaml = False
    use_yaml = True

    if not use_yaml:
        conf = None

        if False:
            parent_folder = '/GPFS/APC/yhidaka/git_repos/nsls2cb/20200123_opt_CLS'

            base_LTE_filepath = os.path.join(
                parent_folder,
                'moga_prod_v1/lattice3_CLSScale_36pm_mod_opt2020-01-23T16-09-20_KickElems.lte')

            runID = 2708 # 5.7e-6 [m^2], 1.84 [hr]
            new_LTE_filepath= f'CLSmod23pm_20200123T160920_moga_prod_v1_{runID:06d}.lte'
            load_parameters = dict(
                filename=os.path.join(parent_folder,
                                      f'moga_prod_v1/moga-{runID:06d}.param'))

            pe.eleutil.save_lattice_after_load_parameters(
                base_LTE_filepath, new_LTE_filepath, load_parameters)

            # Turn off all sextupoles
            input_LTE_filepath = new_LTE_filepath
            zeroSexts_LTE_filepath = input_LTE_filepath.replace('.lte', '_ZeroSexts.lte')
            alter_elements = dict(name='*', type='KSEXT', item='K2', value = 0.0)
            pe.eleutil.save_lattice_after_alter_elements(
                input_LTE_filepath, zeroSexts_LTE_filepath, alter_elements)
        else:
            input_LTE_filepath = 'CLSmod23pm_20200123T160920_moga_prod_v1_002708.lte'
            zeroSexts_LTE_filepath = 'CLSmod23pm_20200123T160920_moga_prod_v1_002708_ZeroSexts.lte'

        rootname = os.path.basename(input_LTE_filepath).replace('.lte', '')
        pdf_file_prefix = rootname

        lin_summary_pkl_filepath = f'{rootname}.lin.pkl'

        if False:

            twiss_plot_opts = [
                dict(print_scalars=[], right_margin_adj=0.85),
                dict(print_scalars=[], slim=[0, 9], right_margin_adj=0.85,
                     disp_elem_names=dict(bends=True, quads=True, sexts=True,
                                          font_size=8, extra_dy_frac=0.05)),
                dict(print_scalars=[], slim=[4, 16], right_margin_adj=0.85,
                     disp_elem_names=dict(bends=True, quads=True, sexts=True,
                                          font_size=8, extra_dy_frac=0.05)),
                dict(print_scalars=[], slim=[14, 23], right_margin_adj=0.85,
                     disp_elem_names=dict(bends=True, quads=True, sexts=True,
                                          font_size=8, extra_dy_frac=0.05)),
            ]
            captions = [
                'Twiss functions for 2 cells (1 super-period).',
                'Twiss functions (' + MathText('0 < s < 9') + ').',
                'Twiss functions (' + MathText('4 < s < 16') + ').',
                'Twiss functions (' + MathText('14 < s < 23') + ').',
            ]
            assert len(twiss_plot_opts) == len(captions)
            captions = [obj if isinstance(obj, str) else obj.dumps_for_caption()
                        for obj in captions]
            meta_twiss = dict(captions=captions)

            d = summarize_lin(
                input_LTE_filepath, E_MeV=3e3, use_beamline_cell='CELL', use_beamline_ring='RING',
                zeroSexts_LTE_filepath=zeroSexts_LTE_filepath, element_divisions=10,
                twiss_plot_opts=twiss_plot_opts,
                beta_elem_names=['M_LS', 'M_SS'],
                #phase_adv_elem_names_indexes = dict(disp=('M_LS', 0, 'M_SS', 0)),
                floor_elem_names_indexes=dict(
                    SS=['M_SS', 0],
                    LS=['M_LS', 1], # Must pick the 2nd index
                    ),
                #length_elem_names={'L@SS': ['OLONG1', 'OS2B', 'OLONG2'],
                                   #'L@LS': ['OLONG2', 'OS2B', 'OLONG1']},
                length_elem_names={'L@Short Straight': ['OLONG1', 'OS2B', 'OLONG2'],
                                   'L@Long Straight': ['OLONG2', 'OS2B', 'OLONG1']},
            )
            lin_versions = d['versions']
            lin_fignums = d['fignums']
            lin_data = d['sel_data']
            abs_input_LTE_filepath = os.path.abspath(input_LTE_filepath)
            LTE_contents = Path(input_LTE_filepath).read_text()

            #fignums = summarize(
                #input_LTE_filepath,
                #zero_sext_LTE_filepath=zeroSexts_LTE_filepath,
                #use_beamline_cell='CELL', use_beamline_ring='RING',
                #N_KICKS=dict(KQUAD=40, KSEXT=8, CSBEND=12),
                #sext_names=('S1', 'S2', 'S1B', 'S1H'),
                #plot_fm_xy=True, plot_fm_px=True, plot_cmap_xy=True, plot_cmap_px=True,
                #plot_tswa=True, plot_nonlin_chrom=True, #chrom_deltalim=[-2e-2, +2e-2],
                #ntasks=200, title='')

            for k, fignum_list in lin_fignums.items():
                pp = PdfPages(f'{pdf_file_prefix}.{k}.pdf')
                if k == 'twiss':
                    meta_twiss['pdf_filepath'] = os.path.abspath(f'{pdf_file_prefix}.{k}.pdf')
                for fignum in fignum_list:
                    pp.savefig(figure=fignum)
                pp.close()

            #plt.show()

            with open(lin_summary_pkl_filepath, 'wb') as f:
                pickle.dump([abs_input_LTE_filepath, LTE_contents,
                             lin_versions, lin_data, meta_twiss], f)
        else:
            with open(lin_summary_pkl_filepath, 'rb') as f:
                (abs_input_LTE_filepath, LTE_contents,
                 lin_versions, lin_data, meta_twiss
                 ) = pickle.load(f)

    else:

        conf = yaml.safe_load(Path('input.yaml').read_text())

        if False: # This section is no longer necessary, after switching from
            # "yaml" of PyYAML 5.3 to "yaml" of ruamel v0.16.

            # Fix the PyYAML bug that a float value in scientific notation
            # without "." and the sign after e/E is treated as a string.
            if not isinstance(conf['E_MeV'], float):
                conf['E_MeV'] = float(conf['E_MeV'])
            if 'nonlin' in conf:
                block_keys_float_keys_list  = [
                    (['map_calc_opts',
                      'fmap_xy_calc_opts', 'fmap_px_calc_opts',
                      'cmap_xy_calc_opts', 'cmap_px_calc_opts',],
                     ['x_offset', 'y_offset', 'xmin', 'xmax', 'ymin', 'ymax',
                      'delta_offset', 'delta_min', 'delta_max']
                     ),
                    (['tswa_calc_opts'], ['abs_x_max', 'abs_ymax']),
                    (['nonlin_chrom_calc_opts'], ['delta_min', 'delta_max'])
                ]
                for block_key_list, float_keys in block_keys_float_keys_list:
                    for block_key in block_key_list:
                        if block_key in conf['nonlin']:
                            for k, v in conf['nonlin'][block_key].items():
                                if (k in float_keys) and isinstance(v, str):
                                    conf['nonlin'][block_key][k] = float(conf['nonlin'][block_key][k])



        assert conf['input_LTE']['filepath'].endswith('.lte')
        input_LTE_filepath = conf['input_LTE']['filepath']

        if conf['input_LTE']['load_param']:
            assert conf['input_LTE']['base_filepath'].endswith('.lte')
            base_LTE_filepath = conf['input_LTE']['base_filepath']

            load_parameters = dict(filename=conf['input_LTE']['param_filepath'])

            pe.eleutil.save_lattice_after_load_parameters(
                base_LTE_filepath, input_LTE_filepath, load_parameters)

        if conf['input_LTE']['generate_zeroSexts']:
            # Turn off all sextupoles

            if 'zeroSexts_filepath' in conf['input_LTE']:
                zeroSexts_LTE_filepath = conf['input_LTE']['zeroSexts_filepath']
            else:
                zeroSexts_LTE_filepath = input_LTE_filepath.replace('.lte', '_ZeroSexts.lte')

            alter_elements = dict(name='*', type='KSEXT', item='K2', value = 0.0)
            pe.eleutil.save_lattice_after_alter_elements(
                input_LTE_filepath, zeroSexts_LTE_filepath, alter_elements)
        else:
            if 'zeroSexts_filepath' in conf['input_LTE']:
                zeroSexts_LTE_filepath = conf['input_LTE']['zeroSexts_filepath']
            else:
                zeroSexts_LTE_filepath = ''

        rootname = os.path.basename(input_LTE_filepath).replace('.lte', '')
        pdf_file_prefix = rootname

        lin_summary_pkl_filepath = f'{rootname}.lin.pkl'
        nonlin_summary_pkl_filepath = f'{rootname}.nonlin.pkl'

        if (not os.path.exists(lin_summary_pkl_filepath)) or \
            conf['lattice_prop']['recalc']:

            assert len(conf['lattice_prop']['twiss_plot_opts']) == \
                   len(conf['lattice_prop']['twiss_plot_captions'])

            captions = [v if isinstance(v, str) else np.sum([
                MathText(s[10:-2]) if s.startswith('MathText(') and s.endswith(')')
                else s for s in v])
                for v in conf['lattice_prop']['twiss_plot_captions']]
            captions = [obj if isinstance(obj, str) else obj.dumps_for_caption()
                        for obj in captions]
            meta_twiss = dict(captions=captions)

            _kwargs = {'E_MeV': conf['E_MeV']}
            for k in [
                'use_beamline_cell', 'use_beamline_ring',
                'element_divisions', 'twiss_plot_opts', 'beta_elem_names',
                'phase_adv_elem_names_indexes', 'floor_elem_names_indexes',
                'length_elem_names']:
                if k in conf['lattice_prop']:
                    _kwargs[k] = conf['lattice_prop'][k]

            d = summarize_lin(
                input_LTE_filepath, zeroSexts_LTE_filepath=zeroSexts_LTE_filepath,
                **_kwargs)
            lin_versions = d['versions']
            lin_fignums = d['fignums']
            lin_data = d['sel_data']
            abs_input_LTE_filepath = os.path.abspath(input_LTE_filepath)
            LTE_contents = Path(input_LTE_filepath).read_text()

            for k, fignum_list in lin_fignums.items():
                pp = PdfPages(f'{pdf_file_prefix}.{k}.pdf')
                if k == 'twiss':
                    meta_twiss['pdf_filepath'] = os.path.abspath(f'{pdf_file_prefix}.{k}.pdf')
                for fignum in fignum_list:
                    pp.savefig(figure=fignum)
                pp.close()

            #plt.show()

            with open(lin_summary_pkl_filepath, 'wb') as f:
                pickle.dump([abs_input_LTE_filepath, LTE_contents,
                             lin_versions, lin_data, meta_twiss], f)
        else:
            with open(lin_summary_pkl_filepath, 'rb') as f:
                (abs_input_LTE_filepath, LTE_contents,
                 lin_versions, lin_data, meta_twiss
                 ) = pickle.load(f)

            if Path(input_LTE_filepath).read_text() != LTE_contents:
                raise RuntimeError(
                    (f'The LTE contents saved in "{lin_summary_pkl_filepath}" '
                     'does NOT exactly match with the currently specified '
                     f'LTE file "{input_LTE_filepath}". Either check the LTE '
                     'file, or re-calculate to create an updated data file.'))

        if 'nonlin' in conf:

            ncf = conf['nonlin']

            all_calc_types = ['fmap_xy', 'fmap_px', 'cmap_xy', 'cmap_px',
                              'tswa', 'nonlin_chrom']

            sel_plots = {k: False for k in all_calc_types}
            for k, v in ncf['include'].items():
                assert k in all_calc_types
                sel_plots[k] = v

            do_calc = {}
            do_plot = {}
            if not os.path.exists(nonlin_summary_pkl_filepath):
                for k, plot_requested in sel_plots.items():
                    if plot_requested:
                        do_calc[k] = True
                        do_plot[k] = True

                        # Make sure to override "recalc" & "replot" in the config
                        ncf['recalc'][k] = True
                        ncf['replot'][k] = True
            else:
                recalc_d = ncf.get('recalc', {})
                replot_d = ncf.get('replot', {})
                for k, plot_requested in sel_plots.items():
                    if plot_requested:
                        if (k in recalc_d) and recalc_d[k]:
                            do_calc[k] = True
                            do_plot[k] = True
                        elif (k in replot_d) and replot_d[k]:
                            do_calc[k] = False
                            do_plot[k] = True

            if do_calc and any(do_calc.values()):
                calc_nonlin_props(input_LTE_filepath, conf['E_MeV'], ncf)

            if do_plot and any(do_plot.values()):
                #(fignums, tswa_page_caption_list
                 #) = plot_nonlin_props(input_LTE_filepath, ncf, pdf_file_prefix)

                #for k, fignum_list in fignums.items():
                    #pp = PdfPages(f'{pdf_file_prefix}.{k}.pdf')
                    #for fignum in fignum_list:
                        #pp.savefig(figure=fignum)
                    #pp.close()

                tswa_page_caption_list = plot_nonlin_props(
                    input_LTE_filepath, ncf, pdf_file_prefix)

                abs_input_LTE_filepath = os.path.abspath(input_LTE_filepath)
                LTE_contents = Path(input_LTE_filepath).read_text()

                with open(nonlin_summary_pkl_filepath, 'wb') as f:
                    pickle.dump([abs_input_LTE_filepath, LTE_contents, conf,
                                 pdf_file_prefix, tswa_page_caption_list], f)

            else:
                with open(nonlin_summary_pkl_filepath, 'rb') as f:
                    (abs_input_LTE_filepath, LTE_contents, saved_conf,
                     pdf_file_prefix, tswa_page_caption_list) = pickle.load(f)

                if Path(input_LTE_filepath).read_text() != LTE_contents:
                    raise RuntimeError(
                        (f'The LTE contents saved in "{nonlin_summary_pkl_filepath}" '
                         'does NOT exactly match with the currently specified '
                         f'LTE file "{input_LTE_filepath}". Either check the LTE '
                         'file, or re-calculate to create an updated data file.'))


    geometry_options = {"vmargin": "1cm", "hmargin": "1.5cm"}
    doc = plx.Document(f'{rootname}_report', geometry_options=geometry_options,
                       documentclass='article')
    doc.preamble.append(plx.Command('usepackage', 'nopageno')) # Suppress page numbering for entire doc
    doc.preamble.append(plx.Package('indentfirst')) # This fixes the problem of the first paragraph not indenting
    doc.preamble.append(plx.Package('seqsplit')) # To split a very long word into multiple lines w/o adding hyphens, like a long file name.

    doc.preamble.append(plx.Command(
        'title', 'ELEGANT Lattice Characterization Report'))
    if conf is None:
        doc.preamble.append(plx.Command('author', 'Yoshiteru Hidaka'))
    else:
        if 'author' in conf:
            doc.preamble.append(plx.Command('author', conf['author']))
    doc.preamble.append(plx.Command('date', plx.NoEscape(r'\today')))
    if False:
        # To have figures/tables be aligned at the top of pages. If you want
        # the default behavior of vertical centering, you should have
        # "0pt plus 1fil" as `extra_arguments`, or just comment this code
        # section out.
        # => But this actually does NOT work, complaining:
        #
        # ! You can't use `\spacefactor' in math mode.
        # \@->\spacefactor

        kwargs = dict(extra_arguments = '0pt')
        doc.preamble.append(
            plx.Command('setlength', plx.Command('@fptop'), **kwargs))
        doc.preamble.append(
            plx.Command('setlength', plx.Command('@fpbot'), **kwargs))

    doc.append(plx.NoEscape(r'\maketitle'))

    #fill_document(doc)

    with doc.create(plx.Section('Lattice Description')):
        mod_LTE_filename = os.path.basename(input_LTE_filepath).replace("_", r"\_")
        paragraph = f'''
        The lattice file being analyzed here is \seqsplit{{"{mod_LTE_filename}"}}.
        This lattice was based on the 36-pm scaled version of a Canadian Light
        Source upgrade lattice candidate (scaling done by G. Wang). It was
        further optimized to 23 picometers. This lattice is still a 30-cell
        periodic lattice, which means the ID source points do not match with
        the current NSLS-II layout.
        '''
        doc.append(plx.NoEscape(prep_multiline_paragraph_str(paragraph)))
        if conf is None:
            print_versions = True
        else:
            print_versions = conf.get('print_versions', True)
        if print_versions:
            doc.append(plx.Command('par'))
            version_sentence = (
                f'This report was generated using PyELEGANT version {pe.__version__["PyELEGANT"]}.')
            doc.append(version_sentence)

    with doc.create(plx.Section('Lattice Elements')):
        create_bend_elements_subsection(doc, lin_data['elem_defs'])
        create_quad_elements_subsection(doc, lin_data['elem_defs'])
        create_sext_elements_subsection(doc, lin_data['elem_defs'])

        create_beamline_elements_list_subsection(
            doc, lin_data['flat_elem_s_name_type_list'])

    #generate_pdf_w_reruns(doc, clean_tex=False, silent=False)

    create_lattice_prop_section(doc, lin_data, meta_twiss, lin_versions)

    #doc.append(plx.NewPage())
    doc.append(plx.Command('clearpage'))

    #doc.generate_pdf(clean_tex=False, silent=False)
    #generate_pdf_w_reruns(doc, clean_tex=False, silent=False)

    if conf is None:
        sub_pdfs = {}
        for k in ['fma_xy', 'fma_px', 'cmap_xy', 'cmap_px', 'tswa', 'nonlin_chrom']:
            sub_pdfs[k] = f'{rootname}.{k}.pdf'

        new_page_required = False

        if os.path.exists(sub_pdfs['fma_xy']) or os.path.exists(sub_pdfs['fma_px']):

            with doc.create(plx.Section('Frequency Map')):

                nx = ny = 201
                n_turns = 1024

                if os.path.exists(sub_pdfs['fma_xy']) and os.path.exists(sub_pdfs['fma_px']):
                    doc.append(f'The number of turns tracked was {n_turns}.')
                    doc.append(plx.Command('vspace', plx.NoEscape('-10pt')))
                    with doc.create(plx.Figure(position='h!')) as fig:
                        doc.append(plx.NoEscape(r'\centering'))
                        for k, caption in [
                            ('fma_xy', 'On-Momentum'), ('fma_px', 'Off-Momentum')]:
                            with doc.create(plx.SubFigure(
                                position='b', width=plx.utils.NoEscape(r'0.5\linewidth'))) as subfig:
                                subfig.add_image(
                                    sub_pdfs[k], width=plx.utils.NoEscape(r'\linewidth'))
                                doc.append(plx.Command('vspace', plx.NoEscape('-10pt')))
                                subfig.add_caption(caption)
                        doc.append(plx.Command('vspace', plx.NoEscape('-10pt')))
                        fig.add_caption('On- & off-momentum frequency maps.')

                    new_page_required = True

                else:
                    for k, subsec_title, caption in [
                        ('fma_xy', 'On Momentum', 'On-momentum frequency map.'),
                        ('fma_px', 'Off Momentum', 'Off-momentum frequency map.')]:


                        if os.path.exists(sub_pdfs[k]):

                            with doc.create(plx.Subsection(subsec_title)):

                                doc.append(f'The number of turns tracked was {n_turns}.')

                                with doc.create(plx.Figure(position='h!')) as fig:

                                    if False: # \setlength{\textfloatsep} does NOT take effect somehow
                                        # Maybe this works if I put these into "preamble", but never tested
                                        # to confirm if it works.

                                        #doc.append(plx.NoEscape(r'\showthe\textfloatsep'))
                                        doc.append(
                                            plx.Command('setlength', plx.Command('textfloatsep'),
                                                        extra_arguments='100pt plus 2.0pt minus 4.0pt'))
                                                        #extra_arguments='0pt plus 0.0pt minus 0.0pt'))
                                    else:
                                        doc.append(plx.Command('vspace', plx.NoEscape('-10pt')))

                                    fig.add_image(sub_pdfs[k],
                                                  width=plx.utils.NoEscape(r'0.6\linewidth'))
                                    fig.add_caption(caption)

                    new_page_required = True

        if os.path.exists(sub_pdfs['cmap_xy']) or os.path.exists(sub_pdfs['cmap_px']):

            if new_page_required:
                doc.append(plx.NewPage())
                new_page_required = False

            with doc.create(plx.Section('Chaos Map')):

                nx = ny = 201
                n_turns = 128

                if os.path.exists(sub_pdfs['cmap_xy']) and os.path.exists(sub_pdfs['cmap_px']):
                    doc.append(f'The number of turns tracked was {n_turns}.')
                    doc.append(plx.Command('vspace', plx.NoEscape('-10pt')))
                    with doc.create(plx.Figure(position='h!')) as fig:
                        doc.append(plx.NoEscape(r'\centering'))
                        for k, caption in [
                            ('cmap_xy', 'On-Momentum'), ('cmap_px', 'Off-Momentum')]:
                            with doc.create(plx.SubFigure(
                                position='b', width=plx.utils.NoEscape(r'0.5\linewidth'))) as subfig:
                                subfig.add_image(
                                    sub_pdfs[k], width=plx.utils.NoEscape(r'\linewidth'))
                                doc.append(plx.Command('vspace', plx.NoEscape('-10pt')))
                                subfig.add_caption(caption)
                        doc.append(plx.Command('vspace', plx.NoEscape('-10pt')))
                        fig.add_caption('On- & off-momentum chaos maps.')

                    new_page_required = True

                else:
                    for k, subsec_title, caption in [
                        ('cmap_xy', 'On Momentum', 'On-momentum chaos map.'),
                        ('cmap_px', 'Off Momentum', 'Off-momentum chaos map.')]:


                        if os.path.exists(sub_pdfs[k]):

                            with doc.create(plx.Subsection(subsec_title)):

                                doc.append(f'The number of turns tracked was {n_turns}.')

                                with doc.create(plx.Figure(position='h!')) as fig:

                                    doc.append(plx.Command('vspace', plx.NoEscape('-10pt')))

                                    fig.add_image(sub_pdfs[k],
                                                  width=plx.utils.NoEscape(r'0.6\linewidth'))
                                    fig.add_caption(caption)

                    new_page_required = True


        #doc.append(plx.Math(inline=True, data=['A_x', '(> 0)'], escape=False))

        if os.path.exists(f'{rootname}.tswa.pdf'):

            if new_page_required:
                doc.append(plx.NewPage())
                new_page_required = False

            with doc.create(plx.Section('Tune Shift with Amplitude')):
                doc.append(f'Some placeholder.')
                doc.append(plx.Command('vspace', plx.NoEscape('-10pt')))

                for plane in ('x', 'y'):

                    with doc.create(plx.Figure(position='h!')) as fig:

                        doc.append(plx.NoEscape(r'\centering'))


                        for iFig, (page, caption) in enumerate([
                            #(1, MathEnglishTextV2(
                                #[GMATH.nu, ' vs. ',
                                 #plx.Math(inline=True, data=[f'A_{plane} (> 0)'])]).dumps_for_caption(),),
                            (1, (MT(r'\nu') + NT(' vs. ') + MT(fr'A_{plane} (> 0)')).dumps_for_caption()),
                            #(3, MathEnglishTextV2(
                                #[GMATH.nu, ' vs. ',
                                 #plx.Math(inline=True, data=[f'A_{plane} (< 0)'])]).dumps_for_caption(),),
                            (3, (MT(r'\nu') + NT(' vs. ') + MT(fr'A_{plane} (< 0)')).dumps_for_caption()),
                            #(2, MathEnglishTextV2(
                                #['Tune footprint vs. ',
                                 #plx.Math(inline=True, data=[f'A_{plane} (> 0)'])]).dumps_for_caption(),),
                            (2, (NT('Tune footprint vs. ') + MT(fr'A_{plane} (> 0)')).dumps_for_caption()),
                            #(4, MathEnglishTextV2(
                                #['Tune footprint vs. ',
                                 #plx.Math(inline=True, data=[f'A_{plane} (< 0)'])]).dumps_for_caption(),),
                            (4, (NT('Tune footprint vs. ') + MT(fr'A_{plane} (< 0)')).dumps_for_caption()),
                            ]):
                            with doc.create(SubFigureForMultiPagePDF(
                                position='b', width=plx.utils.NoEscape(r'0.5\linewidth'))) as subfig:
                                subfig.add_image(
                                    sub_pdfs['tswa'],
                                    page=(page if plane == 'x' else page + 4),
                                    width=plx.utils.NoEscape(r'\linewidth'))
                                doc.append(plx.Command('vspace', plx.NoEscape('-10pt')))
                                subfig.add_caption(caption)

                            if iFig in (1,):
                                doc.append(plx.NewLine())

                        doc.append(plx.Command('vspace', plx.NoEscape('-5pt')))
                        fig.add_caption('Tune-shift with {} amplitude.'.format(
                            'horizontal' if plane == 'x' else 'vertical'))

                    doc.append(plx.NewPage())

        if False:
            # Testing of math mode / text mode mixture
            doc.append(plx.Math(inline=True, data=[plx.NoEscape(r'\nu'), '=', '3.5']))
            doc.append(plx.NewLine())
            doc.append(plx.Math(inline=True, data=[GREEK['nu']]))
            doc.append(plx.NewLine())
            doc.append('    vs. ')
            doc.append(plx.NewLine())
            doc.append(GMATH.nu)
            doc.append('    vs. ')
            doc.append(GMATH.delta)
            doc.append(plx.NewLine())
            if False:
                t = MathEnglishTextV1(doc)
                t.append(GMATH.nu)
                t.append('    vs. ')
                t.append(GMATH.delta)
                t.apply()
            elif False:
                t = MathEnglishTextV2()
                t.append(GMATH.nu)
                t.append('    vs. ')
                t.append(GMATH.delta)
                doc.append(t)
            elif False:
                t = MathEnglishTextV2([GMATH.nu, ' vs. ', GMATH.delta])
                doc.append(t)
            else:
                test = MathText(r'\nu_x')
                print(test)
                test = NormalText(' vs. ')
                print(test)
                test = MathText(r'A_x (> 0)')
                print(test)

                test = MathText(r'\nu_x') + NormalText(' vs. ') + MathText(r'A_x (> 0)')
                doc.append(test)


        if os.path.exists(f'{rootname}.nonlin_chrom.pdf'):

            if new_page_required:
                doc.append(plx.NewPage())
                new_page_required = False

            with doc.create(plx.Section('Nonlinear Chromaticity')):
                doc.append(f'Some placeholder.')
                doc.append(plx.Command('vspace', plx.NoEscape('-10pt')))
                with doc.create(plx.Figure(position='h!')) as fig:

                    doc.append(plx.NoEscape(r'\centering'))

                    for iPage, caption in enumerate([
                        #'Tunes vs. momentum offset',
                        #r'$\nu$ vs. $\delta$',
                        #plx.Math(data=[r'$\nu$']),
                        #plx.NoEscape(MathEnglishTextV2([GMATH.nu, ' vs. ', GMATH.delta]).dumps()),
                        #MathEnglishTextV2([GMATH.nu, ' vs. ', GMATH.delta]).dumps_for_caption(),
                        (MT(r'\nu') + NT(' vs. ') + MT(r'\delta')).dumps_for_caption(),
                        'Off-momentum tune footprint']):
                        with doc.create(SubFigureForMultiPagePDF(
                            position='b', width=plx.utils.NoEscape(r'0.5\linewidth'))) as subfig:
                            subfig.add_image(
                                sub_pdfs['nonlin_chrom'], page=iPage+1,
                                width=plx.utils.NoEscape(r'\linewidth'))
                            doc.append(plx.Command('vspace', plx.NoEscape('-10pt')))
                            subfig.add_caption(caption)
                    doc.append(plx.Command('vspace', plx.NoEscape('-10pt')))
                    fig.add_caption('Nonlinear chromaticity.')


    else:
        ncf = conf['nonlin']

        LTE_contents = Path(input_LTE_filepath).read_text()

        nonlin_data_filepaths = get_nonlin_data_filepaths(input_LTE_filepath, ncf)
        included_types = [k for k, _included in ncf.get('include', {}).items()
                          if _included]
        sub_pdfs = {k: f'{pdf_file_prefix}.{k}.pdf' for k in included_types}

        new_page_required = False

        if ('fmap_xy' in included_types) or ('fmap_px' in included_types):
            with doc.create(plx.Section('Frequency Map')):
                if os.path.exists(sub_pdfs['fmap_xy']) and \
                   os.path.exists(sub_pdfs['fmap_px']):
                    d_xy = pe.util.load_pgz_file(nonlin_data_filepaths['fmap_xy'])
                    d_px = pe.util.load_pgz_file(nonlin_data_filepaths['fmap_px'])

                    assert os.path.basename(d_xy['input']['LTE_filepath']) \
                           == input_LTE_filepath
                    assert os.path.basename(d_px['input']['LTE_filepath']) \
                           == input_LTE_filepath
                    assert d_xy['input']['lattice_file_contents'] == LTE_contents
                    assert d_px['input']['lattice_file_contents'] == LTE_contents

                    n_turns = d_xy['input']['n_turns']
                    nx, ny = d_xy['input']['nx'], d_xy['input']['ny']
                    doc.append((
                        f'The on-momentum frequency map was generated by '
                        f'tracking particles for {n_turns:d} turns at each point '
                        f'in the grid of '))
                    doc.append(MathText((
                        '{nx:d}\, ({xmin_mm:+.3f} \le x_0 [\mathrm{{mm}}] '
                        '\le {xmax_mm:+.3f}) '
                        r'\times '
                        '{ny:d}\, ({ymin_mm:+.3f} \le y_0 [\mathrm{{mm}}] '
                        '\le {ymax_mm:+.3f})').format(
                            nx=nx, ny=ny,
                            xmin_mm=d_xy['input']['xmin'] * 1e3,
                            xmax_mm=d_xy['input']['xmax'] * 1e3,
                            ymin_mm=d_xy['input']['ymin'] * 1e3,
                            ymax_mm=d_xy['input']['ymax'] * 1e3,
                        )))
                    doc.append(plx.NoEscape('\ points'))
                    if d_xy['input']['delta_offset'] != 0.0:
                        doc.append(
                            ', with a constant momentum offset of {:.2g}%.'.format(
                            d_xy['input']['delta_offset'] * 1e2))
                    else:
                        doc.append('.')

                    doc.append(plx.NoEscape('\ '))

                    n_turns = d_px['input']['n_turns']
                    nx, ndelta = d_px['input']['nx'], d_px['input']['ndelta']
                    doc.append((
                        f'The off-momentum frequency map was generated by '
                        f'tracking particles for {n_turns:d} turns at each point '
                        f'in the grid of '))
                    doc.append(MathText((
                        '{ndelta:d}\, ({delta_min:+.3g} \le \delta [\%] '
                        '\le {delta_max:+.3g}) '
                        r'\times '
                        '{nx:d}\, ({xmin_mm:+.3f} \le x_0 [\mathrm{{mm}}] '
                        '\le {xmax_mm:+.3f})').format(
                            nx=nx, ndelta=ndelta,
                            xmin_mm=d_px['input']['xmin'] * 1e3,
                            xmax_mm=d_px['input']['xmax'] * 1e3,
                            delta_min=d_px['input']['delta_min'] * 1e2,
                            delta_max=d_px['input']['delta_max'] * 1e2,
                        )))
                    doc.append(plx.NoEscape('\ points'))
                    if d_px['input']['y_offset'] != 0.0:
                        doc.append(
                            ', with a constant initial vertical offset of {:.3g} mm.'.format(
                            d_px['input']['y_offset'] * 1e3))
                    else:
                        doc.append('.')

                    if d_xy['_version_ELEGANT'] == d_px['_version_ELEGANT']:
                        ver_sentence = (
                            f'ELEGANT version {d_xy["_version_ELEGANT"]} was used '
                            f'to compute the frequency map data.')
                    else:
                        ver_sentence = (
                            f'ELEGANT version {d_xy["_version_ELEGANT"]} was used '
                            f'to compute the on-momentum frequency map data, '
                            f'while ELEGANT version {d_px["_version_ELEGANT"]} '
                            f'was used for the off-momentum frequency map data.')

                    doc.append(plx.Command('par'))
                    doc.append(ver_sentence)
                    #doc.append(f'The number of turns tracked was {n_turns}.')
                    doc.append(plx.VerticalSpace(plx.NoEscape('-10pt')))
                    with doc.create(plx.Figure(position='h!')) as fig:
                        doc.append(plx.NoEscape(r'\centering'))
                        for k, caption in [
                            ('fmap_xy', 'On-Momentum'), ('fmap_px', 'Off-Momentum')]:
                            with doc.create(plx.SubFigure(
                                position='b', width=plx.utils.NoEscape(r'0.5\linewidth'))) as subfig:
                                subfig.add_image(
                                    sub_pdfs[k], width=plx.utils.NoEscape(r'\linewidth'))
                                #doc.append(plx.Command('vspace', plx.NoEscape('-10pt')))
                                doc.append(plx.VerticalSpace(plx.NoEscape('-10pt')))
                                subfig.add_caption(caption)
                        #doc.append(plx.Command('vspace', plx.NoEscape('-10pt')))
                        doc.append(plx.VerticalSpace(plx.NoEscape('-10pt')))
                        fig.add_caption('On- & off-momentum frequency maps.')

                    new_page_required = True
                else:
                    for k, subsec_title, caption in [
                        ('fmap_xy', 'On Momentum', 'On-momentum frequency map.'),
                        ('fmap_px', 'Off Momentum', 'Off-momentum frequency map.')]:

                        d = pe.util.load_pgz_file(nonlin_data_filepaths[k])
                        ver_sentence = (
                            f'ELEGANT version {d["_version_ELEGANT"]} was used '
                            f'to compute the frequency map data.')

                        if os.path.exists(sub_pdfs[k]):
                            with doc.create(plx.Subsection(subsec_title)):
                                doc.append('Description for frequency maps goes here.')
                                doc.append(plx.Command('par'))
                                doc.append(ver_sentence)
                                #doc.append(f'The number of turns tracked was {n_turns}.')

                                with doc.create(plx.Figure(position='h!')) as fig:
                                    #doc.append(plx.Command('vspace', plx.NoEscape('-10pt')))
                                    doc.append(plx.VerticalSpace(plx.NoEscape('-10pt')))

                                    fig.add_image(sub_pdfs[k],
                                                  width=plx.utils.NoEscape(r'0.6\linewidth'))
                                    fig.add_caption(caption)

                    new_page_required = True

        if ('cmap_xy' in included_types) or ('cmap_px' in included_types):
            if new_page_required:
                #doc.append(plx.NewPage())
                doc.append(plx.Command('clearpage'))
                new_page_required = False

            with doc.create(plx.Section('Chaos Map')):

                if os.path.exists(sub_pdfs['cmap_xy']) and \
                   os.path.exists(sub_pdfs['cmap_px']):
                    d_xy = pe.util.load_pgz_file(nonlin_data_filepaths['cmap_xy'])
                    d_px = pe.util.load_pgz_file(nonlin_data_filepaths['cmap_px'])

                    assert os.path.basename(d_xy['input']['LTE_filepath']) \
                           == input_LTE_filepath
                    assert os.path.basename(d_px['input']['LTE_filepath']) \
                           == input_LTE_filepath
                    assert d_xy['input']['lattice_file_contents'] == LTE_contents
                    assert d_px['input']['lattice_file_contents'] == LTE_contents

                    n_turns = d_xy['input']['n_turns']
                    nx, ny = d_xy['input']['nx'], d_xy['input']['ny']
                    doc.append((
                        f'The on-momentum chaos map was generated by tracking '
                        f'particles for {n_turns:d} turns at each point in the '
                        f'grid of '))
                    doc.append(MathText((
                        '{nx:d}\, ({xmin_mm:+.3f} \le x_0 [\mathrm{{mm}}] '
                        '\le {xmax_mm:+.3f}) '
                        r'\times '
                        '{ny:d}\, ({ymin_mm:+.3f} \le y_0 [\mathrm{{mm}}] '
                        '\le {ymax_mm:+.3f})').format(
                            nx=nx, ny=ny,
                            xmin_mm=d_xy['input']['xmin'] * 1e3,
                            xmax_mm=d_xy['input']['xmax'] * 1e3,
                            ymin_mm=d_xy['input']['ymin'] * 1e3,
                            ymax_mm=d_xy['input']['ymax'] * 1e3,
                        )))
                    doc.append(plx.NoEscape('\ points'))
                    if d_xy['input']['delta_offset'] != 0.0:
                        doc.append(
                            ', with a constant momentum offset of {:.2g}%.'.format(
                            d_xy['input']['delta_offset'] * 1e2))
                    else:
                        doc.append('.')

                    doc.append(plx.NoEscape('\ '))

                    n_turns = d_px['input']['n_turns']
                    nx, ndelta = d_px['input']['nx'], d_px['input']['ndelta']
                    doc.append((
                        f'The off-momentum chaos map was generated by tracking '
                        f'particles for {n_turns:d} turns at each point in the '
                        f'grid of '))
                    doc.append(MathText((
                        '{ndelta:d}\, ({delta_min:+.3g} \le \delta [\%] '
                        '\le {delta_max:+.3g}) '
                        r'\times '
                        '{nx:d}\, ({xmin_mm:+.3f} \le x_0 [\mathrm{{mm}}] '
                        '\le {xmax_mm:+.3f})').format(
                            nx=nx, ndelta=ndelta,
                            xmin_mm=d_px['input']['xmin'] * 1e3,
                            xmax_mm=d_px['input']['xmax'] * 1e3,
                            delta_min=d_px['input']['delta_min'] * 1e2,
                            delta_max=d_px['input']['delta_max'] * 1e2,
                        )))
                    doc.append(plx.NoEscape('\ points'))
                    if d_px['input']['y_offset'] != 0.0:
                        doc.append(
                            ', with a constant initial vertical offset of {:.3g} mm.'.format(
                            d_px['input']['y_offset'] * 1e3))
                    else:
                        doc.append('.')

                    if d_xy['_version_ELEGANT'] == d_px['_version_ELEGANT']:
                        ver_sentence = (
                            f'ELEGANT version {d_xy["_version_ELEGANT"]} was used '
                            f'to compute the chaos map data.')
                    else:
                        ver_sentence = (
                            f'ELEGANT version {d_xy["_version_ELEGANT"]} was used '
                            f'to compute the on-momentum chaos map data, '
                            f'while ELEGANT version {d_px["_version_ELEGANT"]} '
                            f'was used for the off-momentum chaos map data.')

                    doc.append(plx.Command('par'))
                    doc.append(ver_sentence)
                    #doc.append(f'The number of turns tracked was {n_turns}.')
                    #doc.append(plx.Command('vspace', plx.NoEscape('-10pt')))
                    doc.append(plx.VerticalSpace(plx.NoEscape('-10pt')))
                    with doc.create(plx.Figure(position='h!')) as fig:
                        doc.append(plx.NoEscape(r'\centering'))
                        for k, caption in [
                            ('cmap_xy', 'On-Momentum'), ('cmap_px', 'Off-Momentum')]:
                            with doc.create(plx.SubFigure(
                                position='b', width=plx.utils.NoEscape(r'0.5\linewidth'))) as subfig:
                                subfig.add_image(
                                    sub_pdfs[k], width=plx.utils.NoEscape(r'\linewidth'))
                                #doc.append(plx.Command('vspace', plx.NoEscape('-10pt')))
                                doc.append(plx.VerticalSpace(plx.NoEscape('-10pt')))
                                subfig.add_caption(caption)
                        #doc.append(plx.Command('vspace', plx.NoEscape('-10pt')))
                        doc.append(plx.VerticalSpace(plx.NoEscape('-10pt')))
                        fig.add_caption('On- & off-momentum chaos maps.')

                    new_page_required = True

                else:
                    for k, subsec_title, caption in [
                        ('cmap_xy', 'On Momentum', 'On-momentum chaos map.'),
                        ('cmap_px', 'Off Momentum', 'Off-momentum chaos map.')]:

                        d = pe.util.load_pgz_file(nonlin_data_filepaths[k])
                        ver_sentence = (
                            f'ELEGANT version {d["_version_ELEGANT"]} was used '
                            f'to compute the chaos map data.')

                        if os.path.exists(sub_pdfs[k]):
                            with doc.create(plx.Subsection(subsec_title)):
                                doc.append('Description for chaos maps goes here.')
                                doc.append(plx.Command('par'))
                                doc.append(ver_sentence)
                                #doc.append(f'The number of turns tracked was {n_turns}.')
                                with doc.create(plx.Figure(position='h!')) as fig:
                                    #doc.append(plx.Command('vspace', plx.NoEscape('-10pt')))
                                    doc.append(plx.VerticalSpace(plx.NoEscape('-10pt')))
                                    fig.add_image(sub_pdfs[k],
                                                  width=plx.utils.NoEscape(r'0.6\linewidth'))
                                    fig.add_caption(caption)

                    new_page_required = True

        if 'tswa' in included_types:

            if new_page_required:
                #doc.append(plx.NewPage())
                doc.append(plx.Command('clearpage'))
                new_page_required = False

            with doc.create(plx.Section('Tune Shift with Amplitude')):
                d = {}
                versions, n_turns_list = [], []
                abs_xmax_list, nx_list, y0_offset_list = [], [], []
                abs_ymax_list, ny_list, x0_offset_list = [], [], []
                for plane in ['x', 'y']:
                    for sign in ['plus', 'minus']:
                        v = d[f'tswa_{plane}{sign}'] = pe.util.load_pgz_file(
                            nonlin_data_filepaths[f'tswa_{plane}{sign}'])

                        versions.append(v['_version_ELEGANT'])

                        assert os.path.basename(v['input']['LTE_filepath']) \
                               == os.path.basename(input_LTE_filepath)
                        assert v['input']['lattice_file_contents'] == LTE_contents

                        n_turns_list.append(v['input']['n_turns'])

                        vv = v['input']['plane_specific_input']
                        if plane == 'x':
                            abs_xmax_list.append(vv['abs_xmax'])
                            nx_list.append(vv['nx'])
                            y0_offset_list.append(vv['y0_offset'])
                        else:
                            abs_ymax_list.append(vv['abs_ymax'])
                            ny_list.append(vv['ny'])
                            x0_offset_list.append(vv['x0_offset'])

                assert len(set(versions)) == 1
                assert len(set(n_turns_list)) == 1
                assert len(set(abs_xmax_list)) == len(set(abs_ymax_list)) == 1
                assert len(set(nx_list)) == len(set(ny_list)) == 1
                assert len(set(y0_offset_list)) == len(set(x0_offset_list)) == 1

                n_turns = n_turns_list[0]
                abs_xmax, abs_ymax = abs_xmax_list[0], abs_ymax_list[0]
                nx, ny = nx_list[0], ny_list[0]
                y0_offset, x0_offset = y0_offset_list[0], x0_offset_list[0]

                ver_sentence = (
                    f'ELEGANT version {versions[0]} was used to compute the '
                    f'tune shift with amplitude.')

                doc.append((
                    f'The plots for tune shift with horizontal amplitude were '
                    f'generated by tracking particles for {n_turns:d} turns at '
                    f'each point in the array of '))
                doc.append(MathText((
                    '{nx:d}\, ({xmin_mm:+.3f} \le x_0 [\mathrm{{mm}}] '
                    '\le {xmax_mm:+.3f}) ').format(
                        nx=nx,
                        xmin_mm=abs_xmax * 1e3 * (-1),
                        xmax_mm=abs_xmax * 1e3,
                    )))
                doc.append(plx.NoEscape('\ points'))
                if y0_offset != 0.0:
                    doc.append(
                        ', with a constant initial vertical offset of {:.3g} mm.'.format(
                        y0_offset * 1e3))
                else:
                    doc.append('.')

                #doc.append(plx.NoEscape('\ '))
                doc.append(plx.Command('par'))

                doc.append((
                    f'The plots for tune shift with vertical amplitude were '
                    f'generated by tracking particles for {n_turns:d} turns at '
                    f'each point in the array of '))
                doc.append(MathText((
                    '{ny:d}\, ({ymin_mm:+.3f} \le y_0 [\mathrm{{mm}}] '
                    '\le {ymax_mm:+.3f}) ').format(
                        ny=ny,
                        ymin_mm=abs_ymax * 1e3 * (-1),
                        ymax_mm=abs_ymax * 1e3,
                    )))
                doc.append(plx.NoEscape('\ points'))
                if x0_offset != 0.0:
                    doc.append(
                        ', with a constant initial horizontal offset of {:.3g} mm.'.format(
                        x0_offset * 1e3))
                else:
                    doc.append('.')

                doc.append(plx.Command('par'))
                doc.append(ver_sentence)
                #doc.append(plx.Command('vspace', plx.NoEscape('-10pt')))
                doc.append(plx.VerticalSpace(plx.NoEscape('-10pt')))

                if False:
                    for plane in ('x', 'y'):

                        with doc.create(plx.Figure(position='h!')) as fig:

                            doc.append(plx.NoEscape(r'\centering'))

                            page_caption_list = [
                                (1, MT(r'\nu') + ' vs. ' + MT(fr'A_{plane} (> 0)')),
                                (3, MT(r'\nu') + ' vs. ' + MT(fr'A_{plane} (< 0)')),
                                (2, 'Tune footprint vs. ' + MT(fr'A_{plane} (> 0)')),
                                (4, 'Tune footprint vs. ' + MT(fr'A_{plane} (< 0)')),
                            ]

                            for iFig, (page, caption) in enumerate(page_caption_list):
                                with doc.create(SubFigureForMultiPagePDF(
                                    position='b', width=plx.utils.NoEscape(r'0.5\linewidth'))) as subfig:
                                    subfig.add_image(
                                        sub_pdfs['tswa'],
                                        page=(page if plane == 'x' else page + 4),
                                        width=plx.utils.NoEscape(r'\linewidth'))
                                    #doc.append(plx.Command('vspace', plx.NoEscape('-10pt')))
                                    doc.append(plx.VerticalSpace(plx.NoEscape('-10pt')))
                                    subfig.add_caption(caption.dumps_for_caption())

                                if iFig in (1,):
                                    doc.append(plx.NewLine())

                            #doc.append(plx.Command('vspace', plx.NoEscape('-5pt')))
                            doc.append(plx.VerticalSpace(plx.NoEscape('-10pt')))
                            fig.add_caption('Tune-shift with {} amplitude.'.format(
                                'horizontal' if plane == 'x' else 'vertical'))

                    #doc.append(plx.NewPage())
                    doc.append(plx.Command('clearpage'))
                else:
                    for plane, page_caption_list in [
                        ('x', tswa_page_caption_list[:2]),
                        ('y', tswa_page_caption_list[2:])]:

                        with doc.create(plx.Figure(position='h!')) as fig:

                            doc.append(plx.NoEscape(r'\centering'))

                            for iFig, (page, caption) in enumerate(page_caption_list):
                                with doc.create(SubFigureForMultiPagePDF(
                                    position='b', width=plx.utils.NoEscape(r'0.5\linewidth'))) as subfig:
                                    subfig.add_image(
                                        sub_pdfs['tswa'], page=page,
                                        width=plx.utils.NoEscape(r'\linewidth'))
                                    doc.append(plx.VerticalSpace(plx.NoEscape('-10pt')))
                                    subfig.add_caption(caption.dumps_for_caption())

                                if iFig in (1,):
                                    doc.append(plx.NewLine())

                            doc.append(plx.VerticalSpace(plx.NoEscape('-10pt')))
                            fig.add_caption('Tune-shift with {} amplitude.'.format(
                                'horizontal' if plane == 'x' else 'vertical'))

                    #doc.append(plx.NewPage())
                    doc.append(plx.Command('clearpage'))




        if 'nonlin_chrom' in included_types:

            if new_page_required:
                #doc.append(plx.NewPage())
                doc.append(plx.Command('clearpage'))
                new_page_required = False

            with doc.create(plx.Section('Nonlinear Chromaticity')):
                d = pe.util.load_pgz_file(nonlin_data_filepaths['nonlin_chrom'])
                ver_sentence = (
                    f'ELEGANT version {d["_version_ELEGANT"]} was used '
                    f'to compute the nonlinear chromaticity data.')

                assert os.path.basename(d['input']['LTE_filepath']) \
                       == os.path.basename(input_LTE_filepath)
                assert d['input']['lattice_file_contents'] == LTE_contents

                n_turns = d['input']['n_turns']
                ndelta = d['input']['ndelta']
                delta_min = d['input']['delta_min']
                delta_max = d['input']['delta_max']
                x0_offset = d['input']['x0_offset']
                y0_offset = d['input']['y0_offset']

                doc.append((
                    f'The plots for nonlinear chromaticity were generated by '
                    f'tracking particles for {n_turns:d} turns at each point in '
                    f'the array of '))
                doc.append(MathText((
                    '{ndelta:d}\, ({delta_min:+.3g} \le \delta [\%] '
                    '\le {delta_max:+.3g}) ').format(
                        ndelta=ndelta,
                        delta_min=delta_min * 1e2, delta_max=delta_max * 1e2,
                    )))
                doc.append(plx.NoEscape('\ points'))
                if (x0_offset != 0.0) or (y0_offset != 0.0):
                    doc.append(
                        (', with a constant initial horizontal and vertical '
                         'offset of {:.3g} and {:.3g} mm, respectively.').format(
                        x0_offset * 1e3, y0_offset * 1e3))
                else:
                    doc.append('.')

                doc.append(plx.Command('par'))
                doc.append(ver_sentence)
                #doc.append(plx.Command('vspace', plx.NoEscape('-10pt')))
                doc.append(plx.VerticalSpace(plx.NoEscape('-10pt')))
                with doc.create(plx.Figure(position='h!')) as fig:

                    doc.append(plx.NoEscape(r'\centering'))

                    for iPage, caption in enumerate([
                        MT(r'\nu') + ' vs. ' + MT(r'\delta'),
                        NT('Off-momentum tune footprint')]):
                        with doc.create(SubFigureForMultiPagePDF(
                            position='b', width=plx.utils.NoEscape(r'0.5\linewidth'))) as subfig:
                            subfig.add_image(
                                sub_pdfs['nonlin_chrom'], page=iPage+1,
                                width=plx.utils.NoEscape(r'\linewidth'))
                            #doc.append(plx.Command('vspace', plx.NoEscape('-10pt')))
                            doc.append(plx.VerticalSpace(plx.NoEscape('-10pt')))
                            subfig.add_caption(caption.dumps_for_caption())
                    #doc.append(plx.Command('vspace', plx.NoEscape('-10pt')))
                    doc.append(plx.VerticalSpace(plx.NoEscape('-10pt')))
                    fig.add_caption('Nonlinear chromaticity.')




    #doc.generate_pdf(clean_tex=False, silent=False)
    generate_pdf_w_reruns(doc, clean_tex=False, silent=False)
