import sys
import os
import numpy as np
import getpass
import matplotlib.pylab as plt
from types import SimpleNamespace
from matplotlib.backends.backend_pdf import PdfPages

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

class MathEnglishText(plx.Math):
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

def summarize_lin(
    LTE_filepath, use_beamline_cell='CELL', use_beamline_ring='RING',
    zeroSexts_LTE_filepath='', element_divisions=10,
    beta_elem_names=None, phase_adv_elem_names_indexes=None, length_elem_names=None,
    floor_elem_names_indexes=None):
    """"""

    fignums = {}

    sel_data = {}
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

    # --- Misc ---
    # L (SS)
    # L (LS)
    #
    # element definitions (only quads/bends/sexts, i.e., no drifts/markers)
    # element list for a cell

    E_MeV = 3e3
    output_filepath = 'test.pgz'

    pe.calc_ring_twiss(
        output_filepath, LTE_filepath, E_MeV, radiation_integrals=True,
        element_divisions=element_divisions, use_beamline=use_beamline_cell)
    pe.plot_twiss(
        output_filepath, print_scalars=[],
        disp_elem_names=dict(bends=True, quads=True, sexts=True,
                             font_size=8, extra_dy_frac=0.05))

    twi = pe.util.load_pgz_file(output_filepath)['data']['twi']

    for k, ele_k in raw_keys['one_period'].items():
        sel_data[k] = twi['scalars'][ele_k]
    for k, ele_k in interm_array_keys['one_period'].items():
        interm_array_data[k] = twi['arrays'][ele_k]

    if zeroSexts_LTE_filepath != '':
        pe.calc_ring_twiss(
            output_filepath, zeroSexts_LTE_filepath, E_MeV, radiation_integrals=True,
            use_beamline=use_beamline_ring)

        twi = pe.util.load_pgz_file(output_filepath)['data']['twi']

        for k, ele_k in raw_keys['ring_natural'].items():
            sel_data[k] = twi['scalars'][ele_k]

    pe.calc_ring_twiss(
        output_filepath, LTE_filepath, E_MeV, radiation_integrals=True,
        use_beamline=use_beamline_ring)

    twi = pe.util.load_pgz_file(output_filepath)['data']['twi']

    for k, ele_k in raw_keys['ring'].items():
        sel_data[k] = twi['scalars'][ele_k]
    for k, ele_k in interm_array_keys['ring'].items():
        interm_array_data[k] = twi['arrays'][ele_k]

    #try:
        #sext_names = [_name.upper() for _name in sext_names]
        #params = pe.util.load_pgz_file(output_filepath)['data']['parameters']
        #K2_vals = np.unique([val for name, param_name, val in
            #zip(params['arrays']['ElementName'], params['arrays']['ElementParameter'],
                #params['arrays']['ParameterValue'])
            #if name in sext_names and param_name == 'K2'])
        #print_lines.append('K2s = ({})'.format(', '.join(['{:+.1f}'.format(K2) for K2 in K2_vals])))
    #except:
        #pass

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





    #par, col = output['twi']['params'], output['twi']['columns']
    #inds = np.where(np.logical_and(
        #output['param']['columns']['ElementParameter'] == 'L',
        #output['param']['columns']['ElementName'] == 'OLONG1'))[0]
    #print('L (Long Str.) [m] = {:.3g}'.format(output['param']['columns']['ParameterValue'][inds[0]]))
    #inds = np.where(np.logical_and(
        #output['param']['columns']['ElementParameter'] == 'L',
        #output['param']['columns']['ElementName'] == 'OSHORT1'))[0]
    #print('L (Short Str.) [m] = {:.3g}'.format(output['param']['columns']['ParameterValue'][inds[0]]))


    return dict(fignums=fignums, sel_data=sel_data)

def summarize_nonlin(
    LTE_filepath, use_beamline_cell='CELL', N_KICKS=None, use_beamline_ring='RING',
    zero_sext_LTE_filepath='', sext_names=('SM1', 'SM2'),
    plot_fm_xy=True, plot_fm_px=True, plot_cmap_xy=True, plot_cmap_px=True,
    plot_tswa=True, plot_nonlin_chrom=True, chrom_deltalim=None,
    ntasks=100, title=None):

    fignums = {}

    print_lines = []

    if N_KICKS is None:
        N_KICKS = dict(KQUAD=40, KSEXT=40, CSBEND=40)

    #E_MeV = 3e3
    #output_filepath = 'test.pgz'

    #pe.calc_ring_twiss(
        #output_filepath, LTE_filepath, E_MeV, radiation_integrals=True,
        #element_divisions=10, use_beamline=use_beamline_cell)
    #pe.plot_twiss(output_filepath, print_scalars=[],
                  #disp_elem_names=dict(bends=True, quads=True, sexts=True, font_size=8, extra_dy_frac=0.05))

    #twi = pe.util.load_pgz_file(output_filepath)['data']['twi']

    #print_lines.append('eps_x [pm] = {:.3f}'.format(twi['scalars']['ex0'] * 1e12))
    #print_lines.append('J_x = {:.3f}'.format(twi['scalars']['Jx']))

    #print_lines.append('beta (x, y) @ straight center [m] = ({:.2f}, {:.2f})'.format(
        #twi['arrays']['betax'][0], twi['arrays']['betay'][0]))

    #try:
        #mdisp_inds = np.where(twi['arrays']['ElementName'] == 'MDISP')[0]
        #dphix = np.diff(twi['arrays']['psix'][mdisp_inds])[0]
        #dphiy = np.diff(twi['arrays']['psiy'][mdisp_inds])[0]

        #print_lines.append('Phase Adv. btw. Disp. Sec. Centers (x, y) = ({:.6f}, {:.6f})'.format(
            #dphix / (2 * np.pi), dphiy / (2 * np.pi)))
    #except:
        #pass


    #if zero_sext_LTE_filepath != '':
        #pe.calc_ring_twiss(
            #output_filepath, zero_sext_LTE_filepath, E_MeV, radiation_integrals=True,
            #use_beamline=use_beamline_ring)

        #twi = pe.util.load_pgz_file(output_filepath)['data']['twi']

        #print_lines.append('Ring Natural Chrom. (x, y) = ({:+.3f}, {:+.3f})'.format(
            #twi['scalars']['dnux/dp'], twi['scalars']['dnuy/dp']))

    #pe.calc_ring_twiss(
        #output_filepath, LTE_filepath, E_MeV, radiation_integrals=True,
        #use_beamline=use_beamline_ring)

    #twi = pe.util.load_pgz_file(output_filepath)['data']['twi']

    #print_lines.append('Ring Corrected Chrom. (x, y) = ({:+.3f}, {:+.3f})'.format(
        #twi['scalars']['dnux/dp'], twi['scalars']['dnuy/dp']))

    #print_lines.append('Ring Tunes (x, y) = ({:.3f}, {:.3f})'.format(
        #twi['scalars']['nux'], twi['scalars']['nuy']))

    #print_lines.append('Momentum Compaction = {:3g}'.format(twi['scalars']['alphac']))
    #print_lines.append('U0 [keV] = {:3g}'.format(twi['scalars']['U0'] * 1e3))
    #print_lines.append('dE/E = {:3e}'.format(twi['scalars']['Sdelta0']))

    #try:
        #sext_names = [_name.upper() for _name in sext_names]
        #params = pe.util.load_pgz_file(output_filepath)['data']['parameters']
        #K2_vals = np.unique([val for name, param_name, val in
            #zip(params['arrays']['ElementName'], params['arrays']['ElementParameter'],
                #params['arrays']['ParameterValue'])
            #if name in sext_names and param_name == 'K2'])
        #print_lines.append('K2s = ({})'.format(', '.join(['{:+.1f}'.format(K2) for K2 in K2_vals])))
    #except:
        #pass

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
                use_beamline=use_beamline_ring, N_KICKS=N_KICKS,
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
                use_beamline=use_beamline_ring, N_KICKS=N_KICKS,
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
                use_beamline=use_beamline_ring, N_KICKS=N_KICKS,
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
                use_beamline=use_beamline_ring, N_KICKS=N_KICKS,
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

def fill_document(doc):
    """Add a section, a subsection and some text to the document.

    :param doc: the document
    :type doc: :class:`pylatex.document.Document` instance
    """
    with doc.create(plx.Section('A section')):
        doc.append('Some regular text and some ')
        doc.append(plx.utils.italic('italic text. '))

        with doc.create(plx.Subsection('A subsection')):
            doc.append('Also some crazy characters: $&#{}')


if __name__ == '__main__':

    if True:

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

        rootname = input_LTE_filepath.replace('.lte', '')
        pdf_file_prefix = rootname

        d = summarize_lin(
            input_LTE_filepath, use_beamline_cell='CELL', use_beamline_ring='RING',
            zeroSexts_LTE_filepath=zeroSexts_LTE_filepath, element_divisions=10,
            beta_elem_names=['M_LS', 'M_SS'],
            #phase_adv_elem_names_indexes = dict(disp=('M_LS', 0, 'M_SS', 0)),
            floor_elem_names_indexes=dict(
                SS=['M_SS', 0],
                LS=['M_LS', 1], # Must pick the 2nd index
                ),
        )
        lin_fignums = d['fignums']

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
            for fignum in fignum_list:
                pp.savefig(figure=fignum)
            pp.close()

        plt.show()

    else:

        rootname = 'CLSmod23pm_20200123T160920_moga_prod_v1_002708'

    geometry_options = {"vmargin": "1cm", "hmargin": "1cm"}
    doc = plx.Document(f'{rootname}_report', geometry_options=geometry_options,
                       documentclass='article')
    doc.preamble.append(plx.Command('usepackage', 'nopageno')) # Suppress page numbering for entire doc

    doc.preamble.append(plx.Command(
        'title', 'ELEGANT Lattice Characterization Report for ?'))
    doc.preamble.append(plx.Command('author', 'Yoshiteru Hidaka'))
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

        with doc.create(plx.Section('Tune Shift with Amplitude')):
            doc.append(f'Some placeholder.')
            doc.append(plx.Command('vspace', plx.NoEscape('-10pt')))

            for plane in ('x', 'y'):

                with doc.create(plx.Figure(position='h!')) as fig:

                    doc.append(plx.NoEscape(r'\centering'))

                    for iFig, (page, caption) in enumerate([
                        (1, MathEnglishText(
                            [GMATH.nu, ' vs. ',
                             plx.Math(inline=True, data=[f'A_{plane} (> 0)'])]).dumps_for_caption(),),
                        (3, MathEnglishText(
                            [GMATH.nu, ' vs. ',
                             plx.Math(inline=True, data=[f'A_{plane} (< 0)'])]).dumps_for_caption(),),
                        (2, MathEnglishText(
                            ['Tune footprint vs. ',
                             plx.Math(inline=True, data=[f'A_{plane} (> 0)'])]).dumps_for_caption(),),
                        (4, MathEnglishText(
                            ['Tune footprint vs. ',
                             plx.Math(inline=True, data=[f'A_{plane} (< 0)'])]).dumps_for_caption(),),
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
            t = MathEnglishText()
            t.append(GMATH.nu)
            t.append('    vs. ')
            t.append(GMATH.delta)
            doc.append(t)
        else:
            t = MathEnglishText([GMATH.nu, ' vs. ', GMATH.delta])
            doc.append(t)


    if os.path.exists(f'{rootname}.nonlin_chrom.pdf'):

        if new_page_required:
            doc.append(plx.NewPage())

        with doc.create(plx.Section('Nonlinear Chromaticity')):
            doc.append(f'Some placeholder.')
            doc.append(plx.Command('vspace', plx.NoEscape('-10pt')))
            with doc.create(plx.Figure(position='h!')) as fig:

                doc.append(plx.NoEscape(r'\centering'))

                for iPage, caption in enumerate([
                    #'Tunes vs. momentum offset',
                    #r'$\nu$ vs. $\delta$',
                    #plx.Math(data=[r'$\nu$']),
                    #plx.NoEscape(MathEnglishText([GMATH.nu, ' vs. ', GMATH.delta]).dumps()),
                    MathEnglishText([GMATH.nu, ' vs. ', GMATH.delta]).dumps_for_caption(),
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


    doc.generate_pdf(clean_tex=False)
