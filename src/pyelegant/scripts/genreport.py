import sys
import os
import numpy as np
import scipy.constants
from scipy.constants import physical_constants
import time
import datetime
import hashlib
import tempfile
import pickle
import fnmatch
from types import SimpleNamespace
import matplotlib.pylab as plt
from matplotlib.backends.backend_pdf import PdfPages
#import yaml
from ruamel import yaml
# ^ ruamel's "yaml" does NOT suffer from the PyYAML(v5.3, YAML v1.1) problem
#   that a float value in scientific notation without "." and the sign after e/E
#   is treated as a string.
from pathlib import Path
import argparse
import xlsxwriter

import pyelegant as pe
pe.disable_stdout()
pe.enable_stderr()
plx = pe.latex

GREEK = dict(
    alpha=chr(0x03b1), beta=chr(0x03b2), Delta=chr(0x0394), delta=chr(0x03b4),
    eta=chr(0x03b7), epsilon=chr(0x03b5), mu=chr(0x03bc), nu=chr(0x03bd),
    phi=chr(0x03c6), pi=chr(0x03c0), psi=chr(0x03c8), rho=chr(0x03c1),
    sigma=chr(0x03c3), tau=chr(0x03c4), theta=chr(0x03b8), xi=chr(0x03be),
)
SYMBOL = dict(
    partial=chr(0x2202), minus=chr(0x2013), # 0x2013 := "en dash"
)

def _yaml_append_map(
    com_map, key, value, eol_comment=None, before_comment=None, before_indent=0):
    """"""

    if before_comment is not None:
        lines = before_comment.splitlines()
        if len(lines) >= 2:
            before_comment = '\n'.join([_s.strip() for _s in lines])
        com_map.yaml_set_comment_before_after_key(
            key, before=before_comment, indent=before_indent)

    com_map.insert(len(com_map), key, value, comment=eol_comment)

def _yaml_set_comment_after_key(com_map, key, comment, indent):
    """
    # Note that an apparent bug in yaml_set_comment_before_after_key() or dump()
    # is preventing the use of "after" keyword from properly being dumped. So,
    # I am relying on the "before" keyword in this function, even though the
    # code logic may seems strange.
    """

    lines = comment.splitlines()
    if len(lines) >= 2:
        comment = '\n'.join([_s.strip() for _s in lines])

    com_map_keys = list(com_map)
    i = com_map_keys.index(key)
    next_key = com_map_keys[i+1]

    com_map.yaml_set_comment_before_after_key(
        next_key, before=comment, indent=indent)




class Report_NSLS2U_Default:
    """"""

    def __init__(self, config_filepath, user_conf=None, example_args=None):
        """Constructor"""

        self.all_nonlin_calc_types = [
            'xy_aper', 'fmap_xy', 'fmap_px', 'cmap_xy', 'cmap_px',
            'tswa', 'nonlin_chrom', 'mom_aper']
        self.all_nonlin_calc_comments = [
            'Dynamic Aperture',
            'On-Momentum Frequency Map', 'Off-Momentum Frequency Map',
            'On-Momentum Chaos Map', 'Off-Momentum Chaos Map',
            'Tune Shift with Amplitude', 'Nonlinear Chromaticity',
            'Momentum Aperture',
        ]
        assert len(self.all_nonlin_calc_types) == len(self.all_nonlin_calc_comments)

        if user_conf:

            self.config_filepath = config_filepath
            self.user_conf = user_conf

            report_version = self.user_conf.get('report_version', None)
            self.conf = self.get_default_config(report_version)

            self.conf.update(self.user_conf)

            # Convert ruamel's CommentedMap object into dict to throw away
            # all the comments.
            with tempfile.NamedTemporaryFile() as tmp:
                yml = yaml.YAML()
                with open(tmp.name, 'w') as f:
                    yml.dump(self.conf, f)
                yml = yaml.YAML(typ='safe')
                self.conf = yml.load(Path(tmp.name).read_text())

            twiss_plot_captions = self.calc_plot()

            self.build(twiss_plot_captions)

        else:
            full_or_min, example_report_version = example_args

            if full_or_min == 'full':
                conf = self.get_default_config(example_report_version, example=True)
            elif full_or_min == 'min':
                conf = self.get_default_config(example_report_version, example=False)
            else:
                raise ValueError('"full_or_min" must be either `True` or `False`')

            yml = yaml.YAML()
            yml.preserve_quotes = True
            yml.width = 70
            yml.boolean_representation = ['False', 'True']

            #yml.dump(conf, sys.stdout)

            with open(config_filepath, 'w') as f:
                yml.dump(conf, f)

    def calc_plot(self):
        """"""

        orig_pe_stdout_enabled = pe.std_print_enabled['out']

        if self.conf.get('enable_pyelegant_stdout', False):
            pe.enable_stdout()
        else:
            pe.disable_stdout()

        self.set_up_lattice()

        self.get_lin_data()

        replot_lattice_props = self.conf['lattice_props'].get('replot', False)

        twiss_pdf_filepath = os.path.join(self.report_folderpath, 'twiss.pdf')
        if (not os.path.exists(twiss_pdf_filepath)) or replot_lattice_props:
            twiss_plot_captions = self.plot_lin_props(skip_plots=False)
        else:
            twiss_plot_captions = self.get_only_lin_props_plot_captions()

        flr_pdf_filepath = os.path.join(self.report_folderpath, 'floor.pdf')
        if (not os.path.exists(flr_pdf_filepath)) or replot_lattice_props:
            self.plot_geom_layout()

        if 'nonlin' in self.conf:

            do_calc, do_plot = self.determine_calc_plot_bools()

            if do_calc and any(do_calc.values()):
                self.calc_nonlin_props(do_calc)

            if do_plot and any(do_plot.values()):
                self.plot_nonlin_props(do_plot)

        if orig_pe_stdout_enabled:
            pe.enable_stdout()
        else:
            pe.disable_stdout()

        return twiss_plot_captions

    def build(self, twiss_plot_captions):
        """"""

        self.init_pdf_report()
        self.init_xlsx_report()

        self.add_xlsx_config()

        self.add_pdf_lattice_description()
        self.add_xlsx_lattice_description()

        self.add_pdf_lattice_elements()
        self.add_xlsx_lattice_elements()

        self.add_pdf_lattice_props(twiss_plot_captions)
        self.add_xlsx_lattice_props()

        self.add_xlsx_geom_layout()

        self.add_xlsx_LTE()

        self.add_xlsx_RF()

        self.add_xlsx_lifetime()

        self.doc.append(plx.ClearPage())

        self.add_pdf_nonlin()
        self.add_xlsx_nonlin()

        plx.generate_pdf_w_reruns(self.doc, clean_tex=False, silent=False)
        self.workbook.close()


    def _convert_multiline_to_oneline(self, multiline_str):
        """"""

        return ' '.join([s.strip() for s in multiline_str.splitlines()
                         if s.strip()])

    def init_pdf_report(self):
        """"""

        conf = self.conf
        report_folderpath = self.report_folderpath
        rootname = self.rootname

        geometry_options = {"vmargin": "1cm", "hmargin": "1.5cm"}
        doc = plx.Document(
            os.path.join(report_folderpath, f'{rootname}_report'),
            geometry_options=geometry_options, documentclass='article')
        doc.preamble.append(plx.Command('usepackage', 'nopageno')) # Suppress page numbering for entire doc
        doc.preamble.append(plx.Package('indentfirst')) # This fixes the problem of the first paragraph not indenting
        doc.preamble.append(plx.Package('seqsplit')) # To split a very long word into multiple lines w/o adding hyphens, like a long file name.
        doc.preamble.append(plx.Command(
            'graphicspath', plx.NoEscape('{'+os.path.abspath(report_folderpath)+'}')))
        # To allow LaTeX to be able to find PDF files in the report folder

        doc.preamble.append(plx.Command(
            'title', 'ELEGANT Lattice Characterization Report'))

        if 'report_author' in conf:
            doc.preamble.append(plx.Command('author', conf['report_author']))

        doc.preamble.append(plx.Command('date', plx.NoEscape(r'\today')))

        doc.append(plx.NoEscape(r'\maketitle'))

        self.doc = doc

    def init_xlsx_report(self):
        """"""

        self.workbook = xlsxwriter.Workbook(
            os.path.join(self.report_folderpath, f'{self.rootname}_report.xlsx'),
            options={'nan_inf_to_errors': True})

        self._build_workbook_formats()

        worksheets = {}
        for ws_key, ws_label in [
            ('lat_params', 'Lattice Parameters'),
            ('mag_params', 'Magnet Parameters'), ('nonlin', 'Nonlinear'),
            ('elems_twiss', 'Elements & Twiss'), ('layout', 'Layout'),
            ('rf', 'RF'), ('report_config', 'Report Config'), ('lte', 'LTE'),
        ]:
            worksheets[ws_key] = self.workbook.add_worksheet(ws_label)

        worksheets['lat_params'].activate()

        self.worksheets = worksheets

        # Placeholder for copying from a worksheet to antoerh sheet
        self.xlsx_map = dict(lat_params={})

        self.lattice_props = {}
        # key   := Excel defined name
        # value := corresponding property value

    def _build_workbook_formats(self):
        """"""

        wb = self.workbook

        default_font_name = 'Times New Roman'

        # Change default cell format
        wb.formats[0].set_font_size(11)
        wb.formats[0].set_font_name(default_font_name)

        wb_txt_fmts = SimpleNamespace()

        wb_txt_fmts.normal = wb.add_format()
        wb_txt_fmts.normal_center = wb.add_format(
            {'align': 'center', 'valign': 'vcenter'})
        wb_txt_fmts.normal_center_wrap = wb.add_format(
            {'align': 'center', 'valign': 'vcenter', 'text_wrap': True})
        wb_txt_fmts.normal_center_border = wb.add_format(
            {'align': 'center', 'valign': 'vcenter', 'border': 1})

        wb_txt_fmts.wrap = wb.add_format({'text_wrap': True})
        wb_txt_fmts.bold_wrap = wb.add_format({'bold': True, 'text_wrap': True})

        wb_txt_fmts.bold_top = wb.add_format({'bold': True, 'align': 'top'})

        wb_txt_fmts.bold = wb.add_format({'bold': True})
        wb_txt_fmts.italic = wb.add_format({'italic': True})
        wb_txt_fmts.bold_italic = wb.add_format({'bold': True, 'italic': True})
        wb_txt_fmts.bold_underline = wb.add_format({'bold': True, 'underline': True})

        wb_txt_fmts.sup = wb.add_format({'font_script': 1})
        wb_txt_fmts.sub = wb.add_format({'font_script': 2})
        wb_txt_fmts.italic_sup = wb.add_format({'italic': True, 'font_script': 1})
        wb_txt_fmts.italic_sub = wb.add_format({'italic': True, 'font_script': 2})
        wb_txt_fmts.bold_sup = wb.add_format({'bold': True, 'font_script': 1})
        wb_txt_fmts.bold_sub = wb.add_format({'bold': True, 'font_script': 2})
        wb_txt_fmts.bold_italic_sup = wb.add_format(
            {'bold': True, 'italic': True, 'font_script': 1})
        wb_txt_fmts.bold_italic_sub = wb.add_format(
            {'bold': True, 'italic': True, 'font_script': 2})

        wb_num_fmts = {}
        for spec in [
            '0.0', '0.00', '0.000', '0.0000', '0.000000', '0.00E+00', '###']:
            wb_num_fmts[spec] = wb.add_format({'num_format': spec})
        wb_num_fmts['bg_yellow_0.0000'] = wb.add_format({
            'num_format': '0.0000', 'bg_color': 'yellow'})

        for k in list(wb_txt_fmts.__dict__):
            fmt = getattr(wb_txt_fmts, k)
            fmt.set_font_name(default_font_name)

        for fmt in wb_num_fmts.values():
            fmt.set_font_name(default_font_name)

        wb_num_fmts['mm/dd/yyyy'] = wb.add_format(
            {'num_format': 'mm/dd/yyyy', 'align': 'left',
             'font_name': default_font_name})

        # From here on, define non-default fonts
        wb_txt_fmts.courier = wb.add_format({'font_name': 'Courier New'})
        wb_txt_fmts.courier_wrap = wb.add_format(
            {'font_name': 'Courier New', 'text_wrap': True})

        self.wb_txt_fmts = wb_txt_fmts
        self.wb_num_fmts = wb_num_fmts

    def add_pdf_lattice_description(self):
        """"""

        doc = self.doc

        input_LTE_filepath = self.input_LTE_filepath
        conf = self.conf

        with doc.create(plx.Section('Lattice Description')):

            mod_LTE_filename = \
                os.path.basename(input_LTE_filepath).replace("_", r"\_")
            ver_str = pe.__version__["PyELEGANT"]

            default_paragraph = plx.NoEscape(
                (f'The lattice file being analyzed here is '
                 f'\seqsplit{{"{mod_LTE_filename}"}}. This report was generated using '
                 f'PyELEGANT version {ver_str}.'))
            doc.append(default_paragraph)

            custom_paragraphs = conf['report_paragraphs'].get('lattice_description', [])
            for para in custom_paragraphs:
                doc.append(plx.NewParagraph())
                doc.append(plx.NoEscape(para.strip()))

    def add_pdf_lattice_elements(self):
        """"""

        doc = self.doc

        with doc.create(plx.Section('Lattice Elements')):
            self.add_pdf_L2_bend_elements()
            self.add_pdf_L2_quad_elements()
            self.add_pdf_L2_sext_elements()
            self.add_pdf_L2_oct_elements()

            self.add_pdf_L2_beamline_elements_list()

    def add_pdf_L2_bend_elements(self):
        """"""

        doc = self.doc
        elem_defs = self.lin_data['elem_defs']

        if elem_defs['bends']:
            d = elem_defs['bends']

            with doc.create(plx.Subsection('Bend Elements')):
                ncol = 6
                table_spec = ' '.join(['l'] * ncol)
                with doc.create(plx.LongTable(table_spec)) as table:
                    table.add_hline()
                    table.add_row([
                        'Name', plx.MathText('L')+' [m]',
                        plx.MathText(r'\theta_{\mathrm{bend}}') + ' [mrad]',
                        plx.MathText(r'\theta_{\mathrm{in}}') + ' [mrad]',
                        plx.MathText(r'\theta_{\mathrm{out}}') + ' [mrad]',
                        plx.MathText('K_1\ [\mathrm{m}^{-2}]')])
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
                            k, plx.MathText(f'{L:.3f}'), plx.MathText(f'{angle:+.3f}'),
                            plx.MathText(f'{e1:+.3f}'), plx.MathText(f'{e2:+.3f}'),
                            plx.MathText(f'{K1:+.4g}')])

    def add_pdf_L2_quad_elements(self):
        """"""

        doc = self.doc
        elem_defs = self.lin_data['elem_defs']

        if elem_defs['quads']:
            d = elem_defs['quads']

            with doc.create(plx.Subsection('Quadrupole Elements')):
                ncol = 3
                table_spec = ' '.join(['l'] * ncol)
                with doc.create(plx.LongTable(table_spec)) as table:
                    table.add_hline()
                    table.add_row([
                        'Name', plx.MathText('L')+' [m]',
                        plx.MathText('K_1\ [\mathrm{m}^{-2}]')])
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
                            k, plx.MathText(f'{L:.3f}'), plx.MathText(f'{K1:+.4g}')])

    def add_pdf_L2_sext_elements(self):
        """"""

        doc = self.doc
        elem_defs = self.lin_data['elem_defs']

        if elem_defs['sexts']:
            d = elem_defs['sexts']

            with doc.create(plx.Subsection('Sextupole Elements')):
                ncol = 3
                table_spec = ' '.join(['l'] * ncol)
                with doc.create(plx.LongTable(table_spec)) as table:
                    table.add_hline()
                    table.add_row([
                        'Name', plx.MathText('L')+' [m]',
                        plx.MathText('K_2\ [\mathrm{m}^{-3}]')])
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
                            k, plx.MathText(f'{L:.3f}'), plx.MathText(f'{K2:+.4g}')])

    def add_pdf_L2_oct_elements(self):
        """"""

        doc = self.doc
        elem_defs = self.lin_data['elem_defs']

        if elem_defs['octs']:
            d = elem_defs['octs']

            with doc.create(plx.Subsection('Octupole Elements')):
                ncol = 3
                table_spec = ' '.join(['l'] * ncol)
                with doc.create(plx.LongTable(table_spec)) as table:
                    table.add_hline()
                    table.add_row([
                        'Name', plx.MathText('L')+' [m]',
                        plx.MathText('K_3\ [\mathrm{m}^{-4}]')])
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
                        L, K3 = d[k]['L'], d[k]['K3']
                        val_str = f'{K3:+.4g}'
                        if 'e' in val_str:
                            val_str = pe.util.pprint_sci_notation(K3, '.4e')
                        table.add_row([
                            k, plx.MathText(f'{L:.3f}'), plx.MathText(val_str)])

    def add_pdf_L2_beamline_elements_list(self):
        """"""

        doc = self.doc
        flat_elem_s_name_type_list = self.lin_data['flat_elem_s_name_type_list']

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
            base_header = [plx.MathText('s')+' [m]', 'Element Name', 'Element Type']
            header_list = []
            header_list.extend(base_header)
            start_ind = 1
            for i, _spec in enumerate(table_spec.split()):
                if _spec == 'c':
                    end_ind = i
                    hline_col_ind_ranges.append((start_ind, end_ind))
                    start_ind = i + 2

                    header_list.append('   ')
                    header_list.extend(base_header)
            end_ind = ncol
            hline_col_ind_ranges.append((start_ind, end_ind))

            with doc.create(plx.LongTable(table_spec)) as table:
                for ind_range in hline_col_ind_ranges:
                    table.add_hline(start=ind_range[0], end=ind_range[1])
                table.add_row(header_list)
                for ind_range in hline_col_ind_ranges:
                    table.add_hline(start=ind_range[0], end=ind_range[1])
                table.end_table_header()
                for ind_range in hline_col_ind_ranges:
                    table.add_hline(start=ind_range[0], end=ind_range[1])
                table.add_row((
                    plx.MultiColumn(ncol, align='r', data='Continued onto Next Page'),))
                for ind_range in hline_col_ind_ranges:
                    table.add_hline(start=ind_range[0], end=ind_range[1])
                table.end_table_footer()
                for ind_range in hline_col_ind_ranges:
                    table.add_hline(start=ind_range[0], end=ind_range[1])
                table.end_table_last_footer()

                for line in folded_list:
                    table.add_row([_s.strip().replace('#', '') for _s in line.split(':')])

    def add_pdf_lattice_props(self, twiss_plot_captions):
        """"""

        doc = self.doc
        conf = self.conf
        lin_data = self.lin_data
        report_folderpath = self.report_folderpath

        table_order = conf['lattice_props'].get('pdf_table_order', None)
        if table_order is None:

            table_order = [
                'E_GeV', # Beam energy
                'eps_x', # Natural horizontal emittance
                'J', # Damping partitions
                'tau', # Damping times
                'nu', # Ring tunes
                'ksi_nat', # Natural chromaticities
                'ksi_cor', # Corrected chromaticities
                'alphac', # Momentum compaction
                'U0', # Energy loss per turn
                'sigma_delta', # Energy spread
                ['req_props', 'beta', 'LS'],
                ['req_props', 'beta', 'SS'],
                'max_beta', # Max beta functions
                'min_beta', # Min beta functions
                'max_min_etax', # Max & Min etax
                ['req_props', 'length', 'LS'],
                ['req_props', 'length', 'SS'],
                'circumf', # Circumference
                ['req_props', 'floor_comparison', 'circumf_change_%'], # Circumference change [%] from Reference Lattice
                'n_periods_in_ring', # Number of super-periods for a full ring
                ['req_props', 'floor_comparison', 'LS'],
                ['req_props', 'floor_comparison', 'SS'],
                'frev' # Revolution frequency
            ]

            for spec in conf['lattice_props'].get(
                'append_opt_props_to_pdf_table', []):
                if spec not in table_order:
                    table_order.append(spec)

        with doc.create(plx.Section('Lattice Properties')):

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

                for row_spec in table_order:
                    label_symb_unit, val_str = \
                        self.get_pdf_lattice_prop_row(row_spec)
                    table.add_row([
                        plx.NoEscape(label_symb_unit), plx.NoEscape(val_str)])


            twiss_pdf_filepath = os.path.join(report_folderpath, 'twiss.pdf')

            if os.path.exists(twiss_pdf_filepath):

                custom_paragraphs = conf['report_paragraphs'].get('lattice_properties', [])
                for iPara, para in enumerate(custom_paragraphs):
                    if iPara != 0:
                        doc.append(plx.NewParagraph())
                    doc.append(para.strip())

                ver_sentence = (
                    f'ELEGANT version {lin_data["_versions"]["ELEGANT"]} was used '
                    f'to compute the lattice properties.')
                doc.append(plx.NewParagraph())
                doc.append(ver_sentence)

                doc.append(plx.VerticalSpace(plx.NoEscape('-10pt')))

                with doc.create(plx.Figure(position='h!t')) as fig:
                    doc.append(plx.NoEscape(r'\centering'))

                    for iPage, caption in enumerate(twiss_plot_captions):
                        if (np.mod(iPage, 2) == 0) and (iPage != 0):
                            doc.append(plx.LineBreak()) # This will move next 2 plots to next row
                        with doc.create(plx.SubFigureForMultiPagePDF(
                            position='b', width=plx.utils.NoEscape(r'0.5\linewidth'))) as subfig:
                            subfig.add_image(
                                os.path.basename(twiss_pdf_filepath), page=iPage+1,
                                width=plx.utils.NoEscape(r'\linewidth'))
                            doc.append(plx.VerticalSpace(plx.NoEscape('-10pt')))
                            subfig.add_caption(caption)
                    doc.append(plx.VerticalSpace(plx.NoEscape('-10pt')))
                    fig.add_caption('Twiss functions.')

    def get_pdf_lattice_prop_row(self, row_spec):
        """"""

        lin_data = self.lin_data
        opt_props = self.conf['lattice_props']['opt_props']

        k = row_spec

        if isinstance(k, list):

            if k[0] == 'req_props':
                required = True
            elif k[0] == 'opt_props':
                required = False
            else:
                raise ValueError('Invalid row specification')

            prop_name, key = k[1], k[2]
            if prop_name == 'beta':
                if required:
                    if key == 'LS':
                        location = 'Long-Straight Center'
                    elif key == 'SS':
                        location = 'Short-Straight Center'
                    else:
                        raise ValueError(f'Invalid 3rd arg for ["{k[0]}", "beta"]')
                    label = fr'$(\beta_x,\, \beta_y)$ at {location} '
                    symbol = ''
                else:
                    label = opt_props[prop_name][key]['pdf_label'] + ' '
                    symbol = ''
                unit = ' [m]'
                val = lin_data[k[0]][k[1]][k[2]]
                val_str = r'$({:.2f},\, {:.2f})$'.format(val['x'], val['y'])
            elif prop_name == 'length':
                if required:
                    if key == 'LS':
                        location = 'Long Straight'
                    elif key == 'SS':
                        location = 'Short Straight'
                    else:
                        raise ValueError(f'Invalid 3rd arg for ["{k[0]}", "length"]')
                    label = f'Length of {location} '
                    symbol = fr'$L_{{\mathrm{{{k[2]}}}}}$'
                else:
                    label = opt_props[prop_name][key]['pdf_label'] + ' '
                    symbol = ''
                unit = ' [m]'
                val = lin_data[k[0]][k[1]][k[2]]['L']
                val_str = '${:.3f}$'.format(val)
            elif prop_name == 'floor_comparison':
                if required:
                    if key == 'circumf_change_%':
                        label = 'Circumference Change '
                        symbol = r'$\Delta C / C$'
                        unit = r' [\%]'
                        val = lin_data[k[0]][k[1]][k[2]]['val']
                        val_str = '${:+.3f}$'.format(val)
                    elif key in ('LS', 'SS'):
                        location = key
                        label = f'Source Point Diff. at {location} '
                        symbol = r'$(\Delta x,\, \Delta z)$'
                        unit = ' [mm]'
                        val = lin_data[k[0]][k[1]][k[2]]
                        val_str = r'$({:+.2f},\, {:+.2f})$'.format(
                            val['x'] * 1e3, val['z'] * 1e3)
                    else:
                        raise ValueError(f'Invalid 3rd arg for ["{k[0]}", "floor_comparison"]')
                else:
                    label = opt_props[prop_name][key]['pdf_label'] + ' '
                    symbol = ''
                    unit = ' [mm]'
                    val = lin_data[k[0]][k[1]][k[2]]
                    val_str = r'$({:+.2f},\, {:+.2f})$'.format(
                        val['x'] * 1e3, val['z'] * 1e3)
            elif prop_name == 'phase_adv':
                if required:
                    raise ValueError('"phase_adv" must be under "opt_props", NOT "req_props".')
                else:
                    label = opt_props[prop_name][key]['pdf_label'] + ' '
                    symbol = ''
                    unit = r' $[2\pi]$'
                    val = lin_data[k[0]][k[1]][k[2]]
                    val_str = r'$({:.6f},\, {:.6f})$'.format(val['x'], val['y'])
            else:
                raise ValueError(f'Invalid 2nd arg for ["{k[0]}"]: {prop_name}')

        elif k == 'E_GeV':
            label, symbol, unit = 'Beam Energy ', '$E$', ' [GeV]'
            val_str = f'${lin_data[k]:.0f}$'
        elif k == 'eps_x':
            label = 'Natural Horizontal Emittance '
            symbol = r'$\epsilon_x$'
            unit = ' [pm-rad]'
            val_str = '${:.1f}$'.format(lin_data[k] * 1e12)
        elif k == 'J':
            label = 'Damping Partitions '
            symbol = r'$(J_x,\, J_y,\, J_{\delta})$'
            unit = ''
            val_str = r'$({:.2f},\, {:.2f},\, {:.2f})$'.format(
                    lin_data['Jx'], lin_data['Jy'], lin_data['Jdelta'])
        elif k == 'tau':
            label = 'Damping Times '
            symbol = r'$(\tau_x,\, \tau_y,\, \tau_{\delta})$'
            unit = ' [ms]'
            val_str = r'$({:.2f},\, {:.2f},\, {:.2f})$'.format(
                    lin_data['taux'] * 1e3, lin_data['tauy'] * 1e3,
                    lin_data['taudelta'] * 1e3)
        elif k == 'nu':
            label = 'Ring Tunes '
            symbol = r'$(\nu_x,\, \nu_y)$'
            unit = ''
            val_str = r'$({:.3f},\, {:.3f})$'.format(
                lin_data['nux'], lin_data['nuy'])
        elif k == 'ksi_nat':
            label = 'Natural Chromaticities '
            symbol = r'$(\xi_x^{\mathrm{nat}},\, \xi_y^{\mathrm{nat}})$'
            unit = ''
            val_str = r'$({:+.3f},\, {:+.3f})$'.format(
                lin_data['ksi_x_nat'], lin_data['ksi_y_nat'])
        elif k == 'ksi_cor':
            label = 'Corrected Chromaticities '
            symbol = r'$(\xi_x^{\mathrm{cor}},\, \xi_y^{\mathrm{cor}})$'
            unit = ''
            val_str = r'$({:+.3f},\, {:+.3f})$'.format(
                lin_data['ksi_x_cor'], lin_data['ksi_y_cor'])
        elif k == 'alphac':
            label = 'Momentum Compaction '
            symbol = r'$\alpha_c$'
            unit = ''
            val_str = '${}$'.format(
                pe.util.pprint_sci_notation(lin_data[k], '.2e'))
        elif k == 'U0':
            label = 'Energy Loss per Turn '
            symbol = r'$U_0$'
            unit = ' [keV]'
            val_str = '${:.0f}$'.format(lin_data['U0_MeV'] * 1e3)
        elif k == 'sigma_delta':
            label = 'Energy Spread '
            symbol = r'$\sigma_{\delta}$'
            unit = r' [\%]'
            val_str = '${:.3f}$'.format(lin_data['dE_E'] * 1e2)
        elif k == 'max_beta':
            label = 'max '
            symbol = r'$(\beta_x,\, \beta_y)$'
            unit = ' [m]'
            val_str = r'$({:.2f},\, {:.2f})$'.format(
                lin_data['max_betax'], lin_data['max_betay'])
        elif k == 'min_beta':
            label = 'min '
            symbol = r'$(\beta_x,\, \beta_y)$'
            unit = ' [m]'
            val_str = r'$({:.2f},\, {:.2f})$'.format(
                lin_data['min_betax'], lin_data['min_betay'])
        elif k == 'max_min_etax':
            label = r'$\eta_x$' + ' (min, max)'
            symbol = ''
            unit = ' [mm]'
            val_str = r'$({:+.1f},\, {:+.1f})$'.format(
                lin_data['min_etax'] * 1e3, lin_data['max_etax'] * 1e3)
        elif k == 'circumf':
            label = 'Circumference '
            symbol = r'$C$'
            unit = ' [m]'
            val_str = '${:.3f}$'.format(lin_data[k])
        elif k == 'n_periods_in_ring':
            label = 'Number of Super-periods'
            symbol, unit = '', ''
            val_str = '${:d}$'.format(lin_data[k])
        elif k == 'f_rev':
            label = 'Revolution Frequency '
            symbol = '$f_{\mathrm{rev}}$'
            unit = ' [kHz]'
            val_str = '${:.3f}$'.format(
                scipy.constants.c / lin_data['circumf'] / 1e3)
        elif k == 'T_rev':
            label = 'Revolution Period '
            symbol = '$T_{\mathrm{rev}}$'
            unit = ' [$\mu$s]'
            f_rev = scipy.constants.c / lin_data['circumf'] # [Hz]
            T_rev = 1.0 / f_rev # [s]
            val_str = '${:.3f}$'.format(T_rev * 1e6)
        else:
            raise RuntimeError(f'Unhandled "pdf_table_order" key: {k}')

        return label + symbol + unit, val_str

    def add_pdf_nonlin(self):
        """"""

        doc = self.doc
        ncf = self.conf['nonlin']
        report_folderpath = self.report_folderpath

        nonlin_data_filepaths = self.get_nonlin_data_filepaths()
        included_types = [k for k, _included in ncf.get('include', {}).items()
                          if _included]
        plots_pdf_paths = {k: os.path.join(report_folderpath, f'{k}.pdf')
                           for k in included_types}

        new_page_required = False

        if 'xy_aper' in included_types:
            self.add_pdf_L2_xy_aper(plots_pdf_paths, nonlin_data_filepaths)

        if ('fmap_xy' in included_types) or ('fmap_px' in included_types):

            self.add_pdf_L2_fmap(plots_pdf_paths, nonlin_data_filepaths)
            new_page_required = True

        if ('cmap_xy' in included_types) or ('cmap_px' in included_types):

            if new_page_required:
                doc.append(plx.ClearPage())

            self.add_pdf_L2_cmap(plots_pdf_paths, nonlin_data_filepaths)
            new_page_required = True

        if 'tswa' in included_types:

            if new_page_required:
                doc.append(plx.ClearPage())

            with open(self.suppl_plot_data_filepath['tswa'], 'rb') as f:
                tswa_plot_captions, tswa_data = pickle.load(f)

            self.add_pdf_L2_tswa(
                plots_pdf_paths, nonlin_data_filepaths, tswa_plot_captions)
            new_page_required = True

        if 'nonlin_chrom' in included_types:

            if new_page_required:
                doc.append(plx.ClearPage())

            self.add_pdf_L2_nonlin_chrom(plots_pdf_paths, nonlin_data_filepaths)
            new_page_required = True

        if 'mom_aper' in included_types:
            self.add_pdf_L2_mom_aper(plots_pdf_paths, nonlin_data_filepaths)

    def add_pdf_L2_xy_aper(self, plots_pdf_paths, nonlin_data_filepaths):
        """"""

        doc = self.doc
        LTE_contents = self.LTE_contents
        input_LTE_filepath = self.input_LTE_filepath

        with doc.create(plx.Section('Dynamic Aperture')):
            if os.path.exists(plots_pdf_paths['xy_aper']):
                d = pe.util.load_pgz_file(nonlin_data_filepaths['xy_aper'])

                assert os.path.basename(d['input']['LTE_filepath']) \
                       == os.path.basename(input_LTE_filepath)
                assert d['input']['lattice_file_contents'] == LTE_contents

                n_turns = d['input']['n_turns']
                abs_xmax = d['input']['xmax']
                abs_ymax = d['input']['ymax']
                n_lines = d['input']['n_lines']
                ini_ndiv = d['input']['ini_ndiv']
                xmin_mm = - abs_xmax * 1e3
                xmax_mm = abs_xmax * 1e3
                if d['input']['neg_y_search']:
                    ymin_mm = - abs_ymax * 1e3
                else:
                    ymin_mm = 0.0
                ymax_mm = abs_ymax * 1e3
                xstep_um = abs_xmax / (ini_ndiv - 1) * 1e6
                ystep_um = abs_ymax / (ini_ndiv - 1) * 1e6
                para = f'''\
                The dynamic aperture was searched by tracking particles for
                {n_turns:d} turns along {n_lines:d} radial lines in the range of
                ${xmin_mm:+.1f} \le x [\mathrm{{mm}}] \le {xmax_mm:+.1f}$ and
                ${ymin_mm:+.1f} \le y [\mathrm{{mm}}] \le {ymax_mm:+.1f}$
                with initial horizontal and vertical step sizes of
                {xstep_um:.1f} and {ystep_um:.1f} $\mu$m, respectively.
                '''
                para = self._convert_multiline_to_oneline(para)
                doc.append(plx.NoEscape(para))

                ver_sentence = f'''\
                ELEGANT version {d["_version_ELEGANT"]} was used to compute the
                dynamic aperture data.
                '''
                ver_sentence = self._convert_multiline_to_oneline(ver_sentence)

                doc.append(plx.NewParagraph())
                doc.append(plx.NoEscape(ver_sentence))
                doc.append(plx.VerticalSpace(plx.NoEscape('-10pt')))
                with doc.create(plx.Figure(position='h!t')) as fig:
                    doc.append(plx.NoEscape(r'\centering'))
                    fig.add_image(os.path.basename(plots_pdf_paths['xy_aper']),
                                  width=plx.utils.NoEscape(r'0.5\linewidth'))
                    doc.append(plx.VerticalSpace(plx.NoEscape('-10pt')))
                    fig.add_caption('Dyanmic Aperture.')

    def add_pdf_L2_fmap(self, plots_pdf_paths, nonlin_data_filepaths):
        """"""

        doc = self.doc
        LTE_contents = self.LTE_contents
        input_LTE_filepath = self.input_LTE_filepath

        with doc.create(plx.Section('Frequency Map')):
            if os.path.exists(plots_pdf_paths['fmap_xy']) and \
               os.path.exists(plots_pdf_paths['fmap_px']):
                d_xy = pe.util.load_pgz_file(nonlin_data_filepaths['fmap_xy'])
                d_px = pe.util.load_pgz_file(nonlin_data_filepaths['fmap_px'])

                assert os.path.basename(d_xy['input']['LTE_filepath']) \
                       == os.path.basename(input_LTE_filepath)
                assert os.path.basename(d_px['input']['LTE_filepath']) \
                       == os.path.basename(input_LTE_filepath)
                assert d_xy['input']['lattice_file_contents'] == LTE_contents
                assert d_px['input']['lattice_file_contents'] == LTE_contents

                n_turns = d_xy['input']['n_turns']
                nx, ny = d_xy['input']['nx'], d_xy['input']['ny']
                doc.append((
                    f'The on-momentum frequency map was generated by '
                    f'tracking particles for {n_turns:d} turns at each point '
                    f'in the grid of '))
                doc.append(plx.MathText((
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
                    f'The off-momentum frequency map was generated by tracking '
                    f'particles for {n_turns:d} turns at each point in the grid of '))
                doc.append(plx.MathText((
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

                doc.append(plx.NewParagraph())
                doc.append(ver_sentence)
                doc.append(plx.VerticalSpace(plx.NoEscape('-10pt')))
                with doc.create(plx.Figure(position='h!t')) as fig:
                    doc.append(plx.NoEscape(r'\centering'))
                    for k, caption in [
                        ('fmap_xy', 'On-Momentum'), ('fmap_px', 'Off-Momentum')]:
                        with doc.create(plx.SubFigure(
                            position='b', width=plx.utils.NoEscape(r'0.5\linewidth'))
                                        ) as subfig:
                            subfig.add_image(
                                os.path.basename(plots_pdf_paths[k]),
                                width=plx.utils.NoEscape(r'\linewidth'))
                            doc.append(plx.VerticalSpace(plx.NoEscape('-10pt')))
                            subfig.add_caption(caption)
                    doc.append(plx.VerticalSpace(plx.NoEscape('-10pt')))
                    fig.add_caption('On- & off-momentum frequency maps.')

            else:
                for k, subsec_title, caption in [
                    ('fmap_xy', 'On Momentum', 'On-momentum frequency map.'),
                    ('fmap_px', 'Off Momentum', 'Off-momentum frequency map.')]:

                    d = pe.util.load_pgz_file(nonlin_data_filepaths[k])
                    ver_sentence = (
                        f'ELEGANT version {d["_version_ELEGANT"]} was used '
                        f'to compute the frequency map data.')

                    if os.path.exists(plots_pdf_paths[k]):
                        with doc.create(plx.Subsection(subsec_title)):
                            doc.append('Description for frequency maps goes here.')
                            doc.append(plx.NewParagraph())
                            doc.append(ver_sentence)

                            with doc.create(plx.Figure(position='h!t')) as fig:
                                doc.append(plx.VerticalSpace(plx.NoEscape('-10pt')))

                                fig.add_image(os.path.basename(plots_pdf_paths[k]),
                                              width=plx.utils.NoEscape(r'0.6\linewidth'))
                                fig.add_caption(caption)

    def add_pdf_L2_cmap(self, plots_pdf_paths, nonlin_data_filepaths):
        """"""

        doc = self.doc
        LTE_contents = self.LTE_contents
        input_LTE_filepath = self.input_LTE_filepath

        with doc.create(plx.Section('Chaos Map')):

            if os.path.exists(plots_pdf_paths['cmap_xy']) and \
               os.path.exists(plots_pdf_paths['cmap_px']):
                d_xy = pe.util.load_pgz_file(nonlin_data_filepaths['cmap_xy'])
                d_px = pe.util.load_pgz_file(nonlin_data_filepaths['cmap_px'])

                assert os.path.basename(d_xy['input']['LTE_filepath']) \
                       == os.path.basename(input_LTE_filepath)
                assert os.path.basename(d_px['input']['LTE_filepath']) \
                       == os.path.basename(input_LTE_filepath)
                assert d_xy['input']['lattice_file_contents'] == LTE_contents
                assert d_px['input']['lattice_file_contents'] == LTE_contents

                n_turns = d_xy['input']['n_turns']
                nx, ny = d_xy['input']['nx'], d_xy['input']['ny']
                doc.append((
                    f'The on-momentum chaos map was generated by tracking particles '
                    f'for {n_turns:d} turns at each point in the grid of '))
                doc.append(plx.MathText((
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
                    f'The off-momentum chaos map was generated by tracking particles '
                    f'for {n_turns:d} turns at each point in the grid of '))
                doc.append(plx.MathText((
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

                doc.append(plx.NewParagraph())
                doc.append(ver_sentence)
                doc.append(plx.VerticalSpace(plx.NoEscape('-10pt')))
                with doc.create(plx.Figure(position='h!t')) as fig:
                    doc.append(plx.NoEscape(r'\centering'))
                    for k, caption in [
                        ('cmap_xy', 'On-Momentum'), ('cmap_px', 'Off-Momentum')]:
                        with doc.create(plx.SubFigure(
                            position='b', width=plx.utils.NoEscape(r'0.5\linewidth'))
                                        ) as subfig:
                            subfig.add_image(
                                os.path.basename(plots_pdf_paths[k]),
                                width=plx.utils.NoEscape(r'\linewidth'))
                            doc.append(plx.VerticalSpace(plx.NoEscape('-10pt')))
                            subfig.add_caption(caption)
                    doc.append(plx.VerticalSpace(plx.NoEscape('-10pt')))
                    fig.add_caption('On- & off-momentum chaos maps.')

            else:
                for k, subsec_title, caption in [
                    ('cmap_xy', 'On Momentum', 'On-momentum chaos map.'),
                    ('cmap_px', 'Off Momentum', 'Off-momentum chaos map.')]:

                    d = pe.util.load_pgz_file(nonlin_data_filepaths[k])
                    ver_sentence = (
                        f'ELEGANT version {d["_version_ELEGANT"]} was used '
                        f'to compute the chaos map data.')

                    if os.path.exists(plots_pdf_paths[k]):
                        with doc.create(plx.Subsection(subsec_title)):
                            doc.append('Description for chaos maps goes here.')
                            doc.append(plx.NewParagraph())
                            doc.append(ver_sentence)
                            with doc.create(plx.Figure(position='h!t')) as fig:
                                doc.append(plx.VerticalSpace(plx.NoEscape('-10pt')))
                                fig.add_image(os.path.basename(plots_pdf_paths[k]),
                                              width=plx.utils.NoEscape(r'0.6\linewidth'))
                                fig.add_caption(caption)


    def add_pdf_L2_tswa(
        self, plots_pdf_paths, nonlin_data_filepaths, tswa_plot_captions):
        """"""

        doc = self.doc
        LTE_contents = self.LTE_contents
        input_LTE_filepath = self.input_LTE_filepath

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
                f'The plots for tune shift with horizontal amplitude were generated '
                f'by tracking particles for {n_turns:d} turns at each point in the '
                f'array of '))
            doc.append(plx.MathText((
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

            doc.append(plx.NewParagraph())

            doc.append((
                f'The plots for tune shift with vertical amplitude were generated by '
                f'tracking particles for {n_turns:d} turns at each point in the '
                f'array of '))
            doc.append(plx.MathText((
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

            doc.append(plx.NewParagraph())
            doc.append(ver_sentence)
            doc.append(plx.VerticalSpace(plx.NoEscape('-10pt')))

            for plane, page_caption_list in [
                ('x', tswa_plot_captions[:2]),
                ('y', tswa_plot_captions[2:])]:

                with doc.create(plx.Figure(position='h!t')) as fig:

                    doc.append(plx.NoEscape(r'\centering'))

                    for iFig, (page, caption) in enumerate(page_caption_list):
                        with doc.create(plx.SubFigureForMultiPagePDF(
                            position='b', width=plx.utils.NoEscape(r'0.5\linewidth'))
                                        ) as subfig:
                            subfig.add_image(
                                os.path.basename(plots_pdf_paths['tswa']), page=page,
                                width=plx.utils.NoEscape(r'\linewidth'))
                            doc.append(plx.VerticalSpace(plx.NoEscape('-10pt')))
                            subfig.add_caption(caption.dumps_for_caption())

                        if iFig in (1,):
                            doc.append(plx.NewLine())

                    doc.append(plx.VerticalSpace(plx.NoEscape('-10pt')))
                    fig.add_caption('Tune-shift with {} amplitude.'.format(
                        'horizontal' if plane == 'x' else 'vertical'))

            doc.append(plx.ClearPage())

    def add_pdf_L2_nonlin_chrom(self, plots_pdf_paths, nonlin_data_filepaths):
        """"""

        doc = self.doc
        LTE_contents = self.LTE_contents
        input_LTE_filepath = self.input_LTE_filepath

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
                f'The plots for nonlinear chromaticity were generated by tracking '
                f'particles for {n_turns:d} turns at each point in the array of '))
            doc.append(plx.MathText((
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

            doc.append(plx.NewParagraph())
            doc.append(ver_sentence)
            doc.append(plx.VerticalSpace(plx.NoEscape('-10pt')))
            with doc.create(plx.Figure(position='h!t')) as fig:

                doc.append(plx.NoEscape(r'\centering'))

                caption_list = [
                    (plx.MathText(r'\nu') + ' vs. ' + plx.MathText(r'\delta')
                     ).dumps_for_caption(),
                    'Off-momentum tune footprint',
                ]

                for iPage, caption in enumerate(caption_list):
                    with doc.create(plx.SubFigureForMultiPagePDF(
                        position='b', width=plx.utils.NoEscape(r'0.5\linewidth'))
                                    ) as subfig:
                        subfig.add_image(
                            os.path.basename(plots_pdf_paths['nonlin_chrom']), page=iPage+1,
                            width=plx.utils.NoEscape(r'\linewidth'))
                        doc.append(plx.VerticalSpace(plx.NoEscape('-10pt')))
                        subfig.add_caption(caption)
                doc.append(plx.VerticalSpace(plx.NoEscape('-10pt')))
                fig.add_caption('Nonlinear chromaticity.')

    def add_pdf_L2_mom_aper(self, plots_pdf_paths, nonlin_data_filepaths):
        """"""

        doc = self.doc
        LTE_contents = self.LTE_contents
        input_LTE_filepath = self.input_LTE_filepath

        with doc.create(plx.Section('Momentum Aperture')):
            if os.path.exists(plots_pdf_paths['mom_aper']):
                d = pe.util.load_pgz_file(nonlin_data_filepaths['mom_aper'])

                assert os.path.basename(d['input']['LTE_filepath']) \
                       == os.path.basename(input_LTE_filepath)
                assert d['input']['lattice_file_contents'] == LTE_contents

                n_turns = d['input']['n_turns']
                x_initial = d['input']['x_initial']
                y_initial = d['input']['y_initial']
                delta_negative_start = d['input']['delta_negative_start']
                delta_negative_limit = d['input']['delta_negative_limit']
                delta_positive_start = d['input']['delta_positive_start']
                delta_positive_limit = d['input']['delta_positive_limit']
                init_delta_step_size = d['input']['init_delta_step_size']
                s_start = d['input']['s_start']
                s_end = d['input']['s_end']
                include_name_pattern = d['input']['include_name_pattern']
                para = f'''\
                The momentum aperture was searched by tracking particles for
                {n_turns:d} turns with initial $(x, y) =
                ({x_initial*1e6:.1f}, {y_initial*1e6:.1f})\, [\mu\mathrm{{m}}]$
                for the elements with the name pattern of "{include_name_pattern}"
                in the range of $ {s_start:.3f} \le s\, [\mathrm{{m}}]\, \le
                {s_end:.3f}$. The positive momentum aperture search started from
                {delta_positive_start*1e2:+.3f}\% up to
                {delta_positive_limit*1e2:+.3f}\%, while the negative momentum
                aperture search started from {delta_negative_start*1e2:-.3f}\%
                up to {delta_negative_limit*1e2:-.3f}\%, with the initial step
                size of {init_delta_step_size*1e2:.6f}\%.
                '''
                para = self._convert_multiline_to_oneline(para)
                doc.append(plx.NoEscape(para))

                ver_sentence = f'''\
                ELEGANT version {d["_version_ELEGANT"]} was used to compute the
                dynamic aperture data.
                '''
                ver_sentence = self._convert_multiline_to_oneline(ver_sentence)

                doc.append(plx.NewParagraph())
                doc.append(plx.NoEscape(ver_sentence))
                doc.append(plx.VerticalSpace(plx.NoEscape('-10pt')))
                with doc.create(plx.Figure(position='h!t')) as fig:
                    doc.append(plx.NoEscape(r'\centering'))
                    fig.add_image(os.path.basename(plots_pdf_paths['mom_aper']),
                                  width=plx.utils.NoEscape(r'0.5\linewidth'))
                    doc.append(plx.VerticalSpace(plx.NoEscape('-10pt')))
                    fig.add_caption('Momentum Aperture.')

    def add_xlsx_config(self):
        """"""

        ws = self.worksheets['report_config']

        bold = self.wb_txt_fmts.bold
        courier = self.wb_txt_fmts.courier

        ws.set_column(0, 0, 15)
        ws.set_column(1, 1, 150)

        yml = yaml.YAML()
        yml.preserve_quotes = True
        yml.width = 110
        yml.boolean_representation = ['False', 'True']
        yml.indent(mapping=2, sequence=2, offset=0) # Default: (mapping=2, sequence=2, offset=0)

        row = 0
        ws.write(row, 0, 'User Config:', bold)

        with tempfile.NamedTemporaryFile() as tmp:
            with open(tmp.name, 'w') as f:
                yml.dump(self.user_conf, f)
            contents = Path(tmp.name).read_text()
        for line in contents.splitlines():
            ws.write(row, 1, line, courier)
            row += 1

        row += 1
        ws.write(row, 0, 'Actual Config:', bold)

        with tempfile.NamedTemporaryFile() as tmp:
            with open(tmp.name, 'w') as f:
                yml.dump(self.conf, f)
            contents = Path(tmp.name).read_text()
        for line in contents.splitlines():
            ws.write(row, 1, line, courier)
            row += 1

    def add_xlsx_lattice_description(self):
        """"""

        ws = self.worksheets['lat_params']

        input_LTE_filepath = self.input_LTE_filepath
        conf = self.conf
        wb_txt_fmts = self.wb_txt_fmts
        wb_num_fmts = self.wb_num_fmts

        description_paras = '\n'.join([
            str(para) for para in
            conf['report_paragraphs'].get('lattice_description', [])])
        property_table_notes = '\n'.join([
            str(para) for para in
            conf['report_paragraphs'].get('lattice_properties', [])])

        bold_top = wb_txt_fmts.bold_top
        bold = wb_txt_fmts.bold
        wrap = wb_txt_fmts.wrap

        courier = wb_txt_fmts.courier

        row = 0
        col = 3
        ws.set_column(col, col, 23)
        ws.set_column(col+1, col+1, 100)
        label_d = dict(
            description='Description:', table_notes='Notes for Table:',
            LTE_filepath='LTE filepath:',
            hash='LTE Hash (SHA-1):', keywords='Keywords:',
            lat_author='LTE Created by:', lat_recv_date='Date LTE Received:',
            elegant_version='ELEGANT Version:',
            pyelegant_version='PyELEGANT Version:',
            report_class='Report Class:', report_version='Report Version:',
            report_author='Report Created by:',
            orig_LTE_filepath='Orig. LTE filepath:',
        )
        for k, label in label_d.items():
            if k in ('description', 'table_notes'):
                ws.write(row, col, label, bold_top)
            else:
                ws.write(row, col, label, bold)

            if k == 'description':
                ws.write(row, col+1, description_paras.strip(), wrap)
            elif k == 'table_notes':
                ws.write(row, col+1, property_table_notes.strip(), wrap)

            elif k == 'hash':
                sha = hashlib.sha1()
                sha.update(Path(input_LTE_filepath).read_text().encode('utf-8'))
                LTE_SHA1 = sha.hexdigest()

                ws.write(row, col+1, LTE_SHA1, courier)

            elif k == 'keywords':
                keywords = ', '.join(conf.get('lattice_keywords', []))
                if keywords:
                    ws.write(row, col+1, keywords)

            elif k == 'lat_author':
                author = conf.get('lattice_author', '')
                if author:
                    ws.write(row, col+1, author)
            elif k == 'report_author':
                author = conf.get('report_author', '')
                if author:
                    ws.write(row, col+1, author)

            elif k == 'lat_recv_date':
                date_str = conf.get('lattice_received_date', '')
                if date_str:
                    try:
                        datenum = datetime.datetime.strptime(date_str, '%m/%d/%Y')
                    except:
                        raise ValueError((
                            'Invalid "lattice_received_date". '
                            'Must be in the fomrat "%m/%d/%Y"'))

                    ws.write(row, col+1, datenum, wb_num_fmts['mm/dd/yyyy'])

            elif k == 'LTE_filepath':
                ws.write(row, col+1, input_LTE_filepath.strip())

            elif k == 'elegant_version':
                ws.write(row, col+1, pe.__version__['ELEGANT'])
            elif k == 'pyelegant_version':
                ws.write(row, col+1, pe.__version__['PyELEGANT'])
            elif k == 'report_class':
                ws.write(row, col+1, conf['report_class'])
            elif k == 'report_version':
                ws.write(row, col+1, conf['report_version'])

            elif k == 'orig_LTE_filepath':
                orig_LTE_fp = conf.get('orig_LTE_filepath', '')
                if orig_LTE_fp:
                    ws.write(row, col+1, orig_LTE_fp.strip())

            else:
                raise ValueError()

            row += 1

    def add_xlsx_lattice_elements(self):
        """"""

        self.add_xlsx_L2_magnet_params()
        self.add_xlsx_L2_beamline_elements_list()

    def add_xlsx_L2_magnet_params(self):
        """"""

        ws = self.worksheets['mag_params']

        lin_data = self.lin_data

        elem_defs = lin_data['elem_defs']

        mag_col_names = [
            'name', 'L', 'K1', 'K2', 'K3', 'US_space', 'DS_space', 'aperture',
            'theta', 'E1', 'E2', 'B', 'rho']
        mag_col_inds = {name: i for i, name in enumerate(mag_col_names)}

        next_row = 0

        excel_elem_list = lin_data['excel_elem_list']
        elem_type_ind = excel_elem_list[0].index('Element Type')
        mod_excel_elem_list = [excel_elem_list[0]] + [
            v for v in excel_elem_list[1:] if v[elem_type_ind] != 'MARK']

        next_row = self.add_xlsx_L3_bend_elements(
            next_row, elem_defs, mag_col_inds, mod_excel_elem_list,
            lin_data['E_GeV'])

        next_row = self.add_xlsx_L3_quad_elements(
            next_row, elem_defs, mag_col_inds, mod_excel_elem_list)

        next_row = self.add_xlsx_L3_sext_elements(
            next_row, elem_defs, mag_col_inds, mod_excel_elem_list)

        next_row = self.add_xlsx_L3_oct_elements(
            next_row, elem_defs, mag_col_inds, mod_excel_elem_list)

        # Adjust column widths
        max_elem_name_width = max([len('Name')] + [
            len(k) for k in list(elem_defs['bends']) + list(elem_defs['quads'])
            + list(elem_defs['sexts']) + list(elem_defs['octs'])])
        c = mag_col_inds['name']
        ws.set_column(c, c, max_elem_name_width + 4)
        c = mag_col_inds['theta']
        ws.set_column(c, c, 11)
        c = mag_col_inds['E1']
        ws.set_column(c, c, 10)
        c = mag_col_inds['E2']
        ws.set_column(c, c, 10)
        c = mag_col_inds['B']
        ws.set_column(c, c, 7)
        c = mag_col_inds['rho']
        ws.set_column(c, c, 13)
        c = mag_col_inds['US_space']
        ws.set_column(c, c, 12)
        c = mag_col_inds['DS_space']
        ws.set_column(c, c, 12)
        c = mag_col_inds['aperture']
        ws.set_column(c, c, 15)

    def _get_US_DS_drift_space(self, elem_name, excel_elem_list):
        """"""

        debug = True

        elem_name_ind = excel_elem_list[0].index('Element Name')
        elem_type_ind = excel_elem_list[0].index('Element Type')
        L_ind = excel_elem_list[0].index('L (m)')
        flat_elem_names = np.array([v[elem_name_ind] for v in excel_elem_list[1:]])

        start_inds, end_inds = [], []
        start = None
        for elem_ind in np.where(flat_elem_names == elem_name)[0]:
            if start is None:
                start = elem_ind
                start_inds.append(start)
            else:
                if start + 1 == elem_ind:
                    start = elem_ind
                else:
                    end_inds.append(start)
                    start = elem_ind
                    start_inds.append(start)
        end_inds.append(elem_ind)
        assert len(start_inds) == len(end_inds)

        if debug:
            print(f'Element Name: {elem_name}')

        magnet_elem_types = (
            'SBEN', 'SBEND', 'RBEN', 'RBEND', 'CSBEND',
            'QUAD', 'KQUAD', 'SEXT', 'KSEXT', 'OCTU', 'KOCT')

        us_drifts, ds_drifts = [], []
        for si, ei in zip(start_inds, end_inds):
            if debug:
                _elem_type_list = [excel_elem_list[si + 1][elem_type_ind]] # +1 to exclude header
            si -= 1 # to look at upstream element
            ds = 0.0
            while si > 0:
                elem = excel_elem_list[si + 1] # +1 to exclude header
                elem_type = elem[elem_type_ind]
                if debug:
                    _elem_type_list.append(elem_type)
                if elem_type in magnet_elem_types:
                    break
                else:
                    ds += elem[L_ind]
                si -= 1
            us_drifts.append(ds)
            if debug:
                print('US drift: ' + ','.join(_elem_type_list))

            if debug:
                _elem_type_list = [excel_elem_list[ei + 1][elem_type_ind]] # +1 to exclude header
            ei += 1 # to look at downstream element
            ds = 0.0
            while ei + 1 < len(excel_elem_list):
                elem = excel_elem_list[ei + 1] # +1 to exclude header
                elem_type = elem[elem_type_ind]
                if debug:
                    _elem_type_list.append(elem_type)
                if elem_type in magnet_elem_types:
                    break
                else:
                    ds += elem[L_ind]
                ei += 1
            ds_drifts.append(ds)
            if debug:
                print('DS drift: ' + ','.join(_elem_type_list))

        if debug:
            print('US drift unique lengths:')
            print(np.unique(us_drifts))
            print('DS drift unique lengths:')
            print(np.unique(ds_drifts))

        return us_drifts, ds_drifts

    def add_xlsx_L3_bend_elements(
        self, next_row, elem_defs, mag_col_inds, excel_elem_list, E_GeV):
        """"""

        if not elem_defs['bends']:
            return next_row

        ws = self.worksheets['mag_params']

        wb_txt_fmts = self.wb_txt_fmts
        wb_num_fmts = self.wb_num_fmts

        bold = wb_txt_fmts.bold
        bold_underline = wb_txt_fmts.bold_underline
        bold_italic = wb_txt_fmts.bold_italic
        bold_sup = wb_txt_fmts.bold_sup
        bold_sub = wb_txt_fmts.bold_sub

        wrap = wb_txt_fmts.wrap
        bold_wrap = wb_txt_fmts.bold_wrap

        ws.write(next_row, 0, 'Bending Magnets', bold_underline)
        next_row += 1

        # Write headers
        for col_name, fragments in [
            ['name', (bold, 'Name')],
            ['L', (bold_italic, 'L', bold, ' (m)')],
            ['theta',
             (bold_italic, GREEK['theta'], bold_sub, 'bend', bold, ' (mrad)')],
            ['E1',
             (bold_italic, GREEK['theta'], bold_sub, 'in', bold, ' (mrad)')],
            ['E2',
             (bold_italic, GREEK['theta'], bold_sub, 'out', bold, ' (mrad)')],
            ['K1',
             (bold_italic, 'K', bold_sub, '1', bold, ' (m', bold_sup, '-2', ')')],
            ['B', (bold_italic, 'B', bold, ' (T)')],
            ['rho',
             (bold, 'Bending\nRadius ', bold_italic, GREEK['rho'], bold, ' (m)',
              wrap)],
            ['US_space', (bold_wrap, 'Min. US\nSpace (mm)')],
            ['DS_space', (bold_wrap, 'Min. DS\nSpace (mm)')],
            ['aperture', (bold_wrap, 'Magnet\nAperture (mm)')],
            ]:

            col = mag_col_inds[col_name]

            if len(fragments) > 2:
                ws.write_rich_string(next_row, col, *fragments)
            elif len(fragments) == 2:
                ws.write(next_row, col, fragments[1], fragments[0])
            elif len(fragments) == 1:
                ws.write(next_row, col, fragments[0])
            else:
                raise ValueError()

            if col_name == 'US_space':
                ws.write_comment(next_row, col, 'US := Upstream')
            elif col_name == 'DS_space':
                ws.write_comment(next_row, col, 'DS := Downstream')
        next_row += 1

        U0_GeV = physical_constants['electron mass energy equivalent in MeV'][0] / 1e3
        Brho = 1e9 / scipy.constants.c * np.sqrt(E_GeV**2 - U0_GeV**2) # [T-m]

        fmt = dict(
            L=wb_num_fmts['0.000'], mrad=wb_num_fmts['0.000'],
            K1=wb_num_fmts['0.000'], B=wb_num_fmts['0.00'], rho=wb_num_fmts['0.00'],
            space=wb_num_fmts['0.0'])

        d = elem_defs['bends']

        m = mag_col_inds

        for k in sorted(list(d)):
            L, angle, e1, e2, K1 = (
                d[k]['L'], d[k]['ANGLE'], d[k]['E1'], d[k]['E2'], d[k]['K1'])
            rho = L / angle # bending radius [m]
            B = Brho / rho # [T]

            us_drifts, ds_drifts = self._get_US_DS_drift_space(k, excel_elem_list)

            ws.write(next_row, m['name'], k)
            ws.write(next_row, m['L'], L, fmt['L'])
            ws.write(next_row, m['theta'], angle * 1e3, fmt['mrad'])
            ws.write(next_row, m['E1'], e1 * 1e3, fmt['mrad'])
            ws.write(next_row, m['E2'], e2 * 1e3, fmt['mrad'])
            ws.write(next_row, m['K1'], K1, fmt['K1'])
            ws.write(next_row, m['B'], B, fmt['B'])
            ws.write(next_row, m['rho'], rho, fmt['rho'])
            ws.write(next_row, m['US_space'], np.min(us_drifts) * 1e3, fmt['space'])
            ws.write(next_row, m['DS_space'], np.min(ds_drifts) * 1e3, fmt['space'])

            next_row += 1

        return next_row

    def add_xlsx_L3_quad_elements(
        self, next_row, elem_defs, mag_col_inds, excel_elem_list):
        """"""

        if not elem_defs['quads']:
            return next_row

        ws = self.worksheets['mag_params']

        wb_txt_fmts = self.wb_txt_fmts
        wb_num_fmts = self.wb_num_fmts

        bold = wb_txt_fmts.bold
        bold_underline = wb_txt_fmts.bold_underline
        bold_italic = wb_txt_fmts.bold_italic
        bold_sup = wb_txt_fmts.bold_sup
        bold_sub = wb_txt_fmts.bold_sub

        bold_wrap = wb_txt_fmts.bold_wrap

        if next_row != 0:
            next_row += 1

        ws.write(next_row, 0, 'Quadrupole Magnets', bold_underline)
        next_row += 1

        # Write headers
        for col_name, fragments in [
            ['name', (bold, 'Name')],
            ['L', (bold_italic, 'L', bold, ' (m)')],
            ['K1',
             (bold_italic, 'K', bold_sub, '1', bold, ' (m', bold_sup, '-2', ')')],
            ['US_space', (bold_wrap, 'Min. US\nSpace (mm)')],
            ['DS_space', (bold_wrap, 'Min. DS\nSpace (mm)')],
            ['aperture', (bold_wrap, 'Magnet\nAperture (mm)')],
            ]:

            col = mag_col_inds[col_name]

            if len(fragments) > 2:
                ws.write_rich_string(next_row, col, *fragments)
            elif len(fragments) == 2:
                ws.write(next_row, col, fragments[1], fragments[0])
            elif len(fragments) == 1:
                ws.write(next_row, col, fragments[0])
            else:
                raise ValueError()
        next_row += 1

        fmt = dict(L=wb_num_fmts['0.000'], K1=wb_num_fmts['0.000'],
                   space=wb_num_fmts['0.0'])

        d = elem_defs['quads']

        m = mag_col_inds

        for k in sorted(list(d)):
            L, K1 = d[k]['L'], d[k]['K1']

            us_drifts, ds_drifts = self._get_US_DS_drift_space(k, excel_elem_list)

            ws.write(next_row, m['name'], k)
            ws.write(next_row, m['L'], L, fmt['L'])
            ws.write(next_row, m['K1'], K1, fmt['K1'])
            ws.write(next_row, m['US_space'], np.min(us_drifts) * 1e3, fmt['space'])
            ws.write(next_row, m['DS_space'], np.min(ds_drifts) * 1e3, fmt['space'])

            next_row += 1

        return next_row

    def add_xlsx_L3_sext_elements(
        self, next_row, elem_defs, mag_col_inds, excel_elem_list):
        """"""

        if not elem_defs['sexts']:
            return next_row

        ws = self.worksheets['mag_params']

        wb_txt_fmts = self.wb_txt_fmts
        wb_num_fmts = self.wb_num_fmts

        bold = wb_txt_fmts.bold
        bold_underline = wb_txt_fmts.bold_underline
        bold_italic = wb_txt_fmts.bold_italic
        bold_sup = wb_txt_fmts.bold_sup
        bold_sub = wb_txt_fmts.bold_sub

        bold_wrap = wb_txt_fmts.bold_wrap

        if next_row != 0:
            next_row += 1

        ws.write(next_row, 0, 'Sextupole Magnets', bold_underline)
        next_row += 1

        # Write headers
        for col_name, fragments in [
            ['name', (bold, 'Name')],
            ['L', (bold_italic, 'L', bold, ' (m)')],
            ['K2',
             (bold_italic, 'K', bold_sub, '2', bold, ' (m', bold_sup, '-3', ')')],
            ['US_space', (bold_wrap, 'Min. US\nSpace (mm)')],
            ['DS_space', (bold_wrap, 'Min. DS\nSpace (mm)')],
            ['aperture', (bold_wrap, 'Magnet\nAperture (mm)')],
            ]:

            col = mag_col_inds[col_name]

            if len(fragments) > 2:
                ws.write_rich_string(next_row, col, *fragments)
            elif len(fragments) == 2:
                ws.write(next_row, col, fragments[1], fragments[0])
            elif len(fragments) == 1:
                ws.write(next_row, col, fragments[0])
            else:
                raise ValueError()
        next_row += 1

        fmt = dict(L=wb_num_fmts['0.000'], K2=wb_num_fmts['0.00'],
                   space=wb_num_fmts['0.0'])

        d = elem_defs['sexts']

        m = mag_col_inds

        for k in sorted(list(d)):
            L, K2 = d[k]['L'], d[k]['K2']

            us_drifts, ds_drifts = self._get_US_DS_drift_space(k, excel_elem_list)

            ws.write(next_row, m['name'], k)
            ws.write(next_row, m['L'], L, fmt['L'])
            ws.write(next_row, m['K2'], K2, fmt['K2'])
            ws.write(next_row, m['US_space'], np.min(us_drifts) * 1e3, fmt['space'])
            ws.write(next_row, m['DS_space'], np.min(ds_drifts) * 1e3, fmt['space'])

            next_row += 1

        return next_row

    def add_xlsx_L3_oct_elements(
        self, next_row, elem_defs, mag_col_inds, excel_elem_list):
        """"""

        if not elem_defs['octs']:
            return next_row

        ws = self.worksheets['mag_params']

        wb_txt_fmts = self.wb_txt_fmts
        wb_num_fmts = self.wb_num_fmts

        bold = wb_txt_fmts.bold
        bold_underline = wb_txt_fmts.bold_underline
        bold_italic = wb_txt_fmts.bold_italic
        bold_sup = wb_txt_fmts.bold_sup
        bold_sub = wb_txt_fmts.bold_sub

        bold_wrap = wb_txt_fmts.bold_wrap

        if next_row != 0:
            next_row += 1

        ws.write(next_row, 0, 'Octupole Magnets', bold_underline)
        next_row += 1

        # Write headers
        for col_name, fragments in [
            ['name', (bold, 'Name')],
            ['L', (bold_italic, 'L', bold, ' (m)')],
            ['K3',
             (bold_italic, 'K', bold_sub, '3', bold, ' (m', bold_sup, '-4', ')')],
            ['US_space', (bold_wrap, 'Min. US\nSpace (mm)')],
            ['DS_space', (bold_wrap, 'Min. DS\nSpace (mm)')],
            ['aperture', (bold_wrap, 'Magnet\nAperture (mm)')],
            ]:

            col = mag_col_inds[col_name]

            if len(fragments) > 2:
                ws.write_rich_string(next_row, col, *fragments)
            elif len(fragments) == 2:
                ws.write(next_row, col, fragments[1], fragments[0])
            elif len(fragments) == 1:
                ws.write(next_row, col, fragments[0])
            else:
                raise ValueError()
        next_row += 1

        fmt = dict(L=wb_num_fmts['0.000'], K3=wb_num_fmts['0.00'],
                   space=wb_num_fmts['0.0'])

        d = elem_defs['octs']

        m = mag_col_inds

        for k in sorted(list(d)):
            L, K3 = d[k]['L'], d[k]['K3']

            us_drifts, ds_drifts = self._get_US_DS_drift_space(k, excel_elem_list)

            ws.write(next_row, m['name'], k)
            ws.write(next_row, m['L'], L, fmt['L'])
            ws.write(next_row, m['K3'], K3, fmt['K3'])
            ws.write(next_row, m['US_space'], np.min(us_drifts) * 1e3, fmt['space'])
            ws.write(next_row, m['DS_space'], np.min(ds_drifts) * 1e3, fmt['space'])

            next_row += 1

        return next_row

    def add_xlsx_L2_beamline_elements_list(self):
        """"""

        ws = self.worksheets['elems_twiss']

        ws.freeze_panes(1, 0) # Freeze the first row

        wb_txt_fmts = self.wb_txt_fmts
        wb_num_fmts = self.wb_num_fmts

        bold = wb_txt_fmts.bold
        bold_italic = wb_txt_fmts.bold_italic
        bold_italic_sub = wb_txt_fmts.bold_italic_sub

        excel_elem_list = self.lin_data['excel_elem_list']

        # Write some data headers.
        header_list = excel_elem_list[0]
        row = 0
        for col, h in enumerate(header_list):
            if h == 's (m)':
                ws.write_rich_string(row, col, bold_italic, 's', bold, ' (m)')
            elif h == 'L (m)':
                ws.write_rich_string(row, col, bold_italic, 'L', bold, ' (m)')
            elif h == 'betax (m)':
                ws.write_rich_string(
                    row, col, bold_italic, GREEK['beta'], bold_italic_sub, 'x',
                    bold, ' (m)')
            elif h == 'betay (m)':
                ws.write_rich_string(
                    row, col, bold_italic, GREEK['beta'], bold_italic_sub, 'y',
                    bold, ' (m)')
            elif h == 'etax (m)':
                ws.write_rich_string(
                    row, col, bold_italic, GREEK['eta'], bold_italic_sub, 'x',
                    bold, ' (mm)')
                etax_col_index = col
            elif h == 'psix (2\pi)':
                ws.write_rich_string(
                    row, col, bold_italic, GREEK['phi'], bold_italic_sub, 'x',
                    bold, ' (2', bold_italic, GREEK['pi'], bold, ')')
            elif h == 'psiy (2\pi)':
                ws.write_rich_string(
                    row, col, bold_italic, GREEK['phi'], bold_italic_sub, 'y',
                    bold, ' (2', bold_italic, GREEK['pi'], bold, ')')
            else:
                ws.write(row, col, h, bold)
        row += 1

        beta_fmt = wb_num_fmts['0.000']
        etax_fmt = wb_num_fmts['0.000']
        psi_fmt = wb_num_fmts['0.0000']

        fmt_list = [None, None, None, None, None,
                    beta_fmt, beta_fmt, etax_fmt, psi_fmt, psi_fmt]
        assert len(fmt_list) == len(excel_elem_list[1])

        # Adjust the column widths for "Element Name" and "Element Type" columns
        for col in [2, 3]:
            max_width = max([len(v[col]) for v in excel_elem_list])
            ws.set_column(col, col, max_width + 1)

        for contents in excel_elem_list[1:]:
            for col, (v, fmt) in enumerate(zip(contents, fmt_list)):
                if col == etax_col_index:
                    v *= 1e3 # convert etax unit from [m] to [mm]
                ws.write(row, col, v, fmt)
            row += 1

        img_height = 25
        row = 0
        #for fp in sorted(Path(self.report_folderpath).glob('twiss_*.svg')):
        for fp in sorted(Path(self.report_folderpath).glob('twiss_*.png')):
            ws.insert_image(row, len(fmt_list) + 1, fp)
            row += img_height

    def calc_xlsx_rf_volt_dep_props(self):
        """"""

        rf_dep_calc_opts = self.conf.get('rf_dep_calc_opts', None)
        if rf_dep_calc_opts is None:
            self.rf_dep_props = None
            return

        rf_volts = np.array(rf_dep_calc_opts['rf_V']) # [V]
        h = rf_dep_calc_opts['harmonic_number']

        c = scipy.constants.c
        m_e_eV = physical_constants[
            'electron mass energy equivalent in MeV'][0] * 1e6

        E_GeV = self.lattice_props['E_GeV'] # [GeV]
        alphac = self.lattice_props['alphac'] # momentum compaction
        U0_eV = self.lattice_props['U0_keV'] * 1e3 # energy loss per turn [eV]
        circumf = self.lattice_props['circumf'] # circumference [m]
        sigma_delta = self.lattice_props['sigma_delta_percent'] * 1e-2 # energy spread [frac]

        f_rf = h * c / circumf # RF frequency [Hz]

        # Synchronous Phase
        synch_phases_deg = np.rad2deg(np.pi - np.arcsin(U0_eV / rf_volts))

        # Synchrotron Tune
        nu_s = np.sqrt(
            -rf_volts / (E_GeV * 1e9) * np.cos(np.deg2rad(synch_phases_deg))
            * alphac * h / (2 * np.pi))

        # Bunch Length
        sigma_z_m = alphac * sigma_delta * circumf / (2 * np.pi * nu_s) # [m]
        sigma_z_ps = sigma_z_m / c * 1e12 # [ps]

        # RF Bucket Height (RF Acceptance)
        #
        # See Section 3.1.4.6 on p.212 of Chao & Tigner, "Handbook
        # of Accelerator Physics and Engineering" for analytical
        # formula of RF bucket height, which is "A_s" in Eq. (32),
        # which is equal to (epsilon_max/E_0) [fraction] in Eq. (33).
        #
        # Note that the slip factor (eta) is approximately equal
        # to momentum compaction in the case of NSLS-II.
        gamma = 1.0 + E_GeV * 1e9 / m_e_eV
        gamma_t = 1.0 / np.sqrt(alphac)
        slip_fac = 1.0 / (gamma_t**2) - 1.0 / (gamma**2) # approx. equal to "mom_compac"
        q = rf_volts / U0_eV # overvoltage factor
        F_q = 2.0 * (np.sqrt(q**2 - 1) - np.arccos(1.0 / q))
        rf_bucket_heights_percent = 1e2 * np.sqrt(
            U0_eV / (np.pi * np.abs(slip_fac) * h * (E_GeV * 1e9)) * F_q)

        self.rf_dep_props = dict(
            rf_volts=rf_volts, h=h, f_rf=f_rf, synch_phases_deg=synch_phases_deg,
            nu_s=nu_s, sigma_z_m=sigma_z_m, sigma_z_ps=sigma_z_ps,
            rf_bucket_heights_percent=rf_bucket_heights_percent,
        )

    def calc_xlsx_lifetime_props(self):
        """"""

        lifetime_calc_opts = self.conf.get('lifetime_calc_opts', None)
        if lifetime_calc_opts is None:
            self.lifetime_props = None
            return

        total_beam_current_mA = lifetime_calc_opts['total_beam_current_mA']
        num_filled_bunches = lifetime_calc_opts['num_filled_bunches']
        beam_current_per_bunch_mA = total_beam_current_mA / num_filled_bunches

        total_charge_C = total_beam_current_mA * 1e-3 * (
            self.lattice_props['T_rev_us'] * 1e-6)
        total_charge_uC = total_charge_C * 1e6
        charge_per_bunch_nC = total_charge_C / num_filled_bunches * 1e9

        eps_ys = np.array(lifetime_calc_opts['eps_y']) # [m-rad]

        eps_0 = self.lattice_props['eps_x_pm'] * 1e-12 # equilibrium emittance

        coupling = eps_ys / (eps_0 - eps_ys) # := "coupling" or "k" (or "kappa")
        # used in ELEGANT's "touschekLifetime" function.
        coupling_percent = coupling * 1e2

        eps_xs = eps_0 / (1 + coupling) # [m-rad]

        tau_hrs = np.full(
            (len(eps_ys), len(self.rf_dep_props['rf_volts'])), np.nan)

        self.lifetime_props = dict(
            total_beam_current_mA=total_beam_current_mA,
            num_filled_bunches=num_filled_bunches,
            beam_current_per_bunch_mA=beam_current_per_bunch_mA,
            total_charge_uC=total_charge_uC,
            charge_per_bunch_nC=charge_per_bunch_nC,
            eps_ys=eps_ys, eps_xs=eps_xs,
            coupling_percent=coupling_percent, tau_hrs=tau_hrs)

    def add_xlsx_lattice_props(self):
        """"""

        wb = self.workbook
        ws = self.worksheets['lat_params']

        wb_txt_fmts = self.wb_txt_fmts

        bold_underline = wb_txt_fmts.bold_underline

        conf = self.conf

        row = 2 # Leave top 2 rows for lattice description & notes for property table

        # Write headers
        ws.write(row, 0, 'Property', bold_underline)
        ws.write(row, 1, 'Value', bold_underline)
        row += 1

        table_order = conf['lattice_props'].get('xlsx_table_order', None)
        if table_order is None:
            table_order = [
                'E_GeV', # Beam energy
                'circumf', # Circumference
                'eps_x', # Natural horizontal emittance
                'nux', 'nuy', # Ring tunes
                'ksi_nat_x', 'ksi_nat_y',
                'ksi_cor_x', 'ksi_cor_y',
                'alphac', # Momentum compaction
                'Jx', 'Jy', 'Jdelta', # Damping partition numbers
                'taux', 'tauy', 'taudelta', # Damping times
                'sigma_delta', # Energy spread
                'U0', # Energy loss per turn
                'f_rev', # Revolution frequency
                'T_rev', # Revolution period
                ['req_props', 'beta', 'LS', 'x'], # Horizontal beta at LS Center
                ['req_props', 'beta', 'LS', 'y'], # Vertical beta at LS Center
                ['req_props', 'beta', 'SS', 'x'], # Horizontal beta at SS Center
                ['req_props', 'beta', 'SS', 'y'], # Vertical beta at SS Center
                'max_betax', 'max_betay', # Max beta
                'min_betax', 'min_betay', # Min beta
                'max_etax', 'min_etax', # Max-Min horizontal dispersion
                ['req_props', 'length', 'LS'], # Length of LS
                ['req_props', 'length', 'SS'], # Length of SS
                'n_periods_in_ring', # Number of super-periods for a full ring
                'straight_frac', # Fraction of straights [Use Excel formulat to compute this]
                ['req_props', 'floor_comparison', 'circumf_change_%'], # Circumference change [%] from Reference Lattice
                ['req_props', 'floor_comparison', 'LS', 'x'],
                ['req_props', 'floor_comparison', 'LS', 'z'],
                ['req_props', 'floor_comparison', 'SS', 'x'],
                ['req_props', 'floor_comparison', 'SS', 'z'],

                # RF voltage
                # RF harmonic number
                # Synchrotron tune
                # Bunch length
                # RF acceptance (Bucket height)
                # Beam Lifetime
                # Lattice momentum accetance
                # DA
            ]

            for spec in conf['lattice_props'].get(
                'append_opt_props_to_xlsx_table', []):
                if spec not in table_order:
                    table_order.append(spec)


        self.defined_names = {
            # (row_spec_str): (Excel defined name)
            'n_periods_in_ring': 'n_periods_in_ring',
            'circumf': 'circumf',
            ('\n'.join(['req_props', 'length', 'LS'])): 'L_LS',
            ('\n'.join(['req_props', 'length', 'SS'])): 'L_SS',
            'U0': 'U0_keV',
            'E_GeV': 'E_GeV',
            'alphac': 'alphac',
            'sigma_delta': 'sigma_delta_percent',
            'eps_x': 'eps_x_pm',
            'T_rev': 'T_rev_us',
        }

        for row_spec_str, defined_name in self.defined_names.items():
            if '\n' in row_spec_str:
                row_spec = row_spec_str.splitlines()
            else:
                row_spec = row_spec_str
            (_, value, _) = self.get_xlsx_lattice_prop_row(row_spec)

            self.lattice_props[defined_name] = value

        # Compute RF-voltage-dependent properties
        self.calc_xlsx_rf_volt_dep_props()

        # Compute lifetime-related properties
        self.calc_xlsx_lifetime_props()

        for row_spec in table_order:

            try:
                (label_fragments, value, num_fmt
                 ) = self.get_xlsx_lattice_prop_row(row_spec)
            except:
                print('# WARNING #: Failed to get info:')
                print(row_spec)
                row += 1
                continue

            ws.write_rich_string(row, 0, *label_fragments)

            col = 1
            ws.write(row, col, value, num_fmt)

            cell = xlsxwriter.utility.xl_rowcol_to_cell(
                row, col, row_abs=True, col_abs=True)
            cell_address = f"='{ws.name}'!{cell}"

            if isinstance(row_spec, list):
                row_spec_str = '\n'.join(row_spec)
            else:
                row_spec_str = row_spec

            self.xlsx_map['lat_params'][row_spec_str] = dict(
                label_fragments=label_fragments, cell_address=cell_address,
                num_fmt=num_fmt)

            if row_spec_str in self.defined_names:
                wb.define_name(self.defined_names[row_spec_str], cell_address)

            row += 1

        ws.set_column(0, 0, 40)

    def _proc_xlsx_yaml_str(self, yaml_str_list):
        """"""

        label = []
        for i, token in enumerate(yaml_str_list):
            if np.mod(i, 2) == 0:
                if 'greek' in token:
                    token = token.replace('_greek', '').replace(
                        'greek_', '')
                    convert_greek = True
                else:
                    convert_greek = False
                label.append(getattr(self.wb_txt_fmts, token))
            else:
                if convert_greek:
                    token = GREEK[token]
                label.append(token)

        return label

    def get_xlsx_lattice_prop_row(self, row_spec):
        """"""

        wb_txt_fmts = self.wb_txt_fmts
        nf = self.wb_num_fmts

        normal = wb_txt_fmts.normal
        italic = wb_txt_fmts.italic
        sup = wb_txt_fmts.sup
        sub = wb_txt_fmts.sub
        italic_sub = wb_txt_fmts.italic_sub

        lin_data = self.lin_data
        opt_props = self.conf['lattice_props']['opt_props']

        k = row_spec

        plane_words = dict(x='Horizontal', y='Vertical', delta='Longitudinal')

        if isinstance(k, list):

            if k[0] == 'req_props':
                required = True
            elif k[0] == 'opt_props':
                required = False
            else:
                raise ValueError('Invalid row specification')

            prop_name, key = k[1], k[2]
            if prop_name == 'beta':
                plane = k[3]
                if required:
                    if key == 'LS':
                        location = 'Long-Straight Center'
                    elif key == 'SS':
                        location = 'Short-Straight Center'
                    else:
                        raise ValueError(f'Invalid 3rd arg for ["{k[0]}", "beta"]')
                    label = [italic, GREEK['beta'], italic_sub, plane,
                             normal, f' at {location} ']
                    symbol = []
                else:
                    label = self._proc_xlsx_yaml_str(
                        opt_props[prop_name][key]['xlsx_label'][plane])
                    symbol = []
                unit = [normal, ' (m)']
                value = lin_data[k[0]][k[1]][k[2]][k[3]]
                num_fmt = nf['0.00']
            elif prop_name == 'length':
                if required:
                    if key == 'LS':
                        location = 'Long Straight'
                    elif key == 'SS':
                        location = 'Short Straight'
                    else:
                        raise ValueError(f'Invalid 3rd arg for ["{k[0]}", "length"]')
                    label = [normal, f'Length of {location} ']
                    symbol = [italic, 'L', sub, k[2]]
                else:
                    label = self._proc_xlsx_yaml_str(
                        opt_props[prop_name][key]['xlsx_label'])
                    symbol = []
                unit = [normal, ' (m)']
                value = lin_data[k[0]][k[1]][k[2]]['L']
                num_fmt = nf['0.000']
            elif prop_name == 'floor_comparison':
                if required:
                    if key == 'circumf_change_%':
                        label = [normal, 'Circumference Change ']
                        symbol = [italic, GREEK['Delta'] + 'C', '/', italic, 'C']
                        unit = [normal, ' (%)']
                        value = lin_data[k[0]][k[1]][k[2]]['val']
                        num_fmt = nf['0.000']
                    elif key in ('LS', 'SS'):
                        location = k[2]
                        plane = k[3]
                        label = [normal, f'Source Point Diff. @ {location} ']
                        symbol = [italic, GREEK['Delta'] + plane, sub, location]
                        unit = [normal, ' (mm)']
                        value = lin_data[k[0]][k[1]][k[2]][k[3]] * 1e3
                        num_fmt = nf['0.00']
                    else:
                        raise ValueError(f'Invalid 3rd arg for ["{k[0]}", "floor_comparison"]')
                else:
                    plane = k[3]
                    label = self._proc_xlsx_yaml_str(
                        opt_props[prop_name][key]['xlsx_label'][plane])
                    symbol = []
                    unit = [normal, ' (mm)']
                    value = lin_data[k[0]][k[1]][k[2]][k[3]] * 1e3
                    num_fmt = nf['0.00']
            elif prop_name == 'phase_adv':
                if required:
                    raise ValueError('"phase_adv" must be under "opt_props", NOT "req_props".')
                else:
                    plane = k[3]
                    label = self._proc_xlsx_yaml_str(
                        opt_props[prop_name][key]['xlsx_label'][plane])
                    symbol = []
                    unit = [normal, ' (2', italic, GREEK['pi'], normal, ')']
                    value = lin_data[k[0]][k[1]][k[2]][k[3]]
                    num_fmt = nf['0.000000']
            else:
                raise ValueError(f'Invalid 2nd arg for ["{k[0]}"]')

        elif k == 'E_GeV':
            label = [normal, 'Beam Energy ']
            symbol = [italic, 'E']
            unit = [normal, ' (GeV)']
            value, num_fmt = lin_data[k], None
        elif k == 'circumf':
            label = [normal, 'Circumference ']
            symbol = [italic, 'C']
            unit = [normal, ' (m)']
            value, num_fmt = lin_data[k], nf['0.000']
        elif k == 'eps_x':
            label = [normal, 'Natural Horizontal Emittance ']
            symbol = [italic, GREEK['epsilon'], italic_sub, 'x']
            unit = [normal, ' (pm-rad)']
            value, num_fmt = lin_data[k] * 1e12, nf['0.00']
        elif k in ('nux', 'nuy'):
            plane = k[-1]
            label = [normal, f'{plane_words[plane]} Tune ']
            symbol = [italic, GREEK['nu'], italic_sub, plane]
            unit = [normal, ' ()']
            value, num_fmt = lin_data[k], nf['0.000']
        elif k in ('ksi_nat_x', 'ksi_nat_y'):
            plane = k[-1]
            label = [normal, f'{plane_words[plane]} Natural Chromaticity ']
            symbol = [italic, GREEK['xi'], italic_sub, plane, sup, 'nat']
            unit = [normal, ' ()']
            value, num_fmt = lin_data[f'ksi_{plane}_nat'], nf['0.000']
        elif k in ('ksi_cor_x', 'ksi_cor_y'):
            plane = k[-1]
            label = [normal, f'{plane_words[plane]} Corrected Chromaticity ']
            symbol = [italic, GREEK['xi'], italic_sub, plane, sup, 'cor']
            unit = [normal, ' ()']
            value, num_fmt = lin_data[f'ksi_{plane}_cor'], nf['0.000']
        elif k == 'alphac':
            label = [normal, 'Momentum Compaction ']
            symbol = [italic, GREEK['alpha'], italic_sub, 'c']
            unit = [normal, ' ()']
            value, num_fmt = lin_data[k], nf['0.00E+00']
        elif k in ('Jx', 'Jy', 'Jdelta'):
            plane = k[1:]
            label = [normal, f'{plane_words[plane]} Damping Partition Number ']
            if plane == 'delta':
                symbol = [italic, 'J', italic_sub, GREEK[plane]]
            else:
                symbol = [italic, 'J', italic_sub, plane]
            unit = [normal, ' ()']
            value, num_fmt = lin_data[k], nf['0.00']
        elif k in ('taux', 'tauy', 'taudelta'):
            plane = k[3:]
            label = [normal, f'{plane_words[plane]} Damping Time ']
            if plane == 'delta':
                symbol = [italic, GREEK['tau'], italic_sub, GREEK[plane]]
            else:
                symbol = [italic, GREEK['tau'], italic_sub, plane]
            unit = [normal, ' (ms)']
            value, num_fmt = lin_data[k] * 1e3, nf['0.00']
        elif k == 'sigma_delta':
            label = [normal, 'Energy Spread ']
            symbol = [italic, GREEK['sigma'], italic_sub, GREEK['delta']]
            unit = [normal, ' (%)']
            value, num_fmt = lin_data['dE_E'] * 1e2, nf['0.000']
        elif k == 'U0':
            label = [normal, 'Energy Loss per Turn ']
            symbol = [italic, 'U', sub, '0']
            unit = [normal, ' (keV)']
            value, num_fmt = lin_data['U0_MeV'] * 1e3, nf['###']
        elif k == 'f_rev':
            label = [normal, 'Revolution Frequency ']
            symbol = [italic, 'f', sub, 'rev']
            unit = [normal, ' (kHz)']
            value, num_fmt = (
                scipy.constants.c / lin_data['circumf'] / 1e3, nf['0.000'])
        elif k == 'T_rev':
            label = [normal, 'Revolution Period ']
            symbol = [italic, 'T', sub, 'rev']
            unit = [normal, ' (', italic, GREEK['mu'], 's)']
            f_rev = scipy.constants.c / lin_data['circumf'] # [Hz]
            T_rev = 1.0 / f_rev # [s]
            value, num_fmt = (T_rev * 1e6, nf['0.000'])
        elif k in ('max_betax', 'max_betay', 'min_betax', 'min_betay'):
            plane = k[-1]
            max_min = ('Maximum' if k.startswith('max_') else 'Minimum')
            label = [normal, f'{max_min} ']
            symbol = [italic, GREEK['beta'], italic_sub, plane]
            unit = [normal, ' (m)']
            value, num_fmt = lin_data[k], nf['0.00']
        elif k in ('max_etax', 'min_etax'):
            max_min = ('Maximum' if k.startswith('max_') else 'Minimum')
            label = [normal, f'{max_min} ']
            symbol = [italic, GREEK['eta'], italic_sub, 'x']
            unit = [normal, ' (mm)']
            value, num_fmt = lin_data[k] * 1e3, nf['0.0']
        elif k == 'n_periods_in_ring':
            label = [normal, 'Number of Super-periods ']
            symbol = []
            unit = [normal, ' ()']
            value, num_fmt = lin_data[k], None
        elif k == 'straight_frac':
            # Use Excel formulat to compute this
            label = [normal, 'Fraction of Straight Sections ']
            symbol = []
            unit = [normal, ' (%)']
            value = '=(L_LS + L_SS) * n_periods_in_ring / circumf * 1e2'
            num_fmt = nf['0.00']
        else:
            raise RuntimeError(f'Unhandled "xlsx_table_order" key: {k}')

        label_fragments = label + symbol + unit

        return label_fragments, value, num_fmt

    def add_xlsx_geom_layout(self):
        """"""

        ws = self.worksheets['layout']

        wb_txt_fmts = self.wb_txt_fmts
        wb_num_fmts = self.wb_num_fmts

        bold = wb_txt_fmts.bold
        bold_italic = wb_txt_fmts.bold_italic
        bold_underline = wb_txt_fmts.bold_underline

        d = self.lin_data['req_props']['floor_comparison']

        header_list = [(bold_italic, 'x', bold, ' (m)'),
                       (bold_italic, 'z', bold, ' (m)'),
                       (bold_italic, GREEK['theta'], bold, ' (deg)'),]

        row, col = 0, 0
        ws.write(row, col, 'Reference', bold_underline)
        col += len(header_list) + 1
        ws.write(row, col, 'Current', bold_underline)

        row = 1
        col = 0
        # Write headers for reference layout
        for fragments in header_list:
            ws.write_rich_string(row, col, *fragments)
            col += 1
        col += 1
        # Write headers for current layout
        for fragments in header_list:
            ws.write_rich_string(row, col, *fragments)
            col += 1

        # Write coordinates for reference layout
        row = 2
        col = 0
        for iRow, (x, z, theta) in enumerate(zip(
            d['_ref_flr_x'], d['_ref_flr_z'], d['_ref_flr_theta'])):
            if iRow not in d['_ref_flr_speical_inds']:
                fmt = wb_num_fmts['0.0000']
            else:
                fmt = wb_num_fmts['bg_yellow_0.0000']
            ws.write(row, col, x, fmt)
            ws.write(row, col+1, z, fmt)
            ws.write(row, col+2, np.rad2deg(theta) * (-1), fmt)
            row += 1

        # Write coordinates for current layout
        row = 2
        col = len(header_list) + 1
        for iRow, (x, z, theta) in enumerate(zip(
            d['_cur_flr_x'], d['_cur_flr_z'], d['_cur_flr_theta'])):
            if iRow not in d['_cur_flr_speical_inds']:
                fmt = wb_num_fmts['0.0000']
            else:
                fmt = wb_num_fmts['bg_yellow_0.0000']
            ws.write(row, col, x, fmt)
            ws.write(row, col+1, z, fmt)
            ws.write(row, col+2, np.rad2deg(theta) * (-1), fmt)
            row += 1

        row = 0
        for iFig, fp in enumerate(
            sorted(Path(self.report_folderpath).glob('floor_*.png'))):
            if iFig == 0:
                img_height = 15
            else:
                img_height = 25
            ws.insert_image(row, len(header_list) * 2 + 2, fp)
            row += img_height

    def add_xlsx_LTE(self):
        """"""

        ws = self.worksheets['lte']

        wb_txt_fmts = self.wb_txt_fmts

        normal = wb_txt_fmts.normal
        bold = wb_txt_fmts.bold
        courier = wb_txt_fmts.courier

        ws.set_column(0, 0, 20)

        ws.write(0, 0, 'Input LTE filepath:', bold)
        ws.write(0, 1, self.input_LTE_filepath, normal)

        ws.write(2, 0, '#' * 100, courier)

        row = 4
        for line in self.LTE_contents.splitlines():
            ws.write(row, 0, line, courier)
            row += 1

    def add_xlsx_RF(self):
        """"""

        ws = self.worksheets['rf']

        wb_txt_fmts = self.wb_txt_fmts
        wb_num_fmts = self.wb_num_fmts

        bold = wb_txt_fmts.bold
        normal = wb_txt_fmts.normal
        normal_center_wrap = wb_txt_fmts.normal_center_wrap
        normal_center_border = wb_txt_fmts.normal_center_border
        italic = wb_txt_fmts.italic
        italic_sub = wb_txt_fmts.italic_sub

        ws.set_column(0, 0, 40)

        row = 0

        ws.write(row, 0, 'Lattice Property', bold)
        ws.write(row, 1, 'Value', bold)
        row += 1

        for row_spec_str in (
            'E_GeV', 'circumf', 'n_periods_in_ring',
            '\n'.join(['req_props', 'length', 'LS']),
            '\n'.join(['req_props', 'length', 'SS']),
            'eps_x', 'alphac', 'U0', 'sigma_delta',
            ):
            d = self.xlsx_map['lat_params'][row_spec_str]
            ws.write_rich_string(row, 0, *d['label_fragments'])
            ws.write(row, 1, d['cell_address'], d['num_fmt'])
            row += 1

        if self.rf_dep_props is not None:

            row += 1

            ws.write(row, 0, 'Voltage-independent Property', bold)
            ws.write(row, 1, 'Value', bold)
            row += 1

            ws.write(row, 0, 'Harmonic Number ()', normal)
            ws.write(row, 1, self.rf_dep_props['h'], None)
            row += 1

            ws.write(row, 0, 'RF Frequency (MHz)', normal)
            ws.write(row, 1, self.rf_dep_props['f_rf'] / 1e6, wb_num_fmts['0.000'])
            row += 1

            row += 1

            ws.write(row, 0, 'Voltage-dependent Property', bold)
            ws.write(row, 1, 'Values', bold)
            row += 1

            ws.write(row, 0, 'RF Voltage (MV)', normal)
            for col, v in enumerate(self.rf_dep_props['rf_volts']):
                ws.write(row, col+1, v / 1e6, wb_num_fmts['0.0'])
            row += 1

            ws.write(row, 0, 'Synchrotron Tune ()', normal)
            for col, v in enumerate(self.rf_dep_props['nu_s']):
                ws.write(row, col+1, v, wb_num_fmts['0.000000'])
            row += 1

            ws.write(row, 0, 'Synchronous Phase (deg)', normal)
            for col, v in enumerate(self.rf_dep_props['synch_phases_deg']):
                ws.write(row, col+1, v, wb_num_fmts['0.00'])
            row += 1

            ws.write(row, 0, 'RF Bucket Height (%)', normal)
            for col, v in enumerate(self.rf_dep_props['rf_bucket_heights_percent']):
                ws.write(row, col+1, v, wb_num_fmts['0.0'])
            row += 1

            ws.write(row, 0, 'Zero-Current RMS Bunch Length (mm)', normal)
            for col, v in enumerate(self.rf_dep_props['sigma_z_m']):
                ws.write(row, col+1, v * 1e3, wb_num_fmts['0.00'])
            row += 1

            ws.write(row, 0, 'Zero-Current RMS Bunch Length (ps)', normal)
            for col, v in enumerate(self.rf_dep_props['sigma_z_ps']):
                ws.write(row, col+1, v, wb_num_fmts['0.00'])
            row += 1

        if self.lifetime_props is not None:

            row += 1

            ws.write(row, 0, 'Beam Current Property', bold)
            ws.write(row, 1, 'Value', bold)
            row += 1

            ws.write(row, 0, 'Number of Filled Bunches ()', normal)
            ws.write(row, 1, self.lifetime_props['num_filled_bunches'],
                     wb_num_fmts['###'])
            row += 1

            ws.write(row, 0, 'Total Beam Current (mA)', normal)
            ws.write(row, 1, self.lifetime_props['total_beam_current_mA'],
                     wb_num_fmts['###'])
            row += 1

            ws.write(row, 0, 'Beam Current per Bunch (mA)', normal)
            ws.write(row, 1, self.lifetime_props['beam_current_per_bunch_mA'],
                     wb_num_fmts['0.00'])
            row += 1

            ws.write_rich_string(row, 0, normal, 'Total Beam Charge (', italic,
                                 GREEK['mu'], normal, 'C)')
            ws.write(row, 1, self.lifetime_props['total_charge_uC'],
                     wb_num_fmts['0.00'])
            row += 1

            ws.write(row, 0, 'Beam Charge per Bunch (nC)', normal)
            ws.write(row, 1, self.lifetime_props['charge_per_bunch_nC'],
                     wb_num_fmts['0.00'])
            row += 1

        if (self.rf_dep_props is not None) and (self.lifetime_props is not None):

            row += 1

            ws.write(row, 0, 'Beam Lifetime (hr)', bold)
            #
            col = 1
            cell_format = normal_center_wrap
            ws.merge_range(row, col, row + 1, col, '', cell_format)
            ws.write_rich_string(
                row, col, italic, GREEK['epsilon'], italic_sub, 'y', ' (pm-rad)',
                cell_format)
            #
            col += 1
            cell_format = normal_center_wrap
            ws.merge_range(row, col, row + 1, col, '', cell_format)
            ws.write_rich_string(
                row, col, italic, GREEK['epsilon'], italic_sub, 'x', ' (pm-rad)',
                cell_format)
            #
            col += 1
            cell_format = normal_center_wrap
            ws.merge_range(row, col, row + 1, col, '', cell_format)
            ws.write_rich_string(
                row, col, 'Coupling ', italic, GREEK['epsilon'], italic_sub, 'y',
                '/', italic, GREEK['epsilon'], italic_sub, 'x', ' (%)',
                cell_format)
            #
            col += 1
            table_col_offset = col
            #
            ws.merge_range(
                row, table_col_offset, row,
                table_col_offset + len(self.rf_dep_props['rf_volts']) - 1,
                'RF Voltage (MV)', normal_center_border)
            row += 1
            #
            for col, v in enumerate(self.rf_dep_props['rf_volts']):
                ws.write(row, col+table_col_offset, v / 1e6, wb_num_fmts['0.0'])
            row += 1
            #
            col = 1
            for row_shift, (eps_y, eps_x, kappa, tau_hr_array) in enumerate(zip(
                self.lifetime_props['eps_ys'], self.lifetime_props['eps_xs'],
                self.lifetime_props['coupling_percent'],
                self.lifetime_props['tau_hrs'])):
                ws.write(row + row_shift, col, eps_y * 1e12, wb_num_fmts['0.0'])
                ws.write(row + row_shift, col+1, eps_x * 1e12, wb_num_fmts['0.0'])
                ws.write(row + row_shift, col+2, kappa, wb_num_fmts['0.0'])
                for col_shift, tau_hr in enumerate(tau_hr_array):
                    ws.write(row + row_shift, table_col_offset + col_shift, tau_hr,
                             wb_num_fmts['0.000'])


    def add_xlsx_lifetime(self):
        """ TODO """

        lt_calc_opts = self.conf.get('lifetime_calc_opts', None)
        if lt_calc_opts is None:
            return

        eps_ys = np.array(lt_calc_opts['eps_y']) # [m-rad]

    def add_xlsx_nonlin(self):
        """"""

        ws = self.worksheets['nonlin']

        wb_txt_fmts = self.wb_txt_fmts
        num_txt_fmts = self.wb_num_fmts

        normal = wb_txt_fmts.normal
        bold = wb_txt_fmts.bold
        italic = wb_txt_fmts.italic
        italic_sub = wb_txt_fmts.italic_sub
        sup = wb_txt_fmts.sup
        sub = wb_txt_fmts.sub

        # Write headers
        row = 0
        ws.write(row, 0, 'Term Name', bold)
        ws.write(row, 1, 'Drives', bold)
        ws.write(row, 2, 'Value', bold)
        row += 1

        ws.set_column(0, 0, 12)
        ws.set_column(1, 1, 12)
        ws.set_column(2, 2, 8)

        minus = SYMBOL['minus']
        delta = [italic, GREEK['delta']]
        nux = [italic, GREEK['nu'], italic_sub, 'x']
        nuy = [italic, GREEK['nu'], italic_sub, 'y']
        nuxy = [italic, GREEK['nu'], italic_sub, 'x,y']
        betax = [italic, GREEK['beta'], italic_sub, 'x']
        betay = [italic, GREEK['beta'], italic_sub, 'y']
        etax = [italic, GREEK['eta'], italic_sub, 'x']
        Jx = [italic, 'J', italic_sub, 'x']
        Jy = [italic, 'J', italic_sub, 'y']
        Jyx = [italic, 'J', italic_sub, 'y,x']

        def p_d(numerator, denominator, order=1):
            """ partial derivative """
            if order == 1:
                return (
                    [italic, SYMBOL['partial']] + numerator + [normal, '/'] +
                    [italic, SYMBOL['partial']] + denominator)
            elif order >= 2:
                return (
                    [italic, SYMBOL['partial'], sup, f'{order:d}'] + numerator +
                    [normal, '/'] +  [italic, SYMBOL['partial']] + denominator +
                    [sup, f'{order:d}'])
            else:
                raise ValueError(f'Invalid order: {order:d}')

        h_drv_explanation = {
            'h11001': p_d(nux, delta), 'h00111': p_d(nuy, delta),
            'h20001': p_d(betax, delta), 'h00201': p_d(betay, delta),
            'h10002': p_d(etax, delta), 'h21000': nux,
            'h30000': [normal, '3'] + nux, 'h10110': nux,
            'h10020': nux + [normal, ' ' + minus + ' 2'] + nuy,
            'h10200': nux + [normal, ' + 2'] + nuy,
            'h22000': p_d(nux, Jx), 'h11110': p_d(nuxy, Jyx),
            'h00220': p_d(nuy, Jy), 'h31000': [normal, '2'] + nux,
            'h40000': [normal, '4'] + nux, 'h20110': [normal, '2'] + nux,
            'h11200': [normal, '2'] + nuy,
            'h20020': [normal, '2'] + nux + [normal, ' ' + minus + ' 2'] + nuy,
            'h20200': [normal, '2'] + nux + [normal, ' + 2'] + nuy,
            'h00310': [normal, '2'] + nuy, 'h00400': [normal, '4'] + nuy}

        for k, frag in h_drv_explanation.items():
            ws.write_rich_string(
                row, 0, normal, '|', italic, 'h', normal, k[1:], normal, '|')
            ws.write_rich_string(row, 1, *frag)
            ws.write(row, 2, self.lin_data[k], num_txt_fmts['0.00E+00'])
            row += 1

        # Start wrting other nonlinear dynamics data
        row = 0
        col_offset = 4

        if os.path.exists(self.suppl_plot_data_filepath['tswa']):

            # Retrieve tracking-based tswa data
            with open(self.suppl_plot_data_filepath['tswa'], 'rb') as f:
                _, tswa_data = pickle.load(f)

            # Write header for tswa
            nfmt = num_txt_fmts['0.00E+00']
            #
            ws.set_column(col_offset  , col_offset  , 11)
            ws.set_column(col_offset+1, col_offset+1, 12)
            ws.set_column(col_offset+2, col_offset+2, 12)
            ws.set_column(col_offset+3, col_offset+3, 12)
            #
            ws.write(row, col_offset, 'Tune Shift with Amplitude', bold)
            row += 1
            ws.write(row, col_offset+1, '&twiss_output', normal)
            ws.write(row, col_offset+2, 'tracking (+)', normal)
            ws.write(row, col_offset+3, f'tracking ({SYMBOL["minus"]})', normal)
            row += 1
            #
            ws.write_rich_string(row, col_offset, *p_d(nux, Jx))
            ws.write(row, col_offset+1, self.lin_data['dnux_dJx'], nfmt)
            ws.write(row, col_offset+2, tswa_data['x']['+']['dnux_dJx0'], nfmt)
            ws.write(row, col_offset+3, tswa_data['x']['-']['dnux_dJx0'], nfmt)
            row += 1
            ws.write_rich_string(row, col_offset, *p_d(nuy, Jx))
            ws.write(row, col_offset+1, self.lin_data['dnux_dJy'], nfmt)
            # ^ Note there is no "dnuy_dJx", so re-using self.lin_data['dnux_dJy'].
            ws.write(row, col_offset+2, tswa_data['x']['+']['dnuy_dJx0'], nfmt)
            ws.write(row, col_offset+3, tswa_data['x']['-']['dnuy_dJx0'], nfmt)
            row += 1
            ws.write_rich_string(row, col_offset, *p_d(nux, Jy))
            ws.write(row, col_offset+1, self.lin_data['dnux_dJy'], nfmt)
            ws.write(row, col_offset+2, tswa_data['y']['+']['dnux_dJy0'], nfmt)
            ws.write(row, col_offset+3, tswa_data['y']['-']['dnux_dJy0'], nfmt)
            row += 1
            ws.write_rich_string(row, col_offset, *p_d(nuy, Jy))
            ws.write(row, col_offset+1, self.lin_data['dnuy_dJy'], nfmt)
            ws.write(row, col_offset+2, tswa_data['y']['+']['dnuy_dJy0'], nfmt)
            ws.write(row, col_offset+3, tswa_data['y']['-']['dnuy_dJy0'], nfmt)
            row += 1

            row += 1

        if os.path.exists(self.suppl_plot_data_filepath['nonlin_chrom']):

            # Retrieve nonlin_chrom data
            with open(self.suppl_plot_data_filepath['nonlin_chrom'], 'rb') as f:
                nonlin_chrom_data = pickle.load(f)

            # Write header for nonlin_chrom
            nfmt = num_txt_fmts['0.00E+00']
            #
            ws.write(row, col_offset, 'Nonlinear Chromaticity', bold)
            row += 1
            #
            for iOrder, deriv_val in enumerate(
                nonlin_chrom_data['fit_coeffs']['x'][::-1]):
                if iOrder == 0:
                    frag = nux + [normal, ' ('] + delta + [normal, ' = 0)']
                else:
                    frag = p_d(nux, delta, iOrder)
                ws.write_rich_string(row, col_offset, *frag)
                ws.write(row, col_offset+1, deriv_val, nfmt)
                row += 1
            for iOrder, deriv_val in enumerate(
                nonlin_chrom_data['fit_coeffs']['y'][::-1]):
                if iOrder == 0:
                    frag = nuy + [normal, ' ('] + delta + [normal, ' = 0)']
                else:
                    frag = p_d(nuy, delta, iOrder)
                ws.write_rich_string(row, col_offset, *frag)
                ws.write(row, col_offset+1, deriv_val, nfmt)
                row += 1

            row += 1

        if os.path.exists(self.suppl_plot_data_filepath['xy_aper']):

            # Retrieve xy_aper data
            with open(self.suppl_plot_data_filepath['xy_aper'], 'rb') as f:
                xy_aper_data = pickle.load(f)

            # Write header for xy_aper
            ws.write(row, col_offset, 'Dynamic Aperture', bold)
            row += 1

            for plane in ['x', 'y']:
                if plane == 'x':
                    min_max_list = ['min', 'max']
                else:
                    if xy_aper_data['neg_y_search']:
                        min_max_list = ['min', 'max']
                    else:
                        min_max_list = ['max']
                for k in min_max_list:
                    frag = [italic, plane, sub, k, normal, ' (mm)']
                    ws.write_rich_string(row, col_offset, *frag)
                    ws.write(row, col_offset+1, xy_aper_data[f'{plane}_{k}'] * 1e3,
                             num_txt_fmts['0.000'])
                    row += 1

            frag = [normal, 'Area (mm', sup, '2', normal, ')']
            ws.write_rich_string(row, col_offset, *frag)
            ws.write(row, col_offset+1, xy_aper_data['area'] * 1e6,
                     num_txt_fmts['0.000'])
            row += 1

            row += 1

        if os.path.exists(self.suppl_plot_data_filepath['mom_aper']):

            # Retrieve mom_aper data
            with open(self.suppl_plot_data_filepath['mom_aper'], 'rb') as f:
                mom_aper_data = pickle.load(f)

            # Write header for mom_aper
            ws.write(row, col_offset, 'Momentum Aperture', bold)
            row += 1

            for m, sign, symb in [
                ('min', '+', '+'), ('max', '+', '+'),
                ('min', '-', SYMBOL['minus']), ('max', '-', SYMBOL['minus'])]:
                frag = [normal, f'{m} '] + delta + [sub, symb, normal, ' (%)']
                ws.write_rich_string(row, col_offset, *frag)
                ws.write(row, col_offset+1, mom_aper_data['delta_percent'][sign][m],
                         num_txt_fmts['0.000'])
                row += 1



        img_height = 25
        img_width = 10
        row = 0
        for calc_type, comment in zip(
            self.all_nonlin_calc_types, self.all_nonlin_calc_comments):

            col = 10
            ws.write(row, col, comment, bold)
            row += 1

            #for fp in sorted(Path(self.report_folderpath).glob(f'{calc_type}_*.svg')):
            for fp in sorted(Path(self.report_folderpath).glob(f'{calc_type}_*.png')):
                ws.insert_image(row, col, fp)
                col += img_width

            row += img_height

    def set_up_lattice(self):
        """"""

        conf = self.conf

        # Allow multi-line definition for a long LTE filepath in YAML
        conf['input_LTE']['filepath'] = ''.join([
            _s.strip() for _s in conf['input_LTE']['filepath'].splitlines()])

        assert conf['input_LTE']['filepath'].endswith('.lte')
        input_LTE_filepath = conf['input_LTE']['filepath']
        if input_LTE_filepath == '?.lte':
            raise ValueError('"input_LTE/filepath" must be specified in the config file')
        self.input_LTE_filepath = input_LTE_filepath

        self.LTE_contents = Path(input_LTE_filepath).read_text()

        if self.config_filepath.endswith('.yml'):
            rootname = os.path.basename(self.config_filepath)[:(-len('.yml'))]
        elif self.config_filepath.endswith('.yaml'):
            rootname = os.path.basename(self.config_filepath)[:(-len('.yaml'))]
        else:
            raise ValueError('Config file name must end with ".yml" or ".yaml".')
        self.rootname = rootname

        report_folderpath = f'report_{rootname}'
        self.report_folderpath = report_folderpath
        Path(report_folderpath).mkdir(exist_ok=True)

        self.suppl_plot_data_filepath = {}
        for calc_type in ['xy_aper', 'tswa', 'nonlin_chrom', 'mom_aper']:
            self.suppl_plot_data_filepath[calc_type] = os.path.join(
                self.report_folderpath, f'{calc_type}.plot_suppl.pkl')

        if conf['input_LTE'].get('load_param', False):
            self.gen_LTE_from_base_LTE_and_param_file()

        if conf['input_LTE'].get('zeroSexts_filepath', ''):
            zeroSexts_LTE_filepath = conf['input_LTE']['zeroSexts_filepath']
            assert os.path.exists(zeroSexts_LTE_filepath)
        else:
            # Turn off all sextupoles
            zeroSexts_LTE_filepath = self.gen_zeroSexts_LTE(
                conf['input_LTE'].get('regenerate_zeroSexts', False))
        self.zeroSexts_LTE_filepath = zeroSexts_LTE_filepath

    def gen_LTE_from_base_LTE_and_param_file(self):
        """"""

        conf = self.conf
        input_LTE_filepath = self.input_LTE_filepath

        assert conf['input_LTE']['base_LTE_filepath'].endswith('.lte')
        base_LTE_filepath = conf['input_LTE']['base_LTE_filepath']

        load_parameters = dict(filename=conf['input_LTE']['param_filepath'])

        pe.eleutil.save_lattice_after_load_parameters(
            base_LTE_filepath, input_LTE_filepath, load_parameters)

    def gen_zeroSexts_LTE(self, regenerate):
        """
        Turn off all sextupoles' K2 values to zero and save a new LTE file.
        """

        input_LTE_filepath = self.input_LTE_filepath
        report_folderpath = self.report_folderpath

        input_LTE_filename = os.path.basename(input_LTE_filepath)
        zeroSexts_LTE_filepath = os.path.join(
            report_folderpath,
            input_LTE_filename.replace('.lte', '_ZeroSexts.lte'))

        if (not os.path.exists(zeroSexts_LTE_filepath)) or regenerate:
            alter_elements = [
                dict(name='*', type='KSEXT', item='K2', value = 0.0,
                     allow_missing_elements=True),
                dict(name='*', type='KOCT', item='K3', value = 0.0,
                     allow_missing_elements=True)
            ]
            pe.eleutil.save_lattice_after_alter_elements(
                input_LTE_filepath, zeroSexts_LTE_filepath, alter_elements)

        return zeroSexts_LTE_filepath

    def get_lin_data(self):
        """"""

        conf = self.conf
        report_folderpath = self.report_folderpath
        input_LTE_filepath = self.input_LTE_filepath

        lin_summary_pkl_filepath = os.path.join(report_folderpath, 'lin.pkl')

        if (not os.path.exists(lin_summary_pkl_filepath)) or \
            conf['lattice_props'].get('recalc', False):

            # Make sure to override "replot" in the config
            conf['lattice_props']['replot'] = True

            d = self.calc_lin_props(
                zeroSexts_LTE_filepath=self.zeroSexts_LTE_filepath)

            lin_data = d['sel_data']
            lin_data['_versions'] = d['versions']
            lin_data['_timestamp'] = d['timestamp']
            abs_input_LTE_filepath = os.path.abspath(input_LTE_filepath)
            LTE_contents = self.LTE_contents

            with open(lin_summary_pkl_filepath, 'wb') as f:
                pickle.dump(
                    [conf, abs_input_LTE_filepath, LTE_contents, lin_data], f)
        else:
            with open(lin_summary_pkl_filepath, 'rb') as f:
                (saved_conf, abs_input_LTE_filepath, LTE_contents, lin_data
                 ) = pickle.load(f)

            if Path(input_LTE_filepath).read_text() != LTE_contents:
                raise RuntimeError(
                    (f'The LTE contents saved in "{lin_summary_pkl_filepath}" '
                     'does NOT exactly match with the currently specified '
                     f'LTE file "{input_LTE_filepath}". Either check the LTE '
                     'file, or re-calculate to create an updated data file.'))

        self.lin_data = lin_data

    def calc_lin_props(self, zeroSexts_LTE_filepath=''):
        """"""

        conf = self.conf['lattice_props']
        E_MeV = self.conf['E_MeV']
        report_folderpath = self.report_folderpath
        LTE_filepath = self.input_LTE_filepath
        use_beamline_cell = self.conf['use_beamline_cell']
        use_beamline_ring = self.conf['use_beamline_ring']

        default_twiss_calc_opts = dict(
            one_period={'use_beamline': use_beamline_cell,
                        'element_divisions': 10},
            ring_natural={'use_beamline': use_beamline_ring},
            ring={'use_beamline': use_beamline_ring},
        )
        conf_twi = conf.get('twiss_calc_opts', default_twiss_calc_opts)

        sel_data = {'E_GeV': E_MeV / 1e3}
        interm_array_data = {} # holds data that will be only used to derive some other quantities

        raw_keys = dict(
            one_period=dict(eps_x='ex0', Jx='Jx', Jy='Jy', Jdelta='Jdelta',
                            taux='taux', tauy='tauy', taudelta='taudelta'),
            ring=dict(
                nux='nux', nuy='nuy', ksi_x_cor='dnux/dp', ksi_y_cor='dnuy/dp',
                alphac='alphac', U0_MeV='U0', dE_E='Sdelta0'),
            ring_natural=dict(ksi_x_nat='dnux/dp', ksi_y_nat='dnuy/dp'),
        )
        #
        tswa_dict = dict(dnux_dJx='dnux/dJx', dnux_dJy='dnux/dJy',
                         dnuy_dJy='dnuy/dJy')
        h_drv_dict = {k: k for k in [
            'h11001', 'h00111', 'h20001', 'h00201', 'h10002', 'h21000', 'h30000',
            'h10110', 'h10020', 'h10200', 'h22000', 'h11110', 'h00220', 'h31000',
            'h40000', 'h20110', 'h11200', 'h20020', 'h20200', 'h00310', 'h00400']}
        raw_keys['ring'].update(tswa_dict)
        raw_keys['ring'].update(h_drv_dict)
        conf_twi['ring']['compute_driving_terms'] = True

        interm_array_keys = dict(
            one_period=dict(
                s_1p='s', betax_1p='betax', betay_1p='betay', etax_1p='etax',
                elem_names_1p='ElementName'),
            ring=dict(
                s_ring='s', elem_names_ring='ElementName',
                betax_ring='betax', betay_ring='betay',
                psix_ring='psix', psiy_ring='psiy'),
        )

        output_filepaths = {}
        for k in list(raw_keys):
            output_filepaths[k] = os.path.join(
                report_folderpath, f'twiss_{k}.pgz')

        for lat_type in ('one_period', 'ring'):
            pe.calc_ring_twiss(
                output_filepaths[lat_type], LTE_filepath, E_MeV,
                radiation_integrals=True, **conf_twi[lat_type])
            twi = pe.util.load_pgz_file(output_filepaths[lat_type])['data']['twi']
            for k, ele_k in raw_keys[lat_type].items():
                sel_data[k] = twi['scalars'][ele_k]
            for k, ele_k in interm_array_keys[lat_type].items():
                interm_array_data[k] = twi['arrays'][ele_k]

        if zeroSexts_LTE_filepath:
            lat_type = 'ring_natural'
            pe.calc_ring_twiss(
                output_filepaths[lat_type], zeroSexts_LTE_filepath, E_MeV,
                radiation_integrals=True, **conf_twi[lat_type])
            twi = pe.util.load_pgz_file(output_filepaths[lat_type])['data']['twi']
            for k, ele_k in raw_keys[lat_type].items():
                sel_data[k] = twi['scalars'][ele_k]

        sel_data['circumf'] = interm_array_data['s_ring'][-1]

        ring_mult = sel_data['circumf'] / interm_array_data['s_1p'][-1]
        sel_data['n_periods_in_ring'] = int(np.round(ring_mult))
        assert np.abs(ring_mult - sel_data['n_periods_in_ring']) < 1e-3

        sel_data['max_betax'] = np.max(interm_array_data['betax_1p'])
        sel_data['min_betax'] = np.min(interm_array_data['betax_1p'])
        sel_data['max_betay'] = np.max(interm_array_data['betay_1p'])
        sel_data['min_betay'] = np.min(interm_array_data['betay_1p'])
        sel_data['max_etax'] = np.max(interm_array_data['etax_1p'])
        sel_data['min_etax'] = np.min(interm_array_data['etax_1p'])

        req_conf = conf['req_props']
        req_data = sel_data['req_props'] = {}

        opt_conf = conf.get('opt_props', None)
        opt_data = sel_data['opt_props'] = {}

        # Compute required property "beta" at LS and SS centers
        _d = req_data['beta'] = {}
        for k2 in ['LS', 'SS']:
            elem_d = req_conf['beta'][k2]
            elem_name = elem_d['name'].upper()
            index = np.where(interm_array_data['elem_names_ring'] ==
                             elem_name)[0][elem_d['occur']]
            _d[k2] = dict(x=interm_array_data['betax_ring'][index],
                          y=interm_array_data['betay_ring'][index])

        if opt_conf:
            # Compute optional property "beta" at user-specified locations
            beta_conf = opt_conf.get('beta', None)
            if beta_conf:
                _d = opt_data['beta'] = {}
                for k2, elem_d in beta_conf.items():
                    elem_name = elem_d['name'].upper()
                    index = np.where(interm_array_data['elem_names_ring'] ==
                                     elem_name)[0][elem_d['occur']]
                    _d[k2] = dict(
                        x=interm_array_data['betax_ring'][index],
                        y=interm_array_data['betay_ring'][index],
                        pdf_label=elem_d['pdf_label'],
                        xlsx_label=elem_d['xlsx_label'])

            # Compute optional property "phase_adv" btw. user-specified locations
            phase_adv_conf = opt_conf.get('phase_adv', None)
            if phase_adv_conf:
                _d = opt_data['phase_adv'] = {}
                for k2, elem_d in phase_adv_conf.items():

                    elem_name_1 = elem_d['elem1']['name'].upper()
                    occur_1 = elem_d['elem1']['occur']
                    elem_name_2 = elem_d['elem2']['name'].upper()
                    occur_2 = elem_d['elem2']['occur']

                    multiplier = elem_d.get('multiplier', 1.0)

                    index_1 = np.where(
                        interm_array_data['elem_names_ring'] ==
                        elem_name_1)[0][occur_1]
                    index_2 = np.where(
                        interm_array_data['elem_names_ring'] ==
                        elem_name_2)[0][occur_2]
                    _d[k2] = dict(
                        x=multiplier * (interm_array_data['psix_ring'][index_2] -
                           interm_array_data['psix_ring'][index_1]) / (2 * np.pi),
                        y=multiplier * (interm_array_data['psiy_ring'][index_2] -
                           interm_array_data['psiy_ring'][index_1]) / (2 * np.pi),
                        pdf_label=elem_d['pdf_label'],
                        xlsx_label=elem_d['xlsx_label']
                    )


        # Compute floor layout & Twiss.
        # Also collect element definitions & flat list.

        ed = pe.elebuilder.EleDesigner()
        ed.add_block('run_setup',
            lattice = LTE_filepath, p_central_mev = E_MeV,
            use_beamline=conf_twi['one_period']['use_beamline'],
            parameters='%s.param')
        ed.add_block('floor_coordinates', filename = '%s.flr')
        ed.add_block('twiss_output', filename='%s.twi', matched=True)
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

        elem_defs = dict(bends={}, quads={}, sexts={}, octs={})
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

            elif elem_type in ('OCTU', 'KOCT'):
                props = {}
                for k in ['L', 'K3']:
                    temp = ed.get_LTE_elem_prop(elem_name, k)
                    if temp is not None:
                        props[k] = temp
                    else:
                        props[k] = 0.0
                elem_defs['octs'][elem_name] = props
        sel_data['elem_defs'] = elem_defs

        titles = dict(s='s [m]', ElementName='Element Name',
                      ElementType='Element Type')
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
        excel_headers = [
            's (m)', 'L (m)', 'Element Name', 'Element Type', 'Element Occurrence',
            'betax (m)', 'betay (m)', 'etax (m)', 'psix (2\pi)', 'psiy (2\pi)']
        excel_elem_list = [excel_headers]
        sel_data['tot_bend_angle_rad_per_period'] = 0.0
        assert np.all(res['twi']['columns']['s'] == res['flr']['columns']['s'])
        for (s, L, elem_name, elem_type, elem_occur, betax, betay, etax,
             psix, psiy) in zip(
            res['flr']['columns']['s'], res['flr']['columns']['ds'],
            res['flr']['columns']['ElementName'],
            res['flr']['columns']['ElementType'],
            res['flr']['columns']['ElementOccurence'],
            res['twi']['columns']['betax'], res['twi']['columns']['betay'],
            res['twi']['columns']['etax'], res['twi']['columns']['psix'],
            res['twi']['columns']['psiy']):

            flat_elem_s_name_type_list.append(
                value_template.format(s, elem_name, elem_type))
            excel_elem_list.append([
                s, L, elem_name, elem_type, elem_occur, betax, betay, etax,
                psix / (2 * np.pi), psiy / (2 * np.pi)])

            if elem_name in elem_defs['bends']:
                sel_data['tot_bend_angle_rad_per_period'] += \
                    elem_defs['bends'][elem_name]['ANGLE']
        #
        assert len(excel_elem_list[0]) == len(excel_elem_list[1]) # Confirm that the
        # length of headers is equal to the length of each list.
        #
        sel_data['flat_elem_s_name_type_list'] = flat_elem_s_name_type_list
        sel_data['excel_elem_list'] = excel_elem_list

        tot_bend_angle_deg_in_ring = np.rad2deg(
            sel_data['tot_bend_angle_rad_per_period']) * sel_data['n_periods_in_ring']
        if np.abs(tot_bend_angle_deg_in_ring - 360.0) > 0.1:
            raise RuntimeError(
                ('Total ring bend angle NOT close to 360 deg: '
                 f'{tot_bend_angle_deg_in_ring:.6f} [deg]'))

        # Compute required property "length" of LS and SS sections
        _d = req_data['length'] = {}
        _s = res['flr']['columns']['s']
        for k2 in ['LS', 'SS']:
            elem_d = req_conf['length'][k2]
            elem_name_list = [name.upper() for name in elem_d['name_list']]
            first_inds = np.where(res['flr']['columns']['ElementName']
                                  == elem_name_list[0])[0]
            for fi in first_inds:
                L = _s[fi] - _s[fi-1]
                for offset, elem_name in enumerate(elem_name_list[1:]):
                    if (res['flr']['columns']['ElementName'][fi + offset + 1]
                        == elem_name):
                        L += _s[fi + offset + 1] - _s[fi + offset]
                    else:
                        break
                else:
                    break
            else:
                raise ValueError(
                    'Invalid "length" dict value for key "{}": {}'.format(
                        k2, ', '.join(elem_name_list)))

            L *= elem_d.get('multiplier', 1.0)
            _d[k2] = dict(L=L)

        # Compute required property "floor_comparison" at LS and SS centers
        _d = req_data['floor_comparison'] = {}
        #
        ref_flr_filepath = req_conf['floor_comparison']['ref_flr_filepath']
        flr_data = pe.sdds.sdds2dicts(ref_flr_filepath)[0]
        ref_X_all = flr_data['columns']['X']
        ref_Z_all = flr_data['columns']['Z']
        ref_theta_all = flr_data['columns']['theta']
        ref_ElemNames = flr_data['columns']['ElementName']
        #
        ref_circumf = flr_data['columns']['s'][-1]
        _d['circumf_change_%'] = dict(
            val=(sel_data['circumf'] / ref_circumf - 1) * 1e2,
            label='Circumference Change ' + plx.MathText(r'\Delta C / C')
        )
        #
        ref_X, ref_Z = {}, {}
        cur_X, cur_Z = {}, {}
        special_inds = dict(ref=[], cur=[])
        for k2 in ['LS', 'SS']:
            elems_d = req_conf['floor_comparison'][k2]

            ref_elem = elems_d['ref_elem']

            ind = np.where(ref_ElemNames ==
                           ref_elem['name'].upper())[0][ref_elem['occur']]

            ref_X[k2] = ref_X_all[ind]
            ref_Z[k2] = ref_Z_all[ind]

            special_inds['ref'].append(ind)

            cur_elem = elems_d['cur_elem']

            ind = np.where(
                res['flr']['columns']['ElementName'] ==
                cur_elem['name'].upper())[0][cur_elem['occur']]

            cur_X[k2] = res['flr']['columns']['X'][ind]
            cur_Z[k2] = res['flr']['columns']['Z'][ind]

            special_inds['cur'].append(ind)

            _d[k2] = dict(x=cur_X[k2] - ref_X[k2],
                          z=cur_Z[k2] - ref_Z[k2])

        _d['_ref_flr_speical_inds'] = special_inds['ref']
        _d['_cur_flr_speical_inds'] = special_inds['cur']

        max_ind = max(special_inds['ref'])
        _d['_ref_flr_x'] = ref_X_all[:max_ind+1]
        _d['_ref_flr_z'] = ref_Z_all[:max_ind+1]
        _d['_ref_flr_theta'] = ref_theta_all[:max_ind+1]

        max_ind = max(special_inds['cur'])
        _d['_cur_flr_x'] = res['flr']['columns']['X'][:max_ind+1]
        _d['_cur_flr_z'] = res['flr']['columns']['Z'][:max_ind+1]
        _d['_cur_flr_theta'] = res['flr']['columns']['theta'][:max_ind+1]
        #_d['_cur_flr_next_elem_types'] = \
            #res['flr']['columns']['NextElementType'][:max_ind+1]


        if opt_conf:
            # Compute optional property "length" of user-specified consecutive elements
            length_conf = opt_conf.get('length', None)
            if length_conf:
                _d = opt_data['length'] = {}

                _s = res['flr']['columns']['s']
                for k2, elem_d in length_conf.items():
                    elem_name_list = [name.upper() for name in elem_d['name_list']]
                    if k2 in _d:
                        raise ValueError(
                            f'Duplicate key "{k2}" found for "length" dict')
                    first_inds = np.where(res['flr']['columns']['ElementName']
                                          == elem_name_list[0])[0]
                    for fi in first_inds:
                        L = _s[fi] - _s[fi-1]
                        for offset, elem_name in enumerate(elem_name_list[1:]):
                            if (res['flr']['columns']['ElementName'][fi + offset + 1]
                                == elem_name):
                                L += _s[fi + offset + 1] - _s[fi + offset]
                            else:
                                break
                        else:
                            break
                    else:
                        raise ValueError(
                            'Invalid "length" dict value for key "{}": {}'.format(
                                k2, ', '.join(elem_name_list)))

                    L *= elem_d.get('multiplier', 1.0)
                    _d[k2] = dict(L=L, pdf_label=elem_d['pdf_label'],
                                  xlsx_label=elem_d['xlsx_label'])


            # Compute optional property "floor_comparison" at user-specified locations
            floor_comparison_conf = opt_conf.get('floor_comparison', None)
            if floor_comparison_conf:
                _d = opt_data['floor_comparison'] = {}

                ref_X, ref_Z = {}, {}
                cur_X, cur_Z = {}, {}
                special_inds = dict(ref=[], cur=[])
                for k2, elems_d in floor_comparison_conf.items():

                    ref_elem = elems_d['ref_elem']

                    ind = np.where(ref_ElemNames ==
                                   ref_elem['name'].upper())[0][ref_elem['occur']]

                    ref_X[k2] = ref_X_all[ind]
                    ref_Z[k2] = ref_Z_all[ind]

                    special_inds['ref'].append(ind)

                    cur_elem = elems_d['cur_elem']

                    ind = np.where(
                        res['flr']['columns']['ElementName'] ==
                        cur_elem['name'].upper())[0][cur_elem['occur']]

                    cur_X[k2] = res['flr']['columns']['X'][ind]
                    cur_Z[k2] = res['flr']['columns']['Z'][ind]

                    special_inds['cur'].append(ind)

                    _d[k2] = dict(x=cur_X[k2] - ref_X[k2],
                                  z=cur_Z[k2] - ref_Z[k2],
                                  pdf_label=elems_d['pdf_label'],
                                  xlsx_label=elems_d['xlsx_label'],
                                  )

                _d['_ref_flr_speical_inds'] = special_inds['ref']
                _d['_cur_flr_speical_inds'] = special_inds['cur']

                max_ind = max(special_inds['ref'])
                _d['_ref_flr_x'] = ref_X_all[:max_ind+1]
                _d['_ref_flr_z'] = ref_Z_all[:max_ind+1]
                _d['_ref_flr_theta'] = ref_theta_all[:max_ind+1]

                max_ind = max(special_inds['cur'])
                _d['_cur_flr_x'] = res['flr']['columns']['X'][:max_ind+1]
                _d['_cur_flr_z'] = res['flr']['columns']['Z'][:max_ind+1]
                _d['_cur_flr_theta'] = res['flr']['columns']['theta'][:max_ind+1]
                #_d['_cur_flr_next_elem_types'] = \
                    #res['flr']['columns']['NextElementType'][:max_ind+1]

        return dict(versions=pe.__version__, sel_data=sel_data,
                    timestamp=time.time())

    def get_only_lin_props_plot_captions(self):
        """"""

        caption_list = self.plot_lin_props(skip_plots=True)

        return caption_list

    def plot_lin_props(self, skip_plots=False):
        """"""

        report_folderpath = self.report_folderpath
        twiss_plot_opts = self.conf['lattice_props']['twiss_plot_opts']
        twiss_plot_captions = self.conf['lattice_props']['twiss_plot_captions']

        existing_fignums = plt.get_fignums()

        caption_list = []

        for lat_type in list(twiss_plot_opts):
            output_filepath = os.path.join(
                report_folderpath, f'twiss_{lat_type}.pgz')

            try:
                assert len(twiss_plot_opts[lat_type]) == \
                       len(twiss_plot_captions[lat_type])
            except AssertionError:
                print(
                    (f'ERROR: Number of yaml["lattice_props"]["twiss_plot_opts"]["{lat_type}"] '
                     f'and that of yaml["lattice_props"]["twiss_plot_captions"]["{lat_type}"] '
                     f'must match.'))
                raise

            for opts, caption in zip(twiss_plot_opts[lat_type],
                                     twiss_plot_captions[lat_type]):

                if not skip_plots:
                    pe.plot_twiss(output_filepath, print_scalars=[], **opts)

                caption_list.append(plx.NoEscape(caption))

        if skip_plots:
            return caption_list

        twiss_pdf_filepath = os.path.join(report_folderpath, 'twiss.pdf')

        fignums_to_delete = []

        pp = PdfPages(twiss_pdf_filepath)
        page = 0
        for fignum in plt.get_fignums():
            if fignum not in existing_fignums:
                pp.savefig(figure=fignum)
                #plt.savefig(os.path.join(report_folderpath, f'twiss_{page:d}.svg'))
                plt.savefig(os.path.join(report_folderpath, f'twiss_{page:d}.png'),
                            dpi=200)
                page += 1
                fignums_to_delete.append(fignum)
        pp.close()

        #plt.show()

        for fignum in fignums_to_delete:
            plt.close(fignum)

        return caption_list

    def plot_geom_layout(self):
        """"""

        report_folderpath = self.report_folderpath

        existing_fignums = plt.get_fignums()

        d = self.lin_data['req_props']['floor_comparison']

        ref_max_ind = np.max(d['_ref_flr_speical_inds'])
        cur_max_ind = np.max(d['_cur_flr_speical_inds'])

        sort_inds_special_inds = np.argsort(d['_ref_flr_speical_inds'])
        for k in ['_ref_flr_speical_inds', '_cur_flr_speical_inds']:
            d[k] = np.array(d[k])[sort_inds_special_inds]

        sel_zpos = dict(ref=[], cur=[])
        sel_xpos = dict(ref=[], cur=[])

        plt.figure(figsize=(9, 3))
        # Plot reference
        plt.plot(d['_ref_flr_z'][:ref_max_ind+1],
                 d['_ref_flr_x'][:ref_max_ind+1], 'b.-', ms=5, label='Reference')
        max_z = np.max(d['_ref_flr_z'][:ref_max_ind+1])
        for i in d['_ref_flr_speical_inds']:
            plt.plot(d['_ref_flr_z'][i], d['_ref_flr_x'][i], 'b+', ms=20)
            sel_zpos['ref'].append(d['_ref_flr_z'][i])
            sel_xpos['ref'].append(d['_ref_flr_x'][i])
        nonbend_to_bend_inds_forward = np.where(
            np.diff(d['_ref_flr_theta']) != 0.0)[0]
        #nonbend_to_bend_inds_backward = np.sort(len(d['_ref_flr_theta']) + np.where(
            #np.diff(d['_ref_flr_theta'][::-1]) != 0.0)[0] * (-1))
        for bend_ind in nonbend_to_bend_inds_forward:
            z_ini = d['_ref_flr_z'][bend_ind]
            x_ini = d['_ref_flr_x'][bend_ind]
            prev_ind = bend_ind - 1
            z_prev = d['_ref_flr_z'][prev_ind]
            x_prev = d['_ref_flr_x'][prev_ind]
            while z_ini == z_prev:
                prev_ind -= 1
                z_prev = d['_ref_flr_z'][prev_ind]
                x_prev = d['_ref_flr_x'][prev_ind]
                if z_prev == 0:
                    break
            slope = (x_ini - x_prev) / (z_ini - z_prev)

            z_fin = max_z
            x_fin = slope * (z_fin - z_ini) + x_ini
            plt.plot([z_ini, z_fin], [x_ini, x_fin], 'b-.', lw=1)
        #
        # Plot current
        plt.plot(d['_cur_flr_z'][:cur_max_ind+1],
                 d['_cur_flr_x'][:cur_max_ind+1], 'r.-', ms=5, label='Current')
        for i in d['_cur_flr_speical_inds']:
            plt.plot(d['_cur_flr_z'][i], d['_cur_flr_x'][i], 'rx', ms=15)
            sel_zpos['cur'].append(d['_cur_flr_z'][i])
            sel_xpos['cur'].append(d['_cur_flr_x'][i])
        nonbend_to_bend_inds_forward = np.where(
            np.diff(d['_cur_flr_theta']) != 0.0)[0]
        for bend_ind in nonbend_to_bend_inds_forward:
            z_ini = d['_cur_flr_z'][bend_ind]
            x_ini = d['_cur_flr_x'][bend_ind]
            prev_ind = bend_ind - 1
            z_prev = d['_cur_flr_z'][prev_ind]
            x_prev = d['_cur_flr_x'][prev_ind]
            while z_ini == z_prev:
                prev_ind -= 1
                z_prev = d['_cur_flr_z'][prev_ind]
                x_prev = d['_cur_flr_x'][prev_ind]
                if z_prev == 0:
                    break
            slope = (x_ini - x_prev) / (z_ini - z_prev)

            z_fin = max_z
            x_fin = slope * (z_fin - z_ini) + x_ini
            plt.plot([z_ini, z_fin], [x_ini, x_fin], 'r:', lw=1)
        #
        plt.legend(loc='best')
        #plt.axis('image')
        plt.axis('equal')
        plt.xlabel(r'$z\, [\mathrm{m}]$', size=20)
        plt.ylabel(r'$x\, [\mathrm{m}]$', size=20)
        plt.tight_layout()

        for iSel, (z_ref, z_cur, x_ref, x_cur) in enumerate(zip(
            sel_zpos['ref'], sel_zpos['cur'], sel_xpos['ref'], sel_xpos['cur'])):

            clip_zrange = 6.0
            viz_margin = 0.03

            zlim = [max([0.0,
                         np.min([z_ref, z_cur]) - clip_zrange]),
                    min([np.max([z_ref, z_cur]) + clip_zrange,
                         np.max(
                             [np.max(d['_ref_flr_z'][:ref_max_ind+1]) + 0.5,
                              np.max(d['_cur_flr_z'][:cur_max_ind+1]) + 0.5])
                         ])
                    ]
            viz_zlim = [min([z_ref, z_cur]) - viz_margin,
                        max([z_ref, z_cur]) + viz_margin]

            plt.figure()
            clip = np.logical_and(d['_ref_flr_z'][:ref_max_ind+1] >= zlim[0],
                                  d['_ref_flr_z'][:ref_max_ind+1] <= zlim[1],)
            plt.plot(d['_ref_flr_z'][:ref_max_ind+1][clip],
                     d['_ref_flr_x'][:ref_max_ind+1][clip], 'b.-', ms=5,
                     label='Reference')
            i = d['_ref_flr_speical_inds'][iSel]
            plt.plot(d['_ref_flr_z'][i], d['_ref_flr_x'][i], 'b+', ms=20)
            clip = np.logical_and(d['_cur_flr_z'][:cur_max_ind+1] >= zlim[0],
                                  d['_cur_flr_z'][:cur_max_ind+1] <= zlim[1],)
            plt.plot(d['_cur_flr_z'][:cur_max_ind+1],
                     d['_cur_flr_x'][:cur_max_ind+1], 'r.-', ms=5, label='Current')
            i = d['_cur_flr_speical_inds'][iSel]
            plt.plot(d['_cur_flr_z'][i], d['_cur_flr_x'][i], 'rx', ms=15)
            plt.xlim(viz_zlim)
            plt.ylim([min([x_ref, x_cur]) - viz_margin,
                      max([x_ref, x_cur]) + viz_margin])
            plt.grid(True, linestyle=':')
            plt.legend(loc='best')
            plt.xlabel(r'$z\, [\mathrm{m}]$', size=20)
            plt.ylabel(r'$x\, [\mathrm{m}]$', size=20)
            plt.tight_layout()

        flr_pdf_filepath = os.path.join(report_folderpath, 'floor.pdf')

        fignums_to_delete = []

        pp = PdfPages(flr_pdf_filepath)
        page = 0
        for fignum in plt.get_fignums():
            if fignum not in existing_fignums:
                pp.savefig(figure=fignum)
                #plt.savefig(os.path.join(report_folderpath, f'floor_{page:d}.svg'))
                plt.savefig(os.path.join(report_folderpath, f'floor_{page:d}.png'),
                            dpi=200)
                page += 1
                fignums_to_delete.append(fignum)
        pp.close()

        #plt.show()

        for fignum in fignums_to_delete:
            plt.close(fignum)

    def determine_calc_plot_bools(self):
        """"""

        ncf = self.conf['nonlin']

        sel_plots = {k: False for k in self.all_nonlin_calc_types}
        for k, v in ncf['include'].items():
            assert k in self.all_nonlin_calc_types
            sel_plots[k] = v

        nonlin_data_filepaths = self.get_nonlin_data_filepaths()

        do_calc = {}
        do_plot = {}

        # First judge whether to calculate or plot based on the existence of result files
        for k, plot_requested in sel_plots.items():
            if plot_requested:

                if k != 'tswa':
                    if not os.path.exists(nonlin_data_filepaths[k]):
                        b_calc = True
                    else:
                        b_calc = False
                else:
                    b_calc = False
                    for k2 in ['tswa_xminus', 'tswa_xplus',
                               'tswa_yminus', 'tswa_yplus']:
                        if not os.path.exists(nonlin_data_filepaths[k2]):
                            b_calc = True
                            break

                if b_calc:
                    b_plot = True
                else:
                    pdf_fp = os.path.join(self.report_folderpath, f'{k}.pdf')
                    if os.path.exists(pdf_fp):
                        b_plot = False
                    else:
                        b_plot = True

                do_calc[k] = b_calc
                do_plot[k] = b_plot

        # Then override to re-calculate or re-plot, if so requested
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

        return do_calc, do_plot

    def get_nonlin_data_filepaths(self):
        """"""

        output_filetype = 'pgz'
        #output_filetype = 'hdf5'

        ncf = self.conf['nonlin']

        suffix_list = []
        data_file_key_list = []
        for calc_type in self.all_nonlin_calc_types:

            if not ncf['include'].get(calc_type, False):
                continue

            opt_name = ncf['selected_calc_opt_names'][calc_type]
            assert opt_name in ncf['calc_opts'][calc_type]

            if calc_type == 'xy_aper':
                suffix_list.append(
                    f'_xy_aper_{opt_name}.{output_filetype}')
                data_file_key_list.append(calc_type)
            elif calc_type.startswith(('fmap', 'cmap')):
                suffix_list.append(
                    f'_{calc_type}_{opt_name}.{output_filetype}')
                data_file_key_list.append(calc_type)
            elif calc_type == 'tswa':
                for plane in ['x', 'y']:
                    for sign in ['plus', 'minus']:
                        suffix_list.append(
                            f'_tswa_{opt_name}_{plane}{sign}.{output_filetype}')
                        data_file_key_list.append(f'tswa_{plane}{sign}')
            elif calc_type == 'nonlin_chrom':
                suffix_list.append(
                    f'_nonlin_chrom_{opt_name}.{output_filetype}')
                data_file_key_list.append(calc_type)
            elif calc_type == 'mom_aper':
                suffix_list.append(
                    f'_mom_aper_{opt_name}.{output_filetype}')
                data_file_key_list.append(calc_type)
            else:
                raise ValueError

        assert len(suffix_list) == len(data_file_key_list)
        nonlin_data_filepaths = {}
        for k, suffix in zip(data_file_key_list, suffix_list):
            filename = suffix[1:] # remove the first "_"
            nonlin_data_filepaths[k] = os.path.join(
                self.report_folderpath, filename)

        return nonlin_data_filepaths

    def calc_nonlin_props(self, do_calc):
        """"""

        ncf = self.conf['nonlin']

        nonlin_data_filepaths = self.get_nonlin_data_filepaths()
        use_beamline = ncf['use_beamline']
        N_KICKS = ncf.get('N_KICKS', dict(CSBEND=40, KQUAD=40, KSEXT=20, KOCT=20))

        common_remote_opts = ncf['common_remote_opts']

        calc_type = 'xy_aper'
        if (calc_type in nonlin_data_filepaths) and \
           (do_calc[calc_type] or
            (not os.path.exists(nonlin_data_filepaths[calc_type]))):

            print(f'\n*** Starting compuation for "{calc_type}" ***\n')
            self.calc_xy_aper(
                use_beamline, N_KICKS, nonlin_data_filepaths, common_remote_opts)

        calc_type = 'fmap_xy'
        if (calc_type in nonlin_data_filepaths) and \
           (do_calc[calc_type] or
            (not os.path.exists(nonlin_data_filepaths[calc_type]))):

            print(f'\n*** Starting compuation for "{calc_type}" ***\n')
            self.calc_fmap_xy(
                use_beamline, N_KICKS, nonlin_data_filepaths, common_remote_opts)

        calc_type = 'fmap_px'
        if (calc_type in nonlin_data_filepaths) and \
           (do_calc[calc_type] or
            (not os.path.exists(nonlin_data_filepaths[calc_type]))):

            print(f'\n*** Starting compuation for "{calc_type}" ***\n')
            self.calc_fmap_px(
                use_beamline, N_KICKS, nonlin_data_filepaths, common_remote_opts)

        calc_type = 'cmap_xy'
        if (calc_type in nonlin_data_filepaths) and \
           (do_calc[calc_type] or
            (not os.path.exists(nonlin_data_filepaths[calc_type]))):

            print(f'\n*** Starting compuation for "{calc_type}" ***\n')
            self.calc_cmap_xy(
                use_beamline, N_KICKS, nonlin_data_filepaths, common_remote_opts)

        calc_type = 'cmap_px'
        if (calc_type in nonlin_data_filepaths) and \
           (do_calc[calc_type] or
            (not os.path.exists(nonlin_data_filepaths[calc_type]))):

            print(f'\n*** Starting compuation for "{calc_type}" ***\n')
            self.calc_cmap_px(
                use_beamline, N_KICKS, nonlin_data_filepaths, common_remote_opts)

        if ('tswa_xplus' in nonlin_data_filepaths) and \
           (do_calc['tswa'] or
            (not os.path.exists(nonlin_data_filepaths['tswa_xplus'])) or
            (not os.path.exists(nonlin_data_filepaths['tswa_xminus'])) or
            (not os.path.exists(nonlin_data_filepaths['tswa_yplus'])) or
            (not os.path.exists(nonlin_data_filepaths['tswa_yminus']))
            ):

            print(f'\n*** Starting compuation for "tswa" ***\n')
            self.calc_tswa(
                use_beamline, N_KICKS, nonlin_data_filepaths, common_remote_opts)

        calc_type = 'nonlin_chrom'
        if (calc_type in nonlin_data_filepaths) and \
           (do_calc[calc_type] or
            (not os.path.exists(nonlin_data_filepaths[calc_type]))):

            print(f'\n*** Starting compuation for "{calc_type}" ***\n')
            self.calc_nonlin_chrom(
                use_beamline, N_KICKS, nonlin_data_filepaths, common_remote_opts)

        calc_type = 'mom_aper'
        if (calc_type in nonlin_data_filepaths) and \
           (do_calc[calc_type] or
            (not os.path.exists(nonlin_data_filepaths[calc_type]))):

            print(f'\n*** Starting compuation for "{calc_type}" ***\n')
            self.calc_mom_aper(
                use_beamline, N_KICKS, nonlin_data_filepaths, common_remote_opts)

        return nonlin_data_filepaths

    def calc_xy_aper(
        self, use_beamline, N_KICKS, nonlin_data_filepaths, common_remote_opts):
        """"""

        LTE_filepath = self.input_LTE_filepath
        E_MeV = self.conf['E_MeV']
        ncf = self.conf['nonlin']

        calc_type = 'xy_aper'

        output_filepath = nonlin_data_filepaths[calc_type]

        opt_name = ncf['selected_calc_opt_names'][calc_type]
        calc_opts = ncf['calc_opts'][calc_type][opt_name]

        n_turns = calc_opts['n_turns']
        xmax = calc_opts['abs_xmax']
        ymax = calc_opts['abs_ymax']
        ini_ndiv = calc_opts['ini_ndiv']
        n_lines = calc_opts['n_lines']
        neg_y_search = calc_opts.get('neg_y_search', False)

        remote_opts = dict(
            use_sbatch=True, exit_right_after_sbatch=False, pelegant=True,
            job_name=calc_type)
        remote_opts.update(pe.util.deepcopy_dict(common_remote_opts))
        remote_opts.update(pe.util.deepcopy_dict(calc_opts.get('remote_opts', {})))

        pe.nonlin.calc_find_aper_nlines(
            output_filepath, LTE_filepath, E_MeV, xmax=xmax, ymax=ymax,
            ini_ndiv=ini_ndiv, n_lines=n_lines, neg_y_search=neg_y_search,
            n_turns=n_turns, use_beamline=use_beamline, N_KICKS=N_KICKS,
            del_tmp_files=True, run_local=False, remote_opts=remote_opts)

    def _calc_map_xy(
        self, map_type, use_beamline, N_KICKS, nonlin_data_filepaths,
        common_remote_opts):
        """"""

        if map_type not in ('c', 'f'):
            raise ValueError(f'Invalid "map_type": {map_type}')

        LTE_filepath = self.input_LTE_filepath
        E_MeV = self.conf['E_MeV']
        ncf = self.conf['nonlin']

        calc_type = f'{map_type}map_xy'

        output_filepath = nonlin_data_filepaths[calc_type]

        opt_name = ncf['selected_calc_opt_names'][calc_type]
        calc_opts = ncf['calc_opts'][calc_type][opt_name]

        n_turns = calc_opts['n_turns']
        nx, ny = calc_opts['nx'], calc_opts['ny']
        x_offset = calc_opts.get('x_offset', 1e-6)
        y_offset = calc_opts.get('y_offset', 1e-6)
        delta_offset = calc_opts.get('delta_offset', 0.0)
        xmin = calc_opts['xmin'] + x_offset
        xmax = calc_opts['xmax'] + x_offset
        ymin = calc_opts['ymin'] + y_offset
        ymax = calc_opts['ymax'] + y_offset

        remote_opts = dict(
            use_sbatch=True, exit_right_after_sbatch=False, pelegant=True,
            job_name=calc_type)
        remote_opts.update(pe.util.deepcopy_dict(common_remote_opts))
        remote_opts.update(pe.util.deepcopy_dict(calc_opts.get('remote_opts', {})))

        if calc_type == 'fmap_xy':
            kwargs = dict(quadratic_spacing=False, full_grid_output=False)
            func = pe.nonlin.calc_fma_xy
        elif calc_type == 'cmap_xy':
            kwargs = dict(forward_backward=True)
            func = pe.nonlin.calc_cmap_xy
        else:
            raise ValueError

        func(
            output_filepath, LTE_filepath, E_MeV, xmin, xmax, ymin, ymax, nx, ny,
            use_beamline=use_beamline, N_KICKS=N_KICKS,
            n_turns=n_turns, delta_offset=delta_offset,
            del_tmp_files=True, run_local=False, remote_opts=remote_opts,
            **kwargs)

    def calc_fmap_xy(
        self, use_beamline, N_KICKS, nonlin_data_filepaths, common_remote_opts):
        """"""

        self._calc_map_xy('f', use_beamline, N_KICKS, nonlin_data_filepaths,
                          common_remote_opts)

    def calc_cmap_xy(
        self, use_beamline, N_KICKS, nonlin_data_filepaths, common_remote_opts):
        """"""

        self._calc_map_xy('c', use_beamline, N_KICKS, nonlin_data_filepaths,
                          common_remote_opts)

    def _calc_map_px(
        self, map_type, use_beamline, N_KICKS, nonlin_data_filepaths,
        common_remote_opts):
        """"""

        if map_type not in ('c', 'f'):
            raise ValueError(f'Invalid "map_type": {map_type}')

        LTE_filepath = self.input_LTE_filepath
        E_MeV = self.conf['E_MeV']
        ncf = self.conf['nonlin']

        calc_type = f'{map_type}map_px'

        output_filepath = nonlin_data_filepaths[calc_type]

        opt_name = ncf['selected_calc_opt_names'][calc_type]
        calc_opts = ncf['calc_opts'][calc_type][opt_name]

        n_turns = calc_opts['n_turns']
        ndelta, nx = calc_opts['ndelta'], calc_opts['nx']
        x_offset = calc_opts.get('x_offset', 1e-6)
        y_offset = calc_opts.get('y_offset', 1e-6)
        delta_offset = calc_opts.get('delta_offset', 0.0)
        delta_min = calc_opts['delta_min'] + delta_offset
        delta_max = calc_opts['delta_max'] + delta_offset
        xmin = calc_opts['xmin'] + x_offset
        xmax = calc_opts['xmax'] + x_offset

        remote_opts = dict(
            use_sbatch=True, exit_right_after_sbatch=False, pelegant=True,
            job_name=calc_type)
        remote_opts.update(pe.util.deepcopy_dict(common_remote_opts))
        remote_opts.update(pe.util.deepcopy_dict(calc_opts.get('remote_opts', {})))

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
             del_tmp_files=True, run_local=False, remote_opts=remote_opts,
             **kwargs)

    def calc_fmap_px(
        self, use_beamline, N_KICKS, nonlin_data_filepaths, common_remote_opts):
        """"""

        self._calc_map_px('f', use_beamline, N_KICKS, nonlin_data_filepaths,
                          common_remote_opts)

    def calc_cmap_px(
        self, use_beamline, N_KICKS, nonlin_data_filepaths, common_remote_opts):
        """"""

        self._calc_map_px('c', use_beamline, N_KICKS, nonlin_data_filepaths,
                          common_remote_opts)

    def calc_tswa(
        self, use_beamline, N_KICKS, nonlin_data_filepaths, common_remote_opts):
        """"""

        LTE_filepath = self.input_LTE_filepath
        E_MeV = self.conf['E_MeV']
        ncf = self.conf['nonlin']

        calc_type = 'tswa'

        opt_name = ncf['selected_calc_opt_names'][calc_type]
        calc_opts = ncf['calc_opts'][calc_type][opt_name]

        n_turns = calc_opts['n_turns']
        save_fft = calc_opts.get('save_fft', False)
        nx, ny = calc_opts['nx'], calc_opts['ny']
        x_offset = calc_opts.get('x_offset', 1e-6)
        y_offset = calc_opts.get('y_offset', 1e-6)
        abs_xmax = calc_opts['abs_xmax']
        abs_ymax = calc_opts['abs_ymax']

        remote_opts = dict(job_name=calc_type)
        remote_opts.update(pe.util.deepcopy_dict(common_remote_opts))
        remote_opts.update(pe.util.deepcopy_dict(calc_opts.get('remote_opts', {})))

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

            plane_specific_remote_opts = pe.util.deepcopy_dict(remote_opts)
            plane_specific_remote_opts['ntasks'] = min([remote_opts['ntasks'], n])

            for sign, sign_symbol in [('plus', '+'), ('minus', '-')]:

                output_filepath = nonlin_data_filepaths[f'{calc_type}_{plane}{sign}']

                mod_kwargs = pe.util.deepcopy_dict(kwargs)
                mod_kwargs[f'{plane}sign'] = sign_symbol

                func(output_filepath, LTE_filepath, E_MeV, abs_max, n,
                     courant_snyder=True, return_fft_spec=save_fft, save_tbt=False,
                     n_turns=n_turns, N_KICKS=N_KICKS,
                     del_tmp_files=True, run_local=False,
                     remote_opts=plane_specific_remote_opts, **mod_kwargs)

    def calc_nonlin_chrom(
        self, use_beamline, N_KICKS, nonlin_data_filepaths, common_remote_opts):
        """"""

        LTE_filepath = self.input_LTE_filepath
        E_MeV = self.conf['E_MeV']
        ncf = self.conf['nonlin']

        calc_type = 'nonlin_chrom'

        output_filepath = nonlin_data_filepaths[calc_type]

        opt_name = ncf['selected_calc_opt_names'][calc_type]
        calc_opts = ncf['calc_opts'][calc_type][opt_name]

        n_turns = calc_opts['n_turns']
        save_fft = calc_opts.get('save_fft', False)
        ndelta = calc_opts['ndelta']
        x_offset = calc_opts.get('x_offset', 1e-6)
        y_offset = calc_opts.get('y_offset', 1e-6)
        delta_offset = calc_opts.get('delta_offset', 0.0)
        delta_min = calc_opts['delta_min'] + delta_offset
        delta_max = calc_opts['delta_max'] + delta_offset

        remote_opts = dict(job_name=calc_type)
        remote_opts.update(pe.util.deepcopy_dict(common_remote_opts))
        remote_opts.update(pe.util.deepcopy_dict(calc_opts.get('remote_opts', {})))
        #
        remote_opts['ntasks'] = min([remote_opts['ntasks'], ndelta])

        pe.nonlin.calc_chrom_track(
            output_filepath, LTE_filepath, E_MeV, delta_min, delta_max, ndelta,
            courant_snyder=True, return_fft_spec=save_fft, save_tbt=False,
            use_beamline=use_beamline, N_KICKS=N_KICKS,
            n_turns=n_turns, x0_offset=x_offset, y0_offset=y_offset,
            del_tmp_files=True, run_local=False, remote_opts=remote_opts)

    def calc_mom_aper(
        self, use_beamline, N_KICKS, nonlin_data_filepaths, common_remote_opts):
        """"""

        LTE_filepath = self.input_LTE_filepath
        E_MeV = self.conf['E_MeV']
        ncf = self.conf['nonlin']

        calc_type = 'mom_aper'

        output_filepath = nonlin_data_filepaths[calc_type]

        opt_name = ncf['selected_calc_opt_names'][calc_type]
        calc_opts = ncf['calc_opts'][calc_type][opt_name]

        n_turns = calc_opts.pop('n_turns')
        # Handle special specifications
        if calc_opts['s_end'] == 'one_period':
            calc_opts['s_end'] = float(
                self.lin_data['circumf'] / self.lin_data['n_periods_in_ring'])
            # Must be converted to Python float, instead of numpy's float.
            # Otherwise, when you try to dump self.conf into YAML file, it crashes.

        remote_opts = dict(
            use_sbatch=True, exit_right_after_sbatch=False, pelegant=True,
            job_name=calc_type)
        remote_opts.update(pe.util.deepcopy_dict(common_remote_opts))
        remote_opts.update(pe.util.deepcopy_dict(calc_opts.get('remote_opts', {})))
        #
        # Warning from ELEGANT: for best parallel efficiency in output_mode=0,
        # the number of elements divided by the number of processors should be
        # an integer or slightly below an integer.
        elem_names = [
            line.split(':')[1].strip() for line in
            self.lin_data['flat_elem_s_name_type_list'][2:]] # exclude header lines
        n_matched = len([
            elem_name for elem_name in elem_names
            if fnmatch.fnmatch(elem_name, calc_opts['include_name_pattern'])])
        assert n_matched >= 1
        remote_opts['ntasks'] = min([remote_opts['ntasks'], n_matched])
        assert remote_opts['ntasks'] >= 1

        pe.nonlin.calc_mom_aper(
            output_filepath, LTE_filepath, E_MeV, n_turns=n_turns,
            use_beamline=use_beamline, N_KICKS=N_KICKS, del_tmp_files=True,
            run_local=False, remote_opts=remote_opts, **calc_opts)

    def _save_nonlin_plots_to_pdf(self, calc_type, existing_fignums):
        """"""

        pdf_filepath = os.path.join(self.report_folderpath, f'{calc_type}.pdf')
        #svg_filepath_template = os.path.join(
            #self.report_folderpath, f'{calc_type}_{{page:d}}.svg')
        png_filepath_template = os.path.join(
            self.report_folderpath, f'{calc_type}_{{page:d}}.png')

        pp = PdfPages(pdf_filepath)

        page = 0
        for fignum in [fignum for fignum in plt.get_fignums()
                       if fignum not in existing_fignums]:
            pp.savefig(figure=fignum)
            #plt.savefig(svg_filepath_template.format(page=page))
            plt.savefig(png_filepath_template.format(page=page), dpi=200)
            page += 1
            plt.close(fignum)

        pp.close()

    def plot_nonlin_props(self, do_plot):
        """"""

        ncf = self.conf['nonlin']

        nonlin_data_filepaths = self.get_nonlin_data_filepaths()

        existing_fignums = plt.get_fignums()

        calc_type = 'xy_aper'
        if (calc_type in nonlin_data_filepaths) and do_plot[calc_type]:
            xy_aper_data = pe.nonlin.plot_find_aper_nlines(
                nonlin_data_filepaths[calc_type], title='', xlim=None, ylim=None)

            self._save_nonlin_plots_to_pdf(calc_type, existing_fignums)

            with open(self.suppl_plot_data_filepath['xy_aper'], 'wb') as f:
                pickle.dump(xy_aper_data, f)

        calc_type = 'fmap_xy'
        if (calc_type in nonlin_data_filepaths) and do_plot[calc_type]:
            pe.nonlin.plot_fma_xy(
                nonlin_data_filepaths[calc_type], title='',
                is_diffusion=True, scatter=False)

            self._save_nonlin_plots_to_pdf(calc_type, existing_fignums)

        calc_type = 'fmap_px'
        if (calc_type in nonlin_data_filepaths) and do_plot[calc_type]:
            pe.nonlin.plot_fma_px(
                nonlin_data_filepaths[calc_type], title='',
                is_diffusion=True, scatter=False)

            self._save_nonlin_plots_to_pdf(calc_type, existing_fignums)

        calc_type = 'cmap_xy'
        if (calc_type in nonlin_data_filepaths) and do_plot[calc_type]:
            _plot_kwargs = ncf.get(f'{calc_type}_plot_opts', {})
            pe.nonlin.plot_cmap_xy(
                nonlin_data_filepaths[calc_type], title='', is_log10=True,
                scatter=False, **_plot_kwargs)

            self._save_nonlin_plots_to_pdf(calc_type, existing_fignums)

        calc_type = 'cmap_px'
        if (calc_type in nonlin_data_filepaths) and do_plot[calc_type]:
            _plot_kwargs = ncf.get(f'{calc_type}_plot_opts', {})
            pe.nonlin.plot_cmap_px(
                nonlin_data_filepaths[calc_type], title='',
                is_log10=True, scatter=False, **_plot_kwargs)

            self._save_nonlin_plots_to_pdf(calc_type, existing_fignums)

        calc_type = 'tswa'
        if (f'tswa_xplus' in nonlin_data_filepaths) and do_plot[calc_type]:
            _plot_kwargs = ncf.get(f'{calc_type}_plot_opts', {})

            plot_plus_minus_combined = _plot_kwargs.pop(
                'plot_plus_minus_combined', True)

            _plot_kwargs['plot_xy0'] = _plot_kwargs.get('plot_xy0', True)

            _plot_kwargs['plot_Axy'] = _plot_kwargs.get('plot_Axy', False)
            # ^ These plots will NOT be included into the main report, but will be
            #   saved to the "tswa" PDF file if set to True.
            _plot_kwargs['use_time_domain_amplitude'] = _plot_kwargs.get(
                'use_time_domain_amplitude', True)
            # ^ Only relevant when "plot_Axy=True"

            _plot_kwargs['plot_fft'] = _plot_kwargs.get('plot_fft', False)
            # ^ If True, it may take a while to save the "tswa" PDF file. But these
            #   FFT color plots will NOT be included into the main report. These
            #   plots may be useful for debugging or for deciding the number of
            #   divisions for x0/y0 arrays.

            _plot_kwargs['footprint_nuxlim'] = _plot_kwargs.get(
                'footprint_nuxlim', [0.0, 1.0])
            _plot_kwargs['footprint_nuylim'] = _plot_kwargs.get(
                'footprint_nuylim', [0.0, 1.0])
            _plot_kwargs['fit_xmin'] = _plot_kwargs.get('fit_xmin', -0.5e-3)
            _plot_kwargs['fit_xmax'] = _plot_kwargs.get('fit_xmax', +0.5e-3)
            _plot_kwargs['fit_ymin'] = _plot_kwargs.get('fit_ymin', -0.25e-3)
            _plot_kwargs['fit_ymax'] = _plot_kwargs.get('fit_ymax', +0.25e-3)

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

            MathText = plx.MathText # for short-hand notation

            tswa_data = {}

            if plot_plus_minus_combined:

                for plane in ['x', 'y']:

                    tswa_data[plane] = {}

                    out = pe.nonlin.plot_tswa_both_sides(
                        nonlin_data_filepaths[f'tswa_{plane}plus'],
                        nonlin_data_filepaths[f'tswa_{plane}minus'],
                        title='', **_plot_kwargs
                    )
                    tswa_data[plane].update(out)

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
            else:
                fit_abs_xmax = dict(plus=_plot_kwargs['fit_xmax'],
                                    minus=np.abs(_plot_kwargs['fit_xmin']))
                fit_abs_ymax = dict(plus=_plot_kwargs['fit_ymax'],
                                    minus=np.abs(_plot_kwargs['fit_ymin']))

                for plane in ['x', 'y']:
                    tswa_data[plane] = {}
                    for sign in ['plus', 'minus']:
                        data_key = f'tswa_{plane}{sign}'
                        if plane == 'x':
                            out = pe.nonlin.plot_tswa(
                                nonlin_data_filepaths[data_key], title='',
                                fit_abs_xmax=fit_abs_xmax[sign], **_plot_kwargs)
                        else:
                            out = pe.nonlin.plot_tswa(
                                nonlin_data_filepaths[data_key], title='',
                                fit_abs_ymax=fit_abs_ymax[sign], **_plot_kwargs)
                        tswa_data[plane].update(out)

                        if sign == 'plus':
                            if _plot_kwargs['plot_xy0']:
                                tswa_captions.append(
                                    MathText(r'\nu') + ' vs. ' + MathText(fr'{plane}_0'))
                                tswa_caption_keys.append(f'nu_vs_{plane}0{sign}')
                            if _plot_kwargs['plot_Axy']:
                                tswa_captions.append(
                                    MathText(r'\nu') + ' vs. ' + MathText(fr'A_{plane} (> 0)'))
                                tswa_caption_keys.append(f'nu_vs_A{plane}{sign}')
                            tswa_captions.append(
                                'Tune footprint vs. ' + MathText(fr'{plane}_0'))
                            tswa_caption_keys.append(f'tunefootprint_vs_{plane}0{sign}')
                            if _plot_kwargs['plot_fft']:
                                tswa_captions.append(
                                    'FFT ' + MathText(r'\nu_x') + ' vs. ' + MathText(fr'{plane}_0'))
                                tswa_caption_keys.append(f'fft_nux_vs_{plane}0{sign}')
                                tswa_captions.append(
                                    'FFT ' + MathText(r'\nu_y') + ' vs. ' + MathText(fr'{plane}_0'))
                                tswa_caption_keys.append(f'fft_nuy_vs_{plane}0{sign}')
                        else:
                            if _plot_kwargs['plot_xy0']:
                                tswa_captions.append(
                                    MathText(r'\nu') + ' vs. ' + MathText(fr'-{plane}_0'))
                                tswa_caption_keys.append(f'nu_vs_{plane}0{sign}')
                            if _plot_kwargs['plot_Axy']:
                                tswa_captions.append(
                                    MathText(r'\nu') + ' vs. ' + MathText(fr'A_{plane} (< 0)'))
                                tswa_caption_keys.append(f'nu_vs_A{plane}{sign}')
                            tswa_captions.append(
                                'Tune footprint vs. ' + MathText(fr'-{plane}_0'))
                            tswa_caption_keys.append(f'tunefootprint_vs_{plane}0{sign}')
                            if _plot_kwargs['plot_fft']:
                                tswa_captions.append(
                                    'FFT ' + MathText(r'\nu_x') + ' vs. ' + MathText(fr'-{plane}_0'))
                                tswa_caption_keys.append(f'fft_nux_vs_{plane}0{sign}')
                                tswa_captions.append(
                                    'FFT ' + MathText(r'\nu_y') + ' vs. ' + MathText(fr'-{plane}_0'))
                                tswa_caption_keys.append(f'fft_nuy_vs_{plane}0{sign}')

            self._save_nonlin_plots_to_pdf(calc_type, existing_fignums)

            tswa_page_caption_list = []
            for k in sel_tswa_caption_keys:
                i = tswa_caption_keys.index(k)
                page = i + 1
                tswa_page_caption_list.append((page, tswa_captions[i]))

            with open(self.suppl_plot_data_filepath['tswa'], 'wb') as f:
                pickle.dump([tswa_page_caption_list, tswa_data], f)

        calc_type = 'nonlin_chrom'
        if (calc_type in nonlin_data_filepaths) and do_plot[calc_type]:
            _plot_kwargs = ncf.get(f'{calc_type}_plot_opts', {})

            _plot_kwargs['plot_fft'] = _plot_kwargs.get('plot_fft', False)
            # ^ If True, it may take a while to save the "nonlin_chrom" PDF file.
            # ^ But these FFT color plots will NOT be included into
            # ^ the main report. These plots may be useful for debugging
            # ^ or for deciding the number of divisions for delta arrays.
            _plot_kwargs['max_chrom_order'] = _plot_kwargs.get('max_chrom_order', 4)
            _plot_kwargs['fit_deltalim'] = _plot_kwargs.get(
                'fit_deltalim', [-2e-2, +2e-2])

            nonlin_chrom_data = pe.nonlin.plot_chrom(
                nonlin_data_filepaths[calc_type], title='', **_plot_kwargs)

            self._save_nonlin_plots_to_pdf(calc_type, existing_fignums)

            with open(self.suppl_plot_data_filepath['nonlin_chrom'], 'wb') as f:
                pickle.dump(nonlin_chrom_data, f)

        calc_type = 'mom_aper'
        if (calc_type in nonlin_data_filepaths) and do_plot[calc_type]:
            mom_aper_data = pe.nonlin.plot_mom_aper(
                nonlin_data_filepaths[calc_type], title='', slim=None,
                deltalim=None)

            self._save_nonlin_plots_to_pdf(calc_type, existing_fignums)

            with open(self.suppl_plot_data_filepath['mom_aper'], 'wb') as f:
                pickle.dump(mom_aper_data, f)


    def get_default_config(self, report_version, example=False):
        """"""

        func_dict = self._get_default_config_func_dict()

        return func_dict[report_version](example=example)

    def _get_default_config_func_dict(self):
        """"""

        # `None` for latest version
        func_dict = {None: self._get_default_config_v1_0,
                     '1.0': self._get_default_config_v1_0}

        return func_dict

    def _get_default_config_v1_0(self, example=False):
        """"""

        report_version = '1.0'

        com_map = yaml.comments.CommentedMap
        com_seq = yaml.comments.CommentedSeq
        sqss = yaml.scalarstring.SingleQuotedScalarString

        anchors = {}

        conf = com_map()

        _yaml_append_map(conf, 'report_class', 'nsls2u_default',
                         eol_comment='REQUIRED')

        _yaml_append_map(conf, 'report_version', sqss(report_version),
                         eol_comment='REQUIRED')

        if example:
            _yaml_append_map(conf, 'report_author', '')
            _yaml_append_map(conf, 'enable_pyelegant_stdout', False)

        _yaml_append_map(conf, 'lattice_author', '', eol_comment='REQUIRED',
                         before_comment='\n')
        _yaml_append_map(conf, 'lattice_keywords', [], eol_comment='REQUIRED')
        _yaml_append_map(
            conf, 'lattice_received_date',
            time.strftime('%m/%d/%Y', time.localtime()), eol_comment='REQUIRED')

        if example:
            _yaml_append_map(conf, 'orig_LTE_filepath', '')

        _yaml_append_map(conf, 'E_MeV', 3e3, eol_comment='REQUIRED',
                         before_comment='\n')

        # ######################################################################

        _yaml_append_map(conf, 'input_LTE', com_map(),
                         eol_comment='REQUIRED', before_comment='\n')
        d = conf['input_LTE']
        #
        _yaml_append_map(d, 'filepath', sqss('?.lte'), eol_comment='REQUIRED')
        #
        if example:
            comment = '''
            If "load_param" is True, you must specify "base_filepath" (%s.lte) and
            "param_filepath" (%s.param).'''
            _yaml_append_map(d, 'load_param', False,
                             before_comment=comment, before_indent=2)
            _yaml_append_map(d, 'base_LTE_filepath', '')
            _yaml_append_map(d, 'param_filepath', '')

            comment = '''
            If "zeroSexts_filepath" is specified and not an empty string, the script
            assumes this the path to the LTE file with all the sextupoles turned off.
            In this case, the value of "regenerate_zeroSexts" will be ignored.
            '''
            _yaml_append_map(d, 'zeroSexts_filepath', '',
                             before_comment=comment, before_indent=2)
            comment = '''\
            Whether "regenerate_zeroSexts" is True or False, if "zeroSexts_filepath"
            is not specified and an already auto-generated zero-sexts LTE file does
            not exist, the script will generate a zero-sexts version of the input
            LTE file. If "regenerate_zeroSexts" is True, the script will regenerate
            the zero-sexts LTE file even when it already exists.
            '''
            _yaml_append_map(d, 'regenerate_zeroSexts', False,
                             before_comment=comment, before_indent=2)

        # ######################################################################

        _yaml_append_map(conf, 'use_beamline_cell', sqss('SUPCELL'),
                         eol_comment='REQUIRED', before_comment='\n')
        conf['use_beamline_cell'].yaml_set_anchor('use_beamline_cell')
        anchors['use_beamline_cell'] = conf['use_beamline_cell']
        _yaml_append_map(conf, 'use_beamline_ring', sqss('RING'),
                         eol_comment='REQUIRED')
        conf['use_beamline_ring'].yaml_set_anchor('use_beamline_ring')
        anchors['use_beamline_ring'] = conf['use_beamline_ring']

        # ##########################################################################

        _yaml_append_map(conf, 'report_paragraphs', com_map(), before_comment='\n')
        d = conf['report_paragraphs']
        if example:
            comment = '''
            Supply paragraphs to be added into the "Lattice Description" section of the
            report as a list of strings without newline characters.'''
            _yaml_append_map(d, 'lattice_description', [],
                             before_comment=comment, before_indent=2)

            comment = '''
            Supply paragraphs to be added into the "Lattice Properties" section of the
            report as a list of strings without newline characters.'''
            _yaml_append_map(d, 'lattice_properties', [],
                             before_comment=comment, before_indent=2)

        # ##########################################################################

        _yaml_append_map(conf, 'lattice_props', com_map(), before_comment='\n')
        d = conf['lattice_props']

        if example:
            _yaml_append_map(d, 'recalc', False, before_comment='\n')
            _yaml_append_map(d, 'replot', False)

        #---------------------------------------------------------------------------

        if example:
            _yaml_append_map(d, 'twiss_calc_opts', com_map(), before_comment='\n')
            d2 = d['twiss_calc_opts']

            _yaml_append_map(d2, 'one_period', com_map())
            d3 = d2['one_period']
            _yaml_append_map(d3, 'use_beamline', anchors['use_beamline_cell'])
            _yaml_append_map(d3, 'element_divisions', 10)

            _yaml_append_map(d2, 'ring_natural', com_map(),
                             eol_comment='K2 values of all sextupoles set to zero')
            d3 = d2['ring_natural']
            _yaml_append_map(d3, 'use_beamline', anchors['use_beamline_ring'])

            _yaml_append_map(d2, 'ring', com_map())
            d3 = d2['ring']
            _yaml_append_map(d3, 'use_beamline', anchors['use_beamline_ring'])

        #---------------------------------------------------------------------------

        _yaml_append_map(d, 'twiss_plot_opts', com_map(), before_comment='\n',
                         eol_comment='REQUIRED')
        d2 = d['twiss_plot_opts']

        _yaml_append_map(d2, 'one_period', com_seq())
        d3 = d2['one_period']
        #
        m = com_map(right_margin_adj = 0.85)
        m.fa.set_flow_style()
        d3.append(m)
        #
        zoom_in = com_map(
            right_margin_adj = 0.85, slim = [0, 9],
            disp_elem_names = {
                'bends': True, 'quads': True, 'sexts': True, 'octs': True,
                'font_size': 8, 'extra_dy_frac': 0.05})
        zoom_in.fa.set_flow_style()
        zoom_in.yaml_set_anchor('zoom_in')
        anchors['zoom_in'] = zoom_in
        d3.append(zoom_in)
        #
        for slim in ([4, 16], [14, 23]):
            sq = com_seq(slim)
            sq.fa.set_flow_style()
            overwrite = com_map(slim = sq)
            overwrite.add_yaml_merge([(0, anchors['zoom_in'])])
            d3.append(overwrite)

        if example:
            _yaml_append_map(d2, 'ring_natural', com_seq())
            _yaml_append_map(d2, 'ring', com_seq())

        #---------------------------------------------------------------------------

        _yaml_append_map(d, 'twiss_plot_captions', com_map(), before_comment='\n',
                         eol_comment='REQUIRED')
        d2 = d['twiss_plot_captions']

        _yaml_append_map(d2, 'one_period', com_seq())
        d3 = d2['one_period']
        #
        d3.append(sqss('Twiss functions for 2 cells (1 super-period).'))
        d3.append(sqss('Twiss functions $(0 \le s \le 9)$.'))
        d3.append(sqss('Twiss functions $(4 \le s \le 16)$.'))
        d3.append(sqss('Twiss functions $(14 \le s \le 23)$.'))

        if example:
            _yaml_append_map(d2, 'ring_natural', com_seq())
            _yaml_append_map(d2, 'ring', com_seq())

        #---------------------------------------------------------------------------

        _yaml_append_map(d, 'req_props', com_map(), before_comment='\n')
        d2 = d['req_props']

        _yaml_append_map(d2, 'beta', com_map())
        d3 = d2['beta']
        #
        spec = com_map(name = sqss('LS_marker_elem_name'), occur = 0)
        _yaml_append_map(d3, 'LS', spec,
                         eol_comment='(betax, betay) @ Long-Straight Center')
        #
        spec = com_map(name = sqss('SS_marker_elem_name'), occur = 0)
        _yaml_append_map(d3, 'SS', spec,
                         eol_comment='(betax, betay) @ Short-Straight Center')

        _yaml_append_map(d2, 'length', com_map())
        d3 = d2['length']
        #
        name_list = com_seq(['drift_elem_name_1', 'drift_elem_name_2'])
        name_list.fa.set_flow_style()
        spec = com_map(name_list = name_list, multiplier = 2.0)
        _yaml_append_map(d3, 'LS', spec, eol_comment='Length for Long Straight')
        #
        name_list = com_seq(['drift_elem_name_1', 'drift_elem_name_2'])
        name_list.fa.set_flow_style()
        spec = com_map(name_list = name_list, multiplier = 2.0)
        _yaml_append_map(d3, 'SS', spec, eol_comment='Length for Short Straight')

        _yaml_append_map(d2, 'floor_comparison', com_map())
        d3 = d2['floor_comparison']
        #
        _yaml_append_map(d3, 'ref_flr_filepath', sqss('?.flr'),
                         eol_comment='REQUIRED if "floor_comparison" is specified')
        d3['ref_flr_filepath'].yaml_set_anchor('ref_flr_filepath')
        anchors['ref_flr_filepath'] = d3['ref_flr_filepath']
        #
        ref_elem = com_map(name = 'LS_center_marker_elem_name_in_ref_lattice',
                           occur = 1)
        ref_elem.fa.set_flow_style()
        cur_elem = com_map(name = 'LS_center_marker_elem_name_in_cur_lattice',
                           occur = 1)
        cur_elem.fa.set_flow_style()
        spec = com_map(
            ref_elem = ref_elem, cur_elem = cur_elem)
        _yaml_append_map(
            d3, 'LS', spec,
            eol_comment='Source Point Diff. @ LS (Delta_x, Delta_z)')
        #
        ref_elem = com_map(name = 'SS_center_marker_elem_name_in_ref_lattice',
                           occur = 0)
        ref_elem.fa.set_flow_style()
        cur_elem = com_map(name = 'SS_center_marker_elem_name_in_cur_lattice',
                           occur = 0)
        cur_elem.fa.set_flow_style()
        spec = com_map(
            ref_elem = ref_elem, cur_elem = cur_elem)
        _yaml_append_map(
            d3, 'SS', spec,
            eol_comment='Source Point Diff. @ SS (Delta_x, Delta_z)')

        #---------------------------------------------------------------------------

        if example:
            _yaml_append_map(d, 'opt_props', com_map(), before_comment='\n')
            d2 = d['opt_props']

            _yaml_append_map(d2, 'beta', com_map())
            d3 = d2['beta']
            #

            xlsx_label = {}
            for plane in ['x', 'y']:
                xlsx_label[plane] = com_seq([
                    'italic_greek', sqss('beta'), 'italic_sub', sqss(plane),
                    'normal', sqss(' at Somewhere')])
                xlsx_label[plane].fa.set_flow_style()
            #
            spec = com_map(
                pdf_label = sqss(r'$(\beta_x, \beta_y)$ at Somewhere'),
                xlsx_label = com_map(**xlsx_label),
                name = sqss('some_marker_elem_name'), occur = 0)
            _yaml_append_map(d3, 'somewhere', spec)


            _yaml_append_map(d2, 'phase_adv', com_map())
            d3 = d2['phase_adv']
            #
            elem1 = com_map(name = 'LS_marker_elem_name', occur = 0)
            elem1.fa.set_flow_style()
            elem2 = com_map(name = 'SS_marker_elem_name', occur = 0)
            elem2.fa.set_flow_style()
            #
            xlsx_label = {}
            for plane, plane_word in [('x', 'Horizontal'), ('y', 'Vertical')]:
                xlsx_label[plane] = com_seq([
                    'normal', sqss(f'{plane_word} Phase Advance btw. Disp. Bumps '),
                    'italic_greek', sqss('Delta'), 'italic_greek', sqss('nu'),
                    'italic_sub', sqss(plane)])
                xlsx_label[plane].fa.set_flow_style()
            #
            spec = com_map(
                pdf_label = sqss(
                    r'Phase Advance btw. Disp. Bumps $(\Delta\nu_x, \Delta\nu_y)$'),
                xlsx_label = com_map(**xlsx_label),
                elem1 = elem1, elem2 = elem2, multiplier = 1.0,
            )
            _yaml_append_map(d3, 'MDISP 0&1', spec)


            _yaml_append_map(d2, 'length', com_map())
            d3 = d2['length']
            #
            name_list = com_seq(['drift_elem_name_1', 'drift_elem_name_2'])
            name_list.fa.set_flow_style()
            #
            xlsx_label = com_seq([
                'normal', sqss('Length of Some Consecutive Elements')])
            xlsx_label.fa.set_flow_style()
            #
            spec = com_map(
                pdf_label = sqss('Length of Some Consecutive Elements'),
                xlsx_label = xlsx_label,
                name_list = name_list,
                multiplier = 2.0,
            )
            _yaml_append_map(d3, 'some_consecutive_elements', spec)


            _yaml_append_map(d2, 'floor_comparison', com_map())
            d3 = d2['floor_comparison']
            #
            _yaml_append_map(
                d3, 'ref_flr_filepath', anchors['ref_flr_filepath'])
            #
            ref_elem = com_map(
                name = 'some_marker_elem_name_in_ref_lattice', occur = 0)
            ref_elem.fa.set_flow_style()
            cur_elem = com_map(
                name = 'some_marker_elem_name_in_cur_lattice', occur = 0)
            cur_elem.fa.set_flow_style()
            #
            xlsx_label = {}
            for plane in ['x', 'z']:
                xlsx_label[plane] = com_seq([
                    'normal', sqss('Source Point Diff. at Somewhere '),
                    'italic_greek', sqss('Delta'), 'italic', sqss(plane)])
                xlsx_label[plane].fa.set_flow_style()
            #
            spec = com_map(
                pdf_label = sqss(
                    r'Source Point Diff. at Somewhere $(\Delta x, \Delta z)$'),
                xlsx_label = com_map(**xlsx_label),
                ref_elem = ref_elem, cur_elem = cur_elem)
            _yaml_append_map(d3, 'somewhere', spec)


        #---------------------------------------------------------------------------

        if example:

            comment = '''
            You can here specify optional properties to be appended to the table
            with the generated report PDF file. The related option "pdf_table_order"
            allows you to fully control which appears and in which order in the
            table, while this option only allows you to append optional properties
            to the table. If "pdf_table_order" is specified, this option will
            be ignored.
            '''
            _yaml_append_map(
                d, 'append_opt_props_to_pdf_table', com_seq(),
                before_comment=comment, before_indent=2)
            d2 = d['append_opt_props_to_pdf_table']
            #
            for i, (prop_name_or_list, comment) in enumerate([
                (['opt_props', 'phase_adv', 'MDISP 0&1'], None),
            ]):

                if isinstance(prop_name_or_list, str):
                    d2.append(sqss(prop_name_or_list))
                else:
                    prop_list = com_seq([sqss(v) for v in prop_name_or_list])
                    prop_list.fa.set_flow_style()
                    d2.append(prop_list)

                _kwargs = dict(column=0)
                if comment is not None:
                    d2.yaml_add_eol_comment(comment, len(d2)-1, **_kwargs)

            comment = '''
            You can here specify the order of the computed lattice property values in
            the table within the generated report PDF file.'''
            _yaml_append_map(
                d, 'pdf_table_order', com_seq(), before_comment=comment,
                before_indent=2)
            d2 = d['pdf_table_order']
            #
            for i, (prop_name_or_list, comment) in enumerate([
                ('E_GeV', 'Beam energy'),
                ('eps_x', 'Natural horizontal emittance'),
                ('J', 'Damping partitions'),
                ('tau', 'Damping times'),
                ('nu', 'Ring tunes'),
                ('ksi_nat', 'Natural chromaticities'),
                ('ksi_cor', 'Corrected chromaticities'),
                ('alphac', 'Momentum compaction'),
                ('U0', 'Energy loss per turn'),
                ('sigma_delta', 'Energy spread'),
                (['req_props', 'beta', 'LS'], None),
                (['req_props', 'beta', 'SS'], None),
                ('max_beta', 'Max beta functions'),
                ('min_beta', 'Min beta functions'),
                ('max_min_etax', 'Max & Min etax'),
                (['opt_props', 'phase_adv', 'MDISP 0&1'], None),
                (['req_props', 'length', 'LS'], None),
                (['req_props', 'length', 'SS'], None),
                ('circumf', 'Circumference'),
                (['req_props', 'floor_comparison', 'circumf_change_%'],
                 'Circumference change [%] from Reference Lattice'),
                ('n_periods_in_ring', 'Number of super-periods for a full ring'),
                (['req_props', 'floor_comparison', 'LS'], None),
                (['req_props', 'floor_comparison', 'SS'], None),
                ('frev', 'Revolution frequency'),
            ]):

                if isinstance(prop_name_or_list, str):
                    d2.append(sqss(prop_name_or_list))
                else:
                    prop_list = com_seq([sqss(v) for v in prop_name_or_list])
                    prop_list.fa.set_flow_style()
                    d2.append(prop_list)

                _kwargs = dict(column=0)
                if comment is not None:
                    d2.yaml_add_eol_comment(comment, len(d2)-1, **_kwargs)

            comment = '''
            You can here specify optional properties to be appended to the table
            with the generated report Excel file. The related option "xlsx_table_order"
            allows you to fully control which appears and in which order in the
            table, while this option only allows you to append optional properties
            to the table. If "xlsx_table_order" is specified, this option will
            be ignored.
            '''
            _yaml_append_map(
                d, 'append_opt_props_to_xlsx_table', com_seq(),
                before_comment=comment, before_indent=2)
            d2 = d['append_opt_props_to_xlsx_table']
            #
            for i, (prop_name_or_list, comment) in enumerate([
                (['opt_props', 'phase_adv', 'MDISP 0&1', 'x'],
                 'Horizontal phase advance btw. dispersion bumps'),
                (['opt_props', 'phase_adv', 'MDISP 0&1', 'y'],
                 'Vertical phase advance btw. dispersion bumps'),
            ]):

                if isinstance(prop_name_or_list, str):
                    d2.append(sqss(prop_name_or_list))
                else:
                    prop_list = com_seq([sqss(v) for v in prop_name_or_list])
                    prop_list.fa.set_flow_style()
                    d2.append(prop_list)

                _kwargs = dict(column=0)
                if comment is not None:
                    d2.yaml_add_eol_comment(comment, len(d2)-1, **_kwargs)

            comment = '''
            You can here specify the order of the computed lattice property values in
            the table within the generated report Excel file.'''
            _yaml_append_map(
                d, 'xlsx_table_order', com_seq(), before_comment=comment,
                before_indent=2)
            d2 = d['xlsx_table_order']
            #
            for i, (prop_name_or_list, comment) in enumerate([
                ('E_GeV', 'Beam energy'),
                ('circumf', 'Circumference'),
                ('eps_x', 'Natural horizontal emittance'),
                ('nux', 'Horizontal tune'),
                ('nuy', 'Vertical tune'),
                ('ksi_nat_x', 'Horizontal natural chromaticity'),
                ('ksi_nat_y', 'Vertical natural chromaticity'),
                ('ksi_cor_x', 'Horizontal corrected chromaticity'),
                ('ksi_cor_y', 'Vertical corrected chromaticity'),
                ('alphac', 'Momentum compaction'),
                ('Jx', 'Horizontal damping partition number'),
                ('Jy', 'Vertical damping partition number'),
                ('Jdelta', 'Longitudinal damping partition number'),
                ('taux', 'Horizontal damping time'),
                ('tauy', 'Vertical damping time'),
                ('taudelta', 'Longitudinal damping time'),
                ('sigma_delta', 'Energy spread'),
                ('U0', 'Energy loss per turn'),
                ('frev', 'Revolution frequency'),
                (['req_props', 'beta', 'LS', 'x'],
                 'Horizontal beta at Long-Straight center'),
                (['req_props', 'beta', 'LS', 'y'],
                 'Vertical beta at Long-Straight center'),
                (['req_props', 'beta', 'SS', 'x'],
                 'Horizontal beta at Short-Straight center'),
                (['req_props', 'beta', 'SS', 'y'],
                 'Vertical beta at Short-Straight center'),
                ('max_betax', 'Max horizontal beta function'),
                ('max_betay', 'Max vertical beta function'),
                ('min_betax', 'Min horizontal beta function'),
                ('min_betay', 'Min vertical beta function'),
                ('max_etax', 'Max horizontal dispersion'),
                ('min_etax', 'Min horizontal dispersion'),
                (['req_props', 'length', 'LS'], 'Length of Long Straight'),
                (['req_props', 'length', 'SS'], 'Length of Short Straight'),
                ('n_periods_in_ring', 'Number of super-periods for a full ring'),
                ('straight_frac', 'Fraction of straight sections'),
                (['req_props', 'floor_comparison', 'circumf_change_%'],
                 'Circumference change [%] from Reference Lattice'),
                (['req_props', 'floor_comparison', 'LS', 'x'],
                 'Source point diff. in x @ LS'),
                (['req_props', 'floor_comparison', 'LS', 'z'],
                 'Source point diff. in z @ LS'),
                (['req_props', 'floor_comparison', 'SS', 'x'],
                 'Source point diff. in x @ SS'),
                (['req_props', 'floor_comparison', 'SS', 'z'],
                 'Source point diff. in z @ SS'),
                (['opt_props', 'phase_adv', 'MDISP 0&1', 'x'],
                 'Horizontal phase advance btw. dispersion bumps'),
                (['opt_props', 'phase_adv', 'MDISP 0&1', 'y'],
                 'Vertical phase advance btw. dispersion bumps'),
            ]):

                if isinstance(prop_name_or_list, str):
                    d2.append(sqss(prop_name_or_list))
                else:
                    prop_list = com_seq([sqss(v) for v in prop_name_or_list])
                    prop_list.fa.set_flow_style()
                    d2.append(prop_list)

                _kwargs = dict(column=0)
                if comment is not None:
                    d2.yaml_add_eol_comment(comment, len(d2)-1, **_kwargs)

        # ##########################################################################

        if example:
            _yaml_append_map(conf, 'rf_dep_calc_opts', com_map(), before_comment='\n')
            d = conf['rf_dep_calc_opts']

            _yaml_append_map(d, 'harmonic_number', 1320)
            array = com_seq([1.5e6, 2e6, 2.5e6, 3e6])
            array.fa.set_flow_style()
            _yaml_append_map(d, 'rf_V', array)

        # ##########################################################################

        if example:
            _yaml_append_map(conf, 'lifetime_calc_opts', com_map(), before_comment='\n')
            d = conf['lifetime_calc_opts']

            _yaml_append_map(d, 'total_beam_current_mA', 5e2)
            _yaml_append_map(d, 'num_filled_bunches', 1200)
            array = com_seq([8e-12, 20e-12])
            array.fa.set_flow_style()
            _yaml_append_map(d, 'eps_y', array)

        # ##########################################################################

        _yaml_append_map(conf, 'nonlin', com_map(), before_comment='\n')
        d = conf['nonlin']

        keys = self.all_nonlin_calc_types
        comments = self.all_nonlin_calc_comments
        assert len(keys) == len(comments)

        m = com_map()
        for k, c in zip(keys, comments):
            _yaml_append_map(m, k, True, eol_comment=c)
        _yaml_append_map(d, 'include', m, eol_comment='REQUIRED')

        if example:
            m = com_map()
            for k, c in zip(keys, comments):
                _yaml_append_map(m, k, False, eol_comment=c)
            _yaml_append_map(
                d, 'recalc', m,
                eol_comment='Will re-calculate potentially time-consuming data')

            m = com_map()
            for k, c in zip(keys, comments):
                _yaml_append_map(m, k, False, eol_comment=c)
            _yaml_append_map(
                d, 'replot', m,
                eol_comment='Will re-plot and save plotss as PDF files')

        _yaml_append_map(d, 'use_beamline', anchors['use_beamline_ring'],
                         eol_comment='REQUIRED',
                         before_comment='\nCommon Options', before_indent=2)

        N_KICKS = com_map(CSBEND=40, KQUAD=40, KSEXT=20, KOCT=20)
        N_KICKS.fa.set_flow_style()
        _yaml_append_map(d, 'N_KICKS', N_KICKS, eol_comment='REQUIRED')

        #---------------------------------------------------------------------------

        default_max_ntasks = 80

        common_remote_opts = com_map()
        _yaml_append_map(common_remote_opts, 'ntasks', default_max_ntasks,
                         eol_comment='REQUIRED')
        if example:
            _yaml_append_map(common_remote_opts, 'partition', sqss('short'))
            _yaml_append_map(common_remote_opts, 'mail_type_end', False,
                             eol_comment='If True, will send email at the end of job.')
            _yaml_append_map(common_remote_opts, 'mail_user', sqss('your_username@bnl.gov'),
                             eol_comment='REQUIRED only if "mail_type_end" is True.')
            nodelist = com_seq(
                ['apcpu-001', 'apcpu-002', 'apcpu-003', 'apcpu-004', 'apcpu-005'])
            nodelist.fa.set_flow_style()
            _yaml_append_map(
                common_remote_opts, 'nodelist', nodelist,
                eol_comment='list of strings for worker node names that will be used for the job')
            _yaml_append_map(
                common_remote_opts, 'time', sqss('2:00:00'),
                eol_comment='Specify max job run time in SLURM time string format')
        comment = '''
        Common parallel options (can be overwritten in the options block
        for each specific calculation type):
        '''
        _yaml_append_map(d, 'common_remote_opts', common_remote_opts,
                         eol_comment='REQUIRED',
                         before_comment=comment, before_indent=2)

        #---------------------------------------------------------------------------

        selected_calc_opt_names = com_map()

        for calc_type in self.all_nonlin_calc_types:
            _yaml_append_map(selected_calc_opt_names, calc_type, sqss('test'))

        comment = '''
        Selected names of nonlinear calculation options
        '''
        _yaml_append_map(d, 'selected_calc_opt_names', selected_calc_opt_names,
                         eol_comment='REQUIRED',
                         before_comment=comment, before_indent=2)

        #---------------------------------------------------------------------------

        calc_opts = com_map()

        # ### Option sets for "xy_aper" ###

        production = com_map(
            n_turns = 1024, abs_xmax = 10e-3, abs_ymax = 10e-3, ini_ndiv = 51,
            n_lines = 21)
        if example:
            _yaml_append_map(production, 'neg_y_search', False,
                             before_comment='Optional (below)', before_indent=6)
        #
        production.yaml_set_anchor('xy_aper_production')
        anchors['xy_aper_production'] = production
        #
        test = com_map(n_turns = 128)
        #test.fa.set_flow_style()
        test.add_yaml_merge([(0, anchors['xy_aper_production'])])
        #
        xy_aper = com_map(production = production, test = test)
        #
        comment = '''
        Option sets for dynamic aperture finding calculation:'''
        _yaml_append_map(calc_opts, 'xy_aper', xy_aper,
                         before_comment=comment, before_indent=2)

        # ### Option sets for "fmap_xy" ###

        production = com_map(
            n_turns = 1024, xmin = -8e-3, xmax = +8e-3, ymin = 0.0, ymax = +2e-3,
            nx = 201, ny = 201)
        if example:
            _yaml_append_map(production, 'x_offset', 1e-6,
                             before_comment='Optional (below)', before_indent=6)
            _yaml_append_map(production, 'y_offset', 1e-6)
            _yaml_append_map(production, 'delta_offset', 0.0)
        production.yaml_set_anchor('map_xy_production')
        anchors['map_xy_production'] = production
        #
        test = com_map(nx = 21, ny = 21)
        #test.fa.set_flow_style()
        test.add_yaml_merge([(0, anchors['map_xy_production'])])
        test.yaml_set_anchor('map_xy_test')
        anchors['map_xy_test'] = test
        #
        fmap_xy = com_map(production = production, test = test)
        #
        comment = '''
        Option sets for on-momentum frequency map calculation:'''
        _yaml_append_map(calc_opts, 'fmap_xy', fmap_xy,
                         before_comment=comment, before_indent=2)

        # ### Option sets for "fmap_px" ###

        production = com_map(
            n_turns = 1024, delta_min = -0.05, delta_max = +0.05,
            xmin = -8e-3, xmax = +8e-3, ndelta = 201, nx = 201)
        if example:
            _yaml_append_map(production, 'x_offset', 1e-6,
                             before_comment='Optional (below)', before_indent=6)
            _yaml_append_map(production, 'y_offset', 1e-6)
            _yaml_append_map(production, 'delta_offset', 0.0)
        production.yaml_set_anchor('map_px_production')
        anchors['map_px_production'] = production
        #
        test = com_map(ndelta = 21, nx = 21)
        #test.fa.set_flow_style()
        test.add_yaml_merge([(0, anchors['map_px_production'])])
        test.yaml_set_anchor('map_px_test')
        anchors['map_px_test'] = test
        #
        fmap_px = com_map(production = production, test = test)
        #
        comment = '''
        Option sets for off-momentum frequency map calculation:'''
        _yaml_append_map(calc_opts, 'fmap_px', fmap_px,
                         before_comment=comment, before_indent=2)

        # ### Option sets for "cmap_xy" ###

        production = com_map(n_turns = 128)
        production.add_yaml_merge([(0, anchors['map_xy_production'])])
        #
        test = com_map(n_turns = 128)
        test.add_yaml_merge([(0, anchors['map_xy_test'])])
        #
        cmap_xy = com_map(production = production, test = test)
        #
        comment = '''
        Option sets for on-momentum chaos map calculation:'''
        _yaml_append_map(calc_opts, 'cmap_xy', cmap_xy,
                         before_comment=comment, before_indent=2)

        # ### Option sets for "cmap_px" ###

        production = com_map(n_turns = 128)
        production.add_yaml_merge([(0, anchors['map_px_production'])])
        #
        test = com_map(n_turns = 128)
        test.add_yaml_merge([(0, anchors['map_px_test'])])
        #
        cmap_px = com_map(production = production, test = test)
        #
        comment = '''
        Option sets for off-momentum chaos map calculation:'''
        _yaml_append_map(calc_opts, 'cmap_px', cmap_px,
                         before_comment=comment, before_indent=2)

        # ### Option sets for "tswa" ###

        production = com_map(
            n_turns = 1024, abs_xmax = 1e-3, nx = 50, abs_ymax = 0.5e-3, ny = 50)
        if example:
            _yaml_append_map(production, 'x_offset', 1e-6,
                             before_comment='Optional (below)', before_indent=6)
            _yaml_append_map(production, 'y_offset', 1e-6)
            _yaml_append_map(production, 'save_fft', False)
            remote_opts = com_map(partition = sqss('short'), time = sqss('30:00'))
            _yaml_append_map(production, 'remote_opts', remote_opts)
        production.yaml_set_anchor('tswa_production')
        anchors['tswa_production'] = production
        #
        tswa = com_map(production = production)
        #
        comment = '''
        Option sets for tune-shift-with-amplitude calculation:'''
        _yaml_append_map(calc_opts, 'tswa', tswa,
                         before_comment=comment, before_indent=2)

        # ### Option sets for "nonlin_chrom" ###

        production = com_map(
            n_turns = 1024, delta_min = -4e-2, delta_max = +3e-2, ndelta = 100)
        if example:
            _yaml_append_map(production, 'x_offset', 1e-6,
                             before_comment='Optional (below)', before_indent=6)
            _yaml_append_map(production, 'y_offset', 1e-6)
            _yaml_append_map(production, 'delta_offset', 0.0)
            _yaml_append_map(production, 'save_fft', False)
            remote_opts = com_map(partition = sqss('short'), time = sqss('30:00'))
            _yaml_append_map(production, 'remote_opts', remote_opts)
        production.yaml_set_anchor('nonlin_chrom_production')
        anchors['nonlin_chrom_production'] = production
        #
        nonlin_chrom = com_map(production = production)
        #
        comment = '''
        Option sets for nonlinear chromaticity calculation:'''
        _yaml_append_map(calc_opts, 'nonlin_chrom', nonlin_chrom,
                         before_comment=comment, before_indent=2)

        # ### Option sets for "mom_aper" ###

        production = com_map(
            n_turns = 1024, x_initial = 10e-6, y_initial = 10e-6,
            delta_negative_start = -0.1e-2, delta_negative_limit = -5e-2,
            delta_positive_start = +0.1e-2, delta_positive_limit = +5e-2,
            init_delta_step_size = 5e-3,
            s_start = 0.0, s_end = sqss('one_period'),
            include_name_pattern = sqss('[QSO]*'),
        )
        production.yaml_set_anchor('mom_aper_production')
        anchors['mom_aper_production'] = production
        #
        test = com_map(n_turns = 16, include_name_pattern = sqss('[SO]*'))
        test.add_yaml_merge([(0, anchors['mom_aper_production'])])
        #
        mom_aper = com_map(production = production, test = test)
        #
        comment = '''
        Option sets for momentum aperture calculation:'''
        _yaml_append_map(calc_opts, 'mom_aper', mom_aper,
                         before_comment=comment, before_indent=2)

        # ### Finally add "calc_opts" to "nonlin" ###

        comment = '''
        '''
        _yaml_append_map(d, 'calc_opts', calc_opts,
                         eol_comment='REQUIRED',
                         before_comment=comment, before_indent=2)

        #---------------------------------------------------------------------------

        if example:

            comment = '''
            ## Plot Options ##'''
            _yaml_append_map(d, 'cmap_xy_plot_opts', com_map(cmin = -24, cmax = -10),
                             before_comment=comment, before_indent=2)

            #---------------------------------------------------------------------------

            _yaml_append_map(d, 'cmap_px_plot_opts', com_map(cmin = -24, cmax = -10))

            #---------------------------------------------------------------------------

            nux_lim = com_seq([0.0, 1.0])
            nux_lim.fa.set_flow_style()
            nuy_lim = com_seq([0.0, 1.0])
            nuy_lim.fa.set_flow_style()
            #
            tswa_plot_opts = com_map(
                plot_plus_minus_combined = True, plot_xy0 = True, plot_Axy = False,
                use_time_domain_amplitude = True, plot_fft = False,
                footprint_nuxlim = nux_lim, footprint_nuylim = nuy_lim,
                fit_xmin = -0.5e-3, fit_xmax = +0.5e-3,
                fit_ymin = -0.25e-3, fit_ymax = +0.25e-3,
            )
            comment = '''\
            ^ Even if True, these plots will NOT be included into the main report,
            ^ but will be saved to the "tswa" PDF file.
            '''
            _yaml_set_comment_after_key(tswa_plot_opts, 'plot_Axy', comment, indent=4)
            #
            tswa_plot_opts.yaml_add_eol_comment(
                'Only relevant when "plot_Axy" is True', 'use_time_domain_amplitude',
                column=0)
            #
            comment = '''\
            ^ If True, it may take a while to save the "tswa" PDF file.
            ^ But these FFT color plots will NOT be included into
            ^ the main report. These plots may be useful for debugging
            ^ or for deciding the number of divisions for x0/y0 arrays.
            ^ Also, this option being True requires you to have set "save_fft"
            ^ as True (False by default) in "tswa_calc_opts".
            '''
            _yaml_set_comment_after_key(tswa_plot_opts, 'plot_fft', comment, indent=4)
            #
            _yaml_append_map(d, 'tswa_plot_opts', tswa_plot_opts)

            #---------------------------------------------------------------------------

            fit_deltalim = com_seq([-2e-2, +2e-2])
            fit_deltalim.fa.set_flow_style()

            nux_lim = com_seq([0.0, 1.0])
            nux_lim.fa.set_flow_style()
            nuy_lim = com_seq([0.0, 1.0])
            nuy_lim.fa.set_flow_style()

            nonlin_chrom_plot_opts = com_map()
            _yaml_append_map(nonlin_chrom_plot_opts, 'plot_fft', False)
            _yaml_append_map(nonlin_chrom_plot_opts, 'max_chrom_order', 4)
            _yaml_append_map(nonlin_chrom_plot_opts, 'fit_deltalim', fit_deltalim)
            _yaml_append_map(nonlin_chrom_plot_opts, 'footprint_nuxlim', nux_lim)
            _yaml_append_map(nonlin_chrom_plot_opts, 'footprint_nuylim', nuy_lim)

            comment = '''\
            ^ If True, it may take a while to save the "nonlin_chrom" PDF file.
            ^ But these FFT color plots will NOT be included into
            ^ the main report. These plots may be useful for debugging
            ^ or for deciding the number of divisions for delta arrays.
            ^ Also, this option being True requires you to have set "save_fft"
            ^ as True (False by default) in "nonlin_chrom_calc_opts".
            '''
            _yaml_set_comment_after_key(nonlin_chrom_plot_opts, 'plot_fft',
                                        comment, indent=4)

            _yaml_append_map(d, 'nonlin_chrom_plot_opts', nonlin_chrom_plot_opts)

        # ##########################################################################

        if False:
            dumper = yaml.YAML()
            dumper.preserve_quotes = True
            dumper.width = 70
            dumper.boolean_representation = ['False', 'True']
            dumper.dump(conf, sys.stdout)

            with open('test.yaml', 'w') as f:
                dumper.dump(conf, f)

        return conf

def get_parsed_args():
    """"""

    parser = argparse.ArgumentParser(
        prog='pyele_report',
        description='Automated report generator for PyELEGANT')
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '-f', '--full-example-config', default=None, type=str,
        help=('Generate a full-example config YAML file for a specified '
              '"report_class" (All available: ["nsls2u_default"]'))
    group.add_argument(
        '-m', '--min-example-config', default=None, type=str,
        help=('Generate a minimum-example config YAML file for a specified '
              '"report_class"'))
    parser.add_argument(
        '--example-report-version', default=None, type=str,
        help=('Version of example config YAML file to be generated. '
              'Ignored if "--full-example-config" and "--min-example-config" '
              'are not specified'))
    parser.add_argument(
        'config_filepath', type=str,
        help='''\
    Path to YAML file that contains configurations for report generation.
    Or, if "--full-example-config" or "--min-example-config" is specified,
    an example config file will be generated and saved at this file path.''')

    args = parser.parse_args()
    if False:
        print(args)
        print(f'Generate Full Example Config? = {args.full_example_config}')
        print(f'Generate Min Example Config? = {args.min_example_config}')
        print(f'Example Report Version? = {args.example_report_version}')
        print(f'Config File = {args.config_filepath}')

    return args

def gen_report(args):
    """"""

    config_filepath = args.config_filepath

    assert config_filepath.endswith(('.yml', '.yaml'))

    if args.full_example_config or args.min_example_config:
        if args.full_example_config:
            report_class = args.full_example_config
            example_type = 'full'
        else:
            report_class = args.min_example_config
            example_type = 'min'

        example_args = [example_type, args.example_report_version]

        if report_class == 'nsls2u_default':
            Report_NSLS2U_Default(config_filepath, example_args=example_args)
        else:
            raise ValueError(f'Invalid "report_class": {report_class}')

    else:
        if not os.path.exists(config_filepath):
            raise OSError(f'Specified config file "{config_filepath}" does not exist.')

        yml = yaml.YAML()
        yml.preserve_quotes = True
        user_conf = yml.load(Path(config_filepath).read_text())

        if user_conf['report_class'] == 'nsls2u_default':
            Report_NSLS2U_Default(config_filepath, user_conf=user_conf)
        else:
            print('\n# Available names for "report_class":')
            for name in ['nsls2u_default']:
                print(f'   {name}')
            raise NotImplementedError(f'Report Class: {user_conf["report_class"]}')

def main():
    """"""

    args = get_parsed_args()
    gen_report(args)

if __name__ == '__main__':

    main()