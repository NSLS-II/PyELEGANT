import sys
import os
import numpy as np
import pickle
import matplotlib.pylab as plt
from matplotlib.backends.backend_pdf import PdfPages
#import yaml
from ruamel import yaml
# ^ ruamel's "yaml" does NOT suffer from the PyYAML(v5.3, YAML v1.1) problem
#   that a float value in scientific notation without "." and the sign after e/E
#   is treated as a string.
from pathlib import Path

import pyelegant as pe
pe.disable_stdout()
pe.enable_stderr()
plx = pe.latex

def gen_LTE_from_base_LTE_and_param_file(conf, input_LTE_filepath):
    """"""

    assert conf['input_LTE']['base_filepath'].endswith('.lte')
    base_LTE_filepath = conf['input_LTE']['base_filepath']

    load_parameters = dict(filename=conf['input_LTE']['param_filepath'])

    pe.eleutil.save_lattice_after_load_parameters(
        base_LTE_filepath, input_LTE_filepath, load_parameters)

def gen_zeroSexts_LTE(conf, input_LTE_filepath):
    """
    Turn off all sextupoles' K2 values to zero and save a new LTE file.
    """

    if 'zeroSexts_filepath' in conf['input_LTE']:
        zeroSexts_LTE_filepath = conf['input_LTE']['zeroSexts_filepath']
    else:
        zeroSexts_LTE_filepath = input_LTE_filepath.replace(
            '.lte', '_ZeroSexts.lte')

    alter_elements = dict(name='*', type='KSEXT', item='K2', value = 0.0)
    pe.eleutil.save_lattice_after_alter_elements(
        input_LTE_filepath, zeroSexts_LTE_filepath, alter_elements)

    return zeroSexts_LTE_filepath

def calc_lin_props(
    LTE_filepath, rootname, E_MeV, lattice_props_conf, zeroSexts_LTE_filepath=''):
    """"""

    conf = lattice_props_conf
    conf_twi = conf['twiss_calc_opts']

    sel_data = {'E_GeV': E_MeV / 1e3}
    interm_array_data = {} # holds data that will be only used to derive some other quantities

    raw_keys = dict(
        one_period=dict(eps_x='ex0', Jx='Jx', Jy='Jy', Jdelta='Jdelta'),
        ring=dict(
            nux='nux', nuy='nuy', ksi_x_cor='dnux/dp', ksi_y_cor='dnuy/dp',
            alphac='alphac', U0_MeV='U0', dE_E='Sdelta0'),
        ring_natural=dict(ksi_x_nat='dnux/dp', ksi_y_nat='dnuy/dp'),
    )

    interm_array_keys = dict(
        one_period=dict(
            s_one_period='s', betax_1p='betax', betay_1p='betay', etax_1p='etax',
            elem_names_1p='ElementName'),
        ring=dict(
            s_ring='s', elem_names_ring='ElementName',
            betax_ring='betax', betay_ring='betay',
            psix_ring='psix', psiy_ring='psiy'),
    )

    output_filepaths = {}
    for k in list(raw_keys):
        output_filepaths[k] = f'{rootname}.twiss_{k}.pgz'

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

    nsls2_data = {}
    nsls2_data['circumf'] = 791.958 # [m]

    sel_data['circumf'] = interm_array_data['s_ring'][-1]
    sel_data['circumf_change_%'] = (
        sel_data['circumf'] / nsls2_data['circumf'] - 1) * 1e2

    ring_mult = sel_data['circumf'] / interm_array_data['s_one_period'][-1]
    sel_data['n_periods_in_ring'] = int(np.round(ring_mult))
    assert np.abs(ring_mult - sel_data['n_periods_in_ring']) < 1e-3

    sel_data['max_betax'] = np.max(interm_array_data['betax_1p'])
    sel_data['min_betax'] = np.min(interm_array_data['betax_1p'])
    sel_data['max_betay'] = np.max(interm_array_data['betay_1p'])
    sel_data['min_betay'] = np.min(interm_array_data['betay_1p'])
    sel_data['max_etax'] = np.max(interm_array_data['etax_1p'])
    sel_data['min_etax'] = np.min(interm_array_data['etax_1p'])

    extra_conf = conf.get('extra_props', None)

    if extra_conf:
        extra_data = sel_data['extra'] = {}

        beta = extra_conf.get('beta', None)
        if beta:
            _d = extra_data['beta'] = {}
            for key, elem_d in beta.items():
                elem_name = elem_d['name'].upper()
                index = np.where(interm_array_data['elem_names_ring'] ==
                                 elem_name)[0][elem_d['occur']]
                _d[key] = dict(betax=interm_array_data['betax_ring'][index],
                               betay=interm_array_data['betay_ring'][index],
                               label=elem_d['label'])

        phase_adv = extra_conf.get('phase_adv', None)
        if phase_adv:
            _d = extra_data['phase_adv'] = {}
            for key, elem_d in phase_adv.items():

                elem_name_1 = elem_d['elem1']['name'].upper()
                occur_1 = elem_d['elem1']['occur']
                elem_name_2 = elem_d['elem2']['name'].upper()
                occur_2 = elem_d['elem2']['occur']

                index_1 = np.where(
                    interm_array_data['elem_names_ring'] ==
                    elem_name_1)[0][occur_1]
                index_2 = np.where(
                    interm_array_data['elem_names_ring'] ==
                    elem_name_2)[0][occur_2]
                _d[key] = dict(
                    dnux=(interm_array_data['psix_ring'][index_2] -
                          interm_array_data['psix_ring'][index_1]) / (2 * np.pi),
                    dnuy=(interm_array_data['psiy_ring'][index_2] -
                          interm_array_data['psiy_ring'][index_1]) / (2 * np.pi),
                    label=elem_d['label']
                )

    ed = pe.elebuilder.EleDesigner()
    ed.add_block('run_setup',
        lattice = LTE_filepath, p_central_mev = E_MeV,
        use_beamline=conf_twi['one_period']['use_beamline'],
        parameters='%s.param')
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

    if extra_conf:

        length = extra_conf.get('length', None)
        if length:
            _d = extra_data['length'] = {}

            _s = res['flr']['columns']['s']
            for key, elem_d in length.items():
                elem_name_list = elem_d['name_list']
                if key in _d:
                    raise ValueError(
                        f'Duplicate key "{key}" found for "length" dict')
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
                            key, ', '.join(elem_name_list)))

                _d[key] = dict(L=L, label=elem_d['label'])


        floor_comparison = extra_conf.get('floor_comparison', None)
        if floor_comparison:
            _d = extra_data['floor_comparison'] = {}

            nsls2_flr_filepath = floor_comparison.pop('ref_flr_filepath')
            flr_data = pe.sdds.sdds2dicts(nsls2_flr_filepath)[0]
            N2_X_all = flr_data['columns']['X']
            N2_Z_all = flr_data['columns']['Z']
            N2_ElemNames = flr_data['columns']['ElementName']

            N2_X, N2_Z = {}, {}
            N2U_X, N2U_Z = {}, {}
            for key, elems_d in floor_comparison.items():

                ref_elem = elems_d['ref_elem']

                ind = np.where(N2_ElemNames ==
                               ref_elem['name'].upper())[0][ref_elem['occur']]

                N2_X[key] = N2_X_all[ind]
                N2_Z[key] = N2_Z_all[ind]

                cur_elem = elems_d['cur_elem']

                ind = np.where(
                    res['flr']['columns']['ElementName'] ==
                    cur_elem['name'].upper())[0][cur_elem['occur']]

                N2U_X[key] = res['flr']['columns']['X'][ind]
                N2U_Z[key] = res['flr']['columns']['Z'][ind]

                _d[key] = dict(dx=N2U_X[key] - N2_X[key],
                               dz=N2U_Z[key] - N2_Z[key],
                               label=elems_d['label'])

    if False:
        # Check whether specified LaTeX labels are valid
        doc = create_header(conf)
        table_order = conf.get('table_order', [])
        for row_spec in table_order:
            label_w_unit, val_str = get_lattice_prop_row(sel_data, row_spec)
            doc.append(label_w_unit)
            doc.append(plx.NewLine())
        plx.generate_pdf_w_reruns(doc, clean_tex=False, silent=False)

    return dict(versions=pe.__version__, sel_data=sel_data)

def get_only_lin_props_plot_captions(
    rootname, twiss_plot_opts, twiss_plot_captions):
    """"""

    caption_list = plot_lin_props(
        rootname, twiss_plot_opts, twiss_plot_captions, skip_plots=True)

    return caption_list

def plot_lin_props(
    rootname, twiss_plot_opts, twiss_plot_captions, skip_plots=False):
    """"""

    existing_fignums = plt.get_fignums()

    caption_list = []

    for lat_type in list(twiss_plot_opts):
        output_filepath = f'{rootname}.twiss_{lat_type}.pgz'

        try:
            assert len(twiss_plot_opts[lat_type]) == len(twiss_plot_captions[lat_type])
        except AssertionError:
            print(
                (f'ERROR: Number of yaml["lattice_props"]["twiss_plot_opts"]["{lat_type}"] '
                 f'and that of yaml["lattice_props"]["twiss_plot_captions"]["{lat_type}"] '
                 f'must match.'))
            raise

        for opts, caption in zip(twiss_plot_opts[lat_type],
                                  twiss_plot_captions[lat_type]):

            if not skip_plots:
                pe.plot_twiss(output_filepath, **opts)

            caption_list.append(plx.NoEscape(caption))

    if skip_plots:
        return caption_list

    twiss_pdf_filepath = f'{rootname}.twiss.pdf'

    fignums_to_delete = []

    pp = PdfPages(twiss_pdf_filepath)
    for fignum in plt.get_fignums():
        if fignum not in existing_fignums:
            pp.savefig(figure=fignum)
            fignums_to_delete.append(fignum)
    pp.close()

    #plt.show()

    for fignum in fignums_to_delete:
        plt.close(fignum)

    return caption_list

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

def build_report(conf, input_LTE_filepath, rootname, lin_data, twiss_plot_captions):
    """"""

    doc = create_header(conf)

    add_lattice_description(doc, conf, input_LTE_filepath)

    add_lattice_elements(doc, lin_data)

    add_lattice_props_section(doc, conf, rootname, lin_data, twiss_plot_captions)

    plx.generate_pdf_w_reruns(doc, clean_tex=False, silent=False)

def create_header(conf):
    """"""

    geometry_options = {"vmargin": "1cm", "hmargin": "1.5cm"}
    doc = plx.Document(
        f'{rootname}_report', geometry_options=geometry_options,
        documentclass='article')
    doc.preamble.append(plx.Command('usepackage', 'nopageno')) # Suppress page numbering for entire doc
    doc.preamble.append(plx.Package('indentfirst')) # This fixes the problem of the first paragraph not indenting
    doc.preamble.append(plx.Package('seqsplit')) # To split a very long word into multiple lines w/o adding hyphens, like a long file name.

    doc.preamble.append(plx.Command(
        'title', 'ELEGANT Lattice Characterization Report'))

    if 'author' in conf:
        doc.preamble.append(plx.Command('author', conf['author']))

    doc.preamble.append(plx.Command('date', plx.NoEscape(r'\today')))

    doc.append(plx.NoEscape(r'\maketitle'))

    return doc

def add_lattice_description(doc, conf, input_LTE_filepath):
    """"""

    with doc.create(plx.Section('Lattice Description')):

        mod_LTE_filename = os.path.basename(input_LTE_filepath).replace("_", r"\_")
        ver_str = pe.__version__["PyELEGANT"]

        default_paragraph = plx.NoEscape(
            (f'The lattice file being analyzed here is '
             f'\seqsplit{{"{mod_LTE_filename}"}}. This report was generated using '
             f'PyELEGANT version {ver_str}.'))
        doc.append(default_paragraph)

        custom_paragraphs = conf['report_paragraphs'].get('lattice_description', [])
        for para in custom_paragraphs:
            doc.append(plx.NewParagraph())
            doc.append(para.strip())

def add_lattice_elements(doc, lin_data):
    """"""

    with doc.create(plx.Section('Lattice Elements')):
        create_bend_elements_subsection(doc, lin_data['elem_defs'])
        create_quad_elements_subsection(doc, lin_data['elem_defs'])
        create_sext_elements_subsection(doc, lin_data['elem_defs'])

        create_beamline_elements_list_subsection(
            doc, lin_data['flat_elem_s_name_type_list'])

def get_lattice_prop_row(lin_data, row_spec):
    """"""

    k = row_spec

    unit = ''

    if isinstance(k, list):

        extra_props_key, sub_key = k

        _d = lin_data['extra'][extra_props_key][sub_key]

        label = _d['label']

        if extra_props_key == 'beta':
            unit = ' [m]'
            val_str = plx.MathText(
                '({:.2f}, {:.2f})'.format(_d['betax'], _d['betay']))

        elif extra_props_key == 'phase_adv':
            unit = plx.MathText(' [2\pi]')
            val_str = plx.MathText(
                '({:.6f}, {:.6f})'.format(_d['dnux'], _d['dnuy']))

        elif extra_props_key == 'length':
            unit = ' [m]'
            val_str = plx.MathText('{:.3f}'.format(_d['L']))

        elif extra_props_key == 'floor_comparison':
            unit = ' [mm]'
            val_str = plx.MathText(
                '({:+.2f}, {:+.2f})'.format(_d['dx'] * 1e3, _d['dz'] * 1e3))

        else:
            raise ValueError('Unexpected key for "extra_props": {extra_props_key}')

    elif k == 'E_GeV':
        label = f'Beam Energy'
        unit = ' [GeV]'
        val_str = plx.MathText(f'{lin_data[k]:.0f}')
    elif k == 'eps_x':
        label = 'Natural Horizontal Emittance ' + plx.MathText(r'\epsilon_x')
        unit = ' [pm]'
        val_str = plx.MathText('{:.1f}'.format(lin_data[k] * 1e12))
    elif k == 'J':
        label = 'Damping Partitions ' + plx.MathText(r'(J_x, J_y, J_{\delta})')
        val_str = plx.MathText('({:.2f}, {:.2f}, {:.2f})'.format(
                lin_data['Jx'], lin_data['Jy'], lin_data['Jdelta']))
    elif k == 'nu':
        label = 'Ring Tunes ' + plx.MathText(r'(\nu_x, \nu_y)')
        val_str = plx.MathText('({:.3f}, {:.3f})'.format(
            lin_data['nux'], lin_data['nuy']))
    elif k.startswith('ksi_'):
        if k == 'ksi_nat':
            label = 'Natural Chromaticities ' + plx.MathText(
                r'(\xi_x^{\mathrm{nat}}, \xi_y^{\mathrm{nat}})')
            val_str = plx.MathText('({:+.3f}, {:+.3f})'.format(
                lin_data['ksi_x_nat'], lin_data['ksi_y_nat']))
        elif k == 'ksi_cor':
            label = 'Corrected Chromatiticities ' + plx.MathText(
                r'(\xi_x^{\mathrm{cor}}, \xi_y^{\mathrm{cor}})')
            val_str = plx.MathText('({:+.3f}, {:+.3f})'.format(
                lin_data['ksi_x_cor'], lin_data['ksi_y_cor']))
        else:
            raise ValueError
    elif k == 'alphac':
        label = 'Momentum Compaction ' + plx.MathText(r'\alpha_c')
        val_str = plx.MathText(
            pe.util.pprint_sci_notation(lin_data[k], '.2e'))
    elif k == 'U0':
        label = 'Energy Loss per Turn ' + plx.MathText(r'U_0')
        unit = ' [keV]'
        val_str = plx.MathText('{:.0f}'.format(lin_data['U0_MeV'] * 1e3))
    elif k == 'sigma_delta':
        label = 'Energy Spread ' + plx.MathText(r'\sigma_{\delta}')
        unit = r' [\%]'
        val_str = plx.MathText('{:.3f}'.format(lin_data['dE_E'] * 1e2))
    elif k == 'max_beta':
        label = 'max ' + plx.MathText(r'(\beta_x, \beta_y)')
        unit = ' [m]'
        val_str = plx.MathText('({:.2f}, {:.2f})'.format(
            lin_data['max_betax'], lin_data['max_betay']))
    elif k == 'min_beta':
        label = 'min ' + plx.MathText(r'(\beta_x, \beta_y)')
        unit = ' [m]'
        val_str = plx.MathText('({:.2f}, {:.2f})'.format(
            lin_data['min_betax'], lin_data['min_betay']))
    elif k == 'max_min_etax':
        label = plx.MathText(r'\eta_x') + ' (min, max)'
        unit = ' [mm]'
        val_str = plx.MathText('({:+.1f}, {:+.1f})'.format(
            lin_data['min_etax'] * 1e3, lin_data['max_etax'] * 1e3))
    elif k == 'circumf':
        label = 'Circumference ' + plx.MathText(r'C')
        unit = ' [m]'
        val_str = plx.MathText('{:.3f}'.format(lin_data[k]))
    elif k == 'circumf_change_%':
        label = 'Circumference Change ' + plx.MathText(r'\Delta C / C')
        unit = r' [\%]'
        val_str = plx.MathText('{:+.3f}'.format(lin_data[k]))
    elif k == 'n_periods_in_ring':
        label = 'Number of Super-periods'
        val_str = plx.MathText('{:d}'.format(lin_data[k]))
    else:
        raise RuntimeError(f'Unhandled "table_order" key: {k}')

    if isinstance(unit, plx.MathText):
        label_w_unit = plx.NoEscape(label + unit.dumps())
    else:
        label_w_unit = label + unit
        if isinstance(label_w_unit, plx.CombinedMathNormalText):
            label_w_unit = label_w_unit.dumps_for_caption()
        else:
            label_w_unit = plx.NoEscape(label_w_unit)

    return label_w_unit, val_str

def add_lattice_props_section(doc, conf, rootname, lin_data, twiss_plot_captions):
    """"""

    table_order = conf['lattice_props'].get('table_order', None)

    if not table_order:
        table_order = [
            'E_GeV', # Beam energy
            'eps_x', # Natural horizontal emittance
            'J', # Damping partitions
            'nu', # Ring tunes
            'ksi_nat', # Natural chromaticities
            'ksi_cor', # Corrected chromaticities
            'alphac', # Momentum compaction
            'U0', # Energy loss per turn
            'sigma_delta', # Energy spread
            'max_beta', # Max beta functions
            'min_beta', # Min beta functions
            'max_min_etax', # Max & Min etax
            'circumf', # Circumference
            'circumf_change_%', # Circumference change [%] from NSLS-II
            'n_periods_in_ring', # Number of super-periods for a full ring
        ]

        extra_props = conf['lattice_props'].get('extra_props', {})

        for extra_props_key in ['beta', 'phase_adv', 'length', 'floor_comparison']:
            for sub_key in sorted(list(extra_props.get(extra_props_key, {}))):
                if (extra_props_key == 'floor_comparison') and \
                   (sub_key == 'ref_flr_filepath'):
                    continue
                table_order.append([extra_props_key, sub_key])

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
                label_w_unit, val_str = get_lattice_prop_row(lin_data, row_spec)
                table.add_row([label_w_unit, val_str])


        twiss_pdf_filepath = f'{rootname}.twiss.pdf'

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
                            twiss_pdf_filepath, page=iPage+1,
                            width=plx.utils.NoEscape(r'\linewidth'))
                        doc.append(plx.VerticalSpace(plx.NoEscape('-10pt')))
                        subfig.add_caption(caption)
                doc.append(plx.VerticalSpace(plx.NoEscape('-10pt')))
                fig.add_caption('Twiss functions.')



if __name__ == '__main__':

    conf = yaml.safe_load(Path('nsls2.yaml').read_text())

    assert conf['input_LTE']['filepath'].endswith('.lte')
    input_LTE_filepath = conf['input_LTE']['filepath']

    if conf['input_LTE']['load_param']:
        gen_LTE_from_base_LTE_and_param_file(conf, input_LTE_filepath)

    if conf['input_LTE']['generate_zeroSexts']:
        # Turn off all sextupoles
        zeroSexts_LTE_filepath = gen_zeroSexts_LTE(conf, input_LTE_filepath)
    else:
        if 'zeroSexts_filepath' in conf['input_LTE']:
            zeroSexts_LTE_filepath = conf['input_LTE']['zeroSexts_filepath']
        else:
            zeroSexts_LTE_filepath = ''

    rootname = os.path.basename(input_LTE_filepath).replace('.lte', '')

    lin_summary_pkl_filepath = f'{rootname}.lin.pkl'
    nonlin_summary_pkl_filepath = f'{rootname}.nonlin.pkl'

    if (not os.path.exists(lin_summary_pkl_filepath)) or \
        conf['lattice_props'].get('recalc', False):

        d = calc_lin_props(
            input_LTE_filepath, rootname, conf['E_MeV'], conf['lattice_props'],
            zeroSexts_LTE_filepath=zeroSexts_LTE_filepath)

        lin_data = d['sel_data']
        lin_data['_versions'] = d['versions']
        abs_input_LTE_filepath = os.path.abspath(input_LTE_filepath)
        LTE_contents = Path(input_LTE_filepath).read_text()

        with open(lin_summary_pkl_filepath, 'wb') as f:
            pickle.dump([abs_input_LTE_filepath, LTE_contents, lin_data], f)
    else:
        with open(lin_summary_pkl_filepath, 'rb') as f:
            (abs_input_LTE_filepath, LTE_contents, lin_data) = pickle.load(f)

        if Path(input_LTE_filepath).read_text() != LTE_contents:
            raise RuntimeError(
                (f'The LTE contents saved in "{lin_summary_pkl_filepath}" '
                 'does NOT exactly match with the currently specified '
                 f'LTE file "{input_LTE_filepath}". Either check the LTE '
                 'file, or re-calculate to create an updated data file.'))

    twiss_pdf_filepath = f'{rootname}.twiss.pdf'

    if (not os.path.exists(twiss_pdf_filepath)) or \
        conf['lattice_props'].get('replot', False):
        twiss_plot_captions = plot_lin_props(
            rootname, conf['lattice_props']['twiss_plot_opts'],
            conf['lattice_props']['twiss_plot_captions'], skip_plots=False)
    else:
        twiss_plot_captions = get_only_lin_props_plot_captions(
            rootname, conf['lattice_props']['twiss_plot_opts'],
            conf['lattice_props']['twiss_plot_captions'])

    if False and ('nonlin' in conf):

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

    build_report(conf, input_LTE_filepath, rootname, lin_data, twiss_plot_captions)