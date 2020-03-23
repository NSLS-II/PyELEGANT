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
import argparse

import pyelegant as pe
pe.disable_stdout()
pe.enable_stderr()
plx = pe.latex

def gen_LTE_from_base_LTE_and_param_file(conf, input_LTE_filepath):
    """"""

    assert conf['input_LTE']['base_LTE_filepath'].endswith('.lte')
    base_LTE_filepath = conf['input_LTE']['base_LTE_filepath']

    load_parameters = dict(filename=conf['input_LTE']['param_filepath'])

    pe.eleutil.save_lattice_after_load_parameters(
        base_LTE_filepath, input_LTE_filepath, load_parameters)

def gen_zeroSexts_LTE(input_LTE_filepath, report_folderpath, regenerate):
    """
    Turn off all sextupoles' K2 values to zero and save a new LTE file.
    """

    input_LTE_filename = os.path.basename(input_LTE_filepath)
    zeroSexts_LTE_filepath = os.path.join(
        report_folderpath,
        input_LTE_filename.replace('.lte', '_ZeroSexts.lte'))

    if (not os.path.exists(zeroSexts_LTE_filepath)) or regenerate:
        alter_elements = dict(name='*', type='KSEXT', item='K2', value = 0.0)
        pe.eleutil.save_lattice_after_alter_elements(
            input_LTE_filepath, zeroSexts_LTE_filepath, alter_elements)

    return zeroSexts_LTE_filepath

def calc_lin_props(
    LTE_filepath, report_folderpath, E_MeV, lattice_props_conf,
    zeroSexts_LTE_filepath=''):
    """"""

    conf = lattice_props_conf

    default_twiss_calc_opts = dict(
        one_period={'use_beamline': conf['use_beamline_cell'],
                    'element_divisions': 10},
        ring_natural={'use_beamline': conf['use_beamline_ring']},
        ring={'use_beamline': conf['use_beamline_ring']},
    )
    conf_twi = conf.get('twiss_calc_opts', default_twiss_calc_opts)

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
    report_folderpath, twiss_plot_opts, twiss_plot_captions):
    """"""

    caption_list = plot_lin_props(
        report_folderpath, twiss_plot_opts, twiss_plot_captions,
        skip_plots=True)

    return caption_list

def plot_lin_props(
    report_folderpath, twiss_plot_opts, twiss_plot_captions, skip_plots=False):
    """"""

    existing_fignums = plt.get_fignums()

    caption_list = []

    for lat_type in list(twiss_plot_opts):
        output_filepath = os.path.join(report_folderpath, f'twiss_{lat_type}.pgz')

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

    twiss_pdf_filepath = os.path.join(report_folderpath, 'twiss.pdf')

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

def add_nonlin_sections(
    doc, conf, report_folderpath, input_LTE_filepath):
    """"""

    ncf = conf['nonlin']

    LTE_contents = Path(input_LTE_filepath).read_text()

    nonlin_data_filepaths = get_nonlin_data_filepaths(report_folderpath, ncf)
    included_types = [k for k, _included in ncf.get('include', {}).items()
                      if _included]
    plots_pdf_paths = {k: os.path.join(report_folderpath, f'{k}.pdf')
                       for k in included_types}

    new_page_required = False

    if ('fmap_xy' in included_types) or ('fmap_px' in included_types):

        add_fmap_section(doc, plots_pdf_paths, nonlin_data_filepaths,
                         input_LTE_filepath, LTE_contents)
        new_page_required = True

    if ('cmap_xy' in included_types) or ('cmap_px' in included_types):

        if new_page_required:
            doc.append(plx.ClearPage())

        add_cmap_section(doc, plots_pdf_paths, nonlin_data_filepaths,
                         input_LTE_filepath, LTE_contents)
        new_page_required = True

    if 'tswa' in included_types:

        if new_page_required:
            doc.append(plx.ClearPage())

        with open(os.path.join(report_folderpath,
                               f'tswa.plot_suppl.pkl'), 'rb') as f:
            tswa_plot_captions = pickle.load(f)

        add_tswa_section(doc, plots_pdf_paths, nonlin_data_filepaths,
                         input_LTE_filepath, LTE_contents, tswa_plot_captions)
        new_page_required = True

    if 'nonlin_chrom' in included_types:

        if new_page_required:
            doc.append(plx.ClearPage())

        add_nonlin_chrom_section(doc, plots_pdf_paths, nonlin_data_filepaths,
                                 input_LTE_filepath, LTE_contents)
        new_page_required = True

def add_fmap_section(
    doc, plots_pdf_paths, nonlin_data_filepaths, input_LTE_filepath, LTE_contents):
    """"""

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

def add_cmap_section(
    doc, plots_pdf_paths, nonlin_data_filepaths, input_LTE_filepath, LTE_contents):
    """"""

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

def add_tswa_section(
    doc, plots_pdf_paths, nonlin_data_filepaths, input_LTE_filepath, LTE_contents,
    tswa_page_captions):
    """"""

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
            ('x', tswa_page_captions[:2]),
            ('y', tswa_page_captions[2:])]:

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

def add_nonlin_chrom_section(
    doc, plots_pdf_paths, nonlin_data_filepaths, input_LTE_filepath, LTE_contents):
    """"""

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

def build_report(conf, input_LTE_filepath, rootname, report_folderpath, lin_data,
                 twiss_plot_captions):
    """"""

    doc = create_header(conf, report_folderpath, rootname)

    add_lattice_description(doc, conf, input_LTE_filepath)

    add_lattice_elements(doc, lin_data)

    add_lattice_props_section(
        doc, conf, report_folderpath, lin_data, twiss_plot_captions)

    doc.append(plx.ClearPage())

    add_nonlin_sections(
        doc, conf, report_folderpath, input_LTE_filepath)

    plx.generate_pdf_w_reruns(doc, clean_tex=False, silent=False)

def create_header(conf, report_folderpath, rootname):
    """"""

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

def add_lattice_props_section(
    doc, conf, report_folderpath, lin_data, twiss_plot_captions):
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

def get_nonlin_data_filepaths(report_folderpath, nonlin_config):
    """"""

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
            suffix_list.append(
                f'_{calc_type[:4]}_{grid_name}_n{n_turns}.{output_filetype}')
            data_file_key_list.append(calc_type)
        elif calc_type == 'tswa':
            for plane in ['x', 'y']:
                for sign in ['plus', 'minus']:
                    suffix_list.append(
                        f'_tswa_{grid_name}_n{n_turns}_{plane}{sign}.{output_filetype}')
                    data_file_key_list.append(f'tswa_{plane}{sign}')
        elif calc_type == 'nonlin_chrom':
            suffix_list.append(
                f'_nonlin_chrom_{grid_name}_n{n_turns}.{output_filetype}')
            data_file_key_list.append(calc_type)
        else:
            raise ValueError

    assert len(suffix_list) == len(data_file_key_list)
    nonlin_data_filepaths = {}
    for k, suffix in zip(data_file_key_list, suffix_list):
        filename = suffix[1:] # remove the first "_"
        nonlin_data_filepaths[k] = os.path.join(report_folderpath, filename)

    return nonlin_data_filepaths

def calc_nonlin_props(LTE_filepath, report_folderpath, E_MeV, nonlin_config,
                      do_calc):
    """"""

    ncf = nonlin_config

    nonlin_data_filepaths = get_nonlin_data_filepaths(report_folderpath, ncf)
    use_beamline = ncf['use_beamline']
    N_KICKS = ncf.get('N_KICKS', dict(KQUAD=40, KSEXT=40, CSBEND=40))

    common_remote_opts = ncf['common_remote_opts']

    calc_type = 'fmap_xy'
    if (calc_type in nonlin_data_filepaths) and \
       (do_calc[calc_type] or
        (not os.path.exists(nonlin_data_filepaths[calc_type]))):

        calc_fmap_xy(LTE_filepath, E_MeV, ncf, use_beamline, N_KICKS,
                     nonlin_data_filepaths, common_remote_opts)

    calc_type = 'fmap_px'
    if (calc_type in nonlin_data_filepaths) and \
       (do_calc[calc_type] or
        (not os.path.exists(nonlin_data_filepaths[calc_type]))):

        calc_fmap_px(LTE_filepath, E_MeV, ncf, use_beamline, N_KICKS,
                     nonlin_data_filepaths, common_remote_opts)

    calc_type = 'cmap_xy'
    if (calc_type in nonlin_data_filepaths) and \
       (do_calc[calc_type] or
        (not os.path.exists(nonlin_data_filepaths[calc_type]))):

        calc_cmap_xy(LTE_filepath, E_MeV, ncf, use_beamline, N_KICKS,
                     nonlin_data_filepaths, common_remote_opts)

    calc_type = 'cmap_px'
    if (calc_type in nonlin_data_filepaths) and \
       (do_calc[calc_type] or
        (not os.path.exists(nonlin_data_filepaths[calc_type]))):

        calc_cmap_px(LTE_filepath, E_MeV, ncf, use_beamline, N_KICKS,
                     nonlin_data_filepaths, common_remote_opts)

    if ('tswa_xplus' in nonlin_data_filepaths) and \
       (do_calc['tswa'] or
        (not os.path.exists(nonlin_data_filepaths['tswa_xplus'])) or
        (not os.path.exists(nonlin_data_filepaths['tswa_xminus'])) or
        (not os.path.exists(nonlin_data_filepaths['tswa_yplus'])) or
        (not os.path.exists(nonlin_data_filepaths['tswa_yminus']))
        ):

        calc_tswa(LTE_filepath, E_MeV, ncf, use_beamline, N_KICKS,
                  nonlin_data_filepaths, common_remote_opts)

    calc_type = 'nonlin_chrom'
    if (calc_type in nonlin_data_filepaths) and \
       (do_calc[calc_type] or
        (not os.path.exists(nonlin_data_filepaths[calc_type]))):

        calc_nonlin_chrom(LTE_filepath, E_MeV, nonlin_config, use_beamline, N_KICKS,
                          nonlin_data_filepaths, common_remote_opts)

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
         del_tmp_files=True, run_local=False, remote_opts=remote_opts, **kwargs)

def calc_tswa(
    LTE_filepath, E_MeV, nonlin_config, use_beamline, N_KICKS,
    nonlin_data_filepaths, common_remote_opts):
    """"""

    ncf = nonlin_config

    calc_type = 'tswa'

    calc_opts = ncf[f'{calc_type}_calc_opts']

    n_turns = calc_opts['n_turns']

    save_fft = calc_opts.get('save_fft', False)

    g = ncf['tswa_grids'][calc_opts['grid_name']]
    nx, ny = g['nx'], g['ny']
    x_offset = g.get('x_offset', 1e-6)
    y_offset = g.get('y_offset', 1e-6)
    abs_xmax = g['abs_xmax']
    abs_ymax = g['abs_ymax']

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
    LTE_filepath, E_MeV, nonlin_config, use_beamline, N_KICKS,
    nonlin_data_filepaths, common_remote_opts):
    """"""

    ncf = nonlin_config

    calc_type = 'nonlin_chrom'

    output_filepath = nonlin_data_filepaths[calc_type]

    calc_opts = ncf[f'{calc_type}_calc_opts']

    n_turns = calc_opts['n_turns']

    save_fft = calc_opts.get('save_fft', False)

    g = ncf['nonlin_chrom_grids'][calc_opts['grid_name']]
    ndelta = g['ndelta']
    x_offset = g.get('x_offset', 1e-6)
    y_offset = g.get('y_offset', 1e-6)
    delta_offset = g.get('delta_offset', 0.0)
    delta_min = g['delta_min'] + delta_offset
    delta_max = g['delta_max'] + delta_offset

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

def _save_nonlin_plots_to_pdf(report_folderpath, calc_type, existing_fignums):
    """"""

    pp = PdfPages(os.path.join(report_folderpath, f'{calc_type}.pdf'))

    for fignum in [fignum for fignum in plt.get_fignums()
                   if fignum not in existing_fignums]:
        pp.savefig(figure=fignum)
        plt.close(fignum)

    pp.close()

def plot_nonlin_props(report_folderpath, nonlin_config, do_plot):
    """"""

    ncf = nonlin_config

    nonlin_data_filepaths = get_nonlin_data_filepaths(report_folderpath, ncf)

    existing_fignums = plt.get_fignums()

    calc_type = 'fmap_xy'
    if (calc_type in nonlin_data_filepaths) and do_plot[calc_type]:
        pe.nonlin.plot_fma_xy(
            nonlin_data_filepaths[calc_type], title='',
            is_diffusion=True, scatter=False)

        _save_nonlin_plots_to_pdf(report_folderpath, calc_type, existing_fignums)


    calc_type = 'fmap_px'
    if (calc_type in nonlin_data_filepaths) and do_plot[calc_type]:
        pe.nonlin.plot_fma_px(
            nonlin_data_filepaths[calc_type], title='',
            is_diffusion=True, scatter=False)

        _save_nonlin_plots_to_pdf(report_folderpath, calc_type, existing_fignums)

    calc_type = 'cmap_xy'
    if (calc_type in nonlin_data_filepaths) and do_plot[calc_type]:
        _plot_kwargs = ncf.get(f'{calc_type}_plot_opts', {})
        pe.nonlin.plot_cmap_xy(
            nonlin_data_filepaths[calc_type], title='', is_log10=True,
            scatter=False, **_plot_kwargs)

        _save_nonlin_plots_to_pdf(report_folderpath, calc_type, existing_fignums)

    calc_type = 'cmap_px'
    if (calc_type in nonlin_data_filepaths) and do_plot[calc_type]:
        _plot_kwargs = ncf.get(f'{calc_type}_plot_opts', {})
        pe.nonlin.plot_cmap_px(
            nonlin_data_filepaths[calc_type], title='',
            is_log10=True, scatter=False, **_plot_kwargs)

        _save_nonlin_plots_to_pdf(report_folderpath, calc_type, existing_fignums)

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

        if plot_plus_minus_combined:

            for plane in ['x', 'y']:

                pe.nonlin.plot_tswa_both_sides(
                    nonlin_data_filepaths[f'tswa_{plane}plus'],
                    nonlin_data_filepaths[f'tswa_{plane}minus'],
                    title='', **_plot_kwargs
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
        else:
            fit_abs_xmax = dict(plus=_plot_kwargs['fit_xmax'],
                                minus=np.abs(_plot_kwargs['fit_xmin']))
            fit_abs_ymax = dict(plus=_plot_kwargs['fit_ymax'],
                                minus=np.abs(_plot_kwargs['fit_ymin']))

            for plane in ['x', 'y']:
                for sign in ['plus', 'minus']:
                    data_key = f'tswa_{plane}{sign}'
                    if plane == 'x':
                        pe.nonlin.plot_tswa(
                            nonlin_data_filepaths[data_key], title='',
                            fit_abs_xmax=fit_abs_xmax[sign], **_plot_kwargs)
                    else:
                        pe.nonlin.plot_tswa(
                            nonlin_data_filepaths[data_key], title='',
                            fit_abs_ymax=fit_abs_ymax[sign], **_plot_kwargs)

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

        _save_nonlin_plots_to_pdf(report_folderpath, calc_type, existing_fignums)

        tswa_page_caption_list = []
        for k in sel_tswa_caption_keys:
            i = tswa_caption_keys.index(k)
            page = i + 1
            tswa_page_caption_list.append((page, tswa_captions[i]))

        with open(os.path.join(
            report_folderpath, 'tswa.plot_suppl.pkl'), 'wb') as f:
            pickle.dump(tswa_page_caption_list, f)

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

        pe.nonlin.plot_chrom(
            nonlin_data_filepaths[calc_type], title='', **_plot_kwargs)

        _save_nonlin_plots_to_pdf(report_folderpath, calc_type, existing_fignums)


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

def get_default_config_and_comments(example=False):
    """
    """

    production = True
    #production = False

    com_map = yaml.comments.CommentedMap
    com_seq = yaml.comments.CommentedSeq
    sqss = yaml.scalarstring.SingleQuotedScalarString

    anchors = {}

    conf = com_map()

    if example:
        _yaml_append_map(conf, 'author', '')

    _yaml_append_map(conf, 'E_MeV', 3e3, eol_comment='REQUIRED')

    # ##########################################################################

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

    _yaml_append_map(d, 'use_beamline_cell', sqss('CELL'), eol_comment='REQUIRED')
    d['use_beamline_cell'].yaml_set_anchor('use_beamline_cell')
    anchors['use_beamline_cell'] = d['use_beamline_cell']
    _yaml_append_map(d, 'use_beamline_ring', sqss('RING'), eol_comment='REQUIRED')
    d['use_beamline_ring'].yaml_set_anchor('use_beamline_ring')
    anchors['use_beamline_ring'] = d['use_beamline_ring']

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
    m = com_map(print_scalars = [], right_margin_adj = 0.85)
    m.fa.set_flow_style()
    d3.append(m)
    #
    zoom_in = com_map(print_scalars = [], right_margin_adj = 0.85, slim = [0, 9],
                      disp_elem_names = {
                          'bends': True, 'quads': True, 'sexts': True,
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

    if example:
        _yaml_append_map(d, 'extra_props', com_map(), before_comment='\n')
        d2 = d['extra_props']

        _yaml_append_map(d2, 'beta', com_map())
        d3 = d2['beta']
        #
        spec = com_map(label = sqss('$(\beta_x, \beta_y)$ at LS Center'),
                       name = sqss('LS_marker_elem_name'), occur = 0)
        _yaml_append_map(d3, 'LS', spec)
        #
        spec = com_map(label = sqss('$(\beta_x, \beta_y)$ at SS Center'),
                       name = sqss('SS_marker_elem_name'), occur = 0)
        _yaml_append_map(d3, 'SS', spec)

        _yaml_append_map(d2, 'phase_adv', com_map())
        d3 = d2['phase_adv']
        #
        elem1 = com_map(name = 'LS_marker_elem_name', occur = 0)
        elem1.fa.set_flow_style()
        elem2 = com_map(name = 'SS_marker_elem_name', occur = 0)
        elem2.fa.set_flow_style()
        spec = com_map(
            label = sqss(r'Phase Advance btw. LS \& SS $(\Delta\nu_x, \Delta\nu_y)$'),
            elem1 = elem1, elem2 = elem2)
        _yaml_append_map(d3, 'LS & SS', spec)

        _yaml_append_map(d2, 'floor_comparison', com_map())
        d3 = d2['floor_comparison']
        #
        _yaml_append_map(d3, 'ref_flr_filepath', sqss('?.flr'),
                         eol_comment='REQUIRED if "floor_comparison" is specified')
        #
        ref_elem = com_map(name = 'SS_center_marker_elem_name_in_ref_lattice', occur = 0)
        ref_elem.fa.set_flow_style()
        cur_elem = com_map(name = 'SS_center_marker_elem_name_in_cur_lattice', occur = 0)
        cur_elem.fa.set_flow_style()
        spec = com_map(
            label = sqss('Source Point Diff. at SS $(\Delta x, \Delta z)$'),
            ref_elem = ref_elem, cur_elem = cur_elem)
        _yaml_append_map(d3, 'SS', spec)
        #
        ref_elem = com_map(name = 'LS_center_marker_elem_name_in_ref_lattice', occur = 1)
        ref_elem.fa.set_flow_style()
        cur_elem = com_map(name = 'LS_center_marker_elem_name_in_cur_lattice', occur = 1)
        cur_elem.fa.set_flow_style()
        spec = com_map(
            label = sqss('Source Point Diff. at LS $(\Delta x, \Delta z)$'),
            ref_elem = ref_elem, cur_elem = cur_elem)
        _yaml_append_map(d3, 'LS', spec)

        _yaml_append_map(d2, 'length', com_map())
        d3 = d2['length']
        #
        name_list = com_seq(['drift_elem_name_1', 'drift_elem_name_2', 'drift_elem_name_3'])
        name_list.fa.set_flow_style()
        spec = com_map(
            label = sqss('Length of Short Straight'),
            name_list = name_list,
        )
        _yaml_append_map(d3, 'L_SS', spec)
        #
        name_list = com_seq(['drift_elem_name_1', 'drift_elem_name_2', 'drift_elem_name_3'])
        name_list.fa.set_flow_style()
        spec = com_map(
            label = sqss('Length of Long Straight'),
            name_list = name_list,
        )
        _yaml_append_map(d3, 'L_LS', spec)

    #---------------------------------------------------------------------------

    if example:

        comment = '''
        You can here specify the order of the computed lattice property values in
        the table within the generated report.'''
        _yaml_append_map(d, 'table_order', com_seq(), before_comment=comment,
                         before_indent=2)
        d2 = d['table_order']
        #
        for i, (prop_name_or_list, comment) in enumerate([
            ('E_GeV', 'Beam energy'),
            ('eps_x', 'Natural horizontal emittance'),
            ('J', 'Damping partitions'),
            ('nu', 'Ring tunes'),
            ('ksi_nat', 'Natural chromaticities'),
            ('ksi_cor', 'Corrected chromaticities'),
            ('alphac', 'Momentum compaction'),
            ('U0', 'Energy loss per turn'),
            ('sigma_delta', 'Energy spread'),
            (['beta', 'LS'], None),
            (['beta', 'SS'], None),
            ('max_beta', 'Max beta functions'),
            ('min_beta', 'Min beta functions'),
            ('max_min_etax', 'Max & Min etax'),
            (['phase_adv', 'LS & SS'], None),
            (['length', 'L_LS'], None),
            (['length', 'L_SS'], None),
            ('circumf', 'Circumference'),
            ('circumf_change_%', 'Circumference change [%] from NSLS-II'),
            ('n_periods_in_ring', 'Number of super-periods for a full ring'),
            (['floor_comparison', 'LS'], None),
            (['floor_comparison', 'SS'], None),
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

    _yaml_append_map(conf, 'nonlin', com_map(), before_comment='\n')
    d = conf['nonlin']

    keys = ['fmap_xy', 'fmap_px', 'cmap_xy', 'cmap_px', 'tswa', 'nonlin_chrom']
    comments = [
        'On-Momentum Frequency Map', 'Off-Momentum Frequency Map',
        'On-Momentum Chaos Map', 'Off-Momentum Chaos Map',
        'Tune Shift with Amplitude', 'Nonlinear Chromaticity',
    ]
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

    N_KICKS = com_map(KQUAD=40, KSEXT=8, CSBEND=12)
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

    xy1 = com_map(xmin = -8e-3, xmax = +8e-3, ymin = 0.0, ymax = +2e-3,
                  nx = 201, ny = 201)
    if example:
        _yaml_append_map(xy1, 'x_offset', 1e-6,
                         before_comment='Optional (below)', before_indent=6)
        _yaml_append_map(xy1, 'y_offset', 1e-6)
        _yaml_append_map(xy1, 'delta_offset', 0.0)
    xy1.yaml_set_anchor('map_xy1')
    anchors['map_xy1'] = xy1
    #
    xyTest = com_map(nx = 21, ny = 21)
    #xyTest.fa.set_flow_style()
    xyTest.add_yaml_merge([(0, anchors['map_xy1'])])
    #
    xy_grids = com_map(xy1 = xy1, xyTest = xyTest)
    #
    comment = '''
    List of 2-D x-y grid specs for fmap & cmap calculations:'''
    _yaml_append_map(d, 'xy_grids', xy_grids,
                     before_comment=comment, before_indent=2)

    #---------------------------------------------------------------------------

    px1 = com_map(
        delta_min = -0.05, delta_max = +0.05, xmin = -8e-3, xmax = +8e-3,
        ndelta = 201, nx = 201)
    if example:
        _yaml_append_map(px1, 'x_offset', 1e-6,
                         before_comment='Optional (below)', before_indent=6)
        _yaml_append_map(px1, 'y_offset', 1e-6)
        _yaml_append_map(px1, 'delta_offset', 0.0)
    px1.yaml_set_anchor('map_px1')
    anchors['map_px1'] = px1
    #
    pxTest = com_map(ndelta = 21, nx = 21)
    #pxTest.fa.set_flow_style()
    pxTest.add_yaml_merge([(0, anchors['map_px1'])])
    #
    px_grids = com_map(px1 = px1, pxTest = pxTest)
    #
    comment = '''
    List of 2-D delta-x grid specs for fmap & cmap calculations:'''
    _yaml_append_map(d, 'px_grids', px_grids,
                     before_comment=comment, before_indent=2)

    #---------------------------------------------------------------------------

    if production:
        opts = com_map(grid_name = sqss('xy1'), n_turns = 1024)
    else:
        opts = com_map(grid_name = sqss('xyTest'), n_turns = 1024)
    comment = '\nOptions specific only for on-momentum frequency map calculation'
    _yaml_append_map(
        d, 'fmap_xy_calc_opts', opts,
        before_comment=comment, before_indent=2)

    #---------------------------------------------------------------------------

    if production:
        opts = com_map(grid_name = sqss('px1'), n_turns = 1024)
    else:
        opts = com_map(grid_name = sqss('pxTest'), n_turns = 1024)
    comment = '\nOptions specific only for off-momentum frequency map calculation'
    _yaml_append_map(
        d, 'fmap_px_calc_opts', opts,
        before_comment=comment, before_indent=2)

    #---------------------------------------------------------------------------

    if production:
        opts = com_map(grid_name = sqss('xy1'), n_turns = 128)
    else:
        opts = com_map(grid_name = sqss('xyTest'), n_turns = 128)
    comment = '\nOptions specific only for on-momentum chaos map calculation'
    _yaml_append_map(
        d, 'cmap_xy_calc_opts', opts,
        before_comment=comment, before_indent=2)

    #---------------------------------------------------------------------------

    if production:
        opts = com_map(grid_name = sqss('px1'), n_turns = 128)
    else:
        opts = com_map(grid_name = sqss('pxTest'), n_turns = 128)
    comment = '\nOptions specific only for off-momentum chaos map calculation'
    _yaml_append_map(
        d, 'cmap_px_calc_opts', opts,
        before_comment=comment, before_indent=2)

    #---------------------------------------------------------------------------

    xy1 = com_map(abs_xmax = 1e-3, nx = 50, abs_ymax = 0.5e-3, ny = 50)
    if example:
        _yaml_append_map(xy1, 'x_offset', 1e-6,
                         before_comment='Optional (below)', before_indent=6)
        _yaml_append_map(xy1, 'y_offset', 1e-6)
    xy1.yaml_set_anchor('tswa_xy1')
    anchors['tswa_xy1'] = xy1
    #
    tswa_grids = com_map(xy1 = xy1)
    #
    comment = '''
    List of 1-D grid specs for tune-shift-with-amplitude calculation:'''
    _yaml_append_map(d, 'tswa_grids', tswa_grids,
                     before_comment=comment, before_indent=2)

    #---------------------------------------------------------------------------

    remote_opts = com_map(time = sqss('7:00'))
    tswa_calc_opts = com_map(grid_name = sqss('xy1'), n_turns = 1024,
                             remote_opts = remote_opts)
    if example:
        _yaml_append_map(tswa_calc_opts, 'save_fft', False)
    #
    comment = 'Options specific only for tune-shift-with-amplitude calculation'
    _yaml_append_map(d, 'tswa_calc_opts', tswa_calc_opts,
                     before_comment=comment, before_indent=2)

    #---------------------------------------------------------------------------

    p1 = com_map(delta_min = -4e-2, delta_max = +3e-2, ndelta = 100)
    if example:
        _yaml_append_map(p1, 'x_offset', 1e-6,
                         before_comment='Optional (below)', before_indent=6)
        _yaml_append_map(p1, 'y_offset', 1e-6)
        _yaml_append_map(p1, 'delta_offset', 0.0)
    p1.yaml_set_anchor('p1')
    anchors['p1'] = p1
    #
    nonlin_chrom_grids = com_map(p1 = p1)
    #
    comment = '''
    List of 1-D grid specs for nonlinear chromaticity calculation:'''
    _yaml_append_map(d, 'nonlin_chrom_grids', nonlin_chrom_grids,
                     before_comment=comment, before_indent=2)

    #---------------------------------------------------------------------------

    remote_opts = com_map(time = sqss('7:00'))
    nonlin_chrom_calc_opts = com_map(grid_name = sqss('p1'), n_turns = 1024,
                                     remote_opts = remote_opts)
    if example:
        _yaml_append_map(nonlin_chrom_calc_opts, 'save_fft', False)
    #
    comment = 'Options specific only for nonlinear chromaticity calculation'
    _yaml_append_map(d, 'nonlin_chrom_calc_opts', nonlin_chrom_calc_opts,
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

        nonlin_chrom_plot_opts = com_map()
        _yaml_append_map(nonlin_chrom_plot_opts, 'plot_fft', False)
        _yaml_append_map(nonlin_chrom_plot_opts, 'max_chrom_order', 4)
        _yaml_append_map(nonlin_chrom_plot_opts, 'fit_deltalim', fit_deltalim)

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

def determine_calc_plot_bools(report_folderpath, nonlin_config, sel_plots):
    """"""

    ncf = nonlin_config

    nonlin_data_filepaths = get_nonlin_data_filepaths(report_folderpath, ncf)

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
                pdf_fp = os.path.join(report_folderpath, f'{k}.pdf')
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

def gen_report_type_0(config_filepath):
    """"""

    conf = get_default_config_and_comments()

    config_loader = yaml.YAML()
    user_conf = config_loader.load(Path(config_filepath).read_text())

    conf.update(user_conf)

    # Allow multi-line definition for a long LTE filepath in YAML
    conf['input_LTE']['filepath'] = ''.join([
        _s.strip() for _s in conf['input_LTE']['filepath'].splitlines()])

    assert conf['input_LTE']['filepath'].endswith('.lte')
    input_LTE_filepath = conf['input_LTE']['filepath']
    if input_LTE_filepath == '?.lte':
        raise ValueError('"input_LTE/filepath" must be specified in the config file')

    rootname = os.path.basename(input_LTE_filepath).replace('.lte', '')

    report_folderpath = f'report_{rootname}'
    Path(report_folderpath).mkdir(exist_ok=True)

    if conf['input_LTE'].get('load_param', False):
        gen_LTE_from_base_LTE_and_param_file(conf, input_LTE_filepath)

    if conf['input_LTE'].get('zeroSexts_filepath', ''):
        zeroSexts_LTE_filepath = conf['input_LTE']['zeroSexts_filepath']
        assert os.path.exists(zeroSexts_LTE_filepath)
    else:
        # Turn off all sextupoles
        zeroSexts_LTE_filepath = gen_zeroSexts_LTE(
            input_LTE_filepath, report_folderpath,
            regenerate=conf['input_LTE'].get('regenerate_zeroSexts', False))

    lin_summary_pkl_filepath = os.path.join(report_folderpath, 'lin.pkl')

    if (not os.path.exists(lin_summary_pkl_filepath)) or \
        conf['lattice_props'].get('recalc', False):

        # Make sure to override "replot" in the config
        conf['lattice_props']['replot'] = True

        d = calc_lin_props(
            input_LTE_filepath, report_folderpath, conf['E_MeV'],
            conf['lattice_props'], zeroSexts_LTE_filepath=zeroSexts_LTE_filepath)

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

    twiss_pdf_filepath = os.path.join(report_folderpath, 'twiss.pdf')

    if (not os.path.exists(twiss_pdf_filepath)) or \
        conf['lattice_props'].get('replot', False):
        twiss_plot_captions = plot_lin_props(
            report_folderpath, conf['lattice_props']['twiss_plot_opts'],
            conf['lattice_props']['twiss_plot_captions'], skip_plots=False)
    else:
        twiss_plot_captions = get_only_lin_props_plot_captions(
            report_folderpath, conf['lattice_props']['twiss_plot_opts'],
            conf['lattice_props']['twiss_plot_captions'])

    if 'nonlin' in conf:

        ncf = conf['nonlin']

        all_calc_types = ['fmap_xy', 'fmap_px', 'cmap_xy', 'cmap_px',
                          'tswa', 'nonlin_chrom']

        sel_plots = {k: False for k in all_calc_types}
        for k, v in ncf['include'].items():
            assert k in all_calc_types
            sel_plots[k] = v

        do_calc, do_plot = determine_calc_plot_bools(report_folderpath, ncf, sel_plots)

        if do_calc and any(do_calc.values()):
            calc_nonlin_props(input_LTE_filepath, report_folderpath,
                              conf['E_MeV'], ncf, do_calc)

        if do_plot and any(do_plot.values()):
            plot_nonlin_props(report_folderpath, ncf, do_plot)

    build_report(conf, input_LTE_filepath, rootname, report_folderpath, lin_data,
                 twiss_plot_captions)

def gen_example_config_file(config_filepath, full_or_min):
    """"""

    if full_or_min == 'full':
        conf = get_default_config_and_comments(example=True)
    elif full_or_min == 'min':
        conf = get_default_config_and_comments(example=False)
    else:
        raise ValueError('"full_or_min" must be either `True` or `False`')

    dumper = yaml.YAML()
    dumper.preserve_quotes = True
    dumper.width = 70
    dumper.boolean_representation = ['False', 'True']

    #dumper.dump(conf, sys.stdout)

    with open(config_filepath, 'w') as f:
        dumper.dump(conf, f)

def get_parsed_args():
    """"""

    parser = argparse.ArgumentParser(
        prog='pyele_report',
        description='Automated report generator for PyELEGANT')
    parser.add_argument('-t', '--type', type=int, default=0,
                        help='Report Type (default: 0, all available: [0])')
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '-f', '--full-example-config', default=False, action='store_true',
        help='Generate a full-example config YAML file')
    group.add_argument(
        '-m', '--min-example-config', default=False, action='store_true',
        help='Generate a minimum-example config YAML file')
    parser.add_argument(
        'config_filepath', type=str,
        help='''\
    Path to YAML file that contains configurations for report generation.
    Or, if "--full-example-config" or "--min-example-config" was specified,
    an example config file will be generated and saved at this file path.''')

    args = parser.parse_args()
    if False:
        print(args)
        print(f'Record Type = {args.type}')
        print(f'Generate Full Example Config? = {args.full_example_config}')
        print(f'Generate Min Example Config? = {args.min_example_config}')
        print(f'Config File = {args.config_filepath}')

    return args

def gen_report(args):
    """"""

    config_filepath = args.config_filepath

    if args.full_example_config:
        gen_example_config_file(config_filepath, 'full')
    elif args.min_example_config:
        gen_example_config_file(config_filepath, 'min')
    else:
        if args.type == 0:
            if not os.path.exists(config_filepath):
                raise OSError(f'Specified config file "{config_filepath}" does not exist.')
            gen_report_type_0(config_filepath)
        else:
            raise NotImplementedError(f'Unexpected report type: {report_type}')

def main():
    """"""

    args = get_parsed_args()
    gen_report(args)

if __name__ == '__main__':

    main()