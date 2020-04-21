from IPython.display import display, clear_output, HTML
import ipywidgets as widgets
from ipywidgets import interact, fixed, Layout

import pyelegant as pe

DATA, META = None, None

def _htag(tag_type, contents, **kwargs):
    opts = ' '.join([f'{k}="{v}"' for k, v in kwargs.items()])
    if opts:
        opts = ' ' + opts
    return f'<{tag_type}{opts}>{contents}</{tag_type}>'

def _font(contents, **kwargs):
    return _htag('font', contents, **kwargs)
def _bold(contents):
    return _htag('b', contents)
def _italic(contents):
    return _htag('i', contents)
def _sup(contents):
    return _htag('sup', contents)
def _sub(contents):
    return _htag('sub', contents)
def _greek(symbol):
    return f'&{symbol};'

def _courier(contents, color='black', size='4', **kwargs):
    return _font(contents, face='courier', color=color, size=size, **kwargs)
def _timesNewRoman(contents, color='black', size='4', **kwargs):
    return _font(contents, face='times new roman', color=color, size=size, **kwargs)

_HELP_STR = {}
_HELP_STR['higher_order_chromaticity'] = f"""
    If True, requests computation of the 2{_sup("nd")} and 3{_sup("rd")} order chromaticity.
    To obtain reliable values, the user should use {_courier(_bold("concat_order=3"))} in this
    namelist and the highest available order for all beamline elements. elegant computes the
    higher-order chromaticity by finding the trace of off-momentum matrices obtained by concantenation
    of the matrix for {_courier(_bold("higher_order_chromaticity_points"))} values of
    {_italic(_greek("delta"))} over the full range {_courier(_bold("higher_order_chromaticity_range"))}.
    If {_courier(_bold("quick_higher_order_chromaticity"))} is True, then a quicker concatenation method is
    used that gives the 2{_sup("nd")}-order chromaticity only.
"""

_HELP_STR['concat_order'] = f"""
    Order ofHELP_STRconcatenation to use for determining matrix for computation
    of Twiss parameters. Using a lower order will result in inaccuracy for nonlinear lattices
    with orbits and/or momentum errors. However, for on-momentum conditions with zero orbit,
    it is much faster to use {_courier(_bold("concat_order=1"))}.
"""

_HELP_STR['compute_driving_terms'] = f"""
    If True, then resonance driving terms and tune shifts with amplitude are computed by summing over dipole,
    quadrupole, sextupole, and octupole elements. For dipoles, only the effects of gradients and sextupole terms
    are included; curvature effects are not present in the theory. In addition, these quantities may be optimized
    by using those names in optimization terms.
"""

_HELP_STR['radiation_integrals'] = f"""
    A flag indicating, if set, that radiation integrals should be computed and included in output.
"""
_HELP_STR['radiation_integrals'] += ' ' + _italic("""
    N.B.: Radiation integral computation is not correct for systems with
    vertical bending, nor does it take into account coupling. See the""")
_HELP_STR['radiation_integrals'] += ' ' + _courier(_bold("moments_output"))
_HELP_STR['radiation_integrals'] += ' ' + _italic("command if you need such computations") + '.'

def print_option_help(opt_name):
    title = _courier(_bold(opt_name), size="5")
    body = _timesNewRoman(_HELP_STR[opt_name])
    display(HTML(f'<br>{title}<br><br>{body}'))

def option_help_funcgen_on_button_click(output, opt_name):

    def on_button_clicked(_):
        """Linking function with output"""
        with output:
            # Things that happen when the button is pressed go here
            clear_output()
            print_option_help(opt_name)

    return on_button_clicked

def show_calc_gui():
    """"""

    out_calc = widgets.Output()

    ini_style = {'description_width': 'initial'}

    E_MeV_text = widgets.FloatText(value=3e3, description='E [MeV]',
                                   style=ini_style)
    E_MeV_text.layout.max_width = '200px'

    LTE_filepath_text = widgets.Text(
        value='lattice3Sext_19pm3p2m_5cell.lte', description='LTE File Path',
        style=ini_style)
    LTE_filepath_text.layout.min_width = '800px'

    output_filepath_text = widgets.Text(
        value='simple_ring_twiss.hdf5',
        description='Output File Path (.hdf5 or .pgz)', style=ini_style)
    output_filepath_text.layout.min_width = '800px'

    box_req = widgets.VBox([E_MeV_text, LTE_filepath_text,
                            output_filepath_text])

    use_beamline_text = widgets.Text(
        value='', description='use_beamline:',
        description_tooltip='Beamline name to be used', style=ini_style)
    use_beamline_text.layout.min_width = '400px'

    rad_integ_chkbox = widgets.Checkbox(
        value=False, description='radiation_integrals', style=ini_style)
    rad_integ_chkbox.layout.max_width = '200px'
    rad_integ_help = widgets.Button(description='?', style=ini_style)
    rad_integ_help.layout.max_width = '30px'
    rad_integ_help.on_click(
        option_help_funcgen_on_button_click(out_calc, 'radiation_integrals'))

    drv_chkbox = widgets.Checkbox(
        value=False, description='compute_driving_terms', style=ini_style)
    drv_chkbox.layout.max_width = '200px'
    drv_help = widgets.Button(description='?', style=ini_style)
    drv_help.layout.max_width = '30px'
    drv_help.on_click(
        option_help_funcgen_on_button_click(out_calc, 'compute_driving_terms'))

    concat_order_int = widgets.IntSlider(
        value=1, min=0, max=3, step=1, description='concat_order',
        style=ini_style)
    concat_order_help = widgets.Button(description='?', style=ini_style)
    concat_order_help.layout.max_width = '30px'
    concat_order_help.on_click(
        option_help_funcgen_on_button_click(out_calc, 'concat_order'))

    nonlin_chrom_chkbox = widgets.Checkbox(
        value=False, description='higher_order_chromaticity', style=ini_style)
    nonlin_chrom_chkbox.layout.max_width = '200px'
    nonlin_chrom_help = widgets.Button(description='?', style=ini_style)
    nonlin_chrom_help.layout.max_width = '30px'
    nonlin_chrom_help.on_click(
        option_help_funcgen_on_button_click(
            out_calc, 'higher_order_chromaticity'))

    elem_div_int = widgets.IntSlider(
        value=0, description='element_divisions:',
        description_tooltip='Number of divisions for each element',
        style=ini_style, min=0, max=50)
    elem_div_int.layout.min_width = '400px'

    del_tmp_chkbox = widgets.Checkbox(
        value=True, description='Delete temporary files (del_tmp_files)',
        style=ini_style)
    del_tmp_chkbox.layout.min_width = '400px'

    box_opt = widgets.VBox([
        use_beamline_text,
        widgets.HBox([rad_integ_chkbox, rad_integ_help]),
        widgets.HBox([drv_chkbox, drv_help]),
        widgets.HBox([concat_order_int, concat_order_help]),
        widgets.HBox([nonlin_chrom_chkbox, nonlin_chrom_help]),
        elem_div_int, del_tmp_chkbox])

    tab = widgets.Tab()
    tab.children = [box_req, box_opt]
    tab.set_title(0, 'Required')
    tab.set_title(1, 'Optional')

    button = widgets.Button(description='Calculate Twiss')

    def on_button_clicked(_):
        """Linking function with output"""
        with out_calc:
            # Things that happen when the button is pressed go here
            clear_output()
            pe.calc_ring_twiss(
                output_filepath_text.value, LTE_filepath_text.value,
                E_MeV_text.value, use_beamline=(
                    use_beamline_text.value
                    if use_beamline_text.value else None),
                radiation_integrals=rad_integ_chkbox.value,
                compute_driving_terms=drv_chkbox.value,
                element_divisions=elem_div_int.value,
                del_tmp_files=del_tmp_chkbox.value,
            )

    # Now linke the button and the function
    button.on_click(on_button_clicked)

    box = widgets.VBox([tab, button, out_calc])
    display(box)

def show_plot_gui():
    """"""

    out_plot = widgets.Output()

    ini_style = {'description_width': 'initial'}

    output_filepath_text = widgets.Text(
        value='simple_ring_twiss.hdf5',
        description='Output File Path (.hdf5 or .pgz)', style=ini_style)
    output_filepath_text.layout.min_width = '800px'
    box_req = widgets.VBox([output_filepath_text])

    s0_m_text = widgets.FloatText(
        value=0.0, description='Initial s-pos offset [m]:', style=ini_style,
        description_tooltip='s0_m')
    s0_m_text.layout.max_width = '250px'
    slim_range_slider = widgets.FloatRangeSlider(
        min=0.0, max=800.0, value=[0.0, 55.0],
        description='Visible s-pos range [m]:', style=ini_style,
        description_tooltip='slim')
    slim_range_slider.layout.min_width = '800px'
    s_margin_m_text = widgets.FloatText(
        value=0.1, description='Visible s-pos margin [m]:',
        style=ini_style, description_tooltip='s_margin_m')
    s_margin_m_text.layout.max_width = '300px'
    tooltip = ('print_scalars: Use "None" to print all available scalars. '
               'Use an empty string to print none of them.')
    print_scalars_text = widgets.Text(
        value='ex0, Jx, nux, nuy, dnux/dp, dnuy/dp',
        description='Scalar Values to be printed:',
        description_tooltip=tooltip, style=ini_style)
    print_scalars_text.layout.min_width = '800px'

    disp_bool_label = widgets.Label(value='Show names of', style=ini_style)
    disp_bool_label.layout.min_width = '100px'
    disp_bends_chkbox = widgets.Checkbox(
        value=False, description='Bends', style=ini_style)
    disp_bends_chkbox.layout.max_width = '70px'
    disp_quads_chkbox = widgets.Checkbox(
        value=False, description='Quads', style=ini_style)
    disp_quads_chkbox.layout.max_width = '70px'
    disp_sexts_chkbox = widgets.Checkbox(
        value=False, description='Sexts', style=ini_style)
    disp_sexts_chkbox.layout.max_width = '70px'
    disp_elem_types_box = widgets.HBox([
        disp_bool_label, disp_bends_chkbox, disp_quads_chkbox, disp_sexts_chkbox])
    disp_elem_font_size = widgets.BoundedIntText(
        value=8, min=1, max=30, step=1, description='Font Size', style=ini_style)
    disp_elem_font_size.layout.max_width = '120px'
    disp_elem_extra_dy_frac = widgets.BoundedFloatText(
        value=0.05, min=0.0, max=1.0, step=0.01,
        description='Extra vertical frac.', style=ini_style)
    disp_elem_extra_dy_frac.layout.max_width = '200px'
    disp_elem_layout_box = widgets.HBox(
        [disp_elem_font_size, disp_elem_extra_dy_frac])
    disp_elem_names_box = widgets.HBox(
        [disp_elem_types_box, disp_elem_layout_box],
        layout=Layout(border='solid 1px', max_width='650px'))

    box_opt = widgets.VBox(
        [s0_m_text, slim_range_slider, s_margin_m_text, print_scalars_text,
         disp_elem_names_box])

    tab = widgets.Tab()
    tab.children = [box_req, box_opt]
    tab.set_title(0, 'Required')
    tab.set_title(1, 'Optional')

    button = widgets.Button(description='Plot')

    def on_button_clicked(_):
        """Linking function with output"""
        with out_plot:
            # Things that happen when the button is pressed go here
            clear_output()

            if print_scalars_text.value.strip().lower() == 'none':
                print_scalars = None
            else:
                print_scalars = [
                    _s.strip() for _s in print_scalars_text.value.split(',')]

            if disp_bends_chkbox.value or disp_quads_chkbox.value or \
               disp_sexts_chkbox.value:
                disp_elem_names = dict(
                    bends=disp_bends_chkbox.value, quads=disp_quads_chkbox.value,
                    sexts=disp_sexts_chkbox.value,
                    font_size=disp_elem_font_size.value,
                    extra_dy_frac=disp_elem_extra_dy_frac.value,
                )
            else:
                disp_elem_names = None

            pe.plot_twiss(
                output_filepath_text.value, slim=slim_range_slider.value,
                s0_m=s0_m_text.value, print_scalars=print_scalars,
                disp_elem_names=disp_elem_names)

            global DATA, META

            result_filepath = output_filepath_text.value
            if result_filepath.endswith(('hdf5', 'h5')):
                (DATA, meta, version
                 ) = pe.util.load_sdds_hdf5_file(result_filepath)
            elif result_file_type.endswith('pgz'):
                _d = pe.util.load_pgz_file(result_filepath)
                DATA = _d['data']
                meta = _d['meta']
            else:
                try:
                    DATA, meta, version = \
                        pe.util.load_sdds_hdf5_file(result_filepath)
                except OSError:
                    _d = pe.util.load_pgz_file(result_filepath)
                    DATA = _d['data']
                    meta = _d['meta']

            print((
                'Raw data can be accessed through the '
                '"pyelegant.jupygui.twi.DATA" dict variable.'))
            print((
                'Metadata can be accessed through the '
                '"pyelegant.jupygui.twi.META" dict variable.'))


    # Now linke the button and the function
    button.on_click(on_button_clicked)

    box = widgets.VBox([tab, button, out_plot])
    display(box)