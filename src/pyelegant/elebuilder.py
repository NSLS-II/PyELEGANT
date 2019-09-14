import re

from . import util

# From https://ops.aps.anl.gov/manuals/elegant_latest/elegant.pdf
_RAW_ELE_BLOCK_INFO_STR = '''
&run_setup
    STRING lattice = NULL;
    STRING use_beamline = NULL;
    STRING rootname = NULL;
    STRING output = NULL;
    STRING centroid = NULL;
    STRING sigma = NULL;
    STRING final = NULL;
    STRING acceptance = NULL;
    STRING losses = NULL;
    STRING magnets = NULL;
    STRING semaphore_file = NULL;
    STRING parameters = NULL;
    long combine_bunch_statistics = 0;
    long wrap_around = 1;
    long final_pass = 0;
    long default_order = 2;
    long concat_order = 0;
    long print_statistics = 0;
    long show_element_timing = 0;
    long monitor_memory_usage = 0;
    long random_number_seed = 987654321;
    long correction_iterations = 1;
    double p_central = 0.0;
    double p_central_mev = 0.0;
    long always_change_p0 = 0;
    STRING expand_for = NULL;
    long tracking_updates = 1;
    long echo_lattice = 0;
    STRING search_path = NULL;
    long element_divisions = 0;
    long load_balancing_on = 0;
&end

&load_parameters
    STRING filename = NULL;
    STRING filename_list = NULL;
    STRING include_name_pattern = NULL;
    STRING exclude_name_pattern = NULL;
    STRING include_item_pattern = NULL;
    STRING exclude_item_pattern = NULL;
    STRING include_type_pattern = NULL;
    STRING exclude_type_pattern = NULL;
    STRING edit_name_command = NULL;
    long change_defined_values = 0;
    long clear_settings = 0;
    long allow_missing_elements = 0;
    long allow_missing_parameters = 0;
    long allow_missing_files = 0;
    long force_occurence_data = 0;
    long verbose = 0;
    long skip_pages = 0;
    long use_first = 0;
&end

&twiss_output
    STRING filename = NULL;
    long matched = 1;
    long output_at_each_step = 0;
    long output_before_tune_correction = 0;
    long final_values_only = 0;
    long statistics = 0;
    long radiation_integrals = 0;
    long concat_order = 3;
    long higher_order_chromaticity = 0;
    long higher_order_chromaticity_points = 5;
    double higher_order_chromaticity_range = 4e-4;
    double chromatic_tune_spread_half_range = 0;
    long quick_higher_order_chromaticity = 0;
    double beta_x = 1;
    double alpha_x = 0;
    double eta_x = 0;
    double etap_x = 0;
    double beta_y = 1;
    double alpha_y = 0;
    double eta_y = 0;
    double etap_y = 0;
    STRING reference_file = NULL;
    STRING reference_element = NULL;
    long reference_element_occurrence = 0;
    long reflect_reference_values = 0;
    long cavities_are_drifts_if_matched = 1;
    long compute_driving_terms = 0;
    long leading_order_driving_terms_only = 0;
    STRING s_dependent_driving_terms_file = NULL;
    long local_dispersion = 1;
&end

&floor_coordinates
    STRING filename = NULL;
    double X0 = 0.0;
    double Z0 = 0.0;
    double theta0 = 0.0;
    long include_vertices = 0;
    long vertices_only = 0;
    long magnet_centers = 0;
    long store_vertices = 0;
&end

&rpn_load
    STRING tag = NULL;
    STRING filename = NULL;
    STRING match_column = NULL;
    STRING match_column_value = NULL;
    long matching_row_number = -1;
    STRING match_parameter = NULL;
    STRING match_parameter_value = NULL;
    long use_row = -1;
    long use_page = -1;
    long load_parameters = 0;
&end

&run_control
    long n_steps = 1;
    double bunch_frequency = 0;
    long n_indices = 0;
    long n_passes = 1;
    long n_passes_fiducial = 0;
    long reset_rf_for_each_step = 1;
    long first_is_fiducial = 0;
    long restrict_fiducialization = 0;
&end

&optimization_setup
    STRING equation = NULL;
    STRING mode = "minimize";
    STRING method = "simplex";
    double tolerance = -0.01;
    double target = 0;
    long center_on_orbit = 0;
    long center_momentum_also = 1;
    long soft_failure = 1;
    long n_passes = 2;
    long n_evaluations = 500;
    long n_restarts = 0;
    long matrix_order = 1;
    STRING log_file = NULL;
    STRING term_log_file = NULL;
    long output_sparsing_factor = 0;
    long balance_terms = 0;
    double restart_worst_term_factor = 1;
    long restart_worst_terms = 1;
    long verbose = 1;
    long balance_terms = 0;
    double simplex_divisor = 3;
    double simplex_pass_range_factor = 1;
    long include_simplex_1d_scans = 1;
    long start_from_simplex_vertex1 = 0;
    long restart_random_numbers = 0;
    STRING interrupt_file = "%s.interrupt";
    long interrupt_file_check_interval = 0;
&end

&parallel_optimization_setup
    STRING method = "simplex";
    double hybrid_simplex_tolerance = -0.01;
    double hybrid_simplex_tolerance_count = 2;
    long hybrid_simplex_comparison_interval = 0;
    double random_factor = 1
    long n_iterations = 10000;
    long max_no_change = 10000;
    long population_size = 100;
    STRING population_log = NULL;
    long print_all_individuals = 0;
    long output_sparsing_factor = 1;
    STRING crossover = "twopoint";
    STRING simplex_log = NULL;
    long simplex_log_interval = 1;
&end

&optimization_variable
    STRING name = NULL;
    STRING item = NULL;
    double lower_limit = 0;
    double upper_limit = 0;
    double step_size = 1;
    long disable = 0;
    long force_inside = 0;
&end

&optimization_covariable
    STRING name = NULL;
    STRING item = NULL;
    STRING equation = NULL;
    long disable = 0;
&end

&optimization_term
    STRING term = NULL;
    double weight = 1.0;
    STRING field_string = NULL;
    long field_initial_value = 0;
    long field_final_value = 0;
    long field_interval = 1;
    STRING input_file = NULL;
    STRING input_column = NULL;
    long verbose = 0;
&end

&optimize
    long summarize_setup = 0;
&end

&bunched_beam
    STRING bunch = NULL;
    long n_particles_per_bunch = 1;
    double time_start = 0;
    STRING matched_to_cell = NULL;
    double emit_x = 0;
    double emit_nx = 0;
    double beta_x = 1.0;
    double alpha_x = 0.0;
    double eta_x = 0.0;
    double etap_x = 0.0;
    double emit_y = 0;
    double emit_ny = 0;
    double beta_y = 1.0;
    double alpha_y = 0.0;
    double eta_y = 0.0;
    double etap_y = 0.0;
    long use_twiss_command_values = 0;
    long use_moments_output_values = 0;
    double Po = 0.0;
    double sigma_dp = 0.0;
    double sigma_s = 0.0;
    double dp_s_coupling = 0;
    double emit_z = 0;
    double beta_z = 0;
    double alpha_z = 0;
    double momentum_chirp = 0;
    long one_random_bunch = 1;
    long symmetrize = 0;
    long halton_sequence[3] = {0, 0, 0};
    long halton_radix[6] = {0, 0, 0, 0, 0, 0};
    long optimized_halton = 0;
    long randomize_order[3] = {0, 0, 0};
    long limit_invariants = 0;
    long limit_in_4d = 0;
    long enforce_rms_values[3] = {0, 0, 0};
    double distribution_cutoff[3] = {2, 2, 2};
    STRING distribution_type[3] = {"gaussian","gaussian","gaussian"};
    double centroid[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    long first_is_fiducial = 0;
    long save_initial_coordinates = 1;
&end

&save_lattice
    STRING filename = NULL;
    long output_seq = 0;
&end

&alter_elements
    STRING name = NULL;
    STRING item = NULL;
    STRING type = NULL;
    STRING exclude = NULL;
    double value = 0;
    STRING string_value = NULL;
    long differential = 0;
    long multiplicative = 0;
    long alter_at_each_step = 0;
    long alter_before_load_parameters = 0;
    long verbose = 0;
    long allow_missing_elements = 0;
    long allow_missing_parameters = 0;
    long start_occurence = 0;
    long end_occurence = 0;
    double s_start = -1;
    double s_end = -1;
    STRING before = NULL;
    STRING after = NULL;
&end

'''

ELE_BLOCK_INFO = {}
# ^ keywords, dtypes, default_vals, recommended
for block_header, rest in re.findall(
    r'&([a-zA-Z_]+)\s+([^&]+)&end', _RAW_ELE_BLOCK_INFO_STR):
    #print(block_header)
    ELE_BLOCK_INFO[block_header] = []
    #print(rest)
    #print(re.findall(r'(\w+)\s+([\w\d\[\]]+)\s+=\s+([\w"\.\-,\{\}\s]+);', rest))
    for dtype, key, default_val in re.findall(
        r'(\w+)\s+([\w\d\[\]]+)\s+=\s+([\w"\.\-,\{\}\s]+);', rest):
        ELE_BLOCK_INFO[block_header].append([key, dtype, default_val, None])
    #print('*********')
#
keys = [v[0] for v in ELE_BLOCK_INFO['run_setup']]
ELE_BLOCK_INFO['run_setup'][keys.index('output')][3] = '%s.out'
ELE_BLOCK_INFO['run_setup'][keys.index('centroid')][3] = '%s.cen'
ELE_BLOCK_INFO['run_setup'][keys.index('sigma')][3] = '%s.sig'
ELE_BLOCK_INFO['run_setup'][keys.index('final')][3] = '%s.fin'
ELE_BLOCK_INFO['run_setup'][keys.index('acceptance')][3] = '%s.acc'
ELE_BLOCK_INFO['run_setup'][keys.index('losses')][3] = '%s.lost'
ELE_BLOCK_INFO['run_setup'][keys.index('magnets')][3] = '%s.mag'
ELE_BLOCK_INFO['run_setup'][keys.index('semaphore_file')][3] = '%s.done'
ELE_BLOCK_INFO['run_setup'][keys.index('parameters')][3] = '%s.param'
keys = [v[0] for v in ELE_BLOCK_INFO['twiss_output']]
ELE_BLOCK_INFO['twiss_output'][keys.index('filename')][3] = '%s.twi'
keys = [v[0] for v in ELE_BLOCK_INFO['floor_coordinates']]
ELE_BLOCK_INFO['floor_coordinates'][keys.index('filename')][3] = '%s.flr'
#
# "&parallel_optimization_setup" also contains all the options for
# "&optimization_setup" as well. So, add those options here.
ELE_BLOCK_INFO['parallel_optimization_setup'] = \
    ELE_BLOCK_INFO['optimization_setup'][:] + ELE_BLOCK_INFO['parallel_optimization_setup']

class EleContents():
    """"""

    def __init__(self, double_format='.12g'):
        """Constructor"""

        self.text = ''

        self.double_format = double_format

    def clear(self):
        """"""

        self.text = ''

    def write(self, output_ele_filepath, nMaxTry=10, sleep=10.0):
        """"""

        util.robust_text_file_write(
            output_ele_filepath, self.text, nMaxTry=nMaxTry, sleep=sleep)

    def newline(self):
        """"""

        self.text += '\n'

    def comment(self, comment):
        """"""

        self.text += '!' + comment + '\n'

    def _should_be_inline_block(self, block_body_line_list):
        """"""

        single_line = False

        if len(block_body_line_list) == 0:
            single_line = True
        elif len(block_body_line_list) == 1:
            if len(block_body_line_list[0]) <= 80:
                single_line = True

        return single_line

    def _get_block_str(self, block_header, **kwargs):
        """"""

        keywords, dtypes, default_vals, recommended = zip(*ELE_BLOCK_INFO[block_header])

        block = []
        for k, v in kwargs.items():
            i = keywords.index(k)
            if dtypes[i] == 'STRING':
                block.append(f'{k} = "{v}"')
            elif dtypes[i] == 'long':
                block.append(f'{k} = {v:d}')
            elif dtypes[i] == 'double':
                block.append(('{k} = {v:%s}' % self.double_format).format(k=k, v=v))
            else:
                raise ValueError('Unexpected data type: {}'.format(dtypes[i]))

        if self._should_be_inline_block(block):
            first_line = f'&{block_header}  '
            final_line = '  &end\n'
            n_indent = 0
        else:
            first_line = f'&{block_header}\n'
            final_line = '\n&end\n'
            n_indent = 4

        block_str = (
            first_line +
            '\n'.join([' ' * n_indent + line for line in block]) +
            final_line)

        return block_str


    def run_setup(self, **kwargs):
        """"""

        self.text += self._get_block_str('run_setup', **kwargs)

    def load_parameters(self, **kwargs):
        """"""

        self.text += self._get_block_str('load_parameters', **kwargs)

    def twiss_output(self, **kwargs):
        """"""

        self.text += self._get_block_str('twiss_output', **kwargs)

    def floor_coordinates(self, **kwargs):
        """"""

        self.text += self._get_block_str('floor_coordinates', **kwargs)

    def rpn_load(self, **kwargs):
        """"""

        self.text += self._get_block_str('rpn_load', **kwargs)

    def run_control(self, **kwargs):
        """"""

        self.text += self._get_block_str('run_control', **kwargs)

    def optimization_setup(self, **kwargs):
        """"""

        self.text += self._get_block_str('optimization_setup', **kwargs)

    def parallel_optimization_setup(self, **kwargs):
        """"""

        self.text += self._get_block_str('parallel_optimization_setup', **kwargs)

    def optimization_variable(self, **kwargs):
        """"""

        self.text += self._get_block_str('optimization_variable', **kwargs)

    def optimization_covariable(self, **kwargs):
        """"""

        self.text += self._get_block_str('optimization_covariable', **kwargs)

    def optimization_term(self, **kwargs):
        """"""

        self.text += self._get_block_str('optimization_term', **kwargs)

    def optimize(self, **kwargs):
        """"""

        self.text += self._get_block_str('optimize', **kwargs)

    def bunched_beam(self, **kwargs):
        """"""

        self.text += self._get_block_str('bunched_beam', **kwargs)

    def save_lattice(self, **kwargs):
        """"""

        self.text += self._get_block_str('save_lattice', **kwargs)

    def alter_elements(self, **kwargs):
        """"""

        self.text += self._get_block_str('alter_elements', **kwargs)












def build_block_run_setup(
    LTE_filepath, E_MeV, use_beamline=None, rootname=None, magnets=None,
    semaphore_file=None, parameters=None, default_order=2, element_divisions=0):
    """"""

    block = []
    block += ['lattice = "{}"'.format(LTE_filepath)]
    if use_beamline is not None:
        block += ['use_beamline = "{}"'.format(use_beamline)]
    if rootname is not None:
        block += ['rootname = "{}"'.format(rootname)]
    if magnets is not None:
        block += ['magnets = "{}"'.format(magnets)]
    if semaphore_file is not None:
        block += ['semaphore_file = "{}"'.format(semaphore_file)]
    if parameters is not None:
        block += ['parameters = "{}"'.format(parameters)]
    block += ['p_central_mev = {:.9g}'.format(E_MeV)]
    if default_order != 2:
        block += ['default_order = {:d}'.format(default_order)]
    if element_divisions != 0:
        block += ['element_divisions = {:d}'.format(element_divisions)]

    ele_contents = '''
&run_setup
{}
&end
'''.format('\n'.join([' ' * 4 + line for line in block]))

    return ele_contents

def build_block_twiss_output(
    matched, filename='%s.twi', radiation_integrals=False,
    compute_driving_terms=False, concat_order=3, higher_order_chromaticity=False,
    beta_x=1.0, alpha_x=0.0, eta_x=0.0, etap_x=0.0, beta_y=1.0, alpha_y=0.0,
    eta_y=0.0, etap_y=0.0, output_at_each_step=False):
    """"""

    block = []
    block += ['filename = "{}"'.format(filename)]
    if output_at_each_step:
        block += ['output_at_each_step = 1']
    block += ['matched = {:d}'.format(1 if matched else 0)]
    block += ['radiation_integrals = {:d}'.format(1 if radiation_integrals else 0)]
    block += ['compute_driving_terms = {:d}'.format(1 if compute_driving_terms else 0)]
    block += ['concat_order = {:d}'.format(concat_order)]
    if higher_order_chromaticity:
        block += ['higher_order_chromaticity = 1']
        if concat_order != 3:
            print('WARNING: When computing higher-order chromaticity, "concat_order" should be set to 3.')
    if not matched:
        block += ['beta_x = {:.9g}'.format(beta_x)]
        block += ['alpha_x = {:.9g}'.format(alpha_x)]
        block += ['eta_x = {:.9g}'.format(eta_x)]
        block += ['etap_x = {:.9g}'.format(etap_x)]
        block += ['beta_y = {:.9g}'.format(beta_y)]
        block += ['alpha_y = {:.9g}'.format(alpha_y)]
        block += ['eta_y = {:.9g}'.format(eta_y)]
        block += ['etap_y = {:.9g}'.format(etap_y)]

    ele_contents = '''
&twiss_output
{}
&end
'''.format('\n'.join([' ' * 4 + line for line in block]))

    return ele_contents

def build_block_alter_elements(alter_elements_list):
    """"""

    block = []

    for d in alter_elements_list:
        block += [
            '&alter_elements name = {name}, type = {type}, item = {item}, '
            'value = {value:.9g} &end'.format(**d)]

    ele_contents = '\n' + '\n'.join(block) + '\n'

    return ele_contents

