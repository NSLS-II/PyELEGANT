import re
import numpy as np

from . import util

########################################################################
class EleBlocks():
    """
    From https://ops.aps.anl.gov/manuals/elegant_latest/elegant.pdf
    """

    #----------------------------------------------------------------------
    def __init__(self):
        """Constructor"""

        self.info = {}
        # ^ Will contain a tuple of (keywords, dtypes, default_vals, recommended)

        self._block_def_strs = {}
        # ^ Will contain raw block defintion strings from Elegant Manual

        self._parse_alter_elements()
        self._parse_bunched_beam()
        self._parse_chromaticity()
        self._parse_correct_tunes()
        self._parse_floor_coordinates()
        self._parse_frequency_map()
        self._parse_load_parameters()
        self._parse_optimize()
        self._parse_optimization_covariable()
        self._parse_optimization_setup()
        self._parse_parallel_optimization_setup()
        self._parse_optimization_term()
        self._parse_optimization_variable()
        self._parse_rpn_load()
        self._parse_run_control()
        self._parse_run_setup()
        self._parse_save_lattice()
        self._parse_transmute_elements()
        self._parse_twiss_output()
        self._parse_track()

        # "&parallel_optimization_setup" also contains all the options for
        # "&optimization_setup" as well. So, add those options here.
        self.info['parallel_optimization_setup'] = \
            self.info['optimization_setup'][:] + \
            self.info['parallel_optimization_setup']

        self._update_output_filepaths()

    #----------------------------------------------------------------------
    def _parse_block_def(self, block_def_str):
        """"""

        for iBlock, (block_header, rest) in enumerate(re.findall(
            r'&([a-zA-Z_]+)\s+([^&]+)&end', block_def_str)):
            #print(block_header)
            self.info[block_header] = []
            #print(rest)
            #print(re.findall(r'(\w+)\s+([\w\d\[\]]+)\s+=\s+([\w"\.\-,\{\}\s%]+);', rest))
            for dtype, key, default_val in re.findall(
                r'(\w+)\s+([\w\d\[\]]+)\s+=\s+([\w"\.\-,\{\}\s%]+);', rest):
                self.info[block_header].append([key, dtype, default_val, None])
            #print('*********')

            if iBlock != 0:
                raise ValueError('"block_def_str" contains more than one block')

        self._block_def_strs[block_header] = block_def_str

    #----------------------------------------------------------------------
    def _update_output_filepaths(self):
        """"""

        self.output_filepaths = {}
        for k, v in self.info.items():
            matched_keys = []
            for L in v:
                if (L[2] is not None) and ('%s' in L[2]):
                    #print(L[0], L[2])
                    matched_keys.append(L[0])
                if (L[3] is not None) and ('%s' in L[3]):
                    #print(L[0], L[3])
                    matched_keys.append(L[0])
            if matched_keys != []:
                self.output_filepaths[k] = np.unique(matched_keys).tolist()

        # Manually add missed output file keys
        self.output_filepaths['optimization_setup'].append('log_file')

    #----------------------------------------------------------------------
    def get_default_str(self, block_name):
        """"""

        lines = [
            '    ' + s.strip() for s in self._block_def_strs[block_name].split('\n')
            if s.strip() != '']
        lines[0] = lines[0].strip()
        lines[-1] = lines[-1].strip()

        return '\n'.join(lines)

    #----------------------------------------------------------------------
    def get_avail_blocks(self):
        """"""

        return list(self.info)

    #----------------------------------------------------------------------
    def _parse_alter_elements(self):
        """"""

        # Elegant Manual Section 7.5
        self._parse_block_def('''
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
        ''')

    #----------------------------------------------------------------------
    def _parse_bunched_beam(self):
        """"""

        # Elegant Manual Section 7.9
        self._parse_block_def('''
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
        ''')

    #----------------------------------------------------------------------
    def _parse_chromaticity(self):
        """"""

        # Elegant Manual Section 7.11
        self._parse_block_def('''
        &chromaticity
            STRING sextupoles = NULL;
            STRING exclude = NULL;
            double dnux_dp = 0;
            double dnuy_dp = 0;
            double sextupole_tweek = 1e-3;
            double correction_fraction = 0.9;
            long n_iterations = 5;
            double tolerance = 0;
            STRING strength_log = NULL;
            long change_defined_values = 0;
            double strength_limit = 0;
            long use_perturbed_matrix = 0;
            long exit_on_failure = 0;
            long update_orbit = 0;
            long verbosity = 1;
            double dK2_weight = 1;
        &end
        ''')

    #----------------------------------------------------------------------
    def _parse_correct_tunes(self):
        """"""

        # Elegant Manual Section 7.15
        self._parse_block_def('''
        &correct_tunes
            STRING quadrupoles = NULL;
            STRING exclude = NULL;
            double tune_x = 0;
            double tune_y = 0;
            long n_iterations = 5;
            double correction_fraction = 0.9;
            double tolerance = 0;
            long step_up_interval = 0;
            double max_correction_fraction = 0.9;
            double delta_correction_fraction = 0.1;
            long update_orbit = 0;
            STRING strength_log = NULL;
            long change_defined_values = 0;
            long use_perturbed_matrix = 0;
            double dK1_weight = 1;
        &end
        ''')

    #----------------------------------------------------------------------
    def _parse_floor_coordinates(self):
        """"""

        # Elegant Manual Section 7.22
        self._parse_block_def('''
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
        ''')

        d = self.info['floor_coordinates']

        # Fill recommended values
        keys = [v[0] for v in d]
        d[keys.index('filename')][3] = '%s.flr'

    #----------------------------------------------------------------------
    def _parse_frequency_map(self):
        """"""

        # Elegant Manual Section 7.23
        self._parse_block_def('''
        &frequency_map
            STRING output = NULL;
            double xmin = -0.1;
            double xmax = 0.1;
            double ymin = 1e-6;
            double ymax = 0.1;
            double delta_min = 0;
            double delta_max = 0;
            long nx = 21;
            long ny = 21;
            long ndelta = 1;
            long verbosity = 1;
            long include_changes = 0;
            long quadratic_spacing = 0;
            long full_grid_output = 0;
        &end
        ''')

        d = self.info['frequency_map']

        # Fill recommended values
        keys = [v[0] for v in d]
        d[keys.index('output')][3] = '%s.fma'

    #----------------------------------------------------------------------
    def _parse_load_parameters(self):
        """"""

        # Elegant Manual Section 7.33
        self._parse_block_def('''
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
        ''')

    #----------------------------------------------------------------------
    def _parse_optimize(self):
        """"""

        # Elegant Manual Section 7.38
        self._parse_block_def('''
        &optimize
            long summarize_setup = 0;
        &end
        ''')

    #----------------------------------------------------------------------
    def _parse_optimization_covariable(self):
        """"""

        # Elegant Manual Section 7.40
        self._parse_block_def('''
        &optimization_covariable
            STRING name = NULL;
            STRING item = NULL;
            STRING equation = NULL;
            long disable = 0;
        &end
        ''')

    #----------------------------------------------------------------------
    def _parse_optimization_setup(self):
        """"""

        # Elegant Manual Section 7.41
        self._parse_block_def('''
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
        ''')

        d = self.info['optimization_setup']

        # Fill recommended values
        keys = [v[0] for v in d]
        d[keys.index('term_log_file')][3] = '%s.tlog'
        d[keys.index('interrupt_file')][3] = '%s.interrupt'

    #----------------------------------------------------------------------
    def _parse_parallel_optimization_setup(self):
        """"""

        # Elegant Manual Section 7.42
        self._parse_block_def('''
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
        ''')

        d = self.info['parallel_optimization_setup']

        # Fill recommended values
        keys = [v[0] for v in d]
        d[keys.index('population_log')][3] = '%s.pop'
        d[keys.index('simplex_log')][3] = '%s.simlog'

    #----------------------------------------------------------------------
    def _parse_optimization_term(self):
        """"""

        # Elegant Manual Section 7.43
        self._parse_block_def('''
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
        ''')

    #----------------------------------------------------------------------
    def _parse_optimization_variable(self):
        """"""

        # Elegant Manual Section 7.44
        self._parse_block_def('''
        &optimization_variable
            STRING name = NULL;
            STRING item = NULL;
            double lower_limit = 0;
            double upper_limit = 0;
            double step_size = 1;
            long disable = 0;
            long force_inside = 0;
        &end
        ''')

    #----------------------------------------------------------------------
    def _parse_rpn_load(self):
        """"""

        # Elegant Manual Section 7.50
        self._parse_block_def('''
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
        ''')

    #----------------------------------------------------------------------
    def _parse_run_control(self):
        """"""

        # Elegant Manual Section 7.51
        self._parse_block_def('''
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
        ''')

    #----------------------------------------------------------------------
    def _parse_run_setup(self):
        """"""

        # Elegant Manual Section 7.52
        self._parse_block_def('''
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
        ''')

        d = self.info['run_setup']

        # Fill recommended values
        keys = [v[0] for v in d]
        d[keys.index('output')][3] = '%s.out'
        d[keys.index('centroid')][3] = '%s.cen'
        d[keys.index('sigma')][3] = '%s.sig'
        d[keys.index('final')][3] = '%s.fin'
        d[keys.index('acceptance')][3] = '%s.acc'
        d[keys.index('losses')][3] = '%s.lost'
        d[keys.index('magnets')][3] = '%s.mag'
        d[keys.index('semaphore_file')][3] = '%s.done'
        d[keys.index('parameters')][3] = '%s.param'


    #----------------------------------------------------------------------
    def _parse_save_lattice(self):
        """"""

        # Elegant Manual Section 7.54
        self._parse_block_def('''
        &save_lattice
            STRING filename = NULL;
            long output_seq = 0;
        &end
        ''')

        d = self.info['save_lattice']

        # Fill recommended values
        keys = [v[0] for v in d]
        d[keys.index('filename')][3] = '%s.new'

    #----------------------------------------------------------------------
    def _parse_transmute_elements(self):
        """"""

        # Elegant Manual Section 7.62
        self._parse_block_def('''
        &transmute_elements
            STRING name = NULL;
            STRING type = NULL;
            STRING exclude = NULL;
            STRING new_type = "DRIF";
            long disable = 0;
            long clear = 0;
        &end
        ''')
        # Note that the following has been modified from the original in the PDF
        # manual to make regex finding consistent, by changing "," to ";":
        #
        #&transmute_elements
            #STRING name = NULL, => ;
            #STRING type = NULL, => ;
            #STRING exclude = NULL, => ;
            #STRING new_type = "DRIF", => ;
            #long disable = 0;
            #long clear = 0;
        #&end

    #----------------------------------------------------------------------
    def _parse_twiss_output(self):
        """"""

        # Elegant Manual Section 7.65
        self._parse_block_def('''
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
        ''')

        d = self.info['twiss_output']

        # Fill recommended values
        keys = [v[0] for v in d]
        d[keys.index('filename')][3] = '%s.twi'

    #----------------------------------------------------------------------
    def _parse_track(self):
        """"""

        # Elegant Manual Section 7.66
        self._parse_block_def('''
        &track
            long center_on_orbit = 0;
            long center_momentum_also = 1;
            long offset_by_orbit = 0;
            long offset_momentum_also = 1;
            long soft_failure = 1;
            long stop_tracking_particle_limit = -1;
            long check_beam_structure = 0;
            STRING interrupt_file = "%s.interrupt";
        &end
        ''')

class EleDesigner():
    """"""

    def __init__(self, double_format='.12g'):
        """Constructor"""

        self.clear()

        self.double_format = double_format

        self.blocks = EleBlocks()

    def clear(self):
        """"""

        self.text = ''

        self.rootname = None
        self.output_filepath_list = []
        self.actual_output_filepath_list = []

    def write(self, output_ele_filepath, nMaxTry=10, sleep=10.0):
        """"""

        util.robust_text_file_write(
            output_ele_filepath, self.text, nMaxTry=nMaxTry, sleep=sleep)

    def add_newline(self):
        """"""

        self.text += '\n'

    def add_comment(self, comment):
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

        if block_header not in self.blocks.info:

            print('* Valid block names are the following:')
            print('\n'.join(self.blocks.get_avail_blocks()))

            raise ValueError(f'Unexpected block name "{block_header}".')

        keywords, dtypes, default_vals, recommended = zip(
            *self.blocks.info[block_header])

        block = []
        for k, v in kwargs.items():

            if k not in keywords:

                print(f'* Valid keys for Block "{block_header}" are the following:')
                print('\n'.join(sorted(keywords)))

                raise ValueError(f'Unexpected key "{k}" for Block "{block_header}"')

            i = keywords.index(k)
            if dtypes[i] == 'STRING':
                if v is None:
                    continue
                else:
                    block.append(f'{k} = "{v}"')
                if (block_header in self.blocks.output_filepaths) and \
                   (k in self.blocks.output_filepaths[block_header]):
                    self.output_filepath_list.append(v)
                if (block_header == 'run_setup') and (k == 'rootname'):
                    self.rootname = v
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

    def update_output_filepaths(self, ele_filepath_wo_ext):
        """"""

        if self.rootname is not None:
            ele_filepath_wo_ext = self.rootname

        self.actual_output_filepath_list = []
        for template_fp in self.output_filepath_list:
            if '%s' in template_fp:
                self.actual_output_filepath_list.append(
                    template_fp.replace('%s', ele_filepath_wo_ext))
            else:
                self.actual_output_filepath_list.append(template_fp)

        self.actual_output_filepath_list = np.unique(
            self.actual_output_filepath_list).tolist()

    #----------------------------------------------------------------------
    def add_block(self, block_name, **kwargs):
        """"""

        self.text += self._get_block_str(block_name, **kwargs)









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

