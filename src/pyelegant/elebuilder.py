from typing import List, Union, Dict
import sys
import re
import numpy as np
from types import SimpleNamespace
from pathlib import Path
import tempfile
import glob
import ast
import pickle
import itertools
import collections

from . import util
from . import sdds
from . import notation
from . import ltemanager

from . import std_print_enabled

UNAVAILABLE_BLOCK_OPTS = collections.defaultdict(list)

#----------------------------------------------------------------------
def set_unavailable_block_options(block_name, option_name):
    """"""

    if option_name not in UNAVAILABLE_BLOCK_OPTS[block_name]:
        UNAVAILABLE_BLOCK_OPTS[block_name].append(option_name)

########################################################################
class EleBlocks():
    """
    From https://ops.aps.anl.gov/manuals/elegant_latest/elegant.pdf
    Program Version 2019.4 (December 10, 2019)
    """

    #----------------------------------------------------------------------
    def __init__(self):
        """Constructor"""

        self.info = {}
        # ^ Will contain a tuple of (keywords, dtypes, default_vals, recommended)

        self._block_def_strs = {}
        # ^ Will contain raw block defintion strings from Elegant Manual

        self.OPTIM_TERM_TERM_FIELD_MAX_STR_LEN = 900
        # Max string length limitation (approximately & empirically determined)
        # for "optimization_term" block.

        self._parse_alter_elements()
        self._parse_bunched_beam()
        self._parse_chaos_map()
        self._parse_chromaticity()
        self._parse_closed_orbit()
        self._parse_correct_tunes()
        self._parse_find_aperture()
        self._parse_floor_coordinates()
        self._parse_frequency_map()
        self._parse_insert_elements()
        self._parse_load_parameters()
        self._parse_matrix_output()
        self._parse_momentum_aperture()
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
        self._parse_vary_element()

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
            #print(re.findall(r'(\w+)\s+([\w\d\[\]]+)\s*=\s*([\w"\.\-,\{\}\s%]+);', rest))
            for dtype, key, default_val in re.findall(
                r'(\w+)\s+([\w\d\[\]]+)\s*=\s*([\w"\.\-\+,\{\}\s%]+);', rest):
                if '[' in key:
                    key, size_sq_bracket_end = key.split('[')
                    assert size_sq_bracket_end[-1] == ']'
                    array_size = int(size_sq_bracket_end[:-1])
                else:
                    array_size = 0 # which means "scalar"
                self.info[block_header].append(
                    [key, dtype, default_val, None, array_size])
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
    def _parse_chaos_map(self):
        """"""

        # Elegant Manual Section 7.11
        self._parse_block_def('''
        &chaos_map
            STRING output = NULL;
            double xmin = -0.1;
            double xmax = 0.1;
            double ymin = 1e-6;
            double ymax = 0.1;
            double delta_min = 0;
            double delta_max = 0;
            long nx = 20;
            long ny = 21;
            long ndelta = 1;
            long forward_backward = 0;
            double change_x = 1e-6;
            double change_y = 1e-6;
            long verbosity = 1;
        &end
        ''')

        d = self.info['chaos_map']

        # Fill recommended values
        keys = [v[0] for v in d]
        d[keys.index('output')][3] = '%s.cmap'

    #----------------------------------------------------------------------
    def _parse_chromaticity(self):
        """"""

        # Elegant Manual Section 7.12
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
    def _parse_closed_orbit(self):
        """"""

        # Elegant Manual Section 7.13
        self._parse_block_def('''
        &closed_orbit
            STRING output = NULL;
            long output_monitors_only = 0;
            long start_from_centroid = 1;
            long start_from_dp_centroid = 0;
            double closed_orbit_accuracy = 1e-12;
            long closed_orbit_iterations = 40;
            long fixed_length = 0;
            long start_from_recirc = 0;
            long verbosity = 0;
            double iteration_fraction = 0.9;
            double fraction_multiplier = 1.05;
            double multiplier_interval = 5;
            long tracking_turns = 0;
            long disable = 0;
        &end
        ''')
        # ^ There was a duplicate line in the manual PDF file for:
        #      long output_monitors_only = 0;
        #   So, the duplicate has been removed manually here.

        d = self.info['closed_orbit']

        # Fill recommended values
        keys = [v[0] for v in d]
        d[keys.index('output')][3] = '%s.clo'

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
    def _parse_find_aperture(self):
        """"""

        # Elegant Manual Section 7.21
        self._parse_block_def('''
        &find_aperture
            STRING output = NULL;
            STRING search_output = NULL;
            STRING boundary = NULL;
            STRING mode = "many-particle";
            double xmin = -0.1;
            double xmax = 0.1;
            double xpmin = 0.0;
            double xpmax = 0.0;
            double ymin = 0.0;
            double ymax = 0.1;
            double ypmin = 0.0;
            double ypmax = 0.0;
            long nx = 21;
            long ny = 11;
            long n_splits = 0;
            double split_fraction = 0.5;
            double desired_resolution = 0.01;
            long assume_nonincreasing = 0;
            long verbosity = 0;
            long offset_by_orbit = 0;
            long n_lines = 11;
            long optimization_mode = 0;
            long full_plane = 0;
        &end
        ''')

        d = self.info['find_aperture']

        # Fill recommended values
        keys = [v[0] for v in d]
        d[keys.index('output')][3] = '%s.aper'
        d[keys.index('search_output')][3] = '%s.apso'
        d[keys.index('boundary')][3] = '%s.bnd'

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
    def _parse_insert_elements(self):
        """"""

        # Elegant Manual Section 7.27
        self._parse_block_def('''
        &insert_elements
            STRING name = NULL;
            STRING type = NULL;
            STRING exclude = NULL;
            double s_start = -1;
            double s_end = -1;
            long skip = 1;
            long disable = 0;
            long insert_before = 0;
            long add_at_end = 0;
            long add_at_start = 0;
            STRING element_def = NULL;
            long total_occurrences = 0;
            long occurrence[100]={0};
        &end
        ''')

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
    def _parse_matrix_output(self):
        """"""

        # Elegant Manual Section 7.35
        self._parse_block_def('''
        &matrix_output
            STRING printout = NULL;
            long printout_order = 1;
            STRING printout_format = "%22.15e ";
            long full_matrix_only = 0;
            long mathematica_full_matrix = 0;
            long print_element_data = 1;
            STRING SDDS_output = NULL;
            long SDDS_output_order = 1;
            long individual_matrices = 0;
            STRING SDDS_output_match = NULL;
            long output_at_each_step = 0;
            STRING start_from = NULL;
            long start_from_occurence = 1;
        &end
        ''')

        d = self.info['matrix_output']

        # Fill recommended values
        keys = [v[0] for v in d]
        d[keys.index('printout')][3] = '%s.mpr'
        d[keys.index('SDDS_output')][3] = '%s.mat'

    #----------------------------------------------------------------------
    def _parse_momentum_aperture(self):
        """"""

        # Elegant Manual Section 7.37
        self._parse_block_def('''
        &momentum_aperture
            STRING output = NULL;
            double x_initial = 0;
            double y_initial = 0;
            double delta_negative_start = 0.0;
            double delta_positive_start = 0.0;
            double delta_negative_limit = -0.10;
            double delta_positive_limit = 0.10;
            double delta_step_size = 0.01;
            long steps_back = 1;
            long splits = 2;
            long split_step_divisor = 10;
            long skip_elements = 0;
            long process_elements = 2147483647;
            double s_start = 0;
            double s_end = DBL_MAX;
            STRING include_name_pattern = NULL;
            STRING include_type_pattern = NULL;
            long fiducialize = 0;
            long verbosity = 1;
            long soft_failure = 0;
            long output_mode = 0;
            long forbid_resonance_crossing = 0;
        &end
        '''.replace('DBL_MAX', str(sys.float_info.max)))

        d = self.info['momentum_aperture']

        # Fill recommended values
        keys = [v[0] for v in d]
        d[keys.index('output')][3] = '%s.mmap'

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

        # Elegant Manual Section 7.45
        self._parse_block_def('''
        &optimization_variable
            STRING name = NULL;
            STRING item = NULL;
            double lower_limit = 0;
            double upper_limit = 0;
            long differential_limits = 0;
            double step_size = 1;
            long disable = 0;
            long force_inside = 0;
            long no_element = 0;
            double initial_value = 0;
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

    #----------------------------------------------------------------------
    def _parse_vary_element(self):
        """"""

        # Elegant Manual Section 7.69
        self._parse_block_def('''
        &vary_element
            long index_number = 0;
            long index_limit = 0;
            STRING name = NULL;
            STRING item = NULL;
            double initial = 0;
            double final = 0;
            long differential = 0;
            long multiplicative = 0;
            long geometric = 0;
            STRING enumeration_file = NULL;
            STRING enumeration_column = NULL;
        &end
        ''')

########################################################################
class InfixEquation():
    """"""

    #----------------------------------------------------------------------
    def __init__(self, variable_str_repr: str, rpn_conv_post_repl: List = None,
                 double_format: str = '.12g'):
        """Constructor"""

        try:
            ast.parse(variable_str_repr)
        except SyntaxError as e:
            print(e.text)
            print(' ' * (e.offset - 1) + '^')
            print('WARNING: Invalid infix expression')

        self.equation_repr = variable_str_repr

        if rpn_conv_post_repl is None:
            self.rpn_conv_post_repl = []
        else:
            self.rpn_conv_post_repl = rpn_conv_post_repl

        self.double_format = double_format

    #----------------------------------------------------------------------
    def copy(self):
        """"""

        copy = InfixEquation(self.equation_repr,
                             rpn_conv_post_repl=self.rpn_conv_post_repl)

        return copy

    #----------------------------------------------------------------------
    def torpn(self):
        """"""

        return notation.convert_infix_to_rpn(
            self.equation_repr, post_repl=self.rpn_conv_post_repl,
            double_format=self.double_format)

    #----------------------------------------------------------------------
    def __repr__(self):
        """"""

        return self.equation_repr

    #----------------------------------------------------------------------
    def __str__(self):
        """"""

        return self.equation_repr

    #----------------------------------------------------------------------
    def __neg__(self):
        """"""

        copy = self.copy()

        copy.equation_repr = f'-({self.equation_repr})'

        return copy

    #----------------------------------------------------------------------
    def __add__(self, other):
        """"""

        copy = self.copy()

        if isinstance(other, (int, np.integer, str)):
            copy.equation_repr = f'{self.equation_repr} + {other}'
        elif isinstance(other, float):
            copy.equation_repr = f'{self.equation_repr} + {other:{self.double_format}}'
        elif isinstance(other, InfixEquation):
            copy.rpn_conv_post_repl = list(set(
                self.rpn_conv_post_repl + other.rpn_conv_post_repl))
            copy.equation_repr = f'{self.equation_repr} + {other.equation_repr}'
        else:
            raise NotImplementedError(f'__add__ for type "{type(other)}"')

        return copy

    #----------------------------------------------------------------------
    def __radd__(self, left):
        """"""

        copy = self.copy()

        if isinstance(left, (int, str)):
            copy.equation_repr = f'{left} + {self.equation_repr}'
        elif isinstance(left, float):
            copy.equation_repr = f'{left:{self.double_format}} + {self.equation_repr}'
        elif isinstance(left, InfixEquation):
            copy.rpn_conv_post_repl = list(set(
                self.rpn_conv_post_repl + left.rpn_conv_post_repl))
            copy.equation_repr = f'{left.equation_repr} + {self.equation_repr}'
        else:
            raise NotImplementedError(f'__radd__ for type "{type(left)}"')

        return copy

    #----------------------------------------------------------------------
    def __sub__(self, other):
        """"""

        copy = self.copy()

        if isinstance(other, (int, np.integer, str)):
            copy.equation_repr = f'{self.equation_repr} - ({other})'
        elif isinstance(other, float):
            copy.equation_repr = f'{self.equation_repr} - ({other:{self.double_format}})'
        elif isinstance(other, InfixEquation):
            copy.rpn_conv_post_repl = list(set(
                self.rpn_conv_post_repl + other.rpn_conv_post_repl))
            copy.equation_repr = f'{self.equation_repr} - ({other.equation_repr})'
        else:
            raise NotImplementedError(f'__sub__ for type "{type(other)}"')

        return copy

    #----------------------------------------------------------------------
    def __rsub__(self, left):
        """"""

        copy = self.copy()

        if isinstance(left, (int, np.integer, str)):
            copy.equation_repr = f'{left} - ({self.equation_repr})'
        elif isinstance(left, float):
            copy.equation_repr = f'{left:{self.double_format}} - ({self.equation_repr})'
        elif isinstance(left, InfixEquation):
            copy.rpn_conv_post_repl = list(set(
                self.rpn_conv_post_repl + left.rpn_conv_post_repl))
            copy.equation_repr = f'{left.equation_repr} - ({self.equation_repr})'
        else:
            raise NotImplementedError(f'__rsub__ for type "{type(left)}"')

        return copy

    #----------------------------------------------------------------------
    def __mul__(self, other):
        """"""

        copy = self.copy()

        if isinstance(other, (int, np.integer, str)):
            copy.equation_repr = f'({self.equation_repr}) * ({other})'
        elif isinstance(other, float):
            copy.equation_repr = f'({self.equation_repr}) * ({other:{self.double_format}})'
        elif isinstance(other, InfixEquation):
            copy.rpn_conv_post_repl = list(set(
                self.rpn_conv_post_repl + other.rpn_conv_post_repl))
            copy.equation_repr = f'({self.equation_repr}) * ({other.equation_repr})'
        else:
            raise NotImplementedError(f'__mul__ for type "{type(other)}"')

        return copy

    #----------------------------------------------------------------------
    def __rmul__(self, left):
        """"""

        copy = self.copy()

        if isinstance(left, (int, np.integer, str)):
            copy.equation_repr = f'({left}) * ({self.equation_repr})'
        elif isinstance(left, float):
            copy.equation_repr = f'({left:{self.double_format}}) * ({self.equation_repr})'
        elif isinstance(left, InfixEquation):
            copy.rpn_conv_post_repl = list(set(
                self.rpn_conv_post_repl + left.rpn_conv_post_repl))
            copy.equation_repr = f'({left.equation_repr}) * ({self.equation_repr})'
        else:
            raise NotImplementedError(f'__rmul__ for type "{type(left)}"')

        return copy

    #----------------------------------------------------------------------
    def __truediv__(self, other):
        """"""

        copy = self.copy()

        if isinstance(other, (int, np.integer, str)):
            copy.equation_repr = f'({self.equation_repr}) / ({other})'
        elif isinstance(other, float):
            copy.equation_repr = f'({self.equation_repr}) / ({other:{self.double_format}})'
        elif isinstance(other, InfixEquation):
            copy.rpn_conv_post_repl = list(set(
                self.rpn_conv_post_repl + other.rpn_conv_post_repl))
            copy.equation_repr = f'({self.equation_repr}) / ({other.equation_repr})'
        else:
            raise NotImplementedError(f'__truediv__ for type "{type(other)}"')

        return copy

    #----------------------------------------------------------------------
    def __rtruediv__(self, left):
        """"""

        copy = self.copy()

        if isinstance(left, (int, np.integer, str)):
            copy.equation_repr = f'({left}) / ({self.equation_repr})'
        elif isinstance(left, float):
            copy.equation_repr = f'({left:{self.double_format}}) / ({self.equation_repr})'
        elif isinstance(left, InfixEquation):
            copy.rpn_conv_post_repl = list(set(
                self.rpn_conv_post_repl + left.rpn_conv_post_repl))
            copy.equation_repr = f'({left.equation_repr}) / ({self.equation_repr})'
        else:
            raise NotImplementedError(f'__rtruediv__ for type "{type(left)}"')

        return copy

    #----------------------------------------------------------------------
    def __floordiv__(self, other):
        """"""

        raise ArithmeticError('There is no floor division operand in RPN')

    #----------------------------------------------------------------------
    def __rfloordiv__(self, other):
        """"""

        raise ArithmeticError('There is no floor division operand in RPN')

    #----------------------------------------------------------------------
    def __pow__(self, exponent):
        """"""

        copy = self.copy()

        if isinstance(exponent, (int, np.integer, str)):
            copy.equation_repr = f'({self.equation_repr})**({exponent})'
        elif isinstance(exponent, float):
            copy.equation_repr = f'({self.equation_repr})**({exponent:{self.double_format}})'
        elif isinstance(exponent, InfixEquation):
            copy.rpn_conv_post_repl = list(set(
                self.rpn_conv_post_repl + exponent.rpn_conv_post_repl))
            copy.equation_repr = f'({self.equation_repr})**({exponent.equation_repr})'
        else:
            raise NotImplementedError(f'__pow__ for type "{type(exponent)}"')

        return copy

    #----------------------------------------------------------------------
    def __rpow__(self, base):
        """"""

        copy = self.copy()

        if isinstance(base, (int, np.integer, str)):
            copy.equation_repr = f'({base})**({self.equation_repr})'
        elif isinstance(base, float):
            copy.equation_repr = f'({base:{self.double_format}})**({self.equation_repr})'
        elif isinstance(base, InfixEquation):
            copy.rpn_conv_post_repl = list(set(
                self.rpn_conv_post_repl + base.rpn_conv_post_repl))
            copy.equation_repr = f'({base.equation_repr})**({self.equation_repr})'
        else:
            raise NotImplementedError(f'__rpow__ for type "{type(base)}"')

        return copy

AST_COMPATIBLE_REPL = [('/', '__SLASH__'), ('.', '__DOT__'), ('#', '__POUND__'),
                       ('$', '__DOLLAR__')]
RPN_CONV_POST_REPL = [(_temp, _orig) for _orig, _temp in AST_COMPATIBLE_REPL]

########################################################################
class RPNVariableDatabase():
    """
    See ".defns.rpn"
    """

    #----------------------------------------------------------------------
    def __init__(self):
        """Constructor"""

        self.dict = {}
        self.namespace = SimpleNamespace(**self.dict)

        self._var_names = BookmarkableList()
        self._ast_compatible_var_names = BookmarkableList()
        self._var_eqs = BookmarkableList()

        self._uncommitted_var_names = [] # This list will hold variable names
        # that have not been fully committed yet as part of a block. For example,
        # when you are building up an "OptimizationTerm" object, you may create
        # new variables sequentially, and want these new variables immediately
        # show up as an available variable in the next line within in the same
        # "&optimization_term" block. This list will help in doing this. Once
        # the fully-built-up "OptimizationTerm" object is added as an
        # "&optimization_term" block, then these variables will be integrated
        # into the database, and removed from this uncommited list.

        self._builtin_vars = [
            'pi', # PI = 3.14...
            'log_10', # ln(10)
            'HUGE', # largest number possible
            'on_div_by_zero', # == HUGE
            #
            # physical constants
            'c_cgs', 'c_mks', 'e_cgs', 'e_mks', 'me_cgs', 'me_mks',
            're_cgs', 're_mks', 'kb_cgs', 'kb_mks', 'mev',
            'hbar_mks', 'hbar_MeVs', 'mp_mks', 'mu_o', 'eps_o'
        ]

        _builtin_dict = {}
        for var_name in self._builtin_vars:
            eq_obj = InfixEquation(var_name)

            _builtin_dict[var_name] = eq_obj

        self._builtin_dict_dumps = pickle.dumps(_builtin_dict)

    #----------------------------------------------------------------------
    def _get_ast_compatible_var_name_and_eq_obj(
        self, var_name: str) -> Union[str, InfixEquation]:
        """"""

        ast_compatible_var_name = var_name

        for _orig, _temp in AST_COMPATIBLE_REPL:
            ast_compatible_var_name = ast_compatible_var_name.replace(
                _orig, _temp)

        eq_obj = InfixEquation(ast_compatible_var_name,
                               rpn_conv_post_repl=RPN_CONV_POST_REPL)

        return ast_compatible_var_name, eq_obj

    #----------------------------------------------------------------------
    def update_base(self, new_var_names: List) -> None:
        """"""

        names_conflict_w_builtins = [
            name for name in new_var_names if name in self._builtin_vars]
        if names_conflict_w_builtins:
            for var_name in names_conflict_w_builtins:
                print(f'* WARNING: RPN variable: name conflict with built-in '
                      f'variable name  "{var_name}".')

        ast_compatible_var_name_list = []
        eq_obj_list = []
        for var_name in new_var_names:

            ast_compatible_var_name, eq_obj = \
                self._get_ast_compatible_var_name_and_eq_obj(var_name)

            ast_compatible_var_name_list.append(ast_compatible_var_name)

            eq_obj_list.append(eq_obj)

        self._var_names.insert(new_var_names)
        self._ast_compatible_var_names.insert(ast_compatible_var_name_list)
        self._var_eqs.insert(eq_obj_list)

        for var_name in new_var_names:
            if var_name in self._uncommitted_var_names:
                self._uncommitted_var_names.remove(var_name)

    #----------------------------------------------------------------------
    def update_accessible(self):
        """"""

        _builtin_dict = pickle.loads(self._builtin_dict_dumps)

        self.dict.clear()
        self.dict.update(_builtin_dict)

        self.namespace.__dict__.clear()
        self.namespace.__dict__.update(_builtin_dict)

        var_name_LoL = self._var_names.get_truncated_list()
        ast_compatible_var_name_LoL = self._ast_compatible_var_names.get_truncated_list()
        eq_obj_LoL = self._var_eqs.get_truncated_list()

        assert len(var_name_LoL) == len(ast_compatible_var_name_LoL) == len(eq_obj_LoL)

        for var_name_list, ast_compatible_var_name_list, eq_obj_list in zip(
            var_name_LoL, ast_compatible_var_name_LoL, eq_obj_LoL):

            assert len(var_name_list) == len(ast_compatible_var_name_list) \
                   == len(eq_obj_list)

            for var_name, ast_compatible_var_name, eq_obj in zip(
                var_name_list, ast_compatible_var_name_list, eq_obj_list):

                self.dict[var_name] = eq_obj
                self.namespace.__dict__[ast_compatible_var_name] = eq_obj

        for var_name in self._uncommitted_var_names:

            ast_compatible_var_name, eq_obj = \
                self._get_ast_compatible_var_name_and_eq_obj(var_name)

            self.dict[var_name] = eq_obj
            self.namespace.__dict__[ast_compatible_var_name] = eq_obj

    #----------------------------------------------------------------------
    def add_uncommitted_var_name(self, new_var_name):
        """"""

        if new_var_name not in self._uncommitted_var_names:
            self._uncommitted_var_names.append(new_var_name)
            self.update_accessible()

    #----------------------------------------------------------------------
    def get_dict(self):
        """"""

        return self.dict

    #----------------------------------------------------------------------
    def get_namespace(self):
        """"""

        return self.namespace

    #----------------------------------------------------------------------
    def clear(self):
        """"""

        self.dict.clear()
        self.namespace.__dict__.clear()

        self._var_names.clear()
        self._ast_compatible_var_names.clear()
        self._var_eqs.clear()

    #----------------------------------------------------------------------
    def set_bookmark(self, bookmark_key):
        """"""

        self._var_names.set_bookmark(bookmark_key)
        self._ast_compatible_var_names.set_bookmark(bookmark_key)
        self._var_eqs.set_bookmark(bookmark_key)

    #----------------------------------------------------------------------
    def delete_bookmark(self):
        """"""

        self._var_names.delete_bookmark()
        self._ast_compatible_var_names.delete_bookmark()
        self._var_eqs.delete_bookmark()

    #----------------------------------------------------------------------
    def seek_bookmark(self, bookmark_key):
        """"""

        self._var_names.seek_bookmark(bookmark_key)
        self._ast_compatible_var_names.seek_bookmark(bookmark_key)
        self._var_eqs.seek_bookmark(bookmark_key)

    #----------------------------------------------------------------------
    def pop_above(self):
        """"""

        self._var_names.pop_above()
        self._ast_compatible_var_names.pop_above()
        self._var_eqs.pop_above()

    #----------------------------------------------------------------------
    def pop_below(self):
        """"""

        self._var_names.pop_below()
        self._ast_compatible_var_names.pop_below()
        self._var_eqs.pop_below()


########################################################################
class RPNFunctionDatabase():
    """
    See ".defns.rpn"
    """

    #----------------------------------------------------------------------
    def __init__(self):
        """Constructor"""

    #----------------------------------------------------------------------
    def _ensure_InfixEquation_type(self, x):
        """"""

        if not isinstance(x, InfixEquation):
            return InfixEquation(f'{x}', rpn_conv_post_repl=RPN_CONV_POST_REPL)
        else:
            x.rpn_conv_post_repl = RPN_CONV_POST_REPL
            return x

    #----------------------------------------------------------------------
    def _simple_multi_args_func(self, func_name, *args):
        """"""

        eq_obj_list = [self._ensure_InfixEquation_type(x) for x in args]

        args_repr = ', '.join([eq.equation_repr for eq in eq_obj_list])

        return InfixEquation(f'{func_name}({args_repr})',
                             rpn_conv_post_repl=RPN_CONV_POST_REPL)

    #----------------------------------------------------------------------
    def ln(self, x):
        """ Natural Log := np.log(x) """
        return self._simple_multi_args_func('ln', x)
    #----------------------------------------------------------------------
    def exp(self, x):
        """ Exponential Function := exp(x) """
        return self._simple_multi_args_func('exp', x)
    #----------------------------------------------------------------------
    def pow(self, x, y):
        """ Power Function := x**y """
        return self._simple_multi_args_func('pow', x, y)
    #----------------------------------------------------------------------
    def ceil(self, x):
        """ Ceil := np.ceil(x) """
        return self._simple_multi_args_func('ceil', x)
    #----------------------------------------------------------------------
    def floor(self, x):
        """ Floor := np.floor(x) """
        return self._simple_multi_args_func('floor', x)
    #----------------------------------------------------------------------
    def int(self, x):
        """ Take integer part := np.floor(x) """
        return self._simple_multi_args_func('int', x)
    #----------------------------------------------------------------------
    def sin(self, x):
        """ Sine := np.sin(x) """
        return self._simple_multi_args_func('sin', x)
    #----------------------------------------------------------------------
    def cos(self, x):
        """ Cosine := np.cos(x) """
        return self._simple_multi_args_func('cos', x)
    #----------------------------------------------------------------------
    def tan(self, x):
        """ Tangent := np.tan(x) """
        return self._simple_multi_args_func('tan', x)
    #----------------------------------------------------------------------
    def asin(self, x):
        """ Arc Sine [rad] := np.arcsin(x) """
        return self._simple_multi_args_func('asin', x)
    #----------------------------------------------------------------------
    def acos(self, x):
        """ Arc Cosine [rad] := np.arccos(x) """
        return self._simple_multi_args_func('acos', x)
    #----------------------------------------------------------------------
    def atan(self, x):
        """ Arc Tangent [rad] := np.arctan(x) """
        return self._simple_multi_args_func('atan', x)
    #----------------------------------------------------------------------
    def atan2(self, x, y):
        """ Arc Tangent [rad] := np.arctan2(y, x)
        Note the order difference between this "atan2" and "np.arctan2".
        """
        return self._simple_multi_args_func('atan2', x, y)
    #----------------------------------------------------------------------
    def dsin(self, x):
        """ Sine := np.sin(np.deg2rad(x)) """
        return self._simple_multi_args_func('dsin', x)
    #----------------------------------------------------------------------
    def dcos(self, x):
        """ Cosine := np.cos(np.deg2rad(x)) """
        return self._simple_multi_args_func('dcos', x)
    #----------------------------------------------------------------------
    def dtan(self, x):
        """ Tangent := np.tan(np.deg2rad(x)) """
        return self._simple_multi_args_func('dtan', x)
    #----------------------------------------------------------------------
    def dasin(self, x):
        """ Arc Sine [deg] := np.rad2deg(np.arcsin(x)) """
        return self._simple_multi_args_func('dasin', x)
    #----------------------------------------------------------------------
    def dacos(self, x):
        """ Arc Cosine [deg] := np.rad2deg(np.arccos(x)) """
        return self._simple_multi_args_func('dacos', x)
    #----------------------------------------------------------------------
    def datan(self, x):
        """ Arc Tangent [deg] := np.rad2deg(np.arctan(x)) """
        return self._simple_multi_args_func('datan', x)
    #----------------------------------------------------------------------
    def sinh(self, x):
        """ Hyperbolic Sine := np.sinh(x) """
        return self._simple_multi_args_func('sinh', x)
    #----------------------------------------------------------------------
    def cosh(self, x):
        """ Hyperbolic Cosine := np.cosh(x) """
        return self._simple_multi_args_func('cosh', x)
    #----------------------------------------------------------------------
    def tanh(self, x):
        """ Hyperbolic Tangent := np.tanh(x) """
        return self._simple_multi_args_func('tanh', x)
    #----------------------------------------------------------------------
    def asinh(self, x):
        """ Inverse Hyperbolic Sine := np.arcsinh(x) """
        return self._simple_multi_args_func('asinh', x)
    #----------------------------------------------------------------------
    def acosh(self, x):
        """ Inverse Hyperbolic Cosine := np.arccosh(x) """
        return self._simple_multi_args_func('acosh', x)
    #----------------------------------------------------------------------
    def atanh(self, x):
        """ Inverse Hyperbolic Tangent := np.arctanh(x) """
        return self._simple_multi_args_func('atanh', x)
    #----------------------------------------------------------------------
    def sqr(self, x):
        """ Square := x**2 """
        return self._simple_multi_args_func('sqr', x)
    #----------------------------------------------------------------------
    def sqrt(self, x):
        """ Square Root := np.sqrt(x) """
        return self._simple_multi_args_func('sqrt', x)
    #----------------------------------------------------------------------
    def abs(self, x):
        """ Absolute := np.abs(x) """
        return self._simple_multi_args_func('abs', x)
    #----------------------------------------------------------------------
    def segt(self, v1, v2, tol):
        """ Soft-edge "greater-than" :=
            if v1 < v2: 0
            else      : ((v1 - v2) / tol)**2
        """
        return self._simple_multi_args_func('segt', v1, v2, tol)
    #----------------------------------------------------------------------
    def selt(self, v1, v2, tol):
        """ Soft-edge "less-than" :=
            if v1 > v2: 0
            else      : ((v1 - v2) / tol)**2
        """
        return self._simple_multi_args_func('selt', v1, v2, tol)
    #----------------------------------------------------------------------
    def sene(self, v1, v2, tol):
        """ Soft-edge "not-equal-to" :=
            if np.abs(v1 - v2) < tol: 0
            else:
                if v1 > v2: ((v1 - (v2 + tol)) / tol)**2
                else      : ((v2 - (v1 + tol)) / tol)**2
        """
        return self._simple_multi_args_func('sene', v1, v2, tol)
    #----------------------------------------------------------------------
    def chs(self, x):
        """ Change sign := x * (-1) """
        return self._simple_multi_args_func('chs', x)
    #----------------------------------------------------------------------
    def rec(self, x):
        """ Take reciprocal := 1 / x """
        return self._simple_multi_args_func('rec', x)
    #----------------------------------------------------------------------
    def rtod(self, x):
        """ Convert radians to degrees := np.rad2deg(x) """
        return self._simple_multi_args_func('rtod', x)
    #----------------------------------------------------------------------
    def dtor(self, x):
        """ Convert degrees to radians := np.deg2rad(x) """
        return self._simple_multi_args_func('dtor', x)
    #----------------------------------------------------------------------
    def hypot(self, x, y):
        """ hypot function := np.sqrt(x**2 + y**2) """
        return self._simple_multi_args_func('hypot', x, y)
    #----------------------------------------------------------------------
    def max2(self, x, y):
        """ Maximum of top 2 items on stack := np.max([x, y]) """
        return self._simple_multi_args_func('max2', x, y)
    #----------------------------------------------------------------------
    def min2(self, x, y):
        """ Minimum of top 2 items on stack := np.min([x, y]) """
        return self._simple_multi_args_func('min2', x, y)
    #----------------------------------------------------------------------
    def maxn(self, *args):
        """ Maximum of top N items on stack := np.max([x0, x1, ...]) """
        n = len(args)
        mod_args = list(args) + [n]
        return self._simple_multi_args_func('maxn', *mod_args)
    #----------------------------------------------------------------------
    def minn(self, *args):
        """ Minimum of top N items on stack := np.min([x0, x1, ...]) """
        n = len(args)
        mod_args = list(args) + [n]
        return self._simple_multi_args_func('minn', *mod_args)

########################################################################
class RPNCalculator():
    """"""

    #----------------------------------------------------------------------
    def __init__(self):
        """Constructor"""

        self.buffer = []

        self.operator_list = ['+', '-', '*', '/', '**']

        self.func_list = [
            'ln', 'exp', 'pow', 'ceil', 'floor', 'int', 'sin', 'cos', 'tan',
            'asin', 'acos', 'atan', 'atan2', 'dsin', 'dcos', 'dtan',
            'dasin', 'dacos', 'datan', 'sinh', 'cosh', 'tanh',
            'asinh', 'acosh', 'atanh', 'sqr', 'sqrt', 'abs',
            'segt', 'selt', 'sene', 'chs', 'rec', 'rtod', 'dtor', 'hypot',
            'max2', 'min2', 'maxn', 'minn']

    #----------------------------------------------------------------------
    def clear_buffer(self):
        """"""

        self.buffer.clear()

    #----------------------------------------------------------------------
    def ln(self):
        """ Natural Log := np.log(x) """
        x = self.buffer.pop()
        return np.log(x)
    #----------------------------------------------------------------------
    def exp(self):
        """ Exponential Function := exp(x) """
        x = self.buffer.pop()
        return np.exp(x)
    #----------------------------------------------------------------------
    def pow(self):
        """ Power Function := x**y """
        y = self.buffer.pop()
        x = self.buffer.pop()
        return x**y
    #----------------------------------------------------------------------
    def ceil(self):
        """ Ceil := np.ceil(x) """
        x = self.buffer.pop()
        return np.ceil(x)
    #----------------------------------------------------------------------
    def floor(self):
        """ Floor := np.floor(x) """
        x = self.buffer.pop()
        return np.floor(x)
    #----------------------------------------------------------------------
    def int(self):
        """ Take integer part := np.floor(x) """
        x = self.buffer.pop()
        return np.floor(x)
    #----------------------------------------------------------------------
    def sin(self):
        """ Sine := np.sin(x) """
        x = self.buffer.pop()
        return np.sin(x)
    #----------------------------------------------------------------------
    def cos(self):
        """ Cosine := np.cos(x) """
        x = self.buffer.pop()
        return np.cos(x)
    #----------------------------------------------------------------------
    def tan(self):
        """ Tangent := np.tan(x) """
        x = self.buffer.pop()
        return np.tan(x)
    #----------------------------------------------------------------------
    def asin(self):
        """ Arc Sine [rad] := np.arcsin(x) """
        x = self.buffer.pop()
        return np.arcsin(x)
    #----------------------------------------------------------------------
    def acos(self):
        """ Arc Cosine [rad] := np.arccos(x) """
        x = self.buffer.pop()
        return np.arccos(x)
    #----------------------------------------------------------------------
    def atan(self):
        """ Arc Tangent [rad] := np.arctan(x) """
        x = self.buffer.pop()
        return np.arctan(x)
    #----------------------------------------------------------------------
    def atan2(self):
        """ Arc Tangent [rad] := np.arctan2(y, x)
        Note the order difference between this "atan2" and "np.arctan2".
        """
        y = self.buffer.pop()
        x = self.buffer.pop()
        return np.arctan2(y, x)
    #----------------------------------------------------------------------
    def dsin(self):
        """ Sine := np.sin(np.deg2rad(x)) """
        x = self.buffer.pop()
        return np.sin(np.deg2rad(x))
    #----------------------------------------------------------------------
    def dcos(self):
        """ Cosine := np.cos(np.deg2rad(x)) """
        x = self.buffer.pop()
        return np.cos(np.deg2rad(x))
    #----------------------------------------------------------------------
    def dtan(self):
        """ Tangent := np.tan(np.deg2rad(x)) """
        x = self.buffer.pop()
        return np.tan(np.deg2rad(x))
    #----------------------------------------------------------------------
    def dasin(self):
        """ Arc Sine [deg] := np.rad2deg(np.arcsin(x)) """
        x = self.buffer.pop()
        return np.rad2deg(np.arcsin(x))
    #----------------------------------------------------------------------
    def dacos(self):
        """ Arc Cosine [deg] := np.rad2deg(np.arccos(x)) """
        x = self.buffer.pop()
        return np.rad2deg(np.arccos(x))
    #----------------------------------------------------------------------
    def datan(self):
        """ Arc Tangent [deg] := np.rad2deg(np.arctan(x)) """
        x = self.buffer.pop()
        return np.rad2deg(np.arctan(x))
    #----------------------------------------------------------------------
    def sinh(self):
        """ Hyperbolic Sine := np.sinh(x) """
        x = self.buffer.pop()
        return np.sinh(x)
    #----------------------------------------------------------------------
    def cosh(self):
        """ Hyperbolic Cosine := np.cosh(x) """
        x = self.buffer.pop()
        return np.cosh(x)
    #----------------------------------------------------------------------
    def tanh(self):
        """ Hyperbolic Tangent := np.tanh(x) """
        x = self.buffer.pop()
        return np.tanh(x)
    #----------------------------------------------------------------------
    def asinh(self):
        """ Inverse Hyperbolic Sine := np.arcsinh(x) """
        x = self.buffer.pop()
        return np.arcsinh(x)
    #----------------------------------------------------------------------
    def acosh(self):
        """ Inverse Hyperbolic Cosine := np.arccosh(x) """
        x = self.buffer.pop()
        return np.arccosh(x)
    #----------------------------------------------------------------------
    def atanh(self):
        """ Inverse Hyperbolic Tangent := np.arctanh(x) """
        x = self.buffer.pop()
        return np.arctanh(x)
    #----------------------------------------------------------------------
    def sqr(self):
        """ Square := x**2 """
        x = self.buffer.pop()
        return x**2
    #----------------------------------------------------------------------
    def sqrt(self):
        """ Square Root := np.sqrt(x) """
        x = self.buffer.pop()
        return np.sqrt(x)
    #----------------------------------------------------------------------
    def abs(self):
        """ Absolute := np.abs(x) """
        x = self.buffer.pop()
        return np.abs(x)
    #----------------------------------------------------------------------
    def segt(self):
        """ Soft-edge "greater-than" :=
            if v1 < v2: 0
            else      : ((v1 - v2) / tol)**2
        """
        tol = self.buffer.pop()
        v2 = self.buffer.pop()
        v1 = self.buffer.pop()
        if v1 < v2:
            return 0.0
        else:
            return ((v1 - v2) / tol)**2
    #----------------------------------------------------------------------
    def selt(self):
        """ Soft-edge "less-than" :=
            if v1 > v2: 0
            else      : ((v1 - v2) / tol)**2
        """
        tol = self.buffer.pop()
        v2 = self.buffer.pop()
        v1 = self.buffer.pop()
        if v1 > v2:
            return 0.0
        else:
            return ((v1 - v2) / tol)**2
    #----------------------------------------------------------------------
    def sene(self):
        """ Soft-edge "not-equal-to" :=
            if np.abs(v1 - v2) < tol: 0
            else:
                if v1 > v2: ((v1 - (v2 + tol)) / tol)**2
                else      : ((v2 - (v1 + tol)) / tol)**2
        """
        tol = self.buffer.pop()
        v2 = self.buffer.pop()
        v1 = self.buffer.pop()
        if v1 > v2:
            return ((v1 - (v2 + tol)) / tol)**2
        else:
            return ((v2 - (v1 + tol)) / tol)**2
    #----------------------------------------------------------------------
    def chs(self):
        """ Change sign := x * (-1) """
        x = self.buffer.pop()
        return x * (-1.0)
    #----------------------------------------------------------------------
    def rec(self):
        """ Take reciprocal := 1 / x """
        x = self.buffer.pop()
        return 1.0 / x
    #----------------------------------------------------------------------
    def rtod(self):
        """ Convert radians to degrees := np.rad2deg(x) """
        x = self.buffer.pop()
        return np.rad2deg(x)
    #----------------------------------------------------------------------
    def dtor(self):
        """ Convert degrees to radians := np.deg2rad(x) """
        x = self.buffer.pop()
        return np.deg2rad(x)
    #----------------------------------------------------------------------
    def hypot(self):
        """ hypot function := np.sqrt(x**2 + y**2) """
        y = self.buffer.pop()
        x = self.buffer.pop()
        return np.sqrt(x**2 + y**2)
    #----------------------------------------------------------------------
    def max2(self):
        """ Maximum of top 2 items on stack := np.max([x, y]) """
        y = self.buffer.pop()
        x = self.buffer.pop()
        return np.max([x, y])
    #----------------------------------------------------------------------
    def min2(self):
        """ Minimum of top 2 items on stack := np.min([x, y]) """
        y = self.buffer.pop()
        x = self.buffer.pop()
        return np.min([x, y])
    #----------------------------------------------------------------------
    def maxn(self):
        """ Maximum of top N items on stack := np.max([x0, x1, ...]) """
        n = int(self.buffer.pop())
        list_to_compare = []
        for _ in range(n):
            list_to_compare.append(self.buffer.pop())
        return np.max(list_to_compare)
    #----------------------------------------------------------------------
    def minn(self):
        """ Minimum of top N items on stack := np.min([x0, x1, ...]) """
        n = int(self.buffer.pop())
        list_to_compare = []
        for _ in range(n):
            list_to_compare.append(self.buffer.pop())
        return np.min(list_to_compare)

    #----------------------------------------------------------------------
    def _operate(self, op_name):
        """"""

        v2 = self.buffer.pop()
        v1 = self.buffer.pop()

        if op_name == '+':
            return v1 + v2
        elif op_name == '-':
            return v1 - v2
        elif op_name == '*':
            return v1 * v2
        elif op_name == '/':
            return v1 / v2
        elif op_name == '**':
            return v1 ** v2
        else:
            raise NotImplementedError(f'Unknown operator type: {op_name}')

    #----------------------------------------------------------------------
    def get_buffer(self, rpn_expr_str, num_dict=None):
        """"""

        if num_dict is None:
            num_dict = {}

        token_list = rpn_expr_str.split()

        for token in token_list:

            if token in self.func_list:
                self.buffer.append(getattr(self, token)())
            elif token in self.operator_list:
                self.buffer.append(self._operate(token))
            elif token in num_dict:
                self.buffer.append(num_dict[token])
            else:
                self.buffer.append(float(token))

        return self.buffer[:]

########################################################################
class BookmarkableObject:
    """"""

    #----------------------------------------------------------------------
    def __init__(self, obj):
        """Constructor"""

        self.obj = obj

        self.bookmark = None

    #----------------------------------------------------------------------
    def get_object(self):
        """"""

        return self.obj

    #----------------------------------------------------------------------
    def get_bookmark(self):
        """"""

        return self.bookmark

    #----------------------------------------------------------------------
    def set_bookmark(self, bookmark_key):
        """"""

        self.bookmark = bookmark_key

########################################################################
class BookmarkableList:
    """"""

    #----------------------------------------------------------------------
    def __init__(self):
        """Constructor"""

        bottom_obj = BookmarkableObject(None)
        bottom_obj.set_bookmark('bottom')

        self._list = [bottom_obj]
        self._bookmarks = ['bottom']

        self._insert_index = 0
        self._item_index = 0

    #----------------------------------------------------------------------
    def clear(self):
        """"""

        # Keep just "bottom" bookmarkable object
        self._list = self._list[-1:]
        self._bookmarks = self._bookmarks[-1:]

        # Reset indexes to point to "bottom"
        self._insert_index = 0
        self._item_index = 0

    #----------------------------------------------------------------------
    def __len__(self):
        """"""

        # Exclude the special "bottom" BookmarkableObject
        return len(self._list) - 1

    #----------------------------------------------------------------------
    def __iter__(self):
        """"""

        return iter(self._list[:-1])
        # ^ Last element is excluded, as it is the "bottom"
        #   BookmarkableObject, which contains no meaningful data.

    #----------------------------------------------------------------------
    def _assert_no_duplicate_bookmarks(self, bookmark_key):
        """"""

        if bookmark_key:

            if bookmark_key in ('top', 'bottom'):
                raise ValueError(f'Bookmark key "{bookmark_key}" is reserved, and cannot be used.')

            existing_bookmarks = set(self._bookmarks)
            if None in existing_bookmarks:
                existing_bookmarks.remove(None)

            if bookmark_key in existing_bookmarks:
                raise ValueError(f'Bookmark key "{bookmark_key}" already exists, and cannot be used.')

    #----------------------------------------------------------------------
    def _at_top(self):
        """"""

        if self._item_index is None:
            return True
        else:
            return False

    #----------------------------------------------------------------------
    def _at_bottom(self):
        """"""

        if self._item_index == self._insert_index:
            return True
        else:
            return False

    #----------------------------------------------------------------------
    def get_bookmark(self):
        """"""

        if self._at_top():
            return 'top'
        elif self._at_bottom():
            return 'bottom'
        else:
            return self._bookmarks[self._item_index]

    #----------------------------------------------------------------------
    def set_bookmark(self, bookmark_key):
        """"""

        self._assert_no_duplicate_bookmarks(bookmark_key)

        if self._at_top():
            if self._bookmarks[0] == 'bottom':
                raise RuntimeError('No item found to bookmark')
            self._list[0].set_bookmark(bookmark_key)
            self._bookmarks[0] = bookmark_key
        elif self._at_bottom():
            if self._item_index == 0:
                raise RuntimeError('No item found to bookmark')
            self._list[self._item_index - 1].set_bookmark(bookmark_key)
            self._bookmarks[self._item_index - 1] = bookmark_key
        else:
            self._list[self._item_index].set_bookmark(bookmark_key)
            self._bookmarks[self._item_index] = bookmark_key

    #----------------------------------------------------------------------
    def delete_bookmark(self):
        """"""

        if self._at_top():
            raise RuntimeError('You are at "top" bookmark, which cannot be deleted')
        elif self._at_bottom():
            raise RuntimeError('You are at "bottom" bookmark, which cannot be deleted')
        else:
            self._list[self._item_index].set_bookmark(None)
            self._bookmarks[self._item_index] = None

    #----------------------------------------------------------------------
    def seek_bookmark(self, bookmark_key):
        """"""

        if bookmark_key == 'top':
            self._item_index = None
            self._insert_index = 0

        elif bookmark_key == 'bottom':

            self._item_index = self._insert_index = len(self._list) - 1

        else:
            if bookmark_key not in self._bookmarks:
                raise ValueError(f'A bookmark with the name "{bookmark_key}" does NOT exist.')

            self._item_index = self._bookmarks.index(bookmark_key)
            self._insert_index = self._item_index + 1

    #----------------------------------------------------------------------
    def insert(self, obj):
        """"""

        if not isinstance(obj, BookmarkableObject):
            obj = BookmarkableObject(obj)

        bookmark_key = obj.get_bookmark()
        self._assert_no_duplicate_bookmarks(bookmark_key)

        self._list.insert(self._insert_index, obj)
        self._bookmarks.insert(self._insert_index, bookmark_key)

        if self._at_top():
            self._item_index = 0
        else:
            self._item_index += 1

        self._insert_index += 1

    #----------------------------------------------------------------------
    def pop_above(self):
        """"""

        if self._at_top():
            raise RuntimeError('At the top. No item above to pop')

        elif self._at_bottom():
            if self._item_index == 0:
                raise RuntimeError('No item found to pop')
            self._bookmarks.pop(index=self._item_index)
            obj = self._list.pop(index=self._item_index)

            # Reset indexes to "bottom" again
            self._item_index = self._insert_index = len(self._list) - 1

        else:
            self._bookmarks.pop(index=self._item_index)
            obj = self._list.pop(index=self._item_index)

            # Update indexes to reflect the removal
            self._item_index -= 1
            self._insert_index -= 1

        return obj

    #----------------------------------------------------------------------
    def pop_below(self):
        """"""

        if self._at_top():
            if self._bookmarks[0] == 'bottom':
                raise RuntimeError('No item found to pop')
            self._bookmarks.pop(index=0)
            obj = self._list.pop(index=0)

            # No need to update indexes in this case

        elif self._at_bottom():
            raise RuntimeError('At the bottom. No item below to pop')

        else:
            if self._bookmarks[self._item_index + 1] == 'botoom':
                raise RuntimeError('No item below to pop')

            self._bookmarks.pop(index=self._item_index + 1)
            obj = self._list.pop(index=self._item_index + 1)

            # No need to update indexes in this case

        return obj

    #----------------------------------------------------------------------
    def get_truncated_list(self):
        """"""

        return [b_obj.get_object() for b_obj in self._list[:(self._item_index+1)]
                if b_obj.get_bookmark() != 'bottom']

    ##----------------------------------------------------------------------
    #def index(self, value):
        #"""
        #First search for a match in the bookmark keys. If not found,
        #then search for a match in the objects in the list.
        #"""

        #if value in self._bookmarks:
            #return self._bookmarks.index(value)
        #elif value in self._list:
            #return self._list.index(value)
        #else:
            #raise ValueError(
                #('Specified value exists neither in bookmark keys or '
                 #'in the list of objects.'))

    ##----------------------------------------------------------------------
    #def remove(self, value):
        #"""
        #First search for a match in the bookmark keys. If not found,
        #then search for a match in the objects in the list.
        #"""

        #index = self.index(value)

        #self._bookmarks.pop(index=index)
        #self._list.pop(index=index)

        #self._cur_pos = self._get_positive_index(index) - 1

########################################################################
class EleDesigner():
    """"""

    #----------------------------------------------------------------------
    def __init__(
        self, ele_filepath: str = '', ele_folderpath: str = '',
        ele_prefix: str = 'tmp', double_format: str ='.12g',
        auto_print_on_add=True, adj_optim_var_limits_to_init=False):
        """Constructor"""

        self._adj_optim_var_limits_to_init = adj_optim_var_limits_to_init

        if ele_filepath:
            self.ele_filepath = str(Path(ele_filepath).absolute())
            if ele_folderpath:
                print('Since "ele_filepath" is specified, "ele_folderpath" arg will be ignored.')
        else:
            if not ele_folderpath:
                ele_folderpath = Path.cwd()
                # ^ Note that the default "dir" option for tempfile.NamedTemporaryFile()
                #   below should be cwd, NOT None, which results in /tmp.
                #   This is because the temporary ELE file generated by EleDesigner
                #   may well need to be accessible by a worker machine in a cluster.
                #   If the temporary ELE file is generated in /tmp, which is actually
                #   in the memory of the local machine, the file will NOT be readable
                #   by a worker machine.

            tmp = tempfile.NamedTemporaryFile(
                dir=ele_folderpath, delete=True, prefix=ele_prefix, suffix='.ele')
            self.ele_filepath = tmp.name
            tmp.close()

        self.blocks = EleBlocks()

        self.rpnfuncs = RPNFunctionDatabase()

        self.rpnvars = {}
        self.rpnvars['optimization_term'] = RPNVariableDatabase()
        # Variables that will be available within the definition of
        #   "term" in "&optimization_term"

        self.rpnvars['optimization_covariable'] = RPNVariableDatabase()
        # Variables that will be available within the definition of
        #   "equation" in "&optimization_covariable"

        self._text_blocks = BookmarkableList()

        self._text = ''
        self._last_block_text = ''

        self.rootname = None
        self.output_filepath_list = []
        self.actual_output_filepath_list = []

        self.clear()

        self.double_format = double_format
        self.auto_print_on_add = auto_print_on_add

    #----------------------------------------------------------------------
    def clear(self):
        """"""

        for k, v in self.rpnvars.items():
            v.clear()

        self._text_blocks.clear()

        self._text = ''
        self._last_block_text = ''

        self.rootname = None
        self.output_filepath_list.clear()
        self.actual_output_filepath_list.clear()

    #----------------------------------------------------------------------
    def get_rpn_vars(self, block_header):
        """"""

        if block_header not in self.rpnvars:
            raise KeyError('Invalid block header')

        return self.rpnvars[block_header]

    #----------------------------------------------------------------------
    def _update_text(self):
        """"""

        self._text = ''.join([b_obj.get_object() for b_obj in self._text_blocks])

    #----------------------------------------------------------------------
    def print_whole(self):
        """"""

        self._update_text()
        print(self._text)

    #----------------------------------------------------------------------
    def print_last_block(self):
        """"""

        print(self._last_block_text[:-1] if self._last_block_text.endswith('\n')
              else self._last_block_text)

    #----------------------------------------------------------------------
    def write(self, nMaxTry=10, sleep=10.0):
        """"""

        self.update_output_filepaths()

        self._update_text()

        util.robust_text_file_write(
            self.ele_filepath, self._text, nMaxTry=nMaxTry, sleep=sleep)

    #----------------------------------------------------------------------
    def get_bookmark(self):
        """"""

        return self._text_blocks.get_bookmark()

    #----------------------------------------------------------------------
    def set_bookmark(self, bookmark_key):
        """"""

        self._text_blocks.set_bookmark(bookmark_key)
        for v in self.rpnvars.values():
            v.set_bookmark(bookmark_key)

    #----------------------------------------------------------------------
    def delete_bookmark(self):
        """"""

        self._text_blocks.delete_bookmark()
        for v in self.rpnvars.values():
            v.delete_bookmark()

    #----------------------------------------------------------------------
    def seek_bookmark(self, bookmark_key):
        """"""

        self._text_blocks.seek_bookmark(bookmark_key)
        for v in self.rpnvars.values():
            v.seek_bookmark(bookmark_key)

        self._update_accessible_rpnvars()

    #----------------------------------------------------------------------
    def delete_ele_file(self):
        """"""

        try:
            Path(self.ele_filepath).unlink()
        except:
            print(f'Failed to delete "{self.ele_filepath}"')

    #----------------------------------------------------------------------
    def delete_temp_files(self):
        """"""

        for fp in self.actual_output_filepath_list:

            if fp.startswith('/dev'):
                continue

            if fp.endswith('.simlog'):
                for sub_fp in glob.glob(f'{fp}-*'):
                    try:
                        Path(sub_fp).unlink()
                    except:
                        print(f'Failed to delete "{sub_fp}"')

            else:
                try:
                    Path(fp).unlink()
                except:
                    print(f'Failed to delete "{fp}"')

    #----------------------------------------------------------------------
    def load_sdds_output_files(self):
        """"""

        output, meta = {}, {}
        for sdds_fp in self.actual_output_filepath_list:
            if sdds_fp.startswith('/dev/'):
                continue
            print(f'Processing "{sdds_fp}"...')
            ext = sdds_fp.split('.')[-1]
            try:
                output[ext], meta[ext] = sdds.sdds2dicts(sdds_fp)
            except:
                continue

        return dict(data=output, meta=meta)

    #----------------------------------------------------------------------
    def add_newline(self):
        """"""

        self._last_block_text = '\n'

        self._text_blocks.insert(self._last_block_text)
        for v in self.rpnvars.values():
            v.update_base([])

        if std_print_enabled['out'] and self.auto_print_on_add:
            self.print_last_block()

    #----------------------------------------------------------------------
    def add_comment(self, comment):
        """"""

        self._last_block_text = f'! {comment}\n'

        self._text_blocks.insert(self._last_block_text)
        for v in self.rpnvars.values():
            v.update_base([])

        if std_print_enabled['out'] and self.auto_print_on_add:
            self.print_last_block()

    #----------------------------------------------------------------------
    def _should_be_inline_block(self, block_body_line_list):
        """"""

        single_line = False

        if len(block_body_line_list) == 0:
            single_line = True
        elif len(block_body_line_list) == 1:
            if len(block_body_line_list[0]) <= 80:
                single_line = True

        return single_line

    #----------------------------------------------------------------------
    def _get_block_str(self, block_header, **kwargs) -> str:
        """"""

        if block_header not in self.blocks.info:

            print('* Valid block names are the following:')
            print('\n'.join(self.blocks.get_avail_blocks()))

            raise ValueError(f'Unexpected block name "{block_header}".')

        keywords, dtypes, default_vals, recommended, array_sizes = zip(
            *self.blocks.info[block_header])

        block = []
        for k, v in kwargs.items():

            if k not in keywords:
                print(f'* Valid keys for Block "{block_header}" are the following:')
                print('\n'.join(sorted(keywords)))

                raise ValueError(f'Unexpected key "{k}" for Block "{block_header}"')

            if k in UNAVAILABLE_BLOCK_OPTS[block_header]:
                print(
                    f'* Option "{k}" for Block "{block_header}" is set to be '
                    f'unavailable in the current setup. So, this option will '
                    f'NOT be added to the block.')
                continue

            i = keywords.index(k)

            is_scalar = (array_sizes[i] == 0)

            if dtypes[i] == 'STRING':
                if v is None:
                    continue
                elif (
                    (block_header == 'optimization_covariable') and (k == 'equation')
                    ) or (
                    (block_header == 'optimization_term') and (k == 'term')
                    ):
                    if isinstance(v, InfixEquation):
                        rpn_str = v.torpn()
                    else: # Either "str" or "OptimizationTermBlockEquation" object
                        rpn_str = v
                    block.append(f'{k} = "{rpn_str}"')
                else:
                    block.append(f'{k} = "{v}"')
                if (block_header in self.blocks.output_filepaths) and \
                   (k in self.blocks.output_filepaths[block_header]):
                    self.output_filepath_list.append(v)
                if (block_header == 'run_setup') and (k == 'rootname'):
                    self.rootname = v

                if (block_header == 'optimization_term') and (k == 'term'):
                    if len(block[-1]) > self.blocks.OPTIM_TERM_TERM_FIELD_MAX_STR_LEN:
                        raise ValueError(
                            ('Ther number of characters for "term" field in "optimization_term" block '
                             f'cannot exceed {self.blocks.OPTIM_TERM_TERM_FIELD_MAX_STR_LEN:d}'))

            elif dtypes[i] == 'long':
                block.append(f'{k} = {v:d}')
            elif dtypes[i] == 'double':
                if is_scalar:
                    try:
                        block.append(('{k} = {v:%s}' % self.double_format).format(k=k, v=v))
                    except ValueError:
                        if v.startswith('<') and v.endswith('>'): # macro definition
                            block.append(f'{k} = {v}')
                        else:
                            raise
                else:
                    max_array_size = array_sizes[i]
                    for array_index, scalar_val in v.items():
                        assert 0 <= array_index < max_array_size
                        try:
                            block.append(
                                ('{k}[{array_index:d}] = {v:%s}'
                                 % self.double_format).format(
                                     k=k, v=v[array_index], array_index=array_index))
                        except ValueError:
                            v_str = v[array_index]
                            if v_str.startswith('<') and v_str.endswith('>'): # macro definition
                                block.append(f'{k}[{array_index:d}] = {v_str}')
                            else:
                                raise


            else:
                raise ValueError('Unexpected data type: {}'.format(dtypes[i]))

        # Check whether initial value is within specified limits
        if (block_header == 'optimization_variable') and \
           (not kwargs.get('differential_limits', False)):
            lower_limit = kwargs.get('lower_limit', 0.0)
            upper_limit = kwargs.get('upper_limit', 0.0)
            if lower_limit == upper_limit:
                pass # Parameter range is unlimited, so any initial value would be fine.
            else:
                name, item = kwargs.get('name'), kwargs.get('item')
                init_val = self.get_LTE_elem_prop(name, item)
                if init_val:
                    if init_val < lower_limit:
                        if self._adj_optim_var_limits_to_init:
                            _i = [_i for _i, _s in enumerate(block)
                                  if _s.startswith('lower_limit')][0]
                            block[_i] = f'lower_limit={init_val:.12g}'
                        else:
                            raise ValueError(
                                (f'Initial value ({init_val:{self.double_format}}) cannot be '
                                 f'smaller than "lower_limit" ({lower_limit:{self.double_format}})'))
                    elif init_val > upper_limit:
                        if self._adj_optim_var_limits_to_init:
                            _i = [_i for _i, _s in enumerate(block)
                                  if _s.startswith('upper_limit')][0]
                            block[_i] = f'upper_limit={init_val:.12g}'
                        else:
                            raise ValueError(
                                (f'Initial value ({init_val:{self.double_format}}) cannot be '
                                 f'larger than "upper_limit" ({upper_limit:{self.double_format}})'))
                else:
                    print(f'{name}.{item} is not defined in LTE. So, initial value check is skipped.')

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

    #----------------------------------------------------------------------
    def _update_accessible_rpnvars(self) -> None:
        """"""

        for v in self.rpnvars.values():
            v.update_accessible()

    #----------------------------------------------------------------------
    def _update_base_rpnvars(self, block_header: str, **kwargs) -> None:
        """"""

        new_var_names = {}
        for k in list(self.rpnvars):
            new_var_names[k] = []

        if block_header == 'floor_coordinates':
            for fitpoint_name in self._fitpoint_names:
                for quantity in ['X', 'Y', 'Z', 'theta', 'phi', 'psi']:
                    # ^ These are, respectively, the three position coordinates,
                    #   the three angle coordinates, and the total arch length at
                    #   the marker location.
                    new_var_names['optimization_term'].append(f'{fitpoint_name}.{quantity}')

        elif block_header in ('optimization_setup', 'parallel_optimization_setup'):
            new_var_names['optimization_term'].append('Particles')

            # Add transport matrix elements for the terminal point of the beamline
            matrix_order = kwargs.get('matrix_order', 1)
            if matrix_order > 3:
                raise ValueError(f'"matrix_order" for `{block_header}` cannot be larger than 3')
            if matrix_order == 3:
                new_var_names['optimization_term'].extend(
                    [f'U{i}{j}{k}{l}' for i, j, k, l in itertools.product(
                        range(1,6+1), range(1,6+1), range(1,6+1), range(1,6+1))])
            if matrix_order >= 2:
                new_var_names['optimization_term'].extend(
                    [f'T{i}{j}{k}' for i, j, k in itertools.product(
                        range(1,6+1), range(1,6+1), range(1,6+1))])
            new_var_names['optimization_term'].extend(
                [f'R{i}{j}' for i, j in itertools.product(range(1,6+1), range(1,6+1))])

        elif block_header == 'optimization_term':
            if isinstance(kwargs['term'], str):
                terms = kwargs['term'].split()
            elif isinstance(kwargs['term'], InfixEquation):
                terms = kwargs['term'].torpn().split()
            elif isinstance(kwargs['term'], OptimizationTerm):
                terms = str(kwargs['term']).split()
            else:
                raise ValueError('Invalid data type')

            for i, t in enumerate(terms):
                if t == 'sto':
                    #print(f'** Adding new variable "{terms[i+1]}"')
                    new_var_names['optimization_term'].append(terms[i+1])

        elif block_header in ('optimization_variable', 'optimization_covariable'):
            name, item = kwargs['name'].upper(), kwargs['item'].upper()
            new_var_names['optimization_term'].extend(
                [f'{name}.{item}', f'{name}.{item}0'])

            if block_header == 'optimization_variable':
                new_var_names['optimization_covariable'].extend(
                    [f'{name}.{item}', f'{name}.{item}0'])

        elif block_header == 'rpn_load':
            tag = kwargs.get('tag', '')
            tag_dot = tag + '.' if tag != '' else ''
            [_, meta] = sdds.sdds2dicts(kwargs['filename'])
            for col_name, _d in meta['columns'].items():
                if _d['TYPE'] == 'double':
                    new_var_names['optimization_term'].append(f'{tag_dot}{col_name}')

        elif block_header == 'run_setup':

            used_beamline_name = kwargs.get('use_beamline', '')
            used_beamline_name = (
                used_beamline_name if used_beamline_name is not None else '')

            self._LTE = ltemanager.Lattice(
                LTE_filepath=kwargs.get('lattice'),
                used_beamline_name=used_beamline_name)
            self._all_elem_names = [name for name, _, _ in self._LTE.elem_defs]

            self._fitpoint_names = []
            for name, elem_type, prop_str in self._LTE.elem_defs:
                if elem_type.upper() == 'MARK':
                    fitpoints = [int(s) for s in re.findall(
                        'FITPOINT\s*=\s*(\d+)', prop_str, re.IGNORECASE)]
                    if len(fitpoints) == 0:
                        continue
                    elif len(fitpoints) == 1:
                        is_fitpoint = fitpoints[0]
                        if is_fitpoint == 1:
                            n = self._LTE.flat_used_elem_names.count(name)
                            self._fitpoint_names.extend(
                                [f'{name}#{occurrence:d}' for occurrence in range(1, n+1)])
                    else:
                        raise RuntimeError('Unexpected error. Multiple FITPOINT specified')

            # See the definition of "MARK" element in the ELEGANT manual
            quantity_list = [
                'pCentral', 'Cx', 'Cxp', 'Cy', 'Cyp', 'Cs', 'Cdelta',
                'Sx', 'Sxp', 'Sy', 'Syp', 'Ss', 'Sdelta', 'Particles'
                ] + [
                    f's{i}{j}' for i in range(1, 6+1) for j in range(1, 6+1)
                ] + ['betaxBeam', 'alphaxBeam', 'betayBeam', 'alphayBeam'] + [
                    f'R{i}{j}' for i in range(1, 6+1) for j in range(1, 6+1)]
            if kwargs.get('default_order', 2) >= 2:
                quantity_list += [
                    f'T{i}{j}{k}' for i in range(1, 6+1) for j in range(1, 6+1)
                    for k in range(1, 6+1)]
            #
            for fitpoint_name in self._fitpoint_names:
                for quantity in quantity_list:
                    new_var_names['optimization_term'].append(f'{fitpoint_name}.{quantity}')

        elif (block_header == 'twiss_output') and \
             kwargs.get('output_at_each_step', False):

            new_var_names['optimization_term'].extend([
                'nux', 'nuy', 'dnux/dp', 'dnuy/dp', 'alphac', 'alphac2',
            ])

            for statistic in ['min', 'max', 'ave', 'p99', 'p98', 'p96']:
                for twiss_param_name in ['betax', 'alphax', 'betay', 'alphay',
                                         'etax', 'etaxp', 'etay', 'etayp']:
                    new_var_names['optimization_term'].append(f'{statistic}.{twiss_param_name}')

            for fitpoint_name in self._fitpoint_names:
                for twiss_param_name in [
                    'betax', 'alphax', 'betay', 'alphay',
                    'etax', 'etaxp', 'etapx', 'etay', 'etayp', 'etapy',
                    'nux', 'psix', 'nuy', 'psiy']:
                    # ^ Note that "etapx" and "etaxp" are the same, being
                    #   alternate names for etax_prime, and the same is true for
                    #   vertical plane.
                    new_var_names['optimization_term'].append(f'{fitpoint_name}.{twiss_param_name}')

            if kwargs.get('radiation_integrals', False):
                new_var_names['optimization_term'].extend([
                    'ex0', 'Sdelta0', 'Jx', 'Jy', 'Jdelta', 'taux', 'tauy',
                    'taudelta', 'I1', 'I2', 'I3', 'I4', 'I5'])

            if kwargs.get('compute_driving_terms', False):
                new_var_names['optimization_term'].extend([
                    'h11001', 'h00111', 'h20001', 'h00201', 'h10002', 'h21000',
                    'h30000', 'h10110', 'h10020', 'h10200', 'h22000', 'h11110',
                    'h00220', 'h31000', 'h40000', 'h20110', 'h11200', 'h20020',
                    'h20200', 'h00310', 'h00400', 'dnux/dJx', 'dnux/dJy', 'dnuy/dJy'
                ])

        for k, v in self.rpnvars.items():
            v.update_base(new_var_names[k])

        self._update_accessible_rpnvars()

    #----------------------------------------------------------------------
    def getLattice(self):
        """
        Return pyelegant Lattice object
        """

        return self._LTE

    #----------------------------------------------------------------------
    def get_LTE_elem_info(self, elem_name: str):
        """"""

        if elem_name not in self._all_elem_names:
            return None

        matched_index = self._all_elem_names.index(elem_name)

        _, elem_type, prop_str = self._LTE.elem_defs[matched_index]

        return dict(elem_type=elem_type, prop_str=prop_str)

    #----------------------------------------------------------------------
    def get_LTE_elem_prop(self, elem_name: str, prop_name: str):
        """"""

        info = self.get_LTE_elem_info(elem_name)

        if info:
            prop = self._LTE.parse_elem_properties(info['prop_str'])
            if prop_name in prop:
                return prop[prop_name]
            else:
                return None
        else:
            return None

    #----------------------------------------------------------------------
    def get_LTE_all_elem_defs(self):
        """"""

        return self._LTE.elem_defs

    #----------------------------------------------------------------------
    def get_LTE_all_beamline_defs(self):
        """"""

        return self._LTE.beamline_defs

    #----------------------------------------------------------------------
    def get_LTE_used_beamline_name(self):
        """"""

        return self._LTE.used_beamline_name

    #----------------------------------------------------------------------
    def get_LTE_all_elem_names(self):
        """
        Returns all the element names for the used beamline in s-pos order.

        Note that this list of names does NOT start with the "_BEG_" element
        ELEGANT always inserts.
        """

        return self._LTE.flat_used_elem_names

    #----------------------------------------------------------------------
    def get_LTE_all_kickers(self, spos_sorted=False) -> dict:
        """"""

        kickers = {}

        kickers['h'] = [
            (name, elem_type) for name, elem_type, _ in self._LTE.elem_defs
            if elem_type in ('HKICK', 'EHKICK', 'KICKER', 'EKICKER')]

        kickers['v'] = [
            (name, elem_type) for name, elem_type, _ in self._LTE.elem_defs
            if elem_type in ('VKICK', 'EVKICK', 'KICKER', 'EKICKER')]

        if spos_sorted:

            for plane in ['h', 'v']:
                try:
                    assert np.all(np.array(
                        [self._LTE.flat_used_elem_names.count(name)
                         for name, _ in kickers[plane]]) == 1)
                except AssertionError:
                    for name, _ in kickers[plane]:
                        n = self._LTE.flat_used_elem_names.count(name)
                        if n != 1:
                            print(f'* There are {n:d} occurrences of Element "{name}"')

                    print('ERROR: There cannot be multiple occurrences of kicker elements if "spos_sorted" is True.')
                    raise

            for plane in ['h', 'v']:
                num_inds = [self._LTE.flat_used_elem_names.index(name)
                            for name, _ in kickers[plane]]
                sort_inds = np.argsort(num_inds)
                kickers[plane] = [kickers[plane][i] for i in sort_inds]

        return kickers

    #----------------------------------------------------------------------
    def get_LTE_elem_names_by_regex(self, pattern, spos_sorted=False) -> List:
        """"""

        matched_elem_names = [
            name for name, elem_type, _ in self._LTE.elem_defs
            if re.search(pattern, name, flags=re.IGNORECASE)]

        if spos_sorted:
            return self._spos_sort_matched_elem_names(matched_elem_names)
        else:
            return matched_elem_names

    #----------------------------------------------------------------------
    def get_LTE_elem_names_types_by_regex(self, pattern, spos_sorted=False) -> List:
        """"""

        matched_elem_names_types = [
            (name, elem_type) for name, elem_type, _ in self._LTE.elem_defs
            if re.search(pattern, name, flags=re.IGNORECASE)]

        if spos_sorted:
            return self._spos_sort_matched_elem_names_types(matched_elem_names_types)
        else:
            return matched_elem_names_types

    #----------------------------------------------------------------------
    def get_LTE_elem_names_for_elem_type(self, sel_elem_type, spos_sorted=False) -> List:
        """"""

        sel_elem_type = sel_elem_type.upper()

        matched_elem_names = [
            name for name, elem_type, _ in self._LTE.elem_defs
            if elem_type.upper() == sel_elem_type]

        if spos_sorted:
            return self._spos_sort_matched_elem_names(matched_elem_names)
        else:
            return matched_elem_names

    #----------------------------------------------------------------------
    def _spos_sort_matched_elem_names(self, matched_elem_names) -> List:
        """"""

        try:
            assert np.all(np.array(
                [self._LTE.flat_used_elem_names.count(name)
                 for name in matched_elem_names]) == 1)
        except AssertionError:
            for name in matched_elem_names:
                n = self._LTE.flat_used_elem_names.count(name)
                if n != 1:
                    print(f'* There are {n:d} occurrences of Element "{name}"')

            print(
                ('ERROR: There cannot be multiple occurrences of elements '
                 'with the same name if "spos_sorted" is True.'))
            raise

        num_inds = [self._LTE.flat_used_elem_names.index(name)
                    for name in matched_elem_names]
        sort_inds = np.argsort(num_inds)
        matched_elem_names = [matched_elem_names[i] for i in sort_inds]

        return matched_elem_names

    #----------------------------------------------------------------------
    def _spos_sort_matched_elem_names_types(self, matched_elem_names_types) -> List:
        """"""

        try:
            assert np.all(np.array(
                [self._LTE.flat_used_elem_names.count(name)
                 for name, _ in matched_elem_names_types]) == 1)
        except AssertionError:
            for name, _ in matched_elem_names_types:
                n = self._LTE.flat_used_elem_names.count(name)
                if n != 1:
                    print(f'* There are {n:d} occurrences of Element "{name}"')

            print(
                ('ERROR: There cannot be multiple occurrences of elements '
                 'with the same name if "spos_sorted" is True.'))
            raise

        num_inds = [self._LTE.flat_used_elem_names.index(name)
                    for name, _ in matched_elem_names_types]
        sort_inds = np.argsort(num_inds)
        matched_elem_names_types = [matched_elem_names_types[i] for i in sort_inds]

        return matched_elem_names_types

    #----------------------------------------------------------------------
    def get_LTE_elem_count(self, elem_name: str):
        """"""

        return self._LTE.flat_used_elem_names.count(elem_name)

    #----------------------------------------------------------------------
    def update_output_filepaths(self):
        """"""

        if self.rootname is not None:
            ele_filepath_wo_ext = self.rootname
        else:
            assert self.ele_filepath.endswith('.ele')
            ele_filepath_wo_ext = self.ele_filepath[:-4]

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

        self._last_block_text = self._get_block_str(block_name, **kwargs)

        # --- Now update "rpnvars" ---
        self._update_base_rpnvars(block_name, **kwargs)

        self._text_blocks.insert(self._last_block_text)

        if std_print_enabled['out'] and self.auto_print_on_add:
            self.print_last_block()

    #----------------------------------------------------------------------
    def remove_block_above(self):
        """"""

        self._text_blocks.pop_above()

        for v in self.rpnvars.values():
            v.pop_above()
        self._update_accessible_rpnvars()

    #----------------------------------------------------------------------
    def remove_block_below(self):
        """"""

        self._text_blocks.pop_below()

        for v in self.rpnvars.values():
            v.pop_below()
        self._update_accessible_rpnvars()

########################################################################
class OptimizationTerm():
    """"""

    #----------------------------------------------------------------------
    def __init__(self, ele_designer_obj, intermediate=False,
                 suppress_max_str_len_warning=False):
        """Constructor"""

        ed = ele_designer_obj

        self.rpnvars = ed.get_rpn_vars('optimization_term')

        self.assignment_rpn_str_list = []

        self._intermediate = intermediate

        self._final = None

        self._max_str_len = ed.blocks.OPTIM_TERM_TERM_FIELD_MAX_STR_LEN
        # Max string length limitation (approximately & empirically determined)
        # for "optimization_term" block.

        self._suppress_max_str_len_warning = suppress_max_str_len_warning

    def __repr__(self):
        """"""

        if self._final is None:
            output = InfixEquation('0.0')
        else:
            output = self._final

        newline_indent = '\n' + (' ' * 8)
        final_rpn_str = newline_indent.join(
            self.assignment_rpn_str_list + [output.torpn()])

        final_rpn_str = newline_indent + final_rpn_str + ('\n' + (' ' * 4))

        if (not self._suppress_max_str_len_warning) and \
           len(final_rpn_str) > self._max_str_len:
            print('\n## WARNING ##')
            print(f'Expression character length is exceeding {self._max_str_len:d}. ELEGANT will likely fail.')
            print('Try to divide this OptimizationTerm object into multiple intermediate OptimizationTerm objects.')

        return final_rpn_str

    def assign(self, new_var_name, infix_eq_obj):
        """"""

        self.rpnvars.add_uncommitted_var_name(new_var_name)

        rpn_str = f'{infix_eq_obj.torpn()} sto {new_var_name} pop'

        self.assignment_rpn_str_list.append(rpn_str)

    def is_intermediate(self):
        """"""

        return self._intermediate

    def set_as_intermediate(self, true_or_false: bool) -> None:
        """"""

        self._intermediate = true_or_false

    def get_final(self):
        """"""

        return self._final

    def set_final(self, final_val):
        """"""

        if self._intermediate:
            raise RuntimeError('Cannot set a final term for the OptimizationTerm object set as "intermediate".')

        if isinstance(final_val, InfixEquation):
            self._final = final_val
        elif isinstance(final_val, (float, int, np.integer)):
            self._final = InfixEquation(str(final_val))
        elif isinstance(final_val, str):
            self._final = InfixEquation(final_val)
        else:
            raise TypeError('Invalid type')

    def __enter__(self):
        """"""

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """"""



def add_N_KICKS_alter_elements_blocks(ed: EleDesigner, N_KICKS: dict) -> None:
    """
    ed := EleDesigner object
    """

    if N_KICKS is None:
        N_KICKS = dict(CSBEND=40, KQUAD=40, KSEXT=20, KOCT=20)

    for k, v in N_KICKS.items():
        if k.upper() not in ('CSBEND', 'KQUAD', 'KSEXT', 'KOCT'):
            raise ValueError(f'The key "{k}" in N_KICKS dict is invalid. '
                             f'Must be one of CSBEND, KQUAD, KSEXT, or KOCT')
        ed.add_block('alter_elements',
                     name='*', type=k.upper(), item='N_KICKS', value=v,
                     allow_missing_elements=True)

def add_transmute_blocks(ed: EleDesigner, transmute_elements: dict) -> None:
    """"""

    if transmute_elements is None:

        actual_transmute_elems = dict(
            SBEN='CSBEND', RBEN='CSBEND', QUAD='KQUAD', SEXT='KSEXT',
            OCTU='KOCT', RFCA='MARK', SREFFECTS='MARK')
    else:

        actual_transmute_elems = {}
        for old_type, new_type in transmute_elements.items():
            actual_transmute_elems[old_type] = new_type

    for old_type, new_type in actual_transmute_elems.items():
        ed.add_block('transmute_elements',
                     name='*', type=old_type, new_type=new_type)
