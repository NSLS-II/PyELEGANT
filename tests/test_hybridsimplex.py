# Based on nsls2cb/20190315_Pelegant/cbiiMatch04.ele

import sys
import os
import glob

from pyelegant import elebuilder
from pyelegant import remote
from pyelegant import sdds
from pyelegant import util

eb = elebuilder.EleContents(double_format='.12g')

eb.run_setup(
    lattice = 'lattice3.lte', p_central_mev = 3e3,
    use_beamline='ring',
    semaphore_file = '%s.done', parameters = '%s.param', default_order = 2
)

eb.newline()

eb.comment('! Load solution from previous optimization')
eb.load_parameters(
    filename = 'cbiiMatch02.param',
    change_defined_values = 1,
)

eb.newline()

eb.comment('! Ask for twiss parameters (during optimization)')
eb.twiss_output(
    output_at_each_step = 1,
    matched = 1,
    radiation_integrals = 1,
    concat_order = 2,
)

eb.newline()

eb.comment('! Ask for floor coordinates (during optimization)')
eb.floor_coordinates(
    filename = '%s.flr'
)

eb.newline()

eb.comment('! Load floor coordinate data for NSLS-II')
eb.comment('! - First ID')
eb.rpn_load(
    filename = 'nsls2.flr',
    tag = 'flr1',
    match_column = 'ElementName',
    match_column_value = 'MID',
    matching_row_number = 0
)
eb.comment('! - Second ID')
eb.rpn_load(
    filename = 'nsls2.flr',
    tag = 'flr2',
    match_column = 'ElementName',
    match_column_value = 'MID',
    matching_row_number = 1
)

eb.newline()

eb.run_control()

eb.newline()

eb.comment('! Use parallel hybrid simplex optimization')
eb.parallel_optimization_setup(
    mode = 'minimize', method = 'hybridsimplex',
    hybrid_simplex_comparison_interval = 100,
    target = 1e-6, tolerance = 1e-14,
    #n_passes = 3, n_evaluations = 1500, n_restarts = 10,
    n_passes = 3, n_evaluations = 100, n_restarts = 1,
    verbose = 0, log_file = '/dev/tty',
    output_sparsing_factor = 100,
    term_log_file = '%s.tlog',
    population_log = '%s.pop',
    simplex_log = '%s.simlog',
    #!! Depending on your filesystem, you may want to
    #!! increase this in order to improve performance
    #simplex_log_interval = 10,
    simplex_log_interval = 50,
)

eb.newline()

eb.comment('! Dipole angles')
eb.optimization_variable(
    name = 'B1QDH', item = 'ANGLE',
    lower_limit=-0.01, upper_limit=0.03, step_size=1e-5
)
eb.optimization_variable(
    name = 'B2QDH', item = 'ANGLE',
    lower_limit=-0.01, upper_limit=0.03, step_size=1e-5
)
eb.optimization_variable(
    name = 'B3QDH', item = 'ANGLE',
    lower_limit=-0.01, upper_limit=0.03, step_size=1e-5
)
eb.optimization_variable(
    name = 'B4QDH', item = 'ANGLE',
    lower_limit=-0.01, upper_limit=0.03, step_size=1e-5
)
eb.optimization_variable(
    name = 'QF1', item = 'ANGLE',
    lower_limit=-0.01, upper_limit=0.03, step_size=1e-5
)
eb.optimization_variable(
    name = 'QF2', item = 'ANGLE',
    lower_limit=-0.01, upper_limit=0.03, step_size=1e-5
)

eb.newline()

eb.comment('! Set QF4 angle so that the total per cell is 6 degrees')
eb.optimization_covariable(
    name = 'QF4', item = 'ANGLE',
    equation = '6 dtor B1QDH.ANGLE 2 * - B2QDH.ANGLE 2 * - B3QDH.ANGLE 2 * - B4QDH.ANGLE 2 * - QF1.ANGLE - QF2.ANGLE -'
)

eb.newline()

eb.comment('! Constrain QF4 angle within a reasonable range')
eb.optimization_term(
    term = "QF4.ANGLE 0.01 1e-6 selt"
)
eb.optimization_term(
    term = "QF4.ANGLE 0.03 1e-6 segt"
)

eb.newline()

eb.comment('! Vary focusing gradients')
eb.optimization_variable(
    name = 'B1QDH', item = 'K1', lower_limit=-25, upper_limit=25, step_size=1e-4
)
eb.optimization_variable(
    name = 'B2QDH', item = 'K1', lower_limit=-25, upper_limit=25, step_size=1e-4
)
eb.optimization_variable(
    name = 'B3QDH', item = 'K1', lower_limit=-25, upper_limit=25, step_size=1e-4
)
eb.optimization_variable(
    name = 'B4QDH', item = 'K1', lower_limit=-25, upper_limit=25, step_size=1e-4
)
eb.optimization_variable(
    name = 'QF1', item = 'K1', lower_limit=-25, upper_limit=25, step_size=1e-4
)
eb.optimization_variable(
    name = 'QF2', item = 'K1', lower_limit=-25, upper_limit=25, step_size=1e-4
)
eb.optimization_variable(
    name = 'QF3', item = 'K1', lower_limit=-25, upper_limit=25, step_size=1e-4
)
eb.optimization_variable(
    name = 'QF4', item = 'K1', lower_limit=-25, upper_limit=25, step_size=1e-4
)

eb.newline()

eb.optimization_variable(
    name = 'M1G4A', item = 'K1', lower_limit = -7, upper_limit = 7, step_size = 1e-4
)
eb.optimization_variable(
    name = 'M2G4A', item = 'K1', lower_limit = -7, upper_limit = 7, step_size = 1e-4
)

eb.newline()

eb.optimization_variable(
    name = 'Q0', item = 'K1', lower_limit = -7, upper_limit = 7, step_size = 1e-4
)
eb.optimization_variable(
    name = 'QH3G2A', item = 'K1', lower_limit = -7, upper_limit = 7, step_size = 1e-4
)
eb.optimization_variable(
    name = 'QH2G2A', item = 'K1', lower_limit = -7, upper_limit = 7, step_size = 1e-4
)
eb.optimization_variable(
    name = 'QH1G2A', item = 'K1', lower_limit = -7, upper_limit = 7, step_size = 1e-4
)

eb.newline()

eb.optimization_variable(
    name = 'QL0', item = 'K1', lower_limit = -7, upper_limit = 7, step_size = 1e-4
)
eb.optimization_variable(
    name = 'QL3G2A', item = 'K1', lower_limit = -7, upper_limit = 7, step_size = 1e-4
)
eb.optimization_variable(
    name = 'QL2G2A', item = 'K1', lower_limit = -7, upper_limit = 7, step_size = 1e-4
)
eb.optimization_variable(
    name = 'QL1G2A', item = 'K1', lower_limit = -7, upper_limit = 7, step_size = 1e-4
)

eb.newline()

eb.optimization_term(
    term = "dnux/dp 15 * abs 10 /")
eb.optimization_term(
    term = "dnuy/dp 15 * abs 10 /"
)

eb.newline()

eb.comment('! Want Jx:[1, 2]')
eb.optimization_term(term = "Jx 1 1e-6 selt")
eb.optimization_term(term = "Jx 2 1e-6 segt")

eb.newline()

eb.comment('! Want etax to be zero in both ID straights')
eb.optimization_term(term = "MID#1.etax 0 1e-4 sene")
eb.optimization_term(term = "MID#2.etax 0 1e-4 sene")

eb.newline()

eb.comment('! Want etax>0.08 in high-dispersion region (weak requirement)')
eb.optimization_term(term = "MDISP#1.etax .1 .01 selt")
eb.optimization_term(term = "MDISP#2.etax .1 .01 selt")

eb.newline()

eb.comment('! Minimize the emittance')

eb.optimization_term(term = "ex0 1e12 * 50.0 1.0 segt 10 *")

eb.newline()

eb.comment('! Allow straight section lengths to vary (helps matching floor coordinates)')
eb.optimization_variable(
    name = 'ODL1G1A', item = 'L', lower_limit = 3.2, upper_limit = 3.4
)
eb.optimization_variable(
    name = 'ODH1G1A', item = 'L', lower_limit = 4.4, upper_limit = 4.6
)

eb.newline()

eb.comment('! Try to make betax=betay=L/2 in IDs, where L is the *total* straight length')
eb.comment('!&optimization_term term = "MID#1.betax ODL1G1A.L / 1 .1 sene 100 *" &end')
eb.comment('!&optimization_term term = "MID#1.betay ODL1G1A.L / 1 .1 sene 100 *" &end')
eb.comment('!&optimization_term term = "MID#2.betax ODH1G1A.L / 1 .1 sene 100 *" &end')
eb.comment('!&optimization_term term = "MID#2.betay ODH1G1A.L / 1 .1 sene 100 *" &end')
eb.comment('! Try to make betax & betay the same as NSLS-II Day-1 bare')
eb.optimization_term(term = "MID#1.betax 1.846 0.2 sene 5 *")
eb.optimization_term(term = "MID#1.betay 1.171 0.1 sene 5 *")
eb.optimization_term(term = "MID#2.betax 20.466 2.0 sene 5 *")
eb.optimization_term(term = "MID#2.betay 3.369 0.3 sene 5 *")

eb.newline()

eb.comment('! Keep the radius within 1mm of NSLS-II')
eb.optimization_term(
    term = "MID#1.X sqr MID#1.Z sqr + sqrt  flr1.X sqr flr1.Z sqr + sqrt 1e-3 sene")
eb.optimization_term(
    term = "MID#2.X sqr MID#2.Z sqr + sqrt  flr2.X sqr flr2.Z sqr + sqrt 1e-3 sene")

eb.newline()

eb.comment('! Make sure beta is not too large anywhere')
eb.optimization_term(term = "max.betax 35.0 1.0 segt")
eb.optimization_term(term = "max.betay 35.0 1.0 segt")

eb.newline()

eb.comment('! Make sure beta is not too small anywhere')
eb.optimization_term(term = "min.betax 0.2 0.1 selt")
eb.optimization_term(term = "min.betay 0.2 0.1 selt")

eb.newline()

eb.comment("! Ensure that central particle isn't lost")
eb.optimization_term(term = "Particles 1 1e-10 sene")

eb.newline()

eb.comment("! Beam consists of central particle only")
eb.bunched_beam()

eb.newline()

eb.comment("! Start optimization")
eb.optimize()

eb.newline()

eb.comment("! Evaluate the results of optimization")

eb.newline()

eb.run_setup(
    lattice = 'lattice3.lte',
    use_beamline='ring',
    # ^ Note that here I am using the full ring, not 2 ring cells
    p_central_mev = 3e3,
    semaphore_file = '%s.done',
    magnets = '%s.mag',
    default_order = 2,
)

eb.newline()

eb.load_parameters(filename = '%s.param', change_defined_values = 1)

eb.newline()

eb.twiss_output(
    filename = '%s.twi',
    matched = 1,
    radiation_integrals = 1,
    concat_order = 2)

eb.newline()

eb.floor_coordinates(filename = '%s.flr')

eb.newline()

eb.save_lattice(filename = '%s.newlte')

ele_filepath = 'test_hybridsimplex.ele'
eb.write(ele_filepath)

eb.update_output_filepaths(ele_filepath.replace('.ele', ''))
output_filepath_list = eb.actual_output_filepath_list

show_progress_plot = True

if not show_progress_plot: # If you don't care to see the progress of the optimization

    # Run Pelegant
    #
    # Minimal options
    #    remote_opts = dict(pelegant=True, ntasks=50)
    remote_opts = dict(
        use_sbatch=True, exit_right_after_sbatch=False, pelegant=True,
        job_name='job', partition='normal', ntasks=50,
        #mail_type_begin=True,
        mail_type_end=True, mail_user='yhidaka@bnl.gov',
    )

    remote.run(remote_opts, ele_filepath)
    # ^ This will block until the optimization is completed.

else: # If you want to see the progress of the optimization

    # Run Pelegant
    remote_opts = dict(
        use_sbatch=True, exit_right_after_sbatch=True, pelegant=True,
        job_name='job', partition='normal', ntasks=50,
        #mail_type_begin=True,
        #mail_type_end=True, mail_user='yhidaka@bnl.gov',
    )
    job_info = remote.run(remote_opts, ele_filepath)

    # Start plotting simplex optimization progress
    remote.monitor_simplex_log_progress(job_info['job_ID_str'],
                                        ele_filepath.replace('.ele', '.simlog'))

    try:
        os.remove(job_info['sbatch_sh_filepath'])
    except IOError:
        print('* Failed to delete temporary sbatch shell file "{}"'.format(
            job_info['sbatch_sh_filepath']))

# Consolidate data in the generated SDDS files
print('Consolidating SDDS data files...')
sys.stdout.flush()
output, meta = {}, {}
for sdds_fp in output_filepath_list:
    if sdds_fp.startswith('/dev/'):
        continue
    print('Processing "{}"...'.format(sdds_fp))
    ext = sdds_fp.split('.')[-1]
    try:
        output[ext], meta[ext] = sdds.sdds2dicts(sdds_fp)
    except:
        continue
print('Finished.')
sys.stdout.flush()

# Save results into a HDF5 file
print('Writing data to HDF5 file...')
sys.stdout.flush()
output_filepath = 'results.hdf5'
util.robust_sdds_hdf5_write(
    output_filepath, [output, meta], nMaxTry=10, sleep=10.0)
print('Finished.')
sys.stdout.flush()

# Save the dictionaries into a gzipped pickle file
print('Writing data to pgz file...')
sys.stdout.flush()
output_filepath = 'results.pgz'
mod_output = {}
for k, v in output.items():
    mod_output[k] = {}
    if 'params' in v:
        mod_output[k]['scalars'] = v['params']
    if 'columns' in v:
        mod_output[k]['arrays'] = v['columns']
mod_meta = {}
for k, v in meta.items():
    mod_meta[k] = {}
    if 'params' in v:
        mod_meta[k]['scalars'] = v['params']
    if 'columns' in v:
        mod_meta[k]['arrays'] = v['columns']
util.robust_pgz_file_write(
    output_filepath, [mod_output, mod_meta], nMaxTry=10, sleep=10.0)
print('Finished.')
sys.stdout.flush()

# Delete the raw SDDS files
if True:
    for fp in output_filepath_list:
        if fp.startswith('/dev'):
            continue

        if fp.endswith('.simlog'):
            fp_list = glob.glob(fp + '-*')
            for sub_fp in fp_list:
                try:
                    os.remove(sub_fp)
                except:
                    print(f'Failed to delete "{sub_fp}"')

        elif fp.startswith('/dev'):
            continue

        else:
            try:
                os.remove(fp)
            except:
                print(f'Failed to delete "{fp}"')
