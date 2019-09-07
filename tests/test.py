import pyelegant as pe

#from subprocess import Popen, PIPE
#p = Popen('which elegant', shell=True, stdout=PIPE, stderr=PIPE)
#print(p.communicate())
#p = Popen(['which','elegant'], stdout=PIPE, stderr=PIPE)
#print(p.communicate())


LTE_filepath = 'lattice3Sext_19pm3p2m_5cell.lte'
E_MeV = 3e3
use_beamline = None
#use_beamline = 'RING'
radiation_integrals = True
compute_driving_terms = False
concat_order = 1
higher_order_chromaticity = False
ele_filepath = None
twi_filepath = '%s.twi'
rootname = None
magnets = None
semaphore_file = None
parameters = None
element_divisions = 0
macros = None
alter_elements_list = None
#
calc_correct_transport_line_linear_chrom = False
betax0 = 3.0
betay0 = 3.0
alphax0=0.0
alphay0=0.0
etax0=0.0
etay0=0.0
etaxp0=0.0
etayp0=0.0

job_name = 'test'
remote_opts = dict(
    #use_sbatch=False,
    use_sbatch=True,
    pelegant=False,
    job_name=job_name, output=job_name+'.%J.out', error=job_name+'.%J.err',
    ntasks=1, partition='normal', time='10:00',
    #nodelist=['apcpu-001',], exclude=None
    #
    sbatch_err_check_tree=None,
    #sbatch_err_check_tree='default',
    #sbatch_err_check_tree=[
        #['exists', ('semaphore_file',), {}],
        #[
            #['not_empty', ('%s.newlte',), {}],
            #'no_error',
            #'retry'
        #],
        #[
            #['check_slurm_err_log', ('slurm_err_filepath', 'abort_info'), {}],
            #'retry',
            #'abort'
        #]
    #]
)

if False:
    outupt_filepaths = pe.calc_ring_twiss(
        LTE_filepath, E_MeV, use_beamline=use_beamline,
        radiation_integrals=radiation_integrals,
        compute_driving_terms=compute_driving_terms,
        concat_order=concat_order, higher_order_chromaticity=higher_order_chromaticity,
        ele_filepath=ele_filepath, twi_filepath=twi_filepath, rootname=rootname,
        magnets=magnets, semaphore_file=semaphore_file, parameters=parameters,
        element_divisions=element_divisions, macros=macros,
        alter_elements_list=alter_elements_list,
        run_local=True)
elif False:
    outupt_filepaths = pe.calc_ring_twiss(
        LTE_filepath, E_MeV, use_beamline=use_beamline,
        radiation_integrals=radiation_integrals,
        compute_driving_terms=compute_driving_terms,
        concat_order=concat_order, higher_order_chromaticity=higher_order_chromaticity,
        ele_filepath=ele_filepath, twi_filepath=twi_filepath, rootname=rootname,
        magnets=magnets, semaphore_file=semaphore_file, parameters=parameters,
        element_divisions=element_divisions, macros=macros,
        alter_elements_list=alter_elements_list,
        run_local=False, remote_opts=remote_opts)
elif False:
    outupt_filepaths = pe.calc_line_twiss(
        LTE_filepath, E_MeV, betax0, betay0, alphax0=alphax0, alphay0=alphay0,
        etax0=etax0, etay0=etay0, etaxp0=etaxp0, etayp0=etayp0,
        use_beamline=use_beamline, radiation_integrals=radiation_integrals,
        compute_driving_terms=compute_driving_terms, concat_order=concat_order,
        calc_correct_transport_line_linear_chrom=calc_correct_transport_line_linear_chrom,
        ele_filepath=ele_filepath, twi_filepath=twi_filepath,
        rootname=rootname, magnets=magnets, semaphore_file=semaphore_file,
        parameters=parameters, element_divisions=element_divisions,
        macros=macros, alter_elements_list=alter_elements_list,
        run_local=True)
elif True:
    outupt_filepaths = pe.calc_line_twiss(
        LTE_filepath, E_MeV, betax0, betay0, alphax0=alphax0, alphay0=alphay0,
        etax0=etax0, etay0=etay0, etaxp0=etaxp0, etayp0=etayp0,
        use_beamline=use_beamline, radiation_integrals=radiation_integrals,
        compute_driving_terms=compute_driving_terms, concat_order=concat_order,
        calc_correct_transport_line_linear_chrom=calc_correct_transport_line_linear_chrom,
        ele_filepath=ele_filepath, twi_filepath=twi_filepath,
        rootname=rootname, magnets=magnets, semaphore_file=semaphore_file,
        parameters=parameters, element_divisions=element_divisions,
        macros=macros, alter_elements_list=alter_elements_list,
        run_local=False, remote_opts=remote_opts)

print