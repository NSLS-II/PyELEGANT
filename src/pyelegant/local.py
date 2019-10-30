from subprocess import Popen, PIPE
import shlex

from . import sdds
from .remote import remote

def run(ele_filepath, macros=None, print_cmd=False, print_stdout=True, print_stderr=True):
    """"""

    cmd_list = ['elegant', ele_filepath]

    if macros is not None:
        macro_str_list = []
        for k, v in macros.items():
            macro_str_list.append('='.join([k, v]))
        cmd_list.append('-macro=' + ','.join(macro_str_list))

    if print_cmd:
        print('$ ' + ' '.join(cmd_list))

    p = Popen(cmd_list, stdout=PIPE, stderr=PIPE)
    out, err = p.communicate()
    out, err = out.decode('utf-8'), err.decode('utf-8')

    if out and print_stdout:
        print(out)

    if err and print_stderr:
        print('ERROR:')
        print(err)

def geneticOptimizer(
    root_name, parameterName, initialValue, errorLevel, lowerLimit, upperLimit,
    nTotalJobs, nParents, childMultiplier, runScript='', inputFileTemplate='',
    inputFileExtension='', programName='', progressScript='',
    preProcessingScript='', postProcessingScript='',
    contin=False, errorFactor=1.0, reduce=False, multiObjective=False,
    topUp=False, maxRank=1, drain=False, updateOnly=False, pulse=False,
    autoPulse='', staggerTime=0, print_cmd=False, remote_opts=None):
    """
    Interface to the "geneticOptimizer" script.

    root_name: Used as root name for the input SDDS file.

    contin (Default: False):
        If True, then continue optimizing from a previous run.

    errorFactor (Default: 1.0):
        factor by which to multiply the errorLevels from the input file.

    reduce (Default: False):
        If True, then trial jobs that are not selected for breeding are deleted.
        This saves time and disk space.

    multiObjective (Default: False):
        If True, then perform multiobjective optimization.
        The post-processing file *.proc should contain penalty value columns
        one for each object, ending with the string \"Penalty\".  It may or
        may not contain ConstraintViolation (short) columns.

    topUp (Default: False):
        If True, then for continued runs the optimizer will start a sufficient
        number of jobs to ensure that the desired total number is running.  Use
        this only if you have changed you input file in the middle of running to
        increase the number of jobs and want to add jobs now, rather than waiting
        for processing to occur.

    maxRank (Default: 1):
        For multiobjective mode, maximum rank of solutions to include in breeding.

    drain (Default: False):
        The script waits for jobs to complete and updates output files, but
        doesn't submit new jobs.

    updateOnly (Default: False):
        The script updates output files and then exits.

    pulse (Default: False):
        Add a one-time pulse of <number> jobs.

    autoPulse (Default: ""):
        Use script to determine whether to emit a pulse of jobs.

    The following are the parameters that will put in to the input SDDS file:

    ** Parameters **

    nTotalJobs:
        Maximum number of jobs to run.
    nParents:
        Number of parent jobs to use for breeding.
    childMultiplier:
        Number of child jobs to create per parent.
            NB: childMultiplier*nParents gives the number of simultaneous jobs
            that will be run on the queue.
            For multi-objective mode, only this product is important.
    maxRank:
        Overrides maxRank parameter on the commandline.
    sleepTime: (This value is NOT being utilized in the actual script.)
        Seconds to sleep between checking for newly-completed jobs
    staggerTime:
        Seconds to sleep between submission of successive jobs, which can help
        reduce peak load on the system.
    multiObjective:
        If True, perform multiobjective optmization.  See above for requirements.
        Over-rides any commandline value.
    preProcessingScript:
        Script filename to run prior to running the job. (May be blank.)
        The script must accept the following arguments and syntax:
          -rootname <string>       rootname for the job.
          -tagList <string>        list of tags for varied quantities.
          -valueList <string>      list of values associated with tags.
          The tag names are the same as the parameter names given in the
          parameterName column (see below).
        E.g., the script will be called with arguments like these:
          -rootname run-000000 -tagList \"Var1 Var2\" -valueList \"1.7 -2.8\"
    postProcessingScript:
        Script filename to run to postprocess a job.
        Called with one argument giving the rootname of the job.  This script
        must produce a one-row SDDS file named <rootname>.proc that contains at
        least the following columns:
          runName --- the rootname of the run for this file
          penaltyValue --- the value of the penalty function
        This file must also have all the columns from <rootname>.inp sddsxref'd in.
    runScript:
        Script filename to run the simulation job.
        The script must accept the same arguments as the preProcessingScript.
        If this is blank, then you must give inputFileTemplate, inputFileExtension,
        and programName.
        The script must create a file <rootname>.run as soon as it starts and
        <rootname>.done just before finishing.
    inputFileTemplate:
        Name of the template file for creating input files. Sequences of the
        form <name> are substituted with values.
    inputFileExtension:
        Extension to use for input files.
    programName:
        Name of the program to run with the input files. The input filename is
        given as the sole argument of the program.
    progressScript:
        File name of a script to run when new rank=1 solutions are found.
    autoPulse:
        File name of a script to use to determine how many jobs to emit in a pulse

    ** Columns **

    parameterName:
        Names of the parameters to vary in the optimization. These should appear
        in the input file template in the form <name>.
    initialValue:
        Initial values of the parameters.
    errorLevel:
        rms width of the gaussian random errors to add to the parameter values
        as mutations.
    lowerLimit:
        Smallest allowable values for the parameters.
    upperLimit:
        Largest allowable values for the parameters.



    #srun geneticOptimizer -input optim1.sdds -reduce 1
    #srun geneticOptimizer -input optim1.sdds -reduce 1 -contin 1 -updateOnly 1


    root_name = 'test_optim'
    parameterName = [
        'S1AS1', 'S1AS2', 'S1BS3', 'S1BS2', 'S1BS1', 'nuxTarget', 'nuyTarget']
    initialValue = [10.0, -21.0, -10.0, -21.0, 10.0, 36.2, 19.3]
    errorLevel = [0.5, 0.5, 0.5, 0.5, 0.5, 0.02, 0.02]
    lowerLimit = [0.0, -31.5, -21.6, -31.5, 0.0, 36.08, 19.08]
    upperLimit = [21.6, 0.0, 0.0, 0.0, 21.6, 36.42, 19.42]
    nTotalJobs = 60_000
    nParents = 5
    childMultiplier = 8
    runScript = 'runJob1'
    multiObjective = True


    geneticOptimizer(
    root_name, parameterName, initialValue, errorLevel, lowerLimit, upperLimit,
    nTotalJobs, nParents, childMultiplier, runScript='', inputFileTemplate='',
    inputFileExtension='', programName='', progressScript='',
    preProcessingScript='', postProcessingScript='',
    contin=False, errorFactor=1.0, reduce=False, multiObjective=False,
    topUp=False, maxRank=1, drain=False, updateOnly=False, pulse=False,
    autoPulse='', staggerTime=0)
    """

    if runScript == '':
        if inputFileTemplate == '':
            raise ValueError(
                'If "runScript" is an empty string, you must provide "inputFileTemplate"')
        if inputFileExtension == '':
            raise ValueError(
                'If "runScript" is an empty string, you must provide "inputFileExtension"')
        if programName == '':
            raise ValueError(
                'If "runScript" is an empty string, you must provide "programName"')

    remote.write_geneticOptimizer_dot_local(remote_opts)

    input_sdds_filepath = f'{root_name}.sdds'

    input_columns = dict(
        parameterName=parameterName, initialValue=initialValue,
        errorLevel=errorLevel, lowerLimit=lowerLimit, upperLimit=upperLimit)

    input_params = dict(
        nTotalJobs=nTotalJobs, nParents=nParents, childMultiplier=childMultiplier,
        maxRank=maxRank, staggerTime=staggerTime, multiObjective=multiObjective,
        runScript=runScript, inputFileTemplate=inputFileTemplate,
        inputFileExtension=inputFileExtension, programName=programName,
        progressScript=progressScript, preProcessingScript=preProcessingScript,
        postProcessingScript=postProcessingScript, autoPulse=autoPulse,
    )

    cli_params = dict(
        contin=contin, errorFactor=errorFactor, reduce=reduce, topUp=topUp,
        drain=drain, updateOnly=updateOnly, pulse=pulse)

    sdds.dicts2sdds(
        input_sdds_filepath, params=input_params, columns=input_columns,
        outputMode='ascii', suppress_err_msg=True)

    cmd = (
        f'geneticOptimizer -input {input_sdds_filepath} -contin {contin:d} '
        f'-errorFactor {errorFactor:.9g} -reduce {reduce:d} -topUp {topUp:d} '
        f'-drain {drain:d} -updateOnly {updateOnly:d} -pulse {pulse:d}'
    )
    cmd_list = shlex.split(cmd)
    if print_cmd:
        print('\n$ ' + ' '.join(cmd_list) + '\n')

    p = Popen(cmd_list, stdout=PIPE, stderr=PIPE, encoding='utf-8')
    out, err = p.communicate()


