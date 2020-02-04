import sys
import numpy as np

import pyelegant as pe

parameterNames = np.array(['S1_K2', 'S1B_K2', 'nuxTarget', 'nuyTarget'])
lowerLimits   = np.array([ 0.0, -6e2, 78.580, 13.080])
initialValues = np.array([+372,  -55, 78.863, 13.201])
upperLimits   = np.array([+6e2,  0.0, 78.920, 13.420])
errorLevels   = np.array([50.0, 50.0,  0.100,  0.100])

nParams = len(parameterNames)
assert len(lowerLimits) == nParams
assert len(initialValues) == nParams
assert len(upperLimits) == nParams
assert len(errorLevels) == nParams
assert np.all(lowerLimits <= initialValues)
assert np.all(initialValues <= upperLimits)

nTotalJobs = 60_000
nParents = 5
childMultiplier = 4
runScript ='runJob1.py'

#print(sys.argv)

# CLI options
if sys.argv[1] == 'start':
    contin = False
    updateOnly = False
    drain = False
elif sys.argv[1] == 'update_only':
    contin = True
    updateOnly = True
    drain = False
elif sys.argv[1] == 'drain':
    contin = True
    updateOnly = False
    drain = True
elif sys.argv[1] == 'resume':
    contin = True
    updateOnly = False
    drain = False
else:
    raise ValueError()
reduce = True
multiObjective = True

remote_opts=dict(partition='short')

rootname = 'test_optim'

pe.geneopt.run(
    rootname, parameterNames, initialValues, lowerLimits, upperLimits, errorLevels,
    nTotalJobs, nParents, childMultiplier, runScript=runScript,
    contin=contin, reduce=reduce, multiObjective=multiObjective, updateOnly=updateOnly,
    drain=drain,
    remote_opts=remote_opts)
