pyelegant_conda_activate_latest() {

    \local TEMP_CONDA_ENV_NAME=$(python -c "

from glob import glob

prefix = '/opt/conda_envs/pyelegant-'
env_folderpaths = glob(prefix + '*/')
versions = [v[len(prefix):-1] for v in env_folderpaths]
uniform_versions = [v if len(v.split('.')) == 3 else v + '.0' for v in versions]
latest_index = [i for (v, i) in sorted((v, i)
                for (i, v) in enumerate(uniform_versions))][-1]
latest_ver_str = versions[latest_index]
latest_env_name = 'pyelegant-' + latest_ver_str
print(latest_env_name)
")

    #echo $TEMP_CONDA_ENV_NAME

    #eval "$(conda shell.bash hook)"
    conda activate $TEMP_CONDA_ENV_NAME
}


# #!/usr/bin/python
#from glob import glob
#prefix = '/opt/conda_envs/pyelegant-'
##print(sorted(glob(prefix + '*/')))
#if True:
    #env_folderpaths = glob(prefix + '*/')
#else:
    #env_folderpaths = ['/opt/conda_envs/pyelegant-0.3/',
                       #'/opt/conda_envs/pyelegant-0.4.1/',
                       #'/opt/conda_envs/pyelegant-0.4/',
                       #'/opt/conda_envs/pyelegant-0.3.1/']

#versions = [v[len(prefix):-1] for v in env_folderpaths]
#uniform_versions = [v if len(v.split('.')) == 3 else v + '.0' for v in versions]
##print(uniform_versions)
##latest_ver_str = sorted(uniform_versions)[-1]

#latest_index = [i for (v, i) in sorted((v, i)
                #for (i, v) in enumerate(uniform_versions))][-1]
#latest_ver_str = versions[latest_index]
#latest_env_name = 'pyelegant-' + latest_ver_str
#print(latest_env_name)