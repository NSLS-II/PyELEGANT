program_name = 'pyelegant'

from setuptools import setup, find_packages

import sys
import os
import json

#print(sys.argv)

facility_json_filename = 'facility.json'

if ('install' in sys.argv) or ('sdist' in sys.argv):

    facility_name = sys.argv[2]
    available_facility_names = ['nsls2apcluster',]
    if facility_name not in available_facility_names:
        print('* Only the following facility names are available:')
        print('      ' + ', '.join(available_facility_names))
        raise ValueError(
            'Specified facility_name "{}" is not available.'.format(facility_name))

    this_folder = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(
        this_folder, 'src', 'pyelegant', facility_json_filename), 'w') as f:
        json.dump({'name': facility_name}, f)

    sys.argv = sys.argv[:2]

    if 'sdist' in sys.argv:
        # Modify the tarball name to be "pyelegant-{facility_name}-?.?.?.tar.gz"
        program_name += f'-{facility_name}'

setup(
    name = program_name,
    version = '0.1.0',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    #include_package_data = True,
    package_data = {
        'pyelegant': [facility_json_filename,]
    },
    zip_safe=False,
    description = 'Python Interface to Elegant',
    author = 'Yoshiteru Hidaka',
    maintainer = 'Yoshiteru Hidaka',
    maintainer_email = 'yhidaka@bnl.gov'
)
