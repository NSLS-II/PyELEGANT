
def build_block_run_setup(
    LTE_filepath, E_MeV, use_beamline=None, rootname=None, magnets=None,
    semaphore_file=None, parameters=None, element_divisions=0):
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
    if element_divisions != 0:
        block += ['element_divisions = {:d}'.format(element_divisions)]

    ele_contents = '''
&run_setup
{}
&end
'''.format('\n'.join([' ' * 4 + line for line in block]))

    return ele_contents

def build_block_twiss_output(
    matched, twi_filepath='%s.twi', radiation_integrals=False,
    compute_driving_terms=False, concat_order=3, higher_order_chromaticity=False,
    beta_x=1.0, alpha_x=0.0, eta_x=0.0, etap_x=0.0, beta_y=1.0, alpha_y=0.0,
    eta_y=0.0, etap_y=0.0):
    """"""

    block = []
    block += ['filename = "{}"'.format(twi_filepath)]
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

