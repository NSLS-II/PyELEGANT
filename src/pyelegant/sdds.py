import os, sys
import os.path as osp
from subprocess import Popen, PIPE
import re
import numpy as np

#----------------------------------------------------------------------
def strfind(string, pattern):
    """"""

    return [s.start() for s in
            re.finditer(pattern, string)]

#----------------------------------------------------------------------
def str2num(string):
    """"""

    if isinstance(string,str):
        return np.array([float(s) for s in string.split()])
    elif isinstance(string,list):
        string_list = string
        array = [[] for i in string_list]
        for (i, string) in enumerate(string_list):
            array[i] = [float(s) for s in string.split()]
        return np.array(array).flatten()
    else:
        raise TypeError('str2num only accepts a string or a list of strings.')

#----------------------------------------------------------------------
def query(sdds_filepath, suppress_err_msg=False):
    """"""

    p = Popen(['sddsquery', sdds_filepath], stdout=PIPE, stderr=PIPE,
              encoding='utf-8')
    output, error = p.communicate()
    #if isinstance(output, bytes):
        #output = output.decode('utf-8')
        #error = error.decode('utf-8')
    if error and (not suppress_err_msg):
        print('sddsquery stderr:', error)
        print('sddsquery stdout:', output)

    m = re.search(r'(\d+) columns of data:', output)
    if m is not None:
        nColumns = int(m.group(1))
        column_header = m.group(0)
    else:
        nColumns = 0

    m = re.search(r'(\d+) parameters:', output)
    if m is not None:
        nParams = int(m.group(1))
        param_header = m.group(0)
    else:
        nParams = 0

    column_dict = {}
    if nColumns != 0:
        m = re.search(r'(?<='+column_header+r')[\w\W]+', output)
        column_str_list = m.group(0).split('\n')[3:(3+nColumns)]
        for s in column_str_list:
            ss = re.search(r'([^ ]+) +'*6+'(.+)', s).groups()
            column_dict[ss[0]] = {
                'UNITS': ss[1], 'SYMBOL': ss[2], 'FORMAT': ss[3],
                'TYPE': ss[4], 'FIELD LENGTH': ss[5], 'DESCRIPTION': ss[6]}

    param_dict = {}
    if nParams != 0:
        m = re.search(r'(?<='+param_header+r')[\w\W]+', output)
        param_str_list = m.group(0).split('\n')[2:(2+nParams)]
        symbol_unit_pattern = r'[\w\$\(\)/]+'
        for index, s in enumerate(param_str_list):
            ss = re.search(
                r'([\w/]+) +({0:s}) +({0:s}) +(\w+) +(.+)'.format(
                    symbol_unit_pattern), s).groups()
            param_dict[ss[0]] = {'UNITS': ss[1], 'SYMBOL': ss[2], 'TYPE': ss[3],
                                 'DESCRIPTION': ss[4],
                                 '_index': index}

    # deal with the special cases
    if 'enx0' in param_dict:
        param_dict['enx0']['UNITS'] = '{} {}'.format(param_dict['enx0']['UNITS'],
                                                     param_dict['enx0']['SYMBOL'])
        param_dict['enx0']['SYMBOL'] = param_dict['enx0']['TYPE']
        param_dict['enx0']['TYPE'] = param_dict['enx0']['DESCRIPTION'].split()[0]
        param_dict['enx0']['DESCRIPTION'] = ' '.join(
            param_dict['enx0']['DESCRIPTION'].split()[1:])

    return param_dict, column_dict

#----------------------------------------------------------------------
def printout(sdds_filepath, param_name_list=None,
             column_name_list=None, str_format='',
             show_output=False, show_cmd=False,
             suppress_err_msg=False):
    """
    If "str_format" is specified, you must make sure that all the data
    type of the specified paramter or column name list must be the same.

    An example of "str_format" is '%25.16e'.
    """

    if os.name == 'posix':
        newline_char = '\n'
    elif os.name == 'nt':
        newline_char = '\r\n'

    _, column_info_dict = query(sdds_filepath, suppress_err_msg=suppress_err_msg)
    if column_name_list is None:
        column_name_list = list(column_info_dict)

    if column_name_list == []:
        column_option_str = ''
    else:
        column_option_str = '-columns=' + '(' + ','.join(column_name_list) + ')'
        if str_format != '':
            column_option_str += ",format="+str_format

    if param_name_list is None:
        param_info_dict, _ = query(sdds_filepath, suppress_err_msg=suppress_err_msg)
        param_name_list = list(param_info_dict)

    if param_name_list == []:
        param_option_str = ''
    else:
        param_option_str = '-parameters=' + '(' + ','.join(param_name_list) + ')'
        if str_format != '':
            param_option_str += ",format="+str_format

    if (not column_option_str) and (not param_option_str):
        raise ValueError('You must specify at least one of -columns and -parameters.')

    if os.name == 'nt':
        sp = sdds_filepath.split('\\')
        double_backslashed_sdds_filepath = ('\\'*2).join(sp)
        sdds_filepath = double_backslashed_sdds_filepath

    cmd_list = ['sddsprintout', sdds_filepath]
    if param_option_str:
        cmd_list.append(param_option_str)

    if show_cmd:
        print(cmd_list)
    p = Popen(cmd_list, stdout=PIPE, stderr=PIPE, encoding='utf-8')
    output, error = p.communicate()
    #if isinstance(output, bytes):
        #output = output.decode('utf-8')
        #error = error.decode('utf-8')
    if error and (not suppress_err_msg):
        print('sddsprintout stderr:', error)
        print('sddsprintout stdout:', output)

    if (param_name_list == []): # or (param_name_list is None):
        param_dict = {}
    else:
        if False: # old version
            eq_pattern = ' = '
            eq_ind =  strfind(output, eq_pattern)

            param_name_ind = [[] for i in param_name_list]
            for i,n in enumerate(param_name_list):
                param_name_ind[i] = strfind(output,newline_char+n+' ')
                if param_name_ind[i] == []:
                    param_name_ind[i] = strfind(output,' '+n+' ')

            param_val_list = [0.]*len(eq_ind)
            for i in range(len(eq_ind)-1):
                start_ind = eq_ind[i]+len(eq_pattern)
                if param_name_ind[i+1] == []: continue
                end_ind   = param_name_ind[i+1][0]
                val_str = output[start_ind:end_ind]
                if val_str.strip() == '1.#QNAN0e+000': # Elegant's NaN for old version (23.1.2)
                    val_str = 'nan'
                try:
                    param_val_list[i] = float(val_str)
                except ValueError:
                    param_val_list[i] = val_str
            start_ind = eq_ind[-1]+len(eq_pattern)

            end_ind = start_ind + output[start_ind:].find(newline_char)
            val_str = output[start_ind:end_ind]
            if val_str.strip() == '1.#QNAN0e+000': # Elegant's NaN for old version (23.1.2)
                val_str = 'nan'
            param_val_list[-1] = float(val_str)

            param_dict = dict(zip(param_name_list,param_val_list))
            #print(param_dict)
        else:
            param_dict = {}
            for k, v_str in re.findall(
                #'([\w /\(\)\$]+)[ ]+=[ ]+([nae\d\.\+\-]+)[ \n]?',
                '([\w /\(\)\$]+)[ ]*=[ ]*([naife\d\.\+\-]+)[ \n]?',
                output):
                # ^ [n] & [a] is added for digit matching in cases for "nan"
                #   [i] & [f] is added for digit matching in cases for "inf"

                if '(' in k:
                    first_para_ind = k.index('(')
                    k_stripped = k[:first_para_ind].strip()
                else:
                    k_stripped = k.strip()

                # If the parameter name is picking up the previous parameter's
                # non-digit values as characters, remove those here.
                k_stripped = k_stripped.split()[-1]

                if param_info_dict[k_stripped]['TYPE'] == 'double':
                    param_dict[k_stripped] = float(v_str)
                elif param_info_dict[k_stripped]['TYPE'] == 'long':
                    param_dict[k_stripped] = int(v_str)
                elif param_info_dict[k_stripped]['TYPE'] == 'string':
                    pass
                else:
                    raise ValueError(
                        f'Unexpected TYPE: {param_info_dict[k_stripped]["TYPE"]}')

            # Extract string types
            if 'string' in [q_d['TYPE'] for q_d in param_info_dict.values()]:

                ordered_param_name_list = [None] * len(param_name_list)
                for param_name, q_d in param_info_dict.items():
                    ordered_param_name_list[q_d['_index']] = param_name

                for param_name, q_d in param_info_dict.items():

                    if q_d['TYPE'] != 'string':
                        continue

                    _extracted = re.findall(f'{param_name}[ ]*=[ ]*(.+)[=\n]', output)
                    assert len(_extracted) == 1
                    val = _extracted[0].split('=')[0].strip()

                    try:
                        next_param_name = ordered_param_name_list[
                            ordered_param_name_list.index(param_name)+1]
                        val = val.replace(next_param_name, '').strip()
                    except IndexError:
                        pass

                    #print([param_name, val])
                    param_dict[param_name] = val

    # Check if all the specified parameters have been correctly extracted
    _extracted_param_names = list(param_dict)
    for name in param_name_list:
        if name not in _extracted_param_names:
            print(f'* ERROR: Paramter "{name}" was not extracted')
    for name in _extracted_param_names:
        if name not in param_name_list:
            print(f'* WARNING: Unrequested Parameter "{name}" was extracted')

    cmd_list = ['sddsprintout', sdds_filepath]
    if column_option_str:
        cmd_list.append(column_option_str)

    #use_comma_delimiter = False
    use_comma_delimiter = True
    if use_comma_delimiter:
        cmd_list.append("-spreadsheet=(delimiter=',')")

    if show_cmd:
        print(cmd_list)
    p = Popen(cmd_list, stdout=PIPE, stderr=PIPE, encoding='utf-8')
    output, error = p.communicate()
    #if isinstance(output, bytes):
        #output = output.decode('utf-8')
        #error = error.decode('utf-8')
    if error and (not suppress_err_msg):
        print('sddsprintout stderr:', error)
        print('sddsprintout stdout:', output)

    if (column_name_list == []): # or (column_name_list is None):
        column_dict = {}
    else:
        if not use_comma_delimiter:
            column_title_divider_pattern = '---' + newline_char
            column_start_ind = strfind(output, column_title_divider_pattern)
            if column_start_ind != []:
                column_start_ind = column_start_ind[0] + len(column_title_divider_pattern)
                column_data_str = output[column_start_ind:]
                rows_str = column_data_str.splitlines()
                column_dict = dict.fromkeys(column_name_list)
                for col_name in column_name_list:
                    column_dict[col_name] = [[] for i in rows_str]

                col_ind_offset = 0
                row_counter = 0
                for r in rows_str:
                    str_list = [c for c in r.split(' ') if c]

                    for j,st in enumerate(str_list):
                        column_dict[column_name_list[j+col_ind_offset]][row_counter] = st

                    if ( len(str_list)+col_ind_offset ) != len(column_name_list):
                        col_ind_offset += len(str_list)
                    else:
                        col_ind_offset = 0
                        row_counter += 1

                for col_name in column_name_list:
                    column_dict[col_name] = column_dict[col_name][:row_counter] # Make sure to
                    # remove empty elements at the tail
                    #if col_name not in ('ElementName','ElementType'):
                    try:
                        column_dict[col_name] = str2num(column_dict[col_name])
                    except ValueError:
                        pass
        else:
            rows = [s.strip() for s in output.split('\n') if s.strip() != '']

            column_dict = dict.fromkeys(column_name_list)
            for col_name in column_name_list:
                column_dict[col_name] = []

            col_title_rowind = 1
            for row in rows[(col_title_rowind+1):]:
                for col_name, v in zip(column_name_list, row.split("','")):
                    column_dict[col_name].append(v.strip())

            for col_name in column_name_list:
                if column_info_dict[col_name]['TYPE'] == 'double':
                    column_dict[col_name] = str2num(column_dict[col_name])

    # Check if all the specified columns have been correctly extracted
    _extracted_column_names = list(column_dict)
    for name in column_name_list:
        if name not in _extracted_column_names:
            print(f'* ERROR: Column "{name}" was not extracted')
    for name in _extracted_column_names:
        if name not in column_name_list:
            print(f'* WARNING: Unrequested Column "{name}" was extracted')


    if show_output:
        print(output)
        if error and (not suppress_err_msg):
            print(error)

    return param_dict, column_dict

def sdds2dicts(sdds_filepath):
    """"""

    output, meta = {}, {}

    meta_params, meta_columns = query(sdds_filepath, suppress_err_msg=True)
    params, columns = printout(
        sdds_filepath, param_name_list=None, column_name_list=None,
        str_format='', show_output=False, show_cmd=False,
        suppress_err_msg=True)

    meta = {}
    if meta_params != {}:
        meta['params'] = meta_params
    if meta_columns != {}:
        meta['columns'] = meta_columns

    output = {}
    if params != {}:
        output['params'] = params
    if columns != {}:
        for _k, _v in columns.items():
            columns[_k] = np.array(_v)
        output['columns'] = columns

    return output, meta