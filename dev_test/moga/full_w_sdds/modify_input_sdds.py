from pyelegant import sdds

data, meta = sdds.sdds2dicts('optim1.sdds')

data['params']['runScript'] = 'runJob1.py'

#data['params']['childMultiplier'] = 1

sdds.dicts2sdds('mod_optim1.sdds', params=data['params'], columns=data['columns'],
                outputMode='ascii', suppress_err_msg=False)