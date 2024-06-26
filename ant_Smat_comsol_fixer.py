import csv
import numpy as np

# user must specify the frequencies
freqs = np.arange(81,112,1)
filename_label = 'bigscan_full'
filename = f'{filename_label}.csv'
filename_save = f'circ_model/fixed_{filename_label}.csv'


with open(filename, 'r') as f:
    lines = f.readlines()

freq = freqs[0]
freq_i = -1
fmat_string = []
for row in lines:
    if row[0] == ',':
        row = str(freq) + row
        fmat_string.append(row)

    elif (row[0][0] == '%') and (freq_i == -1): # only save headers if its the first time 
        fmat_string.append(row)

    elif row[0][0] == '%':
        pass

    else:
        fmat_string.append(row)
        freq_i += 1
        freq = freqs[freq_i]
        

with open(filename_save, 'w') as f:
    f.writelines(fmat_string)

