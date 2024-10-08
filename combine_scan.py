import numpy as np

# user must specify the frequencies

folder_name = 'bigantscan_CF_13strap_2passive/'
file_prefix = 'bigantscan_CF_13strap_2passive_'
save_file = 'bigscan_full_CF_13strap_2passive.csv'

freqs = np.arange(81,112,1)

lines_out = []
for fr in freqs:
    filename = folder_name + file_prefix + str(fr) + '.csv'

    with open(filename, 'r') as f:
        lines = f.readlines()

    for line in lines:
        lines_out.append(line)
    
#print(len(lines))
# Open the output file in write mode
output_file = open(save_file, 'w')
print(type(lines_out[0][0]))
# Write each line from the list back to the output file
for line in lines_out:
    output_file.write(line)

# Close the output file
output_file.close()