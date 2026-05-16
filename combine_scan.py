import numpy as np

# user must specify the frequencies

folder_name = 'lab3_5_55cmdvac/'
file_prefix = 'lab3_5_55cmdvac_'
save_file = 'lab3_5_55cmdvac.csv'

freqs = np.array([92, 93, 94, 95, 96, 97, 98, 99, 100])

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