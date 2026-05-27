import numpy as np

# user must specify the frequencies

folder_name = 'lab6_11cmdeembed/'
file_prefix = 'lab6_11cmdeembed_'
save_file = 'lab6_11cmdeembed.csv'

# freqs = np.array(['92', '93', '94', '95', '96', '97', '98', '99', '100'])
freqs = np.array(['92', '92_5', 
                  '93', '93_5', 
                  '94', '94_5',
                  '95', '95_5', 
                  '96', '96_5', 
                  '97', '97_5',
                  '98', '98_5',
                  '99', '99_5',
                  '100'])

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