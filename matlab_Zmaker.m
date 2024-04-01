R = readmatrix('Smat_testing_R.txt');
I = readmatrix('Smat_testing_I.txt');
s_params = R + 1j*I;


z_params = s2z(s_params,83.12100040448948);