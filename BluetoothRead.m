% fileId = fopen('usrp_samples_2402.dat');
% plotFFT(fileId, 8000000)

% fileId = fopen('usrp_samples_2426.dat');
% plotFFT(fileId, 12000000)

% fileId = fopen('usrp_samples_2480.dat');
% plotFFT(fileId, 10000000)

% fileId = fopen('usrp_samples_2480_00.dat');
% plotFFT(fileId, 10000000)

fileId = fopen('usrp_samples_2480_01.dat');
plotFFT(fileId, 10000000)

%%
% Eric: FC:FC:48:A6:6C:5E
% Evan: 3C:28:6D:E0:99:3B
% Demba: A4:50:46:BA:97:ED
% David: 2C:33:61:97:C6:0E
%%
fileId = fopen('usrp_samples_2480_01.dat');
ff = fread(fileId, [2, inf], 'float');
ff_real = ff(1,:);
ff_im = ff(2,:);
ff_complex = ff_real + (j * ff_im);
% Filter 0.2 MHz around 8 MHz
N = length(ff);
centerFreq = 10000000;
rect = [zeros(1,(centerFreq - 1000000)), ones(1,2000000), zeros(1, (N - (centerFreq + 1000000)))];
ffilt = ifft(fft(ff_complex)) .* rect;
[bits, accessAddr] = bleIdealReceiver(ffilt');
match = ["["," ","]"];
addr = erase(mat2str(accessAddr'),match);
% addr = '98:E3:E3:E6'
addr = dec2hex(bin2dec(addr));
addr = [addr(1:2),':',addr(3:4),':',addr(5:6),':',addr(7:8)]
%%
fileId = fopen('usrp_samples_2480_00.dat');
ff = fread(fileId, [2, inf], 'float');
ff_real = ff(1,:);
ff_im = ff(2,:);
ff_complex = ff_real + (j * ff_im);
[bits, accessAddr] = bleIdealReceiver(ff_complex');
match = ["["," ","]"];
addr = erase(mat2str(accessAddr'),match);
% addr = '41:2A:66:56'
addr = dec2hex(bin2dec(addr));
addr = [addr(1:2),':',addr(3:4),':',addr(5:6),':',addr(7:8)]
%%
fileId = fopen('usrp_samples_2480.dat');
ff = fread(fileId, [2, inf], 'float');
ff_real = ff(1,:);
ff_im = ff(2,:);
ff_complex = ff_real + (j * ff_im);
[bits, accessAddr] = bleIdealReceiver(ff_complex');
match = ["["," ","]"];
addr = erase(mat2str(accessAddr'),match);
% addr = 
addr = dec2hex(bin2dec(addr));
addr = [addr(1:2),':',addr(3:4),':',addr(5:6),':',addr(7:8)]