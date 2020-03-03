function plotFFT(fileId, centerFreq)
    ff = fread(fileId, [2, inf], 'float');
    ff_real = ff(1,:);
    ff_im = ff(2,:);
    ff_complex = ff_real + (j * ff_im);
    fftA = abs(fftshift(fft(ff_complex)));

    % Filter 0.2 MHz around 8 MHz
    N = length(ff);
    rect = [zeros(1,(centerFreq - 1000000)), ones(1,2000000), zeros(1, (N - (centerFreq + 1000000)))];
    ffilt = abs(fftshift(fft(ff_complex)) .* rect);

    x = (-(N/2) + 1):(N/2);
    
    figure()
%     plot(x, fftA)
    plot(fftA)
    title('Unfiltered FFT')

    figure()
%     plot(x, ffilt)
    plot(ffilt)
    title('Filtered FFT')
end