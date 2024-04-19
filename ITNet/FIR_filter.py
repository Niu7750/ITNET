# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 10:32:24 2020

@author: 77509
"""


import numpy as np
import math as m
from scipy.fftpack import fft
import matplotlib.pyplot as plt


def my_fft(data, fs):
    """
    Fast Fourier Transform

    input:
        data: 1-dimension signal
        fs: the sampling frequency of the signal (Hz)

    output:
        fre: the frequency coordinates
        mode: the amplitude for each discrete frequency point
        ang: the phase for each discrete frequency point
    """

    L = len(data)
    N = np.power(2, np.ceil(np.log2(L))) * 4
    N = N.astype('int')
    FFT_y = (fft(data, N)) / L * 2
    fre = np.arange(int(N / 2)) * fs / N
    FFT_y = FFT_y[range(int(N / 2))]

    mode = abs(FFT_y)
    ang = np.angle(FFT_y)
    return fre, mode, ang


def my_filter(fs, fl, fh, length):
    """
    The FIR band-pass filters based on window function method, where rectangular window is used

    input:
        fs: the sampling frequency (Hz)
        fl: lower-cut-off frequency (Hz)
        fh: higher-cut-off frequency (Hz)
        length: the length of the designed filter (need be set as odd)
    """

    wl = fl/fs*2*m.pi
    wh = fh/fs*2*m.pi
    
    middle = (length-1)/2
    fliter = np.zeros(length)
    
    for i in range(length):
        if i == middle:
            fliter[i] = (wh - wl)
        else:
            fliter[i] = m.sin(wh*(i-middle))/(m.pi*(i-middle)) - m.sin(wl*(i-middle))/(m.pi*(i-middle))

    return fliter


if  __name__ == '__main__':
    filter = my_filter(fs=250, fl=12, fh=14, length=51)
    plt.plot(filter) # plot the elements of the filter array

    Fre, mode, ang = my_fft(filter, 250)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(Fre, mode) # plot amplitude-frequency response
    plt.subplot(1, 2, 2)
    plt.plot(Fre, ang) # plot phase-frequency response
    plt.show()






