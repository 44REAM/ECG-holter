import math

import numpy as np

from .utils import *

def entropy(histogram_bin):
    histogram_bin = np.array(histogram_bin)

    if histogram_bin.size == 0:
        raise ValueError("Array cannot be empty")

    histogram_bin =histogram_bin[histogram_bin!=0]
 
    n_classes = histogram_bin.size

    if n_classes <= 1:
        return 0

    probs = histogram_bin / np.sum(histogram_bin)

    value = 0.

    for i in probs:
        value -= i * np.log(i)

    return value

def sdann_asdnn_old(nn_interval: np.ndarray, index_array: np.ndarray, min_number_of_interval = 3, duration = 5):
    """[summary]

    Args:
        nn_interval (np.ndarray): interval array in millisecond
        index_array (np.ndarray): index array in millisecond
        min_number_of_interval: minimum number of interval for calculate
        duration (int, optional): Duration in minute

    Returns:
        [type]: [description]
    """

    section_msec = minute_to_msec(duration)

    # number of 5 minute sections
    n_section = np.ceil(last_idx = (index_array[-1])/section_msec)

    start_idx = 0

    average_array = []
    std_array = []

    for i in range(n_section):
        end_section = (i+1)*section_msec
        end_idx = np.searchsorted(index_array, end_section,side='right')
        # calculate only if have peak greater than specified
        if len(nn_interval[start_idx:end_idx]) >= min_number_of_interval:

            average_array.append(np.mean(nn_interval[start_idx:end_idx]))
            std_array.append(np.std(nn_interval[start_idx:end_idx]))

        start_idx = end_idx

    if len(average_array) >= min_number_of_interval:
        sdann = np.std(average_array)
        asdnn = np.mean(std_array)
        return sdann, asdnn

    return None, None

# faster than sdann_asdnn_old because dont have to search for index
def sdann_asdnn(nn_interval: np.ndarray, index_array: np.ndarray, min_number_of_interval = 3, duration = 5):
    """[summary]

    Args:
        nn_interval (np.ndarray): interval array in millisecond
        index_array (np.ndarray): index array in millisecond
        min_number_of_interval: minimum number of interval for calculate
        duration (int, optional): Duration in minute

    Returns:
        [type]: [description]
    """
    if len(nn_interval) < min_number_of_interval:
        return None, None
    section_msec = minute_to_msec(duration)
    segment_index = get_segment_index(index_array, section_msec)

    average_array = []
    std_array = []

    start_idx = 0
    for end_idx in segment_index:
        if len(nn_interval[start_idx:end_idx]) >= min_number_of_interval:

            average_array.append(np.mean(nn_interval[start_idx:end_idx]))
            std_array.append(np.std(nn_interval[start_idx:end_idx]))

        start_idx = end_idx

    if len(average_array) >= min_number_of_interval:
        sdann = np.std(average_array)
        asdnn = np.mean(std_array)
        return sdann, asdnn

    return None, None

def sdnn(nn_interval: np.ndarray, min_number_of_interval = 3):
    """[summary]

    Args:
        nn_interval (np.ndarray): interval array in millisecond
        min_number_of_interval: minimum number of interval for calculate

    """
    if len(nn_interval) < min_number_of_interval:
        return None

    std = np.std(nn_interval)
    return std


def rmssd(rr_interval, min_number_of_interval = 3):
    """[summary]

    Args:
        nn_interval (np.ndarray): interval array in millisecond
        min_number_of_interval: minimum number of interval for calculate

    """
    if len(rr_interval) >= min_number_of_interval+1:
        rr_dif = np.diff(rr_interval)

        # return np.sqrt(np.sum(np.array( rr_dif )**2)/len(rr_dif))
        return np.std(rr_dif)
    return None

def hr(rr_interval):
    if len(rr_interval) == 0:
        return None

    mean_rr = np.mean(rr_interval)
    return msec_to_hr(mean_rr)

# ***********************************************************************
# ******************** Ventricular Arrhythmia ***************************
# ***********************************************************************

def tcsc(signals: np.ndarray, segment_sample: int, decision_sample: int, threshold = 0.2, mask = None):
    """
    Arafat MA, Chowdhury AW, Hasan MdK. 
    A simple time domain algorithm for the detection of 
    ventricular fibrillation in electrocardiogram. 
    SIViP. 2011. (With some modification)

    Time domain features
    ** For detect ventricular arrhythmia.

    Not required QRS detection first.
    The signal should be noise free.
    """

    signals = segment_normalize(signals ,segment_sample, absolute=True)

    binary_array = np.zeros_like(signals)
    binary_array[signals >threshold] = 1

    if mask is None:
        return ma(binary_array, decision_sample)
    
    binary_array[mask] = 0

    return ma(binary_array, decision_sample)

def mav(signals: np.ndarray, segment_sample: int, decision_sample: int):
    """
    Anas EMA, Lee SY, Hasan MK. 
    Sequential algorithm for life threatening 
    cardiac pathologies detection based on mean
    signal strength and EMD functions. 
    BioMed Eng OnLine. 2010 .

    Time domain features
    ** For detect ventricular arrhythmia.

    Not required QRS detection first.
    The signal should be noise free.
    """
    signals = segment_normalize(signals ,segment_sample, absolute=True)

    return ma(signals, decision_sample)

def vf_filter(signals: np.ndarray, segment_sample: int):
    """
    Amann A, Tratnig R, Unterkofler K. 
    Reliability of old and new ventricular fibrillation 
    detection algorithms for automated external defibrillators. 
    BioMed Eng OnLine. 2005

    Can be view as frequency domain feature.
    Use filter leakaged.
    ** For detect only ventricular fibrillation.

    Not required QRS detection first.
    The signal should be noise free.
    """
    n_signal = len(signals)
    section = int(np.floor(n_signal/ segment_sample))

    l_array = []

    for i in range(1, section):
        signal_section = signals[i*segment_sample: (i+1)*segment_sample]

        numerator = np.abs(signal_section).sum()
        denominator = np.abs(np.diff(signal_section)).sum()

        try:

            if denominator == 0:
                N = 0
            else:
                N  = np.pi * numerator / denominator + 1/2
                N = int(np.floor(N))
        except Exception as e:
            l_array.append(1)
            continue

        if N > segment_sample:
            # raise ValueError("N cannot greater than section size")

            # TODO
            # Something error 
            # the signal frequency are too low so the N larger than segment_sample
            # this because set signal to constant when preprocessing to remove signal threshold greater than ...
            # append 1 for now
            l_array.append(1)
            continue

        signal_section_shift = signals[i*segment_sample - N: (i+1)*segment_sample - N]

        numerator = np.abs(signal_section + signal_section_shift).sum()
        denominator = np.sum(np.abs(signal_section) + np.abs(signal_section_shift))

        if denominator == 0:
            # TODO
            # Something error 
            # Reason as above
            l_array.append(1)
            continue

        l = numerator / denominator
        
        l_array.append(l)
    
    return np.array(l_array)



