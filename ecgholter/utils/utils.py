import json
import datetime
from typing import List, Tuple
from collections.abc import Mapping

from pytz import timezone
import numpy as np
from scipy import interpolate
from scipy.signal import (butter, 
    sosfilt, sosfiltfilt, lfilter,
    iirnotch, filtfilt, medfilt)
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
from yacs.config import CfgNode

# ======================================================== General utils ========================================================
def count_section_sample(list_of_section):
    count = 0
    for section in list_of_section:
        count+= section[1]-section[0] +1

    return count

def round_or_none(number, r):
    if number != None:
        return round(number, r)
    return number

def format_timedelta(td):
    minutes, seconds = divmod(td.seconds + td.days * 86400, 60)
    hours, minutes = divmod(minutes, 60)
    return '{:d}:{:02d}:{:02d}'.format(hours, minutes, seconds)

def sec_to_time_format(sec):
    conversion = datetime.timedelta(seconds=sec)
    return format_timedelta(conversion)    

def update_parameters(parameters_dict, cfg_new):
    for key, value in parameters_dict.items():
        if isinstance(value, Mapping) and value:
            update_parameters(parameters_dict.get(key, value), cfg_new[key])
        elif value is not None:
            cfg_new[key] = parameters_dict[key]
    return cfg_new

def convert_to_dict(cfg_node, key_list=[]):
    _VALID_TYPES = {tuple, list, str, int, float, bool}
    """ Convert a config node to dictionary """
    if not isinstance(cfg_node, CfgNode):

        return cfg_node
    else:
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = convert_to_dict(v, key_list + [k])
        return cfg_dict

def json_to_dict(path):
    with open(path, "r") as fp:
        dict_data = json.load(fp)
    return dict_data

def sec_to_sample(sec, fs, to_int = True):
    if to_int:
        return int(sec*fs)
    return sec*fs

def msec_to_sample(msec, fs, to_int = True):
    if to_int:
        return int(msec*fs/1000)
    return msec*fs/1000

def msec_to_hr(msec):
    return 60/(msec/1000)

def min_to_sample(min, fs):
    return msec_to_sample(min*60*1000, fs)

def sample_to_msec(sample, fs):
    return sample*1000/fs

def sample_to_sec(sample, fs):
    return sample/fs

def sample_to_hr(sample, fs):
    return int(60/sample_to_sec(sample, fs))

def sample_to_minute(sample, fs):
    return sample/fs/60

def minute_to_msec(minute):
    return minute*60*1000

def timestamp_sec_to_datetime(ts, mode = 'bkk', return_object = True):

    if mode == 'utc':
        mode = timezone('UTC')
    elif mode == "gmt":
        mode = timezone('GMT')
    elif mode == "bkk":
        mode = timezone('Asia/Bangkok')
    else:
        mode = timezone('GMT')

    dt_object = datetime.datetime.fromtimestamp(ts, mode)

    if return_object:
        return dt_object

    return dt_object.strftime("%d/%m/%Y %H:%M:%S")

def timestamp_msec_to_datetime(ts, mode = 'bkk', return_object=True):
    return timestamp_sec_to_datetime(ts/1000, mode = mode, return_object = return_object)

def list_of_list_to_list(list_of_list):
    return [element for mylist in list_of_list for element in mylist] 

# ======================================================== signal utils ============================================

# signals mean square
def signal_mean_square(signals, fs, windows = 20):

    windows = int(windows*fs/1000)
    return  ma(signals*signals, windows) 

def signal_mean_absolute(signals, fs, windows = 20):

    windows = int(windows*fs/1000)
    return  ma(np.abs(signals), windows) 

def signal_second_derivative(signals):
    second_dif_signals = np.zeros_like(signals)
    second_dif_signals[1:-1] = np.diff(np.diff(signals))
    return second_dif_signals

def butter_highpass_parameter(highpass, fs,output ='sos',order = 4):
    high = highpass / (0.5 * fs)
    if output == 'sos':
        sos = butter(order, high, btype='high', analog=False, output=output)
        return sos
    elif output == 'ba':
        b, a = butter(order, high, btype='high', analog=False, output=output)
        return b, a

def butter_highpass(signal, highpass, fs, order = 4, output = 'sos', algo = 'filtfilt'):
    if output == 'sos':
        sos = butter_highpass_parameter(highpass, fs, output=output, order = order)
        if algo == 'filtfilt':
            return sosfiltfilt(sos, signal)
        elif algo == 'filt':
            return sosfilt(sos, signal)

    if output == 'ba':
        b, a = butter_highpass_parameter(highpass, fs, output=output, order = order)
        if algo == 'filtfilt':
            return filtfilt(b, a, signal)
        elif algo == 'filt':
            return lfilter(b,a, signal)
    raise ValueError("Filter algorithm not support")

def butter_lowpass_parameter(lowpass, fs,output ='sos',order = 4):
    low = lowpass / (0.5 * fs)
    if output == 'sos':
        sos = butter(order, low, btype='low', analog=False, output=output)
        return sos
    elif output == 'ba':
        b, a = butter(order, low, btype='low', analog=False, output=output)
        return b, a

def butter_lowpass(signal, lowpass, fs, order = 4, output = 'sos', algo = 'filtfilt'):
    if output == 'sos':
        sos = butter_lowpass_parameter(lowpass, fs, output=output, order = order)
        if algo == 'filtfilt':
            return sosfiltfilt(sos, signal)
        elif algo == 'filt':
            return sosfilt(sos, signal)

    if output == 'ba':
        b, a = butter_lowpass_parameter(lowpass, fs, output=output, order = order)
        if algo == 'filtfilt':
            return filtfilt(b, a, signal)
        elif algo == 'filt':
            return lfilter(b,a, signal)
    raise ValueError("Filter algorithm not support")

def notch_filter_parameter(notch_fs, q_factor, fs):
    b, a = iirnotch(notch_fs, q_factor, fs)
    return b, a

def notch_filter(signal, notch_fs, q_factor, fs):
    b, a = notch_filter_parameter(notch_fs, q_factor, fs)
    return lfilter(b, a, signal)

def butter_bandpass(signal, lowpass, highpass, fs, order = 4, output = 'sos', algo = 'filtfilt'):

    if output == 'sos':
        sos = butter(order, [lowpass, highpass], btype='band', analog=False, output=output, fs = fs)
        if algo == 'filtfilt':
            return sosfiltfilt(sos, signal)
        elif algo == 'filt':
            return sosfilt(sos, signal)
    elif output == 'ba':
        b,a = butter(order, [lowpass, highpass], btype='band', analog=False, output=output, fs = fs)
        if algo == 'filtfilt':
            return filtfilt(b, a, signal)
        elif algo == 'filt':
            return lfilter(b,a, signal)
    raise ValueError("Filter algorithm not support")

def ma(signal, sample, algo = 'cumsum'):
    if algo == 'convolve':
        return ma_convolve(signal, sample)
    elif algo == 'cumsum':
        return ma_cumsum(signal, sample)
    
    raise ValueError(f'{algo} not support. Support algo: "convolve", "cumsum"')

def ma_cumsum(signal, sample):

    back = int(sample/2)
    signal = np.pad(signal, (0, back) ,'edge')

    signal = np.cumsum(signal, dtype=float)
    signal[sample:] = signal[sample:] - signal[:-sample]
    
    return signal[back:]/sample

def ma_cumsum2(signal, sample):
    signal = np.cumsum(signal, dtype=float)
    signal[sample:] = signal[sample:] - signal[:-sample]
    
    for i in range(1,sample):
        signal[i-1] = signal[i-1] / i
    signal[sample - 1:]  = signal[sample - 1:] / sample
    
    return signal

def ma_convolve(signal, sample):
    fil = np.ones(sample)/sample
    return  np.convolve(signal,fil, mode='same') 

def union_list_of_section(section_lists: List):
    list_c = []
    for section_list in section_lists:
        list_c = list_c+section_list
        
    new_list = []
    for begin,end in sorted(list_c):
        if new_list and new_list[-1][1] >= begin - 1:
        
            new_list[-1][1] = max(new_list[-1][1], end)
        else:
            new_list.append([begin, end])
    return new_list

def get_section_mask(list_of_sections, signal_length, invert = False):

    mask = np.ones(signal_length, dtype=bool)
    for section in list_of_sections:
        mask[section[0]:section[1]+1] =False
    if invert:
        return np.invert(mask)
    return mask

def segment_normalize(signals: np.ndarray, segment_sample: int, absolute = True, algo = "max") -> np.ndarray:

    # define algorithm for normalized
    if algo == 'max':
        norm_algo = max_norm
    elif algo == 'max_abs':
        norm_algo = max_abs_norm
    elif algo == "sub_mean":
        norm_algo = substact_mean
    elif algo == "sub_median":
        norm_algo = substact_median
    else:
        raise ValueError(f"No algorithm name {algo}")

    n_signal = len(signals)
    section = int(np.floor(n_signal/ segment_sample))

    if absolute:
        signals = np.abs(signals)
    signals_mod = np.zeros(n_signal)

    for i in range(section):
        signals_mod[i*segment_sample: (i+1)*segment_sample]  = norm_algo(signals[i*segment_sample: (i+1)*segment_sample])

    return signals_mod

def substact_mean(signals):
    return signals-np.mean(signals)

def substact_median(signals):
    return signals-np.median(signals)

def max_abs_norm(signals):
    sig_max = np.max(np.abs(signals))
    if sig_max == 0:
        return signals
    return signals/sig_max

def max_norm(signals):
    sig_max = np.max(signals)
    if sig_max == 0:
        return signals
    return signals/sig_max

def min_max_norm(signals):
    sig_max = np.max(signals)
    sig_min = np.min(signals)
    return (signals/(sig_max-sig_min)*2 - 1) *3

def get_index_where_percentile(signals: np.ndarray, upper_percentile = 80.0, lower_percentile = 20.0):

    if len(signals) != 0:
        lower = np.percentile(signals, lower_percentile)
        upper = np.percentile(signals, upper_percentile)

        indices = np.where((signals>=lower)&(signals<=upper))[0]

        return indices
    else:
        return np.array([])

def remove_outlier(signals: np.ndarray, upper_percentile = 80.0, lower_percentile = 20.0) -> np.ndarray:
    """
    Function for remove outlier

    Args:
        signal (np.ndarray): input signal
        upper_percentile (float, optional): [description]. Defaults to 80.0.
        lower_percentile (float, optional): [description]. Defaults to 20.0.

    Returns:
        np.ndarray: input array that remove outlier
    """

    signals = np.array(signals)
    if len(signals) != 0:
        lower = np.percentile(signals, lower_percentile)
        upper = np.percentile(signals, upper_percentile)

        signals = signals[(signals>=lower)&(signals<=upper)]

        return signals
    else:
        return np.array([])


def mask_segment(signals: np.ndarray, edge: int, n_signals: int, threshold = 1000.0) -> List[Tuple[int, int]]:
    """
    Find segment in signal which greater than threshold.
    (for removed lost signal, noise etc.)

    Args:
        signals (np.ndarray): input signal
        threshold (float, optional): 

    Returns:
        List[Tuple[int, int]]: list of questionmark segment
    """

    clip_binary = np.where(signals > threshold)[0]
    clipping_edges = np.where(np.diff(clip_binary) > 1)[0]

    question_segments = []
    maxidx = n_signals - 1

    for i in range(0, len(clipping_edges)):
        # first clipping segment
        if i == 0:
            question_segments.append(( max(clip_binary[0]-edge,0), 
                                    min(clip_binary[clipping_edges[0]]+edge,maxidx ) ))
        else:
            question_segments.append(( max(clip_binary[clipping_edges[i-1] + 1]-edge,0),
                                    min(clip_binary[clipping_edges[i]]+edge, maxidx)  ))
    # append last segment
    if len(clip_binary)>0 and len(clipping_edges) > 0:
        question_segments.append(( max(clip_binary[clipping_edges[-1] + 1]-edge, 0),
                                    min(clip_binary[-1]+edge, maxidx)   ))
    elif len(clip_binary)>0 and len(clipping_edges) == 0:
        question_segments.append(( max(clip_binary[0]-edge, 0 ),
                                    min(clip_binary[-1]+edge, maxidx)   ))

    return question_segments


def simple_connect_segment(question_segments: List[Tuple[int, int]]) -> List[Tuple[int, int]]:

    """
    Connect questionmark segment if spacing is 
    nearer than 'num_used_signal_for_interpolated'

    Args:
        question_segments (List[Tuple[int, int]]): list of questionmark segment
        num_used_signal_for_interpolated (int):

    Returns:
        List[Tuple(start, end)]: new list of questionmark segment
    """
    n_segment = len(question_segments)
    new_question_segments = []

    idx = 0

    while idx < n_segment:
        if idx == n_segment-1:
            new_question_segments.append(question_segments[idx])
            break
        next_idx = idx+1

        endpoint = question_segments[idx][1]
        while endpoint >= question_segments[next_idx][0]-1:
            endpoint = question_segments[next_idx][1]
            next_idx +=1
            if next_idx>=n_segment:
                break

        new_question_segments.append(( question_segments[idx][0], question_segments[next_idx-1][1])) 

        idx = next_idx
 
    return  new_question_segments

def simple_const_signal(signals, question_segments):
    """[summary]
    Used after simple_connect_segments
    """

    n_signals = len(signals)
    n_segment = len(question_segments)

    for i, segment in enumerate(question_segments):
        start = segment[0]
        end = segment[1]+1

        if start == 0 and end >= n_signals-1:
            return np.zeros_like(signals), []

        if start == 0:
            signals[start:end] = signals[end+1]

        elif end == n_signals:
            signals[start:end] = signals[start-1]

        else:
            signals[start:end] = (signals[end+1] + signals[start-1])/2

    return signals, []

def interpolate_signal(signals: np.ndarray, question_segments,
    num_used_signal_for_interpolated) -> Tuple[np.ndarray, List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Args:
        signals (np.ndarray): input signal
        num_used_signal_for_interpolated (int): 
            length of signal that used for interpolated.
        threshold (int, optional): signal greater than threshold 
            will be identified as noise.

    Returns:
        np.ndarray: interpolate_signal
        List[Tuple(start, end)]: list of questionmark segment
        List[Tuple(start, end)]: list of unclean segment
    """
    def cannot_interpolated_segment(signals: np.ndarray, segment: Tuple[int, int], interpolate = False):

        if segment[0]-1 >= 0:
            signals[segment[0]:segment[1]+1] = signals[segment[0]-1]
        elif segment[1] +1 < len(signals):
            signals[segment[0]:segment[1]+1] = signals[segment[1] +1]
        else:
            signals[segment[0]:segment[1]+1] = 0
    
    n_segment = len(question_segments)
    n_signal = len(signals)

    unclean_segment = []
    
    for i, segment in enumerate(question_segments):
        # No previous data for interpolated
        if segment[0] < num_used_signal_for_interpolated: 
            unclean_segment.append(segment)
            cannot_interpolated_segment(signals, segment)
            continue
        
        # No forward data for interpolated
        if segment[1] + 1 + num_used_signal_for_interpolated > n_signal:
            unclean_segment.append(segment)
            cannot_interpolated_segment(signals, segment)
            continue
        
        # previous data cannot be cleaned
        if len(unclean_segment)>0:
            if segment[0] - num_used_signal_for_interpolated <= unclean_segment[-1][1]:
                unclean_segment.append(segment)
                cannot_interpolated_segment(signals, segment)
                continue

        # forward data is noise or unclean
        if i < n_segment - 1:
            if segment[1] + num_used_signal_for_interpolated >= question_segments[i+1][0]:
                unclean_segment.append(segment)
                cannot_interpolated_segment(signals, segment)
                continue

        if interpolate:
        
            front_signal = signals[segment[0] - num_used_signal_for_interpolated : segment[0]]
            back_signal = signals[segment[1] + 1 : segment[1] + num_used_signal_for_interpolated + 1]
            interpdata_y = np.concatenate((front_signal, back_signal))

            interpdata_x = np.concatenate(([x for x in range(segment[0] - num_used_signal_for_interpolated, segment[0])],
                                                [x for x in range(segment[1] +1, segment[1] + num_used_signal_for_interpolated +1)]))
            x_new = np.linspace(segment[0] - num_used_signal_for_interpolated,
                                segment[1] + num_used_signal_for_interpolated +1,
                                ((segment[1] - segment[0]) + (2 * num_used_signal_for_interpolated) +1 ))
    
            try:
                interp_func = UnivariateSpline(interpdata_x, interpdata_y, k=3)
                interp_data = interp_func(x_new)
                
                signals[segment[0] - num_used_signal_for_interpolated :
                    segment[1] + num_used_signal_for_interpolated +1 ] = interp_data
            except Exception as e:
                unclean_segment.append(segment)
                cannot_interpolated_segment(signals, segment)

        else:
            unclean_segment.append(segment)
            cannot_interpolated_segment(signals, segment)
    
    return signals, unclean_segment


def connect_segment(question_segments: List[Tuple[int, int]],
        num_used_signal_for_interpolated: int) -> List[Tuple[int, int]]:

    """
    Connect questionmark segment if spacing is 
    nearer than 'num_used_signal_for_interpolated'

    Args:
        question_segments (List[Tuple[int, int]]): list of questionmark segment
        num_used_signal_for_interpolated (int):

    Returns:
        List[Tuple(start, end)]: new list of questionmark segment
    """
    n_segment = len(question_segments)
    new_question_segments = []
    
    idx = 0

    while idx < n_segment:
        if idx == n_segment-1:
            new_question_segments.append(question_segments[idx])
            break
        next_idx = idx+1

        endpoint = question_segments[idx][1]
        while endpoint + num_used_signal_for_interpolated >= question_segments[next_idx][0]:
            endpoint = question_segments[next_idx][1]
            next_idx +=1
            if next_idx>=n_segment:
                break

        new_question_segments.append(( question_segments[idx][0], question_segments[next_idx-1][1])) 

        idx = next_idx
 
    return  new_question_segments


def get_segment_index(index_array, segment_size):
    array_size = len(index_array)
    if array_size ==0:
        raise ValueError("array size cannot lower or equal to 1")
    elif array_size ==1:
        return [array_size] 

    segment_index = []
    segment_count = int(np.floor(index_array[0]/segment_size))
    if segment_count ==0:
        segment_count = 1

    for i, idx in enumerate(index_array):
        if idx > segment_size*segment_count:
            if i != 0:
                segment_index.append(i)
            segment_count += 1

        while idx > segment_size*segment_count:
            segment_count+=1

    if len(segment_index) == 0:
        segment_index.append(len(index_array))

    if len(index_array[segment_index[-1]:]) >0:
        segment_index.append(len(index_array))

    
    return segment_index



# ======================================================== visualize utils ========================================================

def plot_interactive(signals):
    plt.close()
    plt.plot(signals)
    plt.show()

def plot_r_wave(signal, peakarray, fs, show = True, figsize = (8,3), show_t = True, show_mu = True, show_j = True):
    plt.close()

    if isinstance(signal, list):
        signal_array = signal
        signal = signal_array[0]
    else:
        signal_array = signal

    signal = np.array(signal)
    plotx = np.arange(0, len(signal)/fs, 1/fs)
    #check if there's a rounding error causing differing lengths of plotx and signal
    diff = len(plotx) - len(signal)
    if diff < 0:
        #add to linspace
        plotx = np.append(plotx, plotx[-1] + (plotx[-2] - plotx[-1]))
    elif diff > 0:
        #trim linspace
        plotx = plotx[0:-diff]
    
    plt.figure(figsize= figsize)
    plt.ylim(-100,100)
    if isinstance(signal_array, list):
        for sig in signal_array:
            plt.plot(plotx, sig, label='heart rate signal', zorder=-10)
    else:
        plt.plot(plotx, signal, color='b', label='heart rate signal', zorder=-10)

    x_location = peakarray.get_r_index()
    y_location = signal[x_location]

    plt.scatter(np.asarray(x_location)/fs, y_location, color='g')

    if show_t:
        if peakarray.T_WAVE:
            x_location = peakarray.get_t_index()
            y_location = signal[x_location]
            plt.scatter(np.asarray(x_location)/fs, y_location, color='r')

            x_location = peakarray.get_t_end_index()
            y_location = signal[x_location]
            plt.scatter(np.asarray(x_location)/fs, y_location, color='b')

    if show_mu:
        if peakarray.QRS_ONSET:
            x_location = peakarray.get_qrs_onset_index()
            y_location = signal[x_location]
            plt.scatter(np.asarray(x_location)/fs, y_location)

    if show_j:
        if peakarray.QRS_OFFSET:
            
            x_location = peakarray.get_qrs_offset_index()
            y_location = signal[x_location]
            plt.scatter(np.asarray(x_location)/fs, y_location)

    if show:
        plt.show()


def plot_diag(signal, peakarray, fs, show = True, figsize = (8,3),
             ventricular = True, pvc =True, normal = True, unknown = True, pause =True):
    plt.close()
    if isinstance(signal, list):
        signal_array = signal
        signal = signal_array[0]
    else:
        signal_array = signal

    signal = np.array(signal)
    plotx = np.arange(0, len(signal)/fs, 1/fs)
    #check if there's a rounding error causing differing lengths of plotx and signal
    diff = len(plotx) - len(signal)
    if diff < 0:
        #add to linspace
        plotx = np.append(plotx, plotx[-1] + (plotx[-2] - plotx[-1]))
    elif diff > 0:
        #trim linspace
        plotx = plotx[0:-diff]
    
    plt.figure(figsize= figsize)
    plt.ylim(-100,100)
    if isinstance(signal_array, list):
        for sig in signal_array:
            plt.plot(plotx, sig, label='heart rate signal', zorder=-10)
    else:
        plt.plot(plotx, signal, color='b', label='heart rate signal', zorder=-10)

    all_labels = peakarray.get_all_labels(to_timestamp = False)

    if ventricular:
        x_location = all_labels['ventricular']
        y_location = signal[x_location]

        plt.scatter(np.asarray(x_location)/fs, y_location, color='r')

    if pvc:
        x_location = all_labels['pvc']
        y_location = signal[x_location]

        plt.scatter(np.asarray(x_location)/fs, y_location, color='g')

    if normal:
        x_location = all_labels['normal']
        y_location = signal[x_location]

        plt.scatter(np.asarray(x_location)/fs, y_location, color='b')

    if unknown:
        x_location = all_labels['unknown']
        y_location = signal[x_location]

        plt.scatter(np.asarray(x_location)/fs, y_location, color='y')

    if pause:
        x_location = all_labels['pause']
        y_location = signal[x_location]

        plt.scatter(np.asarray(x_location)/fs, y_location, color='y')

    if show:
        plt.show()



