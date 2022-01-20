
import numpy as np

from .utils import signal_mean_absolute, signal_second_derivative
from .config import cfg
from .classifier import PeakClassifier
from .peak_detector import preprocess, PeakDetector
from . import metric

def step24hour(signals, start_ts, return_peakarray = False, return_preprocess = False,
        resample_signals = False, lowpass = True):

    classifier = PeakClassifier()
    detector = PeakDetector()

    try:
        signals, question_segments = preprocess(signals, resample_signals=resample_signals, lowpass = lowpass)
    except Exception as e:
        raise Exception(f"Preprocessing error: {str(e)}")

    # Get peakarray
    try:
        peakarray = detector.detect_peaks(signals)
    except Exception as e:
        raise Exception(f"Detector error: {str(e)}")

    # Detect component
    try:
        signals_abs = signal_mean_absolute(signals, cfg.FS)
        detector.detect_r_peak(signals_abs, peakarray, question_segments)
    except Exception as e:
        raise Exception(f"Detector error: {str(e)}")

    try:
        second_dif_signals = signal_second_derivative(signals_abs)
        detector.detect_qrs_onset_offset(second_dif_signals, peakarray)
    except Exception as e:
        raise Exception(f"QRS onset offset detection error: {str(e)}")

    # ------------------ -------------- diagnosis

    try:
        # using time segment
        segment_index = peakarray.get_segment_index_every_nsec(nsec=cfg.DIAG.HR_TIME_SEGMENT_SEC)
        classifier.tachy_brady(peakarray, segment_index)

        ## using n peak
        # classifier.tachy_brady(peakarray)
    except Exception as e:
        raise Exception(f"TachyBrady error: {str(e)}")

    try:
        if cfg.DIAG.ECTOPIC_TIME_SEGMENT_SEC != cfg.DIAG.HR_TIME_SEGMENT_SEC:
            segment_index = peakarray.get_segment_index_every_nsec(nsec=cfg.DIAG.ECTOPIC_TIME_SEGMENT_SEC)
        classifier.ectopic(peakarray, segment_index)
    except Exception as e:
        raise Exception(f"Ectopic error: {str(e)}")

    try:
        classifier.ventricular_arrhythmia(signals, peakarray)
    except Exception as e:
        raise Exception(f"VTVF error: {str(e)}")

    try:
        detector.detect_t_second_derivative(signals_abs, second_dif_signals, peakarray)
    except Exception as e:
        raise Exception(f"T detection error: {str(e)}")

    # ------------------------------

    peakarray.assign_unknown_to_peak()
    peakarray.assign_start_ts(start_ts)
    # ------------------------------

    # get label
    try:
        all_labels = peakarray.get_all_labels()
    except Exception as e:
        raise Exception(f"Get labels error: {str(e)}")

    try:
        all_metrics = peakarray.get_all_metrics(rm_outlier=True, upper_percentile=95, lower_percentile = 10)
    except Exception as e:
        raise Exception(f"Calculate metric error: {str(e)}")

    try:
        # numpy array cannot to json
        interval_arrays, index_arrays = peakarray.get_interval(to_time = True, report = True)

    except Exception as e:
        raise Exception(f"Get interval error: {str(e)}")

    result = {'diag': all_labels, 
            'interval': {'rr':interval_arrays, 'index': index_arrays},
           'metrics': all_metrics,
           'detail': 'Success'
        }
    
    if return_preprocess and return_peakarray:
        return result, peakarray, signals

    if return_peakarray:
        return result, peakarray

    if return_preprocess:
        return result, signals

    return result


def step10sec(signals, return_peakarray = False):
    try:
        signals, _ = preprocess(signals)
    except Exception as e:
        raise Exception("PREPROCESSING ERROR: " + str(e))


    try:
        detector = PeakDetector()
        peakarray = detector.detect_peaks(signals)
    except Exception as e:
        raise Exception("DETECTION ERROR: " + str(e))

    try:
        rr_interval, _ = peakarray.get_interval(nn=False, to_time=True, report = True)
        hr = metric.hr(rr_interval)
    except Exception as e:
        raise Exception("HR ERROR: " + str(e))

    # TODO QTc

    result = {
        'hr': round(hr, cfg.ROUND)
    }

    if return_peakarray:
        return {'metrics': result, 'detail': 'success'}, peakarray

    return {'metrics': result, 'detail': 'success'}
