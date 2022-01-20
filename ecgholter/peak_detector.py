from typing import List, Tuple

import numpy as np
from scipy import signal
from scipy.interpolate import UnivariateSpline
import pywt
from scipy.signal import resample

from .peakobject import PeakArray, Peak, QRS
from .utils import *
from .config import cfg

def preprocess(signals: np.ndarray,
             threshold: float = 6000, detrend = True, minmax_threshold = 50,
             notch = False, resample_signals = False, normalize = False,
             highpass = False, lowpass = False, reconstruct = 'none', detrend_algo = 'cumsum'):

    num_used_signal_for_interpolated = msec_to_sample(cfg.INTERPOLATED_MSEC, cfg.FS)

    # mask segment with lost signal
    edge = sec_to_sample(2, cfg.FS)
    n_signals = len(signals)
    question_segments = mask_segment(signals,edge, n_signals, threshold = threshold)

    # connect loss or noise segment that near together
    #question_segments = connect_segment(question_segments, num_used_signal_for_interpolated)
    question_segments = simple_connect_segment(question_segments)

    # interpolated
    # signals, unclean_segment = interpolate_signal(signals, question_segments,
    #                     num_used_signal_for_interpolated)
    signals, unclean_segment = interpolate_signal(signals, question_segments,
                        num_used_signal_for_interpolated)

    if resample_signals:
        new_sample = int(cfg.FS_RESAMPLE/cfg.FS*len(signals))
        signals = resample(signals, new_sample)
        cfg['FS'] = cfg['FS_RESAMPLE']

    if detrend:
        signals = signals - ma(signals, msec_to_sample(600, cfg.FS), algo=detrend_algo)

    # Remove artifact
    #TODO
    question_segments2 = mask_segment(signals,edge, n_signals, threshold = 50)
    question_segments2 = simple_connect_segment(question_segments2)
    signals, unclean_segment = interpolate_signal(signals, question_segments2,
                        num_used_signal_for_interpolated)

    question_segments = union_list_of_section([question_segments, question_segments2])

    if normalize:
        sample = sec_to_sample(cfg.NORMALIZE_SEC, cfg.FS)
        signals = segment_normalize(signals, sample, algo="max_abs", absolute=False)

    if highpass:
        signals = butter_highpass(signals, 1, cfg.FS, order = 2)

    if lowpass:
        signals = butter_lowpass(signals, 25, cfg.FS, order = 2)

    if notch:
        signals = notch_filter(signals, cfg.NOTCH.FS, cfg.NOTCH.QF, cfg.FS)

    if reconstruct == 'none':
        pass
    elif reconstruct == 'absolute':
        signals = signal_mean_absolute(signals, cfg.FS )
    elif reconstruct == 'square':
        signals = signal_mean_square(signals, cfg.FS)

    return signals, question_segments

class PeakDetector:

    def detect_peaks(self, signals: np.ndarray,
        return_all_peak = False, return_enhanced = False,
        smooth = False, normalize = False, highpass = False, ma_algo = "cumsum"):
        """
            -> enhance QRS for detection
            -> enhance further: normalize detrend etc.
            -> detect R peak
        """

        # enhancing ECG signal for detection
        signals = self.ecg_enhancer(signals, enhancer = cfg.DETECTOR.ENHANCER)

        if smooth:
            msec_to_sample
            signals = signals - ma(signals, msec_to_sample(400, cfg.FS), algo = ma_algo)

        if highpass:
            signals = butter_highpass(signals, 1.5, cfg.FS, order = 4)

        # dont recomment if signal height have large variation
        if normalize:
            signals = min_max_norm(signals)

        # detect r peak
        labelpeaks, peak_idx_array = self.peak_detector(signals, detector = cfg.DETECTOR.DETECTOR, ma_algo=ma_algo)


        if return_all_peak and return_enhanced:
            return labelpeaks, peak_idx_array, signals

        if return_all_peak:
            return labelpeaks, peak_idx_array

        if return_enhanced:
            return labelpeaks, signals

        return labelpeaks

    @staticmethod
    def detect_r_peak(signals_abs, peakarray: PeakArray, question_segments: List[Tuple]):
        r_search_range = msec_to_sample(50, cfg.FS)

        for peak in peakarray:
            if not isinstance(peak, QRS):
                continue

            if not peak.next_interval:
                continue

            if not peak.interval:
                continue

            start_idx = max(peak.idx - r_search_range, 0)
            end_idx = min(peak.idx + r_search_range, len(signals_abs))

            peak.idx = np.argmax(signals_abs[start_idx: end_idx]) + start_idx

        # add question mark peak
        peakarray.add_questionmask_section(question_segments)

        # automatically assign interval to peak
        peakarray.assign_interval_to_peak()

# ***********************************
# *********** ECG enhancer **********
# ***********************************

    def ecg_enhancer(self, signals: np.ndarray, enhancer = "swt_enhancer", smooth = True) -> np.ndarray:
        """For enhancing signal

        Args:
            signals (np.ndarray): input signal
            enhancer (str, optional): Selected algorithm for enhancing signal.

        Returns:
            np.ndarray: enhanced signal
        """
        if enhancer == "swt_enhancer":
            signals = self.swt_enhancer(signals)
        elif enhancer == "modified_swt_enhancer":
            signals = self.modified_swt_enhancer(signals)
        elif enhancer == "pan_tompkins_enhancer":
            signals = self.pan_tompkins_enhancer(signals)
        elif enhancer == "modified_pan_tompkins_enhancer":
            signals = self.modified_pan_tompkins_enhancer(signals)
        elif enhancer == 'none':
            pass
        else:
            raise ValueError(f"enhance algorithm not support")

        if smooth:
            signals =  signals - ma(signals, msec_to_sample(600, cfg.FS), algo='cumsum')

        return signals

    @staticmethod
    def swt_enhancer(signals: np.ndarray) -> np.ndarray:
        """
        Stationary Wavelet Transform
        based on Vignesh Kalidas and Lakshman Tamil.
        Real-time QRS detector using Stationary Wavelet Transform
        for Automated ECG Analysis.
        In: 2017 IEEE 17th International Conference on
        Bioinformatics and Bioengineering (BIBE).
        Uses the Pan and Tompkins thresolding.

        modified from https://github.com/berndporr/py-ecg-detectors
        """

        swt_level=3
        padding = -1
        for i in range(1000):
            if (len(signals)+i)%2**swt_level == 0:
                padding = i
                break

        if padding > 0:
            signals = np.pad(signals, (0, padding), 'edge')
        elif padding == -1:
            raise ValueError("Padding greater than 1000 required\n")

        signals = pywt.swt(signals, 'db3', level=swt_level)
        signals = np.array(signals)
        signals = signals[0, 1, :]

        signals = signals*signals

        lowpass = cfg.ENHANCER.SWT.LOWPASS
        highpass = cfg.ENHANCER.SWT.HIGHPASS

        signals = butter_bandpass(signals, lowpass, highpass, cfg.FS, order=3)
        return signals

    @staticmethod
    def modified_swt_enhancer(signals: np.ndarray) -> np.ndarray:
        """
        Modified to used absolute instead of square to make
        the height of signal not much different.
        Better I guess (testing in MIT dataset).

        Stationary Wavelet Transform
        based on Vignesh Kalidas and Lakshman Tamil.
        Real-time QRS detector using Stationary Wavelet Transform
        for Automated ECG Analysis.
        In: 2017 IEEE 17th International Conference on
        Bioinformatics and Bioengineering (BIBE).
        Uses the Pan and Tompkins thresolding.

        modified from https://github.com/berndporr/py-ecg-detectors
        """

        swt_level=3
        padding = -1
        for i in range(1000):
            if (len(signals)+i)%2**swt_level == 0:
                padding = i
                break

        if padding > 0:
            signals = np.pad(signals, (0, padding), 'edge')
        elif padding == -1:
            raise ValueError("Padding greater than 1000 required\n")

        signals = pywt.swt(signals, 'db3', level=swt_level)

        signals = np.abs(signals[0][1])

        lowpass = cfg.ENHANCER.SWT.LOWPASS
        highpass = cfg.ENHANCER.SWT.HIGHPASS

        signals = butter_bandpass(signals, lowpass, highpass, cfg.FS, order=3)

        return signals*20

    @staticmethod
    def pan_tompkins_enhancer(signals: np.ndarray) -> np.ndarray:
        """
        Jiapu Pan and Willis J. Tompkins.
        A Real-Time QRS Detection Algorithm.
        In: IEEE Transactions on Biomedical Engineering
        BME-32.3 (1985), pp. 230–236.

        This algorithm used first different for enhancing.

        modified from https://github.com/berndporr/py-ecg-detectors
        """

        lowpass = 5
        highpass = 15

        signals = butter_bandpass(signals, lowpass, highpass, cfg.FS, order=1)
        # enhance using first different
        signals = np.diff(signals)
        signals = signals*signals


        signals = ma(signals, msec_to_sample(120, cfg.FS), algo = 'cumsum')

        signals[:msec_to_sample(200, cfg.FS)] = 0
        return signals

    @staticmethod
    def modified_pan_tompkins_enhancer(signals: np.ndarray, ma_algo = "cumsum"):
        """
        Jiapu Pan and Willis J. Tompkins.
        A Real-Time QRS Detection Algorithm.
        In: IEEE Transactions on Biomedical Engineering
        BME-32.3 (1985), pp. 230–236.

        Modified to used absolute instead of squared.

        modified from https://github.com/berndporr/py-ecg-detectors
        """

        lowpass = 5
        highpass = 15

        signals = butter_bandpass(signals, lowpass, highpass, cfg.FS, order=1)
        # enhance using first different
        signals = np.diff(signals)

        signals = np.abs(signals)

        signals = ma(signals, msec_to_sample(120, cfg.FS), algo = ma_algo)

        signals[:msec_to_sample(200, cfg.FS)] = 0
        return signals*100

# ***********************************
# *********** ECG detector **********
# ***********************************

    def peak_detector(self, signals: np.ndarray, detector = "above_mean", ma_algo = "cumsum") -> Tuple[PeakArray, List]:
        """For detect R peak and peak

        Args:
            signals (np.ndarray): [description]
            detector (str, optional): [description]. Defaults to "above_mean".

        Returns:
            labelpeaks (PeakArray): R peak label
            peakarray (PeakArray): All peak (not have to be R peak)
        """

        if detector == 'above_mean':
            labelpeaks, peak_idx_array = self.above_mean_detector(signals, ma_algo=ma_algo)
        elif detector == 'above_mean_fix':
            labelpeaks, peak_idx_array = self.above_mean_fix(signals, ma_algo=ma_algo)
        elif detector == 'pan_tompkins':
            labelpeaks, peak_idx_array = self.pan_tompkins_detector(signals)
        elif detector == "two_average":
            labelpeaks, peak_idx_array = self.two_average_detector(signals,ma_algo = ma_algo)
        else:
            labelpeaks, peak_idx_array = self.above_mean_detector(signals, ma_algo=ma_algo)

        return labelpeaks, peak_idx_array

    @staticmethod
    def above_mean_fix(signals: np.ndarray, smooth = None, ma_algo = "cumsum"):

        if smooth is None:
            smooth = ma(signals, msec_to_sample(cfg.DETECTOR.AMF_SMOOTH_MSEC, cfg.FS), algo=ma_algo)

        peak_idx_array = []
        width_threshold = msec_to_sample(cfg.DETECTOR.AMF_WIDTH_THRESHOLD_MSEC,cfg.FS, to_int=False)

        smooth = np.abs(smooth)+cfg.DETECTOR.AMF_THRESHOLD

        # TODO add more condition

        start = 0
        starting = False

        for i, point in enumerate(signals):
            if point > smooth[i]:
                if not starting:
                    start = i
                    starting = True
                continue
            elif i-start>width_threshold and starting:
                window = signals[start:i]
                max_idx = np.argmax(window)

                if window[max_idx] - smooth[i-len(window) + max_idx+1] <0.2*window[max_idx]:

                    starting = False
                    start = 0
                    continue

                beat_idx = i - len(window) + max_idx+1
                peak_idx_array.append(beat_idx)
            starting = False
            start = 0

        # ------------------------------ detect r peak
        labelpeaks = PeakArray([Peak(0)])

        # define spacing of ECG
        r_spacing = msec_to_sample(cfg.DETECTOR.RPEAK_SPACING_MSEC, cfg.FS, to_int=False)

        for i, peak_idx in enumerate(peak_idx_array):
            # if signal greater than threshold and new peak not near old R peak
            if (peak_idx-labelpeaks[-1].idx)>r_spacing:

                # Insert new QRS peak
                labelpeaks.append_qrs(peak_idx)

        labelpeaks.pop(0)

        return labelpeaks, peak_idx_array

    @staticmethod
    def above_mean_detector(signals: np.ndarray, smooth = None, ma_algo = "cumsum", normalize = True):

        # ----------------------------------- detect peak first

        if normalize:
            sample = sec_to_sample(cfg.NORMALIZE_SEC, cfg.FS)
            signals = segment_normalize(signals, sample, algo="max_abs", absolute=False)

        if smooth is None:
            smooth = ma(signals, msec_to_sample(cfg.DETECTOR.AM_SMOOTH_MSEC, cfg.FS), algo=ma_algo)

        window = []
        peak_idx_array = []

        for i, point in enumerate(signals):
            mean_signal = smooth[i]
            if (point <= mean_signal) and (len(window) <= 1):
                pass
            elif (point > mean_signal):
                window.append(point)
            else:

                max_idx = np.argmax(window)
                beat_idx = i - len(window) + max_idx
                peak_idx_array.append(beat_idx)
                window = []

        # ------------------------------ detect r peak
        labelpeaks = PeakArray([Peak(0)])
        qrs_index_array = []
        qrs_index = 0

        # define spacing of ECG
        r_spacing = cfg.DETECTOR.RPEAK_SPACING_MSEC*cfg.FS/1000

        threshold_I1 = cfg.DETECTOR.AM_START_THRESHOLD
        threshold_I2 = threshold_I1*0.5
        intercept = cfg.DETECTOR.AM_THRESHOLD_INTERCEPT1
        intercept2 = cfg.DETECTOR.AM_THRESHOLD_INTERCEPT2

        SPKI = 0
        NPKI = 0
        RR_missed = 0

        for i, peak_idx in enumerate(peak_idx_array):

            # if signal greater than threshold and new peak not near old R peak

            if signals[peak_idx]-smooth[peak_idx]>threshold_I1 + intercept and (peak_idx-labelpeaks[-1].idx)>r_spacing:

                # Insert new QRS peak
                labelpeaks.append_qrs(peak_idx)
                qrs_index_array.append(i)

                # save amplitude of the signal
                if (signals[peak_idx]- smooth[peak_idx])<0:
                    raise Exception("something error")
                SPKI = 0.125*(signals[peak_idx]- smooth[peak_idx]) + 0.875*SPKI

                if RR_missed!=0:
                    if labelpeaks[-1]-labelpeaks[-2]>RR_missed:
                        missed_section_peaks = peak_idx_array[qrs_index_array[-2]+1:qrs_index_array[-1]]
                        missed_section_peaks2 = []

                        # missed_peak is Peak object
                        for missed_peak_idx in missed_section_peaks:
                            if ((   missed_peak_idx -labelpeaks[-2].idx) > r_spacing
                                    and labelpeaks[-1].idx-missed_peak_idx > r_spacing
                                    and signals[missed_peak_idx] > threshold_I2 +intercept2):

                                missed_section_peaks2.append(missed_peak_idx)

                        if len(missed_section_peaks2)>0:

                            missed_peak_idx = missed_section_peaks2[np.argmax(signals[missed_section_peaks2])]

                            labelpeaks.append(labelpeaks[-1])
                            labelpeaks[-2] = QRS(missed_peak_idx)

            else:

                NPKI = 0.125*(signals[peak_idx]- smooth[peak_idx]) + 0.875*NPKI

            threshold_I1 = NPKI + 0.25*(SPKI-NPKI)
            threshold_I2 = 0.5*threshold_I1

            if len(labelpeaks)>8:
                RR = np.diff(labelpeaks[-9:])
                RR_ave = int(np.mean(RR))
                RR_missed = int(1.66*RR_ave)

            qrs_index = qrs_index+1

        labelpeaks.pop(0)

        return labelpeaks, peak_idx_array

    @staticmethod
    def pan_tompkins_detector(signals: np.ndarray):
        """
        Jiapu Pan and Willis J. Tompkins.
        A Real-Time QRS Detection Algorithm.
        In: IEEE Transactions on Biomedical Engineering
        BME-32.3 (1985), pp. 230–236.

        modified from https://github.com/berndporr/py-ecg-detectors
        """

        # ------------------ Peak detector
        peak_idx_array = []
        n_signal = len(signals)
        for i in range(n_signal):
            if i>0 and i < n_signal-1:
                pass
            else:
                continue
            if signals[i-1] < signals[i] and signals[i+1] < signals[i]:
                peak_idx_array.append(i)

        # -------------- Rpeak detector
        labelpeaks = PeakArray([Peak(0)])
        qrs_index_array = []
        qrs_index = 0

        # define spacing of ECG
        r_spacing = cfg.DETECTOR.RPEAK_SPACING_MSEC*cfg.FS/1000

        threshold_I1 = cfg.DETECTOR.AM_START_THRESHOLD
        threshold_I2 = threshold_I1*0.5
        intercept = cfg.DETECTOR.AM_THRESHOLD_INTERCEPT1
        intercept2 = cfg.DETECTOR.AM_THRESHOLD_INTERCEPT2

        SPKI = 0
        NPKI = 0
        RR_missed = 0

        for i, peak_idx in enumerate(peak_idx_array):

            # if signal greater than threshold and new peak not near old R peak

            if signals[peak_idx]>threshold_I1 + intercept and (peak_idx-labelpeaks[-1].idx)>r_spacing:

                # Insert new QRS peak
                labelpeaks.append_qrs(peak_idx)
                qrs_index_array.append(i)

                SPKI = 0.125*signals[peak_idx] + 0.875*SPKI

                if RR_missed!=0:
                    if labelpeaks[-1]-labelpeaks[-2]>RR_missed:
                        missed_section_peaks = peak_idx_array[qrs_index_array[-2]+1:qrs_index_array[-1]]
                        missed_section_peaks2 = []

                        # missed_peak is Peak object
                        for missed_peak_idx in missed_section_peaks:
                            if ((   missed_peak_idx -labelpeaks[-2].idx) > r_spacing
                                    and labelpeaks[-1].idx-missed_peak_idx > r_spacing
                                    and signals[missed_peak_idx] > threshold_I2 +intercept2):

                                missed_section_peaks2.append(missed_peak_idx)

                        if len(missed_section_peaks2)>0:

                            missed_peak_idx = missed_section_peaks2[np.argmax(signals[missed_section_peaks2])]

                            labelpeaks.append(labelpeaks[-1])
                            labelpeaks[-2] = QRS(missed_peak_idx)

            else:

                NPKI = 0.125*signals[peak_idx] + 0.875*NPKI

            threshold_I1 = NPKI + 0.25*(SPKI-NPKI)
            threshold_I2 = 0.5*threshold_I1

            if len(labelpeaks)>8:
                RR = np.diff(labelpeaks[-9:])
                RR_ave = int(np.mean(RR))
                RR_missed = int(1.66*RR_ave)

            qrs_index = qrs_index+1

        labelpeaks.pop(0)
        return labelpeaks, peak_idx_array

    @staticmethod
    def two_average_detector(signals: np.ndarray, ma_algo = "cumsum"):

        """
        Elgendi, Mohamed & Jonkman,
        Mirjam & De Boer, Friso. (2010).
        Frequency Bands Effects on QRS Detection.
        The 3rd International Conference on Bio-inspired Systems
        and Signal Processing (BIOSIGNALS2010). 428-431.
        
        modified from https://github.com/berndporr/py-ecg-detectors
        """

        lowpass = 8
        highpass = 20

        r_spacing = cfg.DETECTOR.RPEAK_SPACING_MSEC*cfg.FS/1000
        r_width = msec_to_sample(cfg.DETECTOR.TA_PEAK_WIDTH, cfg.FS)

        signals = butter_bandpass(signals, lowpass, highpass, cfg.FS, order = 2, output ='sos')
        signals = abs(signals)

        mwa_qrs = ma(signals, msec_to_sample(cfg.DETECTOR.TA_SHORT_MSEC, cfg.FS), algo=ma_algo)
        mwa_beat = ma(signals, msec_to_sample(cfg.DETECTOR.TA_LONG_MSEC, cfg.FS), algo=ma_algo)

        labelpeaks = PeakArray([])
        start = 0
        for i in range(1, len(signals)):
            if mwa_qrs[i-1] <= mwa_beat[i-1] and mwa_qrs[i] >  mwa_beat[i]:
                start = i

            elif mwa_qrs[i-1] >  mwa_beat[i-1] and mwa_qrs[i] <= mwa_beat[i]:
                end = i-1

                if end-start>r_width:
                    peak_idx = np.argmax(signals[start:end+1])+start

                    if labelpeaks:
                        if peak_idx-labelpeaks[-1].idx > r_spacing:
                            labelpeaks.append_qrs(peak_idx)
                    else:
                        labelpeaks.append_qrs(peak_idx)

        return labelpeaks, []


    @staticmethod
    def detect_t_second_derivative(signals_abs, second_dif_signals, peakarray: PeakArray):
        """Detect T wave

        Detect T wave for calculate QTc

        """

        if not peakarray.ASSIGN_INTERVAL:
            raise ValueError("peakarray must be assigned interval before detect T wave")


        start_t = msec_to_sample(cfg.DETECT_T.MYALGO_LANDMASK_FROM_RPEAK_MSEC, cfg.FS)
        n_signal = len(second_dif_signals)
        landmark_ratio = cfg.DETECT_T.MYALGO_LANDMASK_INTERVAL_RATIO

        for peak in peakarray:
            if not isinstance(peak, QRS):
                continue
            if not peak.next_interval:
                continue
            # QTc work only normal peak
            if not peak.normal:
                continue

            end_t = int(peak.next_interval*landmark_ratio)

            if end_t <= start_t:
                continue

            start_idx = min(peak.idx+start_t, n_signal-1)
            end_idx = min(peak.idx+end_t, n_signal)

            landmask = signals_abs[start_idx: peak.idx+end_t]

            if len(landmask) !=0:
                # assign T peak
                t_peak = np.argmax(landmask)+ peak.idx + start_t
                peak.t = t_peak
            else:
                continue

            # find end of T
            end_idx = min(int(0.3* peak.next_interval)+t_peak, end_t+peak.idx)

            end_t_landmask = second_dif_signals[t_peak: end_idx]


            if len(end_t_landmask) != 0:
                peak.t_end = np.argmax(end_t_landmask) + t_peak

        peakarray.T_WAVE = True

    @staticmethod
    def detect_qrs_onset_offset(second_dif_signals: np.ndarray, peakarray: PeakArray):
        """
        Hermans BJM, Vink AS, Bennis FC, Filippini LH, Meijborg VMF, Wilde AAM, et al.
        The development and validation of an easy
        to use automatic QT-interval algorithm.
        Baumert M, editor. PLoS ONE. 2017.

        """
        if not peakarray.ASSIGN_INTERVAL:
            raise ValueError("Assign interval before search mu")

        search_near_r_long =  msec_to_sample(100, cfg.FS)
        search_near_r_short =  msec_to_sample(20, cfg.FS)

        for peak in peakarray:
            if not isinstance(peak, QRS):
                continue

            if not peak.next_interval:
                continue

            if not peak.interval:
                continue

            start_idx = max(peak.idx-search_near_r_long, 0)
            end_idx = max(peak.idx-search_near_r_short, 0)

            onset_landmask = second_dif_signals[start_idx: end_idx]
            if len(onset_landmask) !=0 :
                mu_idx = np.argmax(onset_landmask) + start_idx
                peak.mu = mu_idx

            start_idx = min(peak.idx + search_near_r_short, len(second_dif_signals) - 1)
            end_idx = min(peak.idx + search_near_r_long, len(second_dif_signals))

            offset_landmask = second_dif_signals[start_idx: end_idx]
            if len(offset_landmask) !=0 :
                j_idx = np.argmax(offset_landmask) + start_idx
                peak.j = j_idx

        peakarray.QRS_ONSET = True
        peakarray.QRS_OFFSET = True



