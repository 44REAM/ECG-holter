import os
from copy import deepcopy

import numpy as np

from .peak import ECGLabel, Event, Peak, QRS, QuestionMark
from ..config.config import cfg
from ..utils import *
from .. import metric

SUPPORT_PEAK_IDX_TYPE = (int, np.int64)
ROUND = 3


class PeakArray(list):
    """Object for store Peak

    """
    def __init__(self, data, assign_interval = False, fs = cfg.FS,
        p_wave = False, qrs_onset = False, qrs_offset = False, t_wave = False,
        vtvf_section = None, q_section = None, af_section = None, start_ts = None):
        super().__init__(data)

        # property below can transfer to new peakarray
        self.START_TS = start_ts
        self.fs = fs

        # Use to check if we already assign attibute to peak

        self.ASSIGN_INTERVAL = assign_interval
        self.P_WAVE = p_wave
        self.QRS_ONSET = qrs_onset
        self.QRS_OFFSET = qrs_offset
        self.T_WAVE = t_wave

        # property below cannot transfer to new peakarray
        self.Q_SECTION = q_section
        self.VTVF_SECTION = vtvf_section
        self.AF_SECTION = af_section
        self.HR_MAX = None
        self.HR_MIN = None
        self.HR_MIN_IDX = None
        self.HR_MAX_IDX = None

    def get_peak_timestamp(self, index):

        if self.START_TS is None:
            raise ValueError("start timestamp is not yet assigned")

        # TODO
        # ********** USE self.fs ********************
        return sample_to_msec(self[index].idx, self.fs) + self.START_TS

    def assign_start_ts(self, start_ts):
        self.START_TS = start_ts

    @property
    def q_section(self):
        if self.Q_SECTION != None:
            return self.Q_SECTION
        raise ValueError("Questionmark not yet assign")

    @property
    def vtvf_section(self):
        if self.VTVF_SECTION != None:
            return self.VTVF_SECTION
        raise ValueError("VTVF not yet assign")

    @property
    def af_section(self):
        if self.AF_SECTION != None:
            return self.AF_SECTION
        raise ValueError("AF not yet assign")

    def __getitem__(self, key):
        if isinstance(key, slice):
            return PeakArray(super().__getitem__(key), 
                assign_interval=self.ASSIGN_INTERVAL, fs=self.fs,
                p_wave = self.P_WAVE, qrs_onset = self.QRS_ONSET,
                qrs_offset = self.QRS_OFFSET, t_wave = self.T_WAVE,
                start_ts = self.START_TS)

        return super().__getitem__(key)

    def __repr__(self) -> str:
        return f"PeakArray({super().__repr__()})"

    def get_hr_minmax(self, rm_outlier = True, upper_percentile = 98, lower_percentile = 2):

        if self.START_TS is None:
            raise ValueError("Start timestamp not yet assigned")

        hr_array = np.array([peak.hr for peak in self if isinstance(peak, QRS) and peak.hr])
        idx_array = np.array([peak.idx for peak in self if isinstance(peak, QRS) and peak.hr])

        if len(hr_array) == 0:
            return None, None, None, None

        if rm_outlier:
            indices = get_index_where_percentile(hr_array, upper_percentile=upper_percentile, lower_percentile=lower_percentile)

            if len(indices) ==0:
                pass
            else:
                hr_array = hr_array[indices]
                idx_array = idx_array[indices]

        maxidx = np.argmax(hr_array)
        minidx = np.argmin(hr_array)

        max_hr = hr_array[maxidx]
        min_hr = hr_array[minidx]

        maxhr_ts = idx_array[maxidx]
        minhr_ts = idx_array[minidx]

        maxhr_ts = sample_to_msec(maxhr_ts, self.fs) + self.START_TS
        minhr_ts = sample_to_msec(minhr_ts, self.fs) + self.START_TS

        return max_hr, maxhr_ts, min_hr, minhr_ts

    def get_index(self, index = None):
        if isinstance(index, slice):
            return [peak.idx for peak in self[index]]
        if isinstance(index, int):
            return self[index].idx
        return [peak.idx for peak in self]

    def get_diagnosis_time(self, diag, index = None):
        if isinstance(index, slice):
            return [peak.get_bkk_time(self.START_TS, self.fs) for peak in self if peak.diagnosis == diag]

        if not index:
            return [peak.get_bkk_time(self.START_TS, self.fs) for peak in self if peak.diagnosis == diag]

        if self[index].diagnosis != diag:
            raise ValueError(f"The peak is not {diag}")


    def get_peak_index(self, index = None):
        """[summary]
        Get all Peak except Event Peak.
        """
            
        if isinstance(index, slice):
            return [peak.idx for peak in self[index] if isinstance(peak, QRS) or isinstance(peak, QuestionMark)]

        if not index:
            return [peak.idx for peak in self if isinstance(peak, QRS) or isinstance(peak, QuestionMark)]

        if (not isinstance(self[index], QRS)) and (not isinstance(self[index], QuestionMark)):
            raise ValueError("The peak is not QRS")

        return self[index].idx

    def get_r_index(self, index = None):
        """
        Get R peak
        different from method get_peak_index is get_peak_index also
        include Questionmark
        """
            
        if isinstance(index, slice):
            return [peak.idx for peak in self[index] if isinstance(peak, QRS)]

        if not index:
            return [peak.idx for peak in self if isinstance(peak, QRS)]

        if not isinstance(self[index], QRS):
            raise ValueError("The peak is not QRS")

        return self[index].idx

    def get_t_index(self, index = None):
        if not self.T_WAVE:
            return []
            
        if isinstance(index, slice):
            return [peak.t for peak in self[index] if isinstance(peak, QRS) and peak.t ]

        if not index:
            return [peak.t for peak in self if isinstance(peak, QRS) and peak.t]

        if not isinstance(self[index], QRS):
            raise ValueError("The peak is not QRS")

        if not self[index].t:
            raise ValueError("The peak not have t peak")
        
        return self[index].t

    def get_t_end_index(self, index = None):
        if not self.T_WAVE:
            return []

        if not index:
            return [peak.t_end for peak in self if isinstance(peak, QRS) and peak.t_end]
            
        if isinstance(index, slice):
            return [peak.t_end for peak in self[index] if isinstance(peak, QRS) and peak.t_end ]

        if not isinstance(self[index], QRS):
            raise ValueError("The peak is not QRS")

        if not self[index].t_end:
            raise ValueError("The peak not have t peak")
        
        return self[index].t_end

    def get_qrs_onset_index(self, index = None):
        if not self.QRS_ONSET:
            return []

        if not index:
            return [peak.mu for peak in self if isinstance(peak, QRS) and peak.mu]
            
        if isinstance(index, slice):
            return [peak.mu for peak in self[index] if isinstance(peak, QRS) and peak.mu ]

        if not isinstance(self[index], QRS):
            raise ValueError("The peak is not QRS")

        if not self[index].mu:
            raise ValueError("The peak not have t peak")
        
        return self[index].mu

    def shift(self, shift):
        for peak in self:
            peak.shift(shift)

    def get_qrs_offset_index(self, index = None):
        if not self.QRS_OFFSET:
            return []

        if not index:
            return [peak.j for peak in self if isinstance(peak, QRS) and peak.j]
            
        if isinstance(index, slice):
            return [peak.j for peak in self[index] if isinstance(peak, QRS) and peak.j ]

        if not isinstance(self[index], QRS):
            raise ValueError("The peak is not QRS")

        if not self[index].j:
            raise ValueError("The peak not have t peak")
        
        return self[index].j


    # def append(self, input_data):
    #     if isinstance(input_data, Peak):
    #         super().append(input_data) 
    #     else:
    #         raise ValueError("Input to PeakArray should be Peak")

    ##  slow. append only integer if not need peak.
    def append_qrs(self, input_data):
        if isinstance(input_data, Peak):
            self.append(QRS(input_data.idx))
        elif isinstance(input_data, SUPPORT_PEAK_IDX_TYPE):
            self.append(QRS(input_data))
        else:
            raise ValueError("input data should be Peak object or int (index)")

    def append_peak(self, input_data):
        if isinstance(input_data, Peak):
            self.append(Peak(input_data.idx))
        elif isinstance(input_data, SUPPORT_PEAK_IDX_TYPE):
            self.append(Peak(input_data))
        else:
            raise ValueError("input data should be Peak object or int (index)")

    def append_question(self, input_data):
        if isinstance(input_data, Peak):
            self.append(QuestionMark(input_data.idx))
        elif isinstance(input_data, SUPPORT_PEAK_IDX_TYPE):
            self.append(QuestionMark(input_data))
        else:
            raise ValueError("input data should be Peak object or int (index)")

    def append_event(self, input_data):
        if isinstance(input_data, Peak):
            self.append(Event(input_data.idx))
        elif isinstance(input_data, SUPPORT_PEAK_IDX_TYPE):
            self.append(Event(input_data))
        else:
            raise ValueError("input data should be Peak object or int (index)")

    #****** 
    #****** recomment: append only integer if not need peak. 
    #****** 

    # def append_qrs(self, input_data):
    #     self.append(QRS(input_data))

    # def append_peak(self, input_data):
    #     self.append(Peak(input_data))

    # def append_question(self, input_data):
    #     self.append(QuestionMark(input_data))

    def _get_section_time(self, list_of_sections, realtime = True, mode = 'bkk', min_section_time = 2000):

        if self.START_TS is None:
            raise ValueError("start timestamp not yet assigned")

        time_list_msec = []
        for section in list_of_sections:

            start_time = sample_to_msec(section[0], self.fs) + self.START_TS
            end_time = sample_to_msec(section[1], self.fs) + self.START_TS

            if end_time-start_time < min_section_time:
                continue

            if realtime:
                start_time = timestamp_msec_to_datetime(start_time, mode=mode)
                end_time = timestamp_msec_to_datetime(end_time, mode=mode)

            time_list_msec.append((start_time, end_time))

        return time_list_msec

    def get_vtvf_section_time(self, realtime = True, mode = 'bkk', min_section_time= 2000):
        if self.VTVF_SECTION is None:
            raise ValueError("VTVF not yet assigned")
        return self._get_section_time(self.VTVF_SECTION, realtime = realtime, mode =mode, min_section_time= min_section_time )

    def get_af_section_time(self, realtime = True, mode = 'bkk', min_section_time=2000):
        if self.AF_SECTION is None:
            raise ValueError("AF not yet assigned")
        return self._get_section_time(self.AF_SECTION, realtime = realtime, mode =mode, min_section_time= min_section_time )

    def get_questionmark_section_time(self, realtime = True, mode = 'bkk', min_section_time= 2000):
        if self.Q_SECTION is None:
            raise ValueError("Questionmark not yet assigned")
        return self._get_section_time(self.Q_SECTION, realtime = realtime, mode =mode, min_section_time=min_section_time )

    def get_all_section_time(self, realtime = True, mode = 'bkk', min_section_time = 2000):

        vtvf_section_time = self.get_vtvf_section_time(realtime = realtime, mode = mode, min_section_time =min_section_time)
        q_section_time = self.get_questionmark_section_time(realtime = realtime, mode = mode, min_section_time= min_section_time)

        section_dict = {
            'vtvf': vtvf_section_time,
            'question': q_section_time
        }

        return section_dict

    def get_segment_index_every_nsec(self, nsec = 10.0):
        if len(self) == 0:
            return []

        segment_sample = sec_to_sample(nsec, self.fs)

        return get_segment_index(self.get_index(), segment_sample)

    def assign_interval_to_peak(self):

        pause_sample = sec_to_sample(cfg.DIAG.PAUSE, self.fs, to_int=False)

        for i in range(1, len(self)):
            if isinstance(self[i-1] ,QRS) and isinstance(self[i],QRS):
                interval = self[i].idx - self[i-1].idx

                self[i-1].next_interval = interval
                self[i].add_previous_interval(interval, pause_sample)

        self.ASSIGN_INTERVAL = True

    def assign_unknown_to_peak(self):
        for peak in self:
            if not isinstance(peak, QRS):
                continue

            peak.set_unknown()


    def assign_peak_by_section(self, list_of_section: List, function):

        section_idx = 0
        n_section = len(list_of_section)


        if n_section == 0:
            return

        for idx, peak in enumerate(self):

            while peak.idx > list_of_section[section_idx][1]:

                section_idx+=1
                if section_idx>= n_section:
                    return

            if peak.idx >= list_of_section[section_idx][0]:
                self[idx] = function(peak)

    def add_event_peak(self, list_of_section):
        
        question_peaks = []
        for section in list_of_section:
            start = section[0]
            end = section[1]
            question_peaks.append(Event(int((start+end)/2)))

        self.extend(question_peaks)
        self.sort()

    def add_questionmask_section(self, question_section):
        def function(peak):
            return QuestionMark(peak.idx)

        self.assign_peak_by_section(question_section, function)
        self.add_event_peak(question_section)

        self.Q_SECTION = question_section

    def add_vtvf_section(self, vtvf_section):
        def function(peak: Peak):
            if not isinstance(peak, QRS):
                return peak

            peak.add_diagnosis(ECGLabel.VENTRICULAR)
            return peak

        
        self.assign_peak_by_section(vtvf_section, function)
        self.VTVF_SECTION = vtvf_section


# ***********************************
# *********** REPORT ****************
# ***********************************
# ****** For general report *********
# ***********************************
    def get_all_labels(self, to_timestamp = True):

        if self.START_TS is None:
            raise ValueError("Start timestamp not yet assigned")

        unknown_marker = []
        q_marker = []
        normal_marker = []
        tachy_marker = []
        brady_marker = []
        v_marker = []
        pause_marker = []
        pvc_marker = []
        pac_marker = []
        af_marker = []

        event_marker = []

        fs = self.fs

        for peak in self:
            # return timestamp have to be integer
            if to_timestamp:
                timestamp = int(sample_to_msec(peak.idx, fs)) + self.START_TS
            else:
                timestamp=peak.idx
            if isinstance(peak, QuestionMark):
                q_marker.append(timestamp)
            elif isinstance(peak, QRS):
                if peak.diagnosis == ECGLabel.NORMAL:
                    normal_marker.append(timestamp)
                elif peak.diagnosis == ECGLabel.UNKNOWN:
                    unknown_marker.append(timestamp)
                elif peak.diagnosis == ECGLabel.TACHYCARDIA:
                    tachy_marker.append(timestamp)
                elif peak.diagnosis == ECGLabel.BRADYCARDIA:
                    brady_marker.append(timestamp)
                elif peak.diagnosis == ECGLabel.PAUSE:
                    pause_marker.append(timestamp)
                elif peak.diagnosis == ECGLabel.PAC:
                    pac_marker.append(timestamp)
                elif peak.diagnosis == ECGLabel.PVC:
                    pvc_marker.append(timestamp)
                elif peak.diagnosis == ECGLabel.AF:
                    af_marker.append(timestamp)
                elif peak.diagnosis == ECGLabel.VENTRICULAR:
                    v_marker.append(timestamp)
            elif isinstance(peak, Event):
                event_marker.append(timestamp)


        all_label = {
            'unknown':unknown_marker,
            'question':q_marker,
            'normal':normal_marker,
            'ventricular':v_marker,
            'pvc':pvc_marker,
            'pac':pac_marker,
            'tachy':tachy_marker,
            'brady':brady_marker,
            'af':af_marker,
            'pause':pause_marker,
            'event':event_marker

        }
        return all_label

    @staticmethod
    def peak_condition(nn, interval = False):

        def nn_cond(peak):
            if isinstance(peak, QRS):
                if peak.normal:
                    return True
            return False

        def normal_cond(peak):
            return isinstance(peak, QRS)

        if nn:
            return nn_cond

        return normal_cond


    def get_different_interval(self, rm_outlier=False, upper_percentile = 98, lower_percentile=5):
        """Get interval

        Args:
            start_ts (float, optional): start timestamp
            nn (bool, optional): specify if used NN interval instead of RR interval

        Returns:
            Array of interval (msec)
            Array of timestamp that corresponse to interval (msec)
        """
        if not self.ASSIGN_INTERVAL:
            raise ValueError("Call assign_interval_to_peak first")

        different_interval = []
        interval = []
        for peak in self:
            if not isinstance(peak, QRS):
                if len(interval) >=10:
                    if rm_outlier:
                        interval = remove_outlier(interval, upper_percentile=upper_percentile, lower_percentile=lower_percentile)

                    different_interval.append(np.diff(interval))
                interval = []
                continue

            if not peak.interval:
                if len(interval) >=10:
                    if rm_outlier:
                        interval = remove_outlier(interval, upper_percentile=upper_percentile, lower_percentile=lower_percentile)
                    different_interval.append(np.diff(interval))
                interval = []
                continue
            
            interval.append(peak.interval)
        
        if len(different_interval) == 0:
            return []

        different_interval = sample_to_msec(np.concatenate(different_interval), self.fs)

        if rm_outlier:
            different_interval = remove_outlier(different_interval, upper_percentile=upper_percentile, lower_percentile=lower_percentile)

        return  different_interval

    def get_interval(self, nn =False, report = False, rm_outlier = False, 
                    upper_percentile = 98, lower_percentile=3, to_time = True):
        """Get interval

        Args:
            start_ts (float, optional): start timestamp
            nn (bool, optional): specify if used NN interval instead of RR interval

        Returns:
            Array of interval (msec)
            Array of timestamp that corresponse to interval (msec)
        """
        if self.START_TS is None:
            start_ts = 0
        else:
            start_ts = self.START_TS
        
        if self.ASSIGN_INTERVAL:
            return self._get_interval_after_assign_interval(nn = nn, start_ts=start_ts, report = report,
                                     rm_outlier = rm_outlier, to_time = to_time,
                                      upper_percentile = upper_percentile, lower_percentile=lower_percentile)
        return self._get_interval_before_assign_interval(nn = nn, start_ts=start_ts)

    def _get_interval_before_assign_interval(self, nn = False, start_ts = 0.0, report = False):
        """Used for get interval before call the function "assign_interval_to_peak"

        """

        index_arrays = []
        interval_arrays = []

        condition = self.peak_condition(nn)

        for i in range(1, len(self)):
            if condition(self[i-1]) and condition(self[i]):
                index_arrays.append(self[i].idx)
                interval_arrays.append(self[i].idx - self[i-1].idx)

        interval_arrays = np.array(interval_arrays)
        index_arrays = np.array(index_arrays)

        interval_arrays = sample_to_msec(interval_arrays, self.fs)
        index_arrays = sample_to_msec(index_arrays,self.fs)

        # plus start timestamp
        index_arrays = index_arrays + start_ts

        if report:
            return interval_arrays.astype(float).tolist(), index_arrays.astype(int).tolist()

        return  interval_arrays, index_arrays

    def _get_interval_after_assign_interval(self, nn = False, start_ts = 0.0, to_time = True,
                              report = False, rm_outlier = False, upper_percentile = 98, lower_percentile=3):
        """Used for get interval after call the function "assign_interval_to_peak"

        """
        condition = self.peak_condition(nn)

        interval_arrays =[peak.interval for peak in self if condition(peak) and peak.interval]
        index_arrays = [peak.idx for peak in self if condition(peak) and peak.interval]

        if len(interval_arrays) == 0:
            return [], []

        if to_time:
            interval_arrays = sample_to_msec(np.array(interval_arrays), self.fs)
            index_arrays = sample_to_msec(np.array(index_arrays), self.fs)
            index_arrays = index_arrays+start_ts

        if rm_outlier:
            indices = get_index_where_percentile(interval_arrays, upper_percentile=upper_percentile, lower_percentile=lower_percentile)
            if len(indices) == 0:
                pass
            else:
                interval_arrays = interval_arrays[indices]
                index_arrays = index_arrays[indices]

        if report:
            return interval_arrays.astype(int).tolist(), list(map(int,index_arrays))

        return  interval_arrays, index_arrays

    def qtc(self, rm_outlier = True, mean = True, upper_percentile = 90, lower_percentile = 10):

        if not self.T_WAVE:
            raise ValueError("T wave must be assigned before calculated QTc")

        if not self.QRS_ONSET:
            raise ValueError("QRS onset must be assigned before calculated QTc")

        qt_array = []
        qtc_array = []
        interval_array = []

        n_self = len(self)

        if n_self <= 1:
            return None, None

        for i, peak in enumerate(self[1:n_self-1]):
            idx = i+1

            if not isinstance(peak, QRS):
                continue

            if not isinstance(self[idx-1], QRS):
                continue

            if not isinstance(self[idx+1], QRS):
                continue

            if not self[idx-1].normal:
                continue

            if not self[idx+1].normal:
                continue

            if not peak.normal:
                continue

            if not peak.t_end:
                continue

            if not peak.mu:
                continue

            qt_array.append(peak.t_end - peak.mu)
            interval_array.append(peak.interval)

        if len(qt_array) ==0:
            return None, None

        qt_array = sample_to_msec(np.array(qt_array), self.fs) + 30
        qtc_array = qt_array/ np.sqrt(sample_to_sec(np.array(interval_array), self.fs))

        if rm_outlier:
            qt_array = remove_outlier(qt_array, upper_percentile=upper_percentile, lower_percentile=lower_percentile)
            qtc_array = remove_outlier(qtc_array, upper_percentile=upper_percentile, lower_percentile=lower_percentile)

        if len(qt_array) ==0 or len(qtc_array) == 0:
            return None, None
        
        if mean:
            return np.mean(qt_array), np.mean(qtc_array)
        return qt_array, qtc_array

    def hr(self):
        rr_interval, _ = self.get_interval(to_time = True)
        return metric.hr(rr_interval)

    def sdnn(self):
        nn_interval, _ = self.get_interval(nn=True, to_time= True)
        sdnn = metric.sdnn(nn_interval, min_number_of_interval = cfg.HRV.MIN_NUMBER_OF_INTERVAL)

        return sdnn

    def sdann_asdnn(self, duration = 5):

        nn_interval, index_array = self.get_interval(nn=True, to_time = True)
        sdann, asdnn = metric.sdann_asdnn(nn_interval, index_array, min_number_of_interval= cfg.HRV.MIN_NUMBER_OF_INTERVAL , duration = duration)
        return sdann, asdnn

    def rmssd(self):
        different_interval = self.get_different_interval()

        rmssd = metric.sdnn(different_interval, min_number_of_interval= cfg.HRV.MIN_NUMBER_OF_INTERVAL)
        return rmssd


    def entropy(self, n_bins = 29, min_hr = 30., max_hr = 250.):
        rr_interval_array, _ = self.get_interval( nn=False, to_time = True)

        if rr_interval_array!= []:

            bin_size = (cfg.AF.HISTMAX - cfg.AF.HISTMIN )/n_bins
            bins = []
            add_bin = cfg.AF.HISTMIN

            for _ in range(n_bins):
                bins.append(add_bin)
                add_bin +=bin_size

            bins.append(cfg.QRS.MAX_RR)
            bins_array = np.histogram(rr_interval_array, bins=bins)[0]

            return metric.entropy(bins_array)
        return 0

    def get_all_metrics(self, duration = 5, rm_outlier = False, upper_percentile = 98, lower_percentile=3):

        sdnn=None
        sdann=None
        asdnn=None
        hr = None
        rmssd = None

        interval, index_array = self.get_interval(nn=True,to_time=True ,rm_outlier=rm_outlier, upper_percentile = upper_percentile, lower_percentile=lower_percentile)
        sdnn = metric.sdnn(interval)
        sdann, asdnn = metric.sdann_asdnn(interval, index_array, duration = duration)

        interval, _ = self.get_interval(nn=False,to_time=True, rm_outlier=rm_outlier, upper_percentile = upper_percentile, lower_percentile=lower_percentile)
        hr = metric.hr(interval)

        interval = self.get_different_interval(rm_outlier=rm_outlier, upper_percentile = upper_percentile, lower_percentile=lower_percentile)
        rmssd = metric.sdnn(interval)

        qt, qtc = self.qtc(rm_outlier=True, upper_percentile = upper_percentile, lower_percentile=lower_percentile)

        max_hr, maxhr_ts, min_hr, minhr_ts = self.get_hr_minmax()

        result = {
            'hr': round_or_none(hr, cfg.ROUND),
            'sdnn': round_or_none(sdnn, cfg.ROUND),
            'sdann': round_or_none(sdann, cfg.ROUND),
            'asdnn': round_or_none(asdnn, cfg.ROUND),
            'rmssd': round_or_none(rmssd, cfg.ROUND),
            'hrmax': round_or_none(max_hr,cfg.ROUND ),
            'hrmin': round_or_none(min_hr, cfg.ROUND),
            'hrmax_ts': maxhr_ts,
            'hrmin_ts': minhr_ts,
            'qt':round_or_none(qt, cfg.ROUND),
            'qtc':round_or_none(qtc, cfg.ROUND)

        }

        return result

    def get_hr_with_time(self, start_idx = None, end_idx = None):
        fs = self.fs

        if start_idx is None:
            start_idx = 0
        if end_idx is None:
            end_idx = len(self)

        landmark_array = self[start_idx:end_idx]

        peak_hr = [peak.hr for peak in landmark_array if isinstance(peak, QRS) and peak.hr and peak.diagnosis != ECGLabel.UNKNOWN]
        peak_time = [peak.get_bkk_time(landmark_array.START_TS, fs, return_object=True) for peak in landmark_array if isinstance(peak, QRS) and peak.hr and peak.diagnosis != ECGLabel.UNKNOWN]

        return peak_hr, peak_time

# ***********************************
# *********** UTILS *****************
# ***********************************

def resample(peakarray: PeakArray, new_fs):
    new_peakarray = deepcopy(peakarray)
    ratio = new_fs/ peakarray.fs
    for peak in new_peakarray:
        peak.rescale(ratio)

    new_peakarray.fs = new_fs
    return new_peakarray

def peakarray_like(peakarray: PeakArray, end, start = 0, shift = False):

    new_peakarray = PeakArray([])

    start_idx = np.searchsorted(peakarray, Peak(start), side = 'left')
    search_array = peakarray[start_idx:]

    for peak in search_array:
        if peak.idx >= end:
            break

        new_peak = deepcopy(peak)
        if isinstance(new_peak, QRS):
            if new_peak.t_end:
                if new_peak.t_end >=end:
                    new_peak.t_end = None

            if new_peak.t:
                if new_peak.t >=end:
                    new_peak.t = None

            if new_peak.j:
                if new_peak.j >=end:
                    new_peak.j = None

        new_peakarray.append(new_peak)

    new_peakarray.START_TS = peakarray.START_TS
    new_peakarray.ASSIGN_INTERVAL = peakarray.ASSIGN_INTERVAL
    new_peakarray.P_WAVE = peakarray.P_WAVE
    new_peakarray.QRS_ONSET = peakarray.QRS_ONSET
    new_peakarray.QRS_OFFSET = peakarray.QRS_OFFSET
    new_peakarray.T_WAVE = peakarray.T_WAVE

    if shift:
        if new_peakarray.START_TS is not None:
            new_peakarray.START_TS += sample_to_msec(start, peakarray.fs)
        new_peakarray.shift(-start)

    return new_peakarray


if __name__ == '__main__':
    pass



