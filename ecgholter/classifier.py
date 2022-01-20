from typing import List

import numpy as np

from .peakobject import ECGLabel
from . import metric
from .config import cfg
from .peakobject import PeakArray, QRS, Peak
from .utils import *

class PeakClassifier:

    def preprocess(self, signals):
        pass

    def tachy_brady(self, peakarray: PeakArray, segment_idx: List[int], algo = "interval"):
        if algo == 'interval':
            self.tachy_brady_time(peakarray, segment_idx)
        elif algo == "peak":
            self.tachy_brachy_peak(peakarray)

    @staticmethod
    def tachy_brady_time(peakarray: PeakArray, segment_idx: List[int]):

        brady_threshold=  cfg.DIAG.BRADY
        tachy_threshold = cfg.DIAG.TACHY
        start_idx = 0
        for end_idx in segment_idx:
            
            peakarray_landmark = peakarray[start_idx: end_idx]

            interval_array, _ = peakarray_landmark.get_interval(rm_outlier=True, upper_percentile=95, lower_percentile=10, to_time=True)
            hr = metric.hr(interval_array)
            if hr == None:
                continue

            for peak in peakarray_landmark:
                if isinstance(peak, QRS):
                    peak.add_hr(hr, tachy_threshold, brady_threshold)

            start_idx = end_idx


    @staticmethod
    def tachy_brachy_peak(peakarray: PeakArray, wait_buffer = False):
        """Assigned HR to peak

        use number of peak
        This method will automatic update diagnosis for 
        tachycardia, bradycardia

        Args:
            wait_buffer (bool, optional): Used for defined if we want to wait for previous data.
        """
        if not peakarray.ASSIGN_INTERVAL:
            raise ValueError("Please assign interval before calculate tachy brady. Call assign_interval_to_peak")

        interval_buffer = []
        fs = cfg.FS
        min_peak_buffer = cfg.DIAG.HR_PEAK_BUFFER
        brady_threshold=  cfg.DIAG.BRADY
        tachy_threshold = cfg.DIAG.TACHY

        for peak in peakarray:

            if not isinstance(peak, QRS):
                if wait_buffer:
                    interval_buffer = []
                continue
            if peak.interval == None:
                if wait_buffer:
                    interval_buffer = []
                continue

            interval_buffer.append(sample_to_msec(peak.interval), fs )
            
            if len(interval_buffer) == min_peak_buffer:
                hr = metric.hr(interval_buffer)
                if hr == None:
                    continue

                # add hr automatically classify tachycardia and bradycardia
                peak.add_hr(hr, tachy_threshold, brady_threshold)
                del(interval_buffer[0])

    def ectopic(self, peakarray, segment_idx, algo = "interval"):
        if algo == "interval":
            self.ectopic_interval(peakarray, segment_idx)

    def ectopic_interval(self, peakarray, segment_idx):
        if not peakarray.ASSIGN_INTERVAL:
            raise ValueError("Please assign interval before calculate tachy brady. Call assign_interval_to_peak")

        if not peakarray.QRS_ONSET:
            raise ValueError("Please assign QRS onset")
        
        if not peakarray.QRS_OFFSET:
            raise  ValueError("Please assign QRS offset")

        start_idx = 0
        ectopic_ratio = cfg.DIAG.ECTOPIC_RATIO
        qrs_width_sample = msec_to_sample(cfg.DIAG.QRS_WIDTH, cfg.FS, to_int=False)
        median = cfg.DIAG.ECTOPIC_MEDIAN

        for end_idx in segment_idx:
            interval_array, _ = peakarray[start_idx: end_idx].get_interval(to_time=False)
            if len(interval_array) ==0:
                continue
            
            if median:
                mean_rr  = np.median(interval_array)
            else:
                mean_rr  = np.mean(interval_array)

            for peak in peakarray[start_idx: end_idx]:
                if not isinstance(peak, QRS):
                    continue
                if not peak.interval:
                    continue
                
                if peak.interval < mean_rr * ectopic_ratio:
                    self.pvc_pac(peak, qrs_width_sample)

            start_idx = end_idx

    def pvc_pac(self, peak: QRS, qrs_width_sample):
        if not peak.mu:
            peak.add_diagnosis(ECGLabel.PAC)
            return

        if not peak.j:
            peak.add_diagnosis(ECGLabel.PAC)
            return

        qrs_width = peak.j - peak.mu

        if qrs_width > qrs_width_sample:
            peak.add_diagnosis(ECGLabel.PVC)
            return

        peak.add_diagnosis(ECGLabel.PVC)


    def ventricular_arrhythmia(self, signals, peakarray: PeakArray, algo = ["tcsc", "vf_filter"]):
        """ventricular arrhythmia

        """
        segment_sample = sec_to_sample(cfg.DIAG.VTFT_TCSC_SEGMENT_SEC, cfg.FS)
        decision_sample = sec_to_sample(cfg.DIAG.VTFT_TCSC_SMOOTH_SEC, cfg.FS)
        tcsc_section = []
        vf_filter_section = []
        n_signals = len(signals)

        if "tcsc" in algo:
            mask = get_section_mask(peakarray.Q_SECTION, len(signals))

            vtvf_signals = metric.tcsc(signals, segment_sample, decision_sample, 
                    threshold=cfg.DIAG.VTFT_TCSC_BINARY_THRESHOLD, mask = mask)
            tcsc_section = mask_segment(vtvf_signals, 0, n_signals, threshold = cfg.DIAG.VTFT_TCSC_THRESHOLD)

        if "vf_filter" in algo:
            vtvf_signals = metric.vf_filter(signals, segment_sample)
            vtvf_signals = vtvf_signals*(-1) +1
            if len(vtvf_signals) !=0:

                tmp_vf_filter_section = mask_segment(vtvf_signals,0,n_signals, threshold = cfg.DIAG.VTFT_VFF_THRESHOLD)

                constant = len(signals)/len(vtvf_signals)
                for section in tmp_vf_filter_section:
                    vf_filter_section.append( (int(section[0]*constant), int(section[1]*constant)))


        vtvf_section = union_list_of_section([tcsc_section, vf_filter_section])
        peakarray.add_vtvf_section(vtvf_section)


    def af(self, peakarray: PeakArray):
        #TODO
        intervals, _ = peakarray.get_interval(to_time=True)
        n_section = int(np.floor(len(intervals)/10))

        for i in range(n_section):
            pass