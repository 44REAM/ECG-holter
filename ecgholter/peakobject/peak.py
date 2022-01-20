from enum import IntEnum
from math import floor

from ..utils import sample_to_msec, sample_to_sec, timestamp_msec_to_datetime

class ECGLabel(IntEnum):
    UNKNOWN = 0
    NORMAL = 1
    BRADYCARDIA = 2
    TACHYCARDIA = 3
    PAC = 4
    PVC = 5
    AF = 6
    VENTRICULAR = 7
    PAUSE = 8
    QUESTION = 9
    EVENT = 10

ANNOTATION = {
    ECGLabel.UNKNOWN: '*',
    ECGLabel.NORMAL: 'N',
    ECGLabel.BRADYCARDIA: 'B',
    ECGLabel.TACHYCARDIA: 'T',
    ECGLabel.PAC: 'A',
    ECGLabel.PVC: 'V',
    ECGLabel.AF: 'a',
    ECGLabel.VENTRICULAR: 'v',
    ECGLabel.PAUSE: 'P',
    ECGLabel.QUESTION: '?'
}

class Peak():
    def __init__(self, idx):
        self.idx = idx
        self.diagnosis = ECGLabel.UNKNOWN

    def __gt__(self, other):
        if self.idx > other.idx:
            return True
        return False

    def __ge__(self, other):
        if self.idx >= other.idx:
            return True
        return False

    def __lt__(self, other):
        if self.idx < other.idx:
            return True
        return False

    def __le__(self, other):
        if self.idx <= other.idx:
            return True
        return False

    def __eq__(self, other):
        if self.idx == other.idx:
            return True
        return False

    def __hash__(self):
        return hash(self.__repr__())
    
    def __ne__(self, other):
        if self.idx != other.idx:
            return True
        return False

    def __sub__(self, other):
        return self.idx - other.idx

    def __add__(self, other):
        return self.idx + other.idx
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{self.idx}"

    def get_timestamp_sec(self, start_ts_sec, fs):
        return start_ts_sec + sample_to_sec(self.idx, fs)

    def get_timestamp_msec(self, start_ts_msec, fs):
        return start_ts_msec + sample_to_msec(self.idx, fs)

    def get_gmt_time(self, start_ts_msec, fs, return_object = True):

        timestamp_msec = self.get_timestamp_msec(start_ts_msec, fs)
        return timestamp_msec_to_datetime(timestamp_msec, mode = 'utc', return_object =return_object)

    def get_bkk_time(self, start_ts_msec, fs, return_object=True):
        timestamp_msec = self.get_timestamp_msec(start_ts_msec, fs)
        return timestamp_msec_to_datetime(timestamp_msec, mode = 'bkk', return_object =return_object)

    def shift(self, shift):
        self.idx += shift

    def rescale(self, ratio):
        self.idx = int(floor(self.idx*ratio))

    def get_annotation(self):
        return ANNOTATION[self.diagnosis]

    @property
    def timestamp(self, start_ts):
        return sample_to_sec(self.idx) + start_ts


class Event(Peak):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.diagnosis = ECGLabel.EVENT

class QuestionMark(Peak):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.diagnosis = ECGLabel.QUESTION

class QRS(Peak):
    def __init__(self, idx, normal=True, diagnosis=ECGLabel.UNKNOWN,
                mu = None, j = None,t=None,u=None, hr = None, interval = None,
                t_end = None, next_interval = None,interval_around_peak = None, *args, **kwargs):
        """
        Rule
        1. Mu and J cannot be None if None it will set to Questionmark
        2. T can be None, if T is None then cannot find QT interval

        """
        super().__init__(idx, *args, **kwargs)

        self.mu = mu
        self.j = j
        self.t = t
        self.t_end = t_end
        self.u = u
        self.hr = hr

        self.interval = interval
        self.next_interval = next_interval

        self.interval_around_peak = interval_around_peak

        self.normal = normal
        self.diagnosis = diagnosis

    def shift(self, shift):
        super().shift(shift)

        if self.mu:
            self.mu +=shift

        if self.j:
            self.j +=shift

        if self.t:
            self.t +=shift

        if self.t_end:
            self.t_end +=shift

        if self.u:
            self.u +=shift

    def rescale(self, ratio):
        #TODO
        super().rescale(ratio)

    def add_interval_around_peak(self, interval: float):
        #TODO
        self.interval_around_peak = interval

    def add_previous_interval(self, interval: float, pause_sample):
        self.interval = interval
        self.diag_pause(pause_sample)
        # if self.interval > pause_sample:
        #     self.add_diagnosis(ECGLabel.PAUSE)

    def diag_pause(self, pause_sample):

        if self.interval > pause_sample:
            self.add_diagnosis(ECGLabel.PAUSE)

    def add_hr(self, hr: int, tachy_threshold, brady_threshold):
        self.hr = hr
        self.diag_tachy_brady(tachy_threshold, brady_threshold)
        # if self.hr > tachy_threshold:
        #     self.add_diagnosis(ECGLabel.TACHYCARDIA)
        # elif self.hr < brady_threshold:
        #     self.add_diagnosis(ECGLabel.BRADYCARDIA)
        # else:
        #     self.add_diagnosis(ECGLabel.NORMAL)

    def conditional_diagnosis(self, property, threshold, greater = True):

        pass
        #TODO
            
    def diag_tachy_brady(self, tachy_threshold, brady_threshold):

        if self.hr > tachy_threshold:
            self.add_diagnosis(ECGLabel.TACHYCARDIA)
        elif self.hr < brady_threshold:
            self.add_diagnosis(ECGLabel.BRADYCARDIA)
        else:
            self.add_diagnosis(ECGLabel.NORMAL)

    def set_unknown(self):

        if self.diagnosis == ECGLabel.UNKNOWN:
            self.normal = False
            return

        if not self.interval:
            self.diagnosis = ECGLabel.UNKNOWN
            self.normal = False
            return
        
        if not self.next_interval:
            self.diagnosis = ECGLabel.UNKNOWN
            self.normal = False
            return

    def add_diagnosis(self, diag: ECGLabel):

        if self.diagnosis < diag:
            self.diagnosis = diag

        if diag in [ECGLabel.PAC, ECGLabel.PVC, ECGLabel.VENTRICULAR]:
            self.normal = False

