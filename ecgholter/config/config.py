
from yacs.config import CfgNode as CN

#--------------------------ECG OPTIONS ------------------------
#--------------------------------------------------------------

_C = CN(new_allowed=True)
cfg = _C

#------------SAMPLING RATE---------
_C.FS = 500
_C.FS_RESAMPLE = 360
_C.ROUND = 3
#--------------------------------------------------------------
#--------------------------PREPROCESS OPTIONS -----------------
#--------------------------------------------------------------

#-----------PREPROCESS FILTER---------

_C.HIGHPASS = 1.0 # Hz
_C.NOTCH = CN()
_C.NOTCH.FS = 50.0 # Hz
_C.NOTCH.QF = 40.0
_C.DETREND = 600.0 # msec
_C.INTERPOLATED_MSEC = 3000.0 # msec
_C.NORMALIZE_SEC = 10 

#------------ ENHANCER---------

_C.ENHANCER = CN()

# SWT enhancer
_C.ENHANCER.SWT = CN()
_C.ENHANCER.SWT.LOWPASS = 0.01
_C.ENHANCER.SWT.HIGHPASS = 8.0

# -------------- DETECTOR ------

_C.DETECTOR = CN()

# above mean (AM) detector
_C.DETECTOR.AM_SMOOTH_MSEC = 500.0
# Threshold recomment [-5, 5]
_C.DETECTOR.AM_START_THRESHOLD = 0.0
_C.DETECTOR.AM_THRESHOLD_INTERCEPT1 = 0.0
_C.DETECTOR.AM_THRESHOLD_INTERCEPT2 = 0.0

_C.DETECTOR.AMF_SMOOTH_MSEC = 1000.0
# Threshold recomment [0, 10]
_C.DETECTOR.AMF_THRESHOLD = 0.2
_C.DETECTOR.AMF_WIDTH_THRESHOLD_MSEC = 60.0

# two average (TA) detector
_C.DETECTOR.TA_SHORT_MSEC = 120.0
_C.DETECTOR.TA_LONG_MSEC = 600.0
_C.DETECTOR.TA_PEAK_WIDTH = 80.0

# R peak should spacing 250ms
_C.DETECTOR.RPEAK_SPACING_MSEC = 250.0

# RECOMMENT combination
# [modified_swt_enhancer, above_mean]
# [modified_swt_enhancer, pan_tompkins]
# [none, two_average]
# [pan_tompkins_enhancer, pan_tompkins]
# [modified_pan_tompkins_enhancer, pan_tompkins]

# usable enhancer [modified_swt_enhancer, swt_enhancer, modified_pan_tompkins_enhancer, pan_tompkins_enhancer, none]
# default: modified_swt_enhancer
_C.DETECTOR.ENHANCER = "modified_swt_enhancer"

# usable detector [above_mean, above_mean_fix, pan_tompkins, two_average]
# above_mean sensitive to large spike
# default: pan_tompkins
_C.DETECTOR.DETECTOR = "above_mean_fix"

_C.DETECT_T = CN()
_C.DETECT_T.MYALGO_LANDMASK_FROM_RPEAK_MSEC = 70.0 # msec (starting point to detect T wave)
_C.DETECT_T.MYALGO_LANDMASK_INTERVAL_RATIO = 0.7 # ratio of RR interval
_C.DETECT_T.MYALGO_SMOOTH_BASELINE = 150.0 # msec

_C.HRV = CN()

# should be one but other said zero
_C.HRV.DDOF = 0
_C.HRV.MIN_NUMBER_OF_INTERVAL = 3

_C.DIAG = CN()
_C.DIAG.TACHY = 150 # bpm
_C.DIAG.BRADY = 40 #bpm
_C.DIAG.PAUSE = 2.0 # second
_C.DIAG.HR_PEAK_BUFFER = 10 # peak
_C.DIAG.HR_TIME_SEGMENT_SEC = 10 # sec

_C.DIAG.ECTOPIC_TIME_SEGMENT_SEC = 10
_C.DIAG.ECTOPIC_RATIO = 0.7
_C.DIAG.ECTOPIC_MEDIAN = True
_C.DIAG.QRS_WIDTH = 120.0 #msec

_C.DIAG.VTFT_TCSC_SEGMENT_SEC = 3.0 # second
_C.DIAG.VTFT_TCSC_SMOOTH_SEC = 6.0 # second
_C.DIAG.VTFT_TCSC_BINARY_THRESHOLD = 0.2 # lower increase sensitivity
_C.DIAG.VTFT_TCSC_THRESHOLD = 0.4 # lower increase sensitivity

_C.DIAG.VTFT_VFF_THRESHOLD = 0.4 # lower increase sensitivity