# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 10:21:15 2026

@author: meval
"""

import numpy as np
import scipy.stats as sci
from . import data, classes

def jarque_bera_test(x, alpha=0.05):
    # test de normalidad de Jarque-Bera
    skew = sci.skew(x)
    kurt = sci.kurtosis(x)
    jb = len(x) / 6 * (skew**2 + 0.25*kurt**2)
    p_value = 1 - sci.chi2.cdf(jb, df=2)
    is_normal = (p_value >= alpha)
    return jb, p_value, is_normal