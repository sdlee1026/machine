#!/usr/bin/env python
# coding: utf-8

import warnings

warnings.filterwarnings(action='ignore')
import collections
from IPython.display import display
from IPython.display import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager, rc
import pandas as pd
import mglearn

mpl.rcParams['axes.unicode_minus'] = False
font_fname = 'fonts/NanumGothic.ttf'
font_name = font_manager.FontProperties(fname=font_fname).get_name()

rc('font', family=font_name)
# size, family
print('font size : ' + str(plt.rcParams['font.size']))
print('font family : ' + str(plt.rcParams['font.family']))
# import default setting
