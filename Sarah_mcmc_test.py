from batman import TransitModel, TransitParams
from batman.openmp import detect
import timeit
import numpy as np
import matplotlib.pyplot as plt

from batman import _nonlinear_ld
from batman import _quadratic_ld
from batman import _uniform_ld
from batman import _logarithmic_ld
from batman import _exponential_ld
from batman import _power2_ld
from batman import _custom_ld
from batman import _rsky
from batman import _eclipse
from math import pi
import multiprocessing
from batman import openmp
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 300
