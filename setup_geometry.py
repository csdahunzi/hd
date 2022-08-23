import numpy as np
from pywarp import *
from pygeo import *
from pyspline import *
import warnings
from scipy.linalg import solve
warnings.filterwarnings("ignore")


DVGeo = DVGeometry_Mode('basis.txt',FullModes=True)

DVGeo.addGeoDVLocal('fullmodes', lower=-5.0, upper=5.0, scale=1.0)

