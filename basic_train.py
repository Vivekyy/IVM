from utils import getXy
import pandas as pd
import numpy as np
from sklearn.utils import shuffle

X,y = getXy()

X,y = shuffle(X, y)
