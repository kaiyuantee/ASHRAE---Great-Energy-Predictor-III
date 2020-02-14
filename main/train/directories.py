import os
from .utils import *

if not os.path.exists(DATA_ROOT):
    os.makedirs(DATA_ROOT)

if not os.path.exists(OUTPUT_ROOT):
    os.makedirs(OUTPUT_ROOT)

if not os.path.exists(KERAS_ROOT):
    os.makedirs(KERAS_ROOT)