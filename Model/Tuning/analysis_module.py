import os

ANALYSIS_PATH = os.path.join('..', '..', 'Analysis')

pwd = os.getcwd()
os.chdir(ANALYSIS_PATH)

from analysis import plot

os.chdir(pwd)