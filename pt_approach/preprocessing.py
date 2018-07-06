import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#load
from sklearn.model_selection import train_test_split

import nltk
from nltk.stem.rslp import RSLPStemmer
from sentimento import SentiLex
from textstat.textstat import *
import seaborn as sn
import pickle
from sklearn import metrics
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.pipeline import FeatureUnion

from sklearn.feature_extraction.text import TfidfVectorizer