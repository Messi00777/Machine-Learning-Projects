import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from pylab import rcParams
rcParams['figure.figsize'] = 10,5
from sklearn.ensemble import BaggingClassifier
import scikitplot as skplt
import warnings 
warnings.filterwarnings('ignore')
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.metrics import make_scorer, recall_score, precision_score, confusion_matrix, classification_report, accuracy_score


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
         print(os.patth.join(dirname, filename))



data = pd.read_csv('/Users/fcbarcelona/Downloads/new_train.csv')
print(data.head())

predict = pd.read_csv('/Users/fcbarcelona/Downloads/new_test.csv')
print(predict.head())