import pandas as pd
pd.set_option('display.max_columns', 999)
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import confusion_matrix

import joblib

# import model
fitted_best_GBC = joblib.load('fitted_best_GBC.pkl')

# read in the data
x = pd.read_csv('../../data/02_x.csv')
y = pd.read_csv('../../data/03_y.csv')
test_x = pd.read_csv('../../data/06_test_x.csv')
test_y = pd.read_csv('../../data/07_test_y.csv')
test_x_missing = pd.read_csv('../../data/08_test_x_missing.csv')

# predict
test_y_prediction = fitted_best_GBC.predict(test_x)
test_y_probability_prediction = fitted_best_GBC.predict_proba(test_x)[:,1]

# histograms
n_total = len(test_y_prediction)
tn, fp, fn, tp = confusion_matrix(test_y, test_y_prediction).ravel()
fig, ax = plt.subplots(figsize=(7,5))
bins=np.linspace(0,1,20)
ax.hist(test_y_probability_prediction[test_y.values.reshape((-1,))==1],
        bins=bins, alpha=0.5, label='blue win', facecolor='C0')
ax.hist(test_y_probability_prediction[test_y.values.reshape((-1,))==0],
        bins=bins, alpha=0.5, label='blue loss', facecolor='C1')
ax.set_xlabel('predicted blue win probability')
ax.legend()
ax.set_title('all test data')
ax.text(0.01,0.97,'true positive  = {}/{} ({:.1f}%)'.format(tp, n_total, (tp/n_total)*100),
        horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
ax.text(0.01,0.92,'true negative  = {}/{} ({:.1f}%)'.format(tn, n_total, (tn/n_total)*100),
        horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
ax.text(0.01,0.87,'false positive = {}/{} ({:.1f}%)'.format(fp, n_total, (fp/n_total)*100),
        horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
ax.text(0.01,0.82,'false negative = {}/{} ({:.1f}%)'.format(fn, n_total, (fn/n_total)*100),
        horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
ax.text(0.01,0.77,'correct predictions = {}/{} ({:.1f}%)'.format(tp+tn, n_total, ((tp+tn)/n_total)*100),
        horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontweight='bold')
fig.savefig('../../visualizations/all-test-prediction-histograms.jpg', bbox_inches='tight', dpi=300)
plt.show()

n_total = len(test_y_prediction[test_x_missing.values.reshape((-1,))==0])
tn, fp, fn, tp = confusion_matrix(test_y[test_x_missing.values.reshape((-1,))==0],
                                  test_y_prediction[test_x_missing.values.reshape((-1,))==0]).ravel()
fig, ax = plt.subplots(figsize=(7,5))
bins=np.linspace(0,1,20)
ax.hist(test_y_probability_prediction[(test_y.values.reshape((-1,))==1)&(test_x_missing.values.reshape((-1,))==0)],
        bins=bins, alpha=0.5, label='blue win', facecolor='C0')
ax.hist(test_y_probability_prediction[(test_y.values.reshape((-1,))==0)&(test_x_missing.values.reshape((-1,))==0)],
        bins=bins, alpha=0.5, label='blue loss', facecolor='C1')
ax.set_xlabel('predicted blue win probability')
ax.legend()
ax.set_title('robust test data')
ax.text(0.01,0.97,'true positive  = {}/{} ({:.1f}%)'.format(tp, n_total, (tp/n_total)*100),
        horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
ax.text(0.01,0.92,'true negative  = {}/{} ({:.1f}%)'.format(tn, n_total, (tn/n_total)*100),
        horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
ax.text(0.01,0.87,'false positive = {}/{} ({:.1f}%)'.format(fp, n_total, (fp/n_total)*100),
        horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
ax.text(0.01,0.82,'false negative = {}/{} ({:.1f}%)'.format(fn, n_total, (fn/n_total)*100),
        horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
ax.text(0.01,0.77,'correct predictions = {}/{} ({:.1f}%)'.format(tp+tn, n_total, ((tp+tn)/n_total)*100),
        horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontweight='bold')
fig.savefig('../../visualizations/robust-test-prediction-histograms.jpg', bbox_inches='tight', dpi=300)
plt.show()