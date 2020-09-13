import pandas as pd
pd.set_option('display.max_columns', 999)
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict

import joblib

# read in the data
x = pd.read_csv('../../data/02_x.csv')
y = pd.read_csv('../../data/03_y.csv')

# import model
best_GBC = joblib.load('best_GBC.pkl')

# the folds
kfolds = KFold(n_splits=4, shuffle=True, random_state=1992)

# ROC curve
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

fig, ax = plt.subplots(figsize=(7,7))
for i, (train, test) in enumerate(kfolds.split(x.values, y.values)):
    best_GBC.fit(x.values[train], y.values[train])
    viz = plot_roc_curve(best_GBC, x.values[test], y.values[test],
                         name='fold {}'.format(i),
                         alpha=0.3, lw=1, ax=ax)
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(mean_fpr, mean_tpr, color='b',
        label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
        lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                label=r'$\pm$ 1 std. dev.')

ax.axis('equal')
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.05)
ax.legend(loc="lower right")
fig.savefig('../../visualizations/ROC-curve.jpg', bbox_inches='tight', dpi=300)
plt.show()

# cross-validated predictions
y_predictions = cross_val_predict(best_GBC, x.values, y.values.reshape((-1,)),
                                  cv=kfolds, method='predict_proba')
y_predictions = y_predictions[:,1]

fig, ax = plt.subplots()
bins=np.linspace(0,1,40)
ax.hist(y_predictions[y.values.reshape((-1,))==1],
        bins=bins, alpha=0.5, label='blue win', facecolor='C0')
ax.hist(y_predictions[y.values.reshape((-1,))==0],
        bins=bins, alpha=0.5, label='blue loss', facecolor='C1')
ax.set_xlabel('predicted blue win probability')
ax.legend()
fig.savefig('../../visualizations/prediction-histograms.jpg', bbox_inches='tight', dpi=300)
plt.show()

# feature importances
features = x.columns
feature_importances = best_GBC.feature_importances_
sort_ind = np.argsort(feature_importances)[::-1]
features = features[sort_ind]
feature_importances = feature_importances[sort_ind]

fig, ax = plt.subplots(figsize=(10,4))
n_top_features = len(feature_importances)
x_inds = np.arange(n_top_features)
ax.bar(x_inds, feature_importances[:n_top_features])
ax.set_xticks(x_inds)
ax.set_xticklabels(features[:n_top_features], rotation=270)
ax.set_ylabel('feature importance')
fig.savefig('../../visualizations/feature-importances.jpg', bbox_inches='tight', dpi=300)
plt.show(fig)