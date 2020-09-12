import pandas as pd
pd.set_option('display.max_columns', 999)
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import roc_auc_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import uniform
from scipy.stats import loguniform

import joblib

# read in the data
x = pd.read_csv('../../data/02_x.csv')
y = pd.read_csv('../../data/03_y.csv')

# define a scorer - in this case ROC AUC
def get_AUC(Y, Y_prediction):
    AUC = roc_auc_score(Y, Y_prediction)
    return AUC

# initialize the scorer
scorer = make_scorer(get_AUC, greater_is_better=True)

# the folds
kfolds = KFold(n_splits=4, shuffle=True, random_state=1992)

# random search function
def random_search(estimator, param_distributions, n_iter):
    RSCV = RandomizedSearchCV(estimator=estimator,
                              n_iter=n_iter,
                              param_distributions=param_distributions,
                              cv=kfolds,
                              scoring=scorer,
                              random_state=2020,
                              verbose=1)
    RSCV.fit(x.values, y.values)
    RSCV_results = pd.DataFrame(RSCV.cv_results_)
    
    return RSCV, RSCV_results

# make simple plots of the random search
def plot_random_search_results(RSCV_results, ylim=None):
    n_params = 0
    for col in RSCV_results.columns:
        if 'param_' in col:
            n_params = n_params + 1
            
    fig, ax = plt.subplots(nrows=n_params, ncols=1, sharey=True, figsize=(4,10))
    i = 0
    for col in RSCV_results.columns:
        if 'param_' in col:
            ax[i].scatter(RSCV_results[col], RSCV_results['mean_test_score'])
            ax[i].set_ylabel(col)
            if ylim!=None:
                ax[i].set_ylim(ylim)
            i = i + 1
    fig.tight_layout()
    fig.savefig('../../visualizations/tuning-loss.jpg', bbox_inches='tight', dpi=300)
    plt.show()

# define the search space
param_distributions = [{'learning_rate': loguniform(a=0.05,b=0.2),
                        'subsample': np.linspace(0.5,1.0,6),
                        'max_features': np.linspace(0.5,1.0,6),
                        'min_samples_leaf': np.arange(2,10+2,2,dtype=int),
                        'max_depth': np.arange(2,5+1,1,dtype=int)},]
GBC = GradientBoostingClassifier(random_state=2020, n_estimators=100)

# run the search
n_iter = 50
GBC_RSCV, GBC_RSCV_results = random_search(GBC, param_distributions, n_iter)

# plot the results
plot_random_search_results(GBC_RSCV_results)

# show the estimator
print(GBC_RSCV.best_estimator_)

# the best performing classifier
best_GBC = GBC_RSCV.best_estimator_
joblib.dump(best_GBC, 'best_GBC.pkl') 