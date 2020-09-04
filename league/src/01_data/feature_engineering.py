import pandas as pd
pd.set_option('display.max_columns', 999)
import numpy as np
import matplotlib.pyplot as plt

prematch_data = pd.read_csv('../../data/01_data.csv')
print('shape : {}'.format(prematch_data.shape))

# how often does blue win?
BW_frac = (prematch_data['blue_win']==1).sum() / len(prematch_data)
print('blue win fraction : {:.3f}'.format(BW_frac))

# blue win fraction plotter
def BW_frac_calc(feature):
    unique_feature = prematch_data[feature].unique()
    unique_feature_BW_frac = np.zeros(len(unique_feature))
    for i in range(len(unique_feature)):
        feature_slice = prematch_data[prematch_data[feature]==unique_feature[i]]['blue_win']
        unique_feature_BW_frac[i] = (feature_slice==1).sum() / len(feature_slice)
    sort_inds = np.argsort(unique_feature_BW_frac)
    unique_feature = unique_feature[sort_inds]
    unique_feature_BW_frac = unique_feature_BW_frac[sort_inds]
    return unique_feature, unique_feature_BW_frac

def BW_frac_plot(feature, unique_feature, unique_feature_BW_frac):
    fig, ax = plt.subplots()
    x_inds = np.arange(len(unique_feature))
    ax.bar(x_inds, unique_feature_BW_frac)
    ax.set_xticks(x_inds)
    ax.set_xticklabels(unique_feature, rotation=270)
    ax.set_xlabel(feature)
    ax.axhline(0.5, ls='--', c='k')
    ax.set_ylabel('blue win fraction')
    plt.show(fig)
    
    

# how do patches correlate with wins?
unique_patch, unique_patch_BW_frac = BW_frac_calc('patch')
BW_frac_plot('patch', unique_patch, unique_patch_BW_frac)

# obviously teams correlate with wins, so we won't plot that up...

# how do picks correlate with wins?
unique_blue_pick1, unique_blue_pick1_BW_frac = BW_frac_calc('blue_pick1')
BW_frac_plot('blue_pick1', unique_blue_pick1, unique_blue_pick1_BW_frac)

# we see that there are outliers for champs that are likely only rarely picked
# so create features that shows pick/ban rate
