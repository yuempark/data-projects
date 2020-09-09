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
# so create features that shows pick/ban rate per patch
picks = ['1','2','3','4','5']
for pick in picks:
    prematch_data['blue_pick'+pick+'_rate'] = np.zeros(len(prematch_data))
    prematch_data['blue_pick'+pick+'_ban_rate'] = np.zeros(len(prematch_data))
    prematch_data['red_pick'+pick+'_rate'] = np.zeros(len(prematch_data))
    prematch_data['red_pick'+pick+'_ban_rate'] = np.zeros(len(prematch_data))
    for patch in unique_patch:
        patch_slice = prematch_data[prematch_data['patch']==patch]
        total_patch_games = len(patch_slice)
        blue_pick_rate = np.zeros(total_patch_games)
        blue_pick_ban_rate = np.zeros(total_patch_games)
        red_pick_rate = np.zeros(total_patch_games)
        red_pick_ban_rate = np.zeros(total_patch_games)
        for i in range(total_patch_games):
            blue_pick = patch_slice['blue_pick'+pick].iloc[i]
            red_pick = patch_slice['red_pick'+pick].iloc[i]
            blue_pick_rate[i] = np.sum(patch_slice['blue_pick'+pick]==blue_pick)/ \
                                total_patch_games
            blue_pick_ban_rate[i] = (np.sum(patch_slice['blue_pick'+pick]==blue_pick)+ \
                                     np.sum(patch_slice['blue_ban1']==blue_pick)+ \
                                     np.sum(patch_slice['blue_ban2']==blue_pick)+ \
                                     np.sum(patch_slice['blue_ban3']==blue_pick)+ \
                                     np.sum(patch_slice['blue_ban4']==blue_pick)+ \
                                     np.sum(patch_slice['blue_ban5']==blue_pick)+ \
                                     np.sum(patch_slice['red_ban1']==blue_pick)+ \
                                     np.sum(patch_slice['red_ban2']==blue_pick)+ \
                                     np.sum(patch_slice['red_ban3']==blue_pick)+ \
                                     np.sum(patch_slice['red_ban4']==blue_pick)+ \
                                     np.sum(patch_slice['red_ban5']==blue_pick))/ \
                                    total_patch_games
            red_pick_rate[i] = np.sum(patch_slice['red_pick'+pick]==red_pick)/ \
                                      total_patch_games
            red_pick_ban_rate[i] = (np.sum(patch_slice['red_pick'+pick]==red_pick)+ \
                                    np.sum(patch_slice['blue_ban1']==red_pick)+ \
                                    np.sum(patch_slice['blue_ban2']==red_pick)+ \
                                    np.sum(patch_slice['blue_ban3']==red_pick)+ \
                                    np.sum(patch_slice['blue_ban4']==red_pick)+ \
                                    np.sum(patch_slice['blue_ban5']==red_pick)+ \
                                    np.sum(patch_slice['red_ban1']==red_pick)+ \
                                    np.sum(patch_slice['red_ban2']==red_pick)+ \
                                    np.sum(patch_slice['red_ban3']==red_pick)+ \
                                    np.sum(patch_slice['red_ban4']==red_pick)+ \
                                    np.sum(patch_slice['red_ban5']==red_pick))/ \
                                   total_patch_games
        prematch_data.loc[prematch_data['patch']==patch, 'blue_pick'+pick+'_rate'] = blue_pick_rate
        prematch_data.loc[prematch_data['patch']==patch, 'blue_pick'+pick+'_ban_rate'] = blue_pick_ban_rate
        prematch_data.loc[prematch_data['patch']==patch, 'red_pick'+pick+'_rate'] = red_pick_rate
        prematch_data.loc[prematch_data['patch']==patch, 'red_pick'+pick+'_ban_rate'] = red_pick_ban_rate
    
# distribution of pick/ban rates
fig, ax = plt.subplots(nrows=5, ncols=2, figsize=(10,10), sharex=True, sharey=True)
for i in range(len(picks)):
    blue_win_slice = prematch_data[(prematch_data['blue_win']==1)]
    blue_loss_slice = prematch_data[(prematch_data['blue_win']==0)]
    bins = np.arange(0,1.05,0.05)
    ax[i][0].hist(blue_win_slice['blue_pick'+picks[i]+'_ban_rate'], bins=bins, facecolor='C0', label='blue win', alpha=0.5)
    ax[i][0].hist(blue_loss_slice['blue_pick'+picks[i]+'_ban_rate'], bins=bins, facecolor='C1', label='blue loss', alpha=0.5)
    ax[i][0].set_ylabel('blue_pick'+picks[i]+'_ban_rate')
    ax[i][1].hist(blue_win_slice['red_pick'+picks[i]+'_ban_rate'], bins=bins, facecolor='C0', label='blue win', alpha=0.5)
    ax[i][1].hist(blue_loss_slice['red_pick'+picks[i]+'_ban_rate'], bins=bins, facecolor='C1', label='blue loss', alpha=0.5)
    ax[i][1].set_ylabel('red_pick'+picks[i]+'_ban_rate')
ax[0][0].set_xlim(0,1)
ax[0][0].legend()
plt.show(fig)

# split X and Y
y = prematch_data['blue_win'].copy()
x = prematch_data.drop(columns=['gameid','blue_win']).copy()

# target encode the categorical features
def target_encode(col, weight=None):
    total_mean = np.mean(y)
    cats = x[col].unique()
    if weight==None:
        weight = int((len(x)/len(cats))/2)
    for cat in cats:
        if pd.isnull(cat):
            if len(x[x[col].isna()])==0:
                x[col].fillna(total_mean, inplace=True)
            else:
                cat_mean = np.mean(y[x[col].isna()])
                count = len(x[x[col].isna()])
                x.loc[x[col].isna(), col] = (count*cat_mean + weight*total_mean)/(count+weight)
        else:
            if len(x[x[col]==cat])==0:
                x.loc[x[col]==cat, col] = total_mean
            else:
                cat_mean = np.mean(y[x[col]==cat])
                count = len(x[x[col]==cat])
                x.loc[x[col]==cat, col] = (count*cat_mean + weight*total_mean)/(count+weight)
                
TE_features = ['league','split','blue_team','red_team',
               'blue_ban1','blue_ban2','blue_ban3','blue_ban4','blue_ban5',
               'red_ban1','red_ban2','red_ban3','red_ban4','red_ban5',
               'blue_pick1','blue_pick2','blue_pick3','blue_pick4','blue_pick5',
               'red_pick1','red_pick2','red_pick3','red_pick4','red_pick5']
for col in TE_features:
    target_encode(col)
    
x.to_csv('../../data/02_x.csv', index=False)
y.to_csv('../../data/03_y.csv', index=False)