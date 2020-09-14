import pandas as pd
pd.set_option('display.max_columns', 999)
import numpy as np

# read in the training data
prematch_data = pd.read_csv('../../data/01_data.csv')
x = pd.read_csv('../../data/02_x.csv')
y = pd.read_csv('../../data/03_y.csv')

# read in the test data
prematch_test_data = pd.read_csv('../../data/05_test_data.csv')

# features that need to have their values copied over from the training data
TE_features = ['league','split','blue_team','red_team',
               'blue_ban1','blue_ban2','blue_ban3','blue_ban4','blue_ban5',
               'red_ban1','red_ban2','red_ban3','red_ban4','red_ban5',
               'blue_pick1','blue_pick2','blue_pick3','blue_pick4','blue_pick5',
               'red_pick1','red_pick2','red_pick3','red_pick4','red_pick5']
rate_features = ['blue_pick1_rate','blue_pick2_rate','blue_pick3_rate','blue_pick4_rate','blue_pick5_rate',
                 'red_pick1_rate','red_pick2_rate','red_pick3_rate','red_pick4_rate','red_pick5_rate',
                 'blue_pick1_ban_rate','blue_pick2_ban_rate','blue_pick3_ban_rate','blue_pick4_ban_rate','blue_pick5_ban_rate',
                 'red_pick1_ban_rate','red_pick2_ban_rate','red_pick3_ban_rate','red_pick4_ban_rate','red_pick5_ban_rate']

# copy the rate values over for the correct patch
unique_patch = prematch_test_data['patch'].unique()

picks = ['1','2','3','4','5']
for pick in picks:
    prematch_test_data['blue_pick'+pick+'_rate'] = np.zeros(len(prematch_test_data))
    prematch_test_data['blue_pick'+pick+'_ban_rate'] = np.zeros(len(prematch_test_data))
    prematch_test_data['red_pick'+pick+'_rate'] = np.zeros(len(prematch_test_data))
    prematch_test_data['red_pick'+pick+'_ban_rate'] = np.zeros(len(prematch_test_data))

    for patch in unique_patch:
        
        prematch_test_data_patch_slice = prematch_test_data[prematch_test_data['patch']==patch]
        total_patch_games = len(prematch_test_data_patch_slice)
        blue_pick_rate = np.zeros(total_patch_games)
        blue_pick_ban_rate = np.zeros(total_patch_games)
        red_pick_rate = np.zeros(total_patch_games)
        red_pick_ban_rate = np.zeros(total_patch_games)
        
        # if the patch is in the training data
        if patch in prematch_data['patch'].values:
            prematch_data_patch_slice = prematch_data[prematch_data['patch']==patch]
            x_patch_slice = x[x['patch']==patch]
        
        # if it isn't use the latest patch in the training data
        else:
            print('patch {} not in training data.'.format(patch))
            train_patch = np.max(prematch_data['patch'].unique())
            prematch_data_patch_slice = prematch_data[prematch_data['patch']==train_patch]
            x_patch_slice = x[x['patch']==train_patch]
        
        # iterate through the test data
        for i in range(total_patch_games):
            blue_pick = prematch_test_data_patch_slice['blue_pick'+pick].iloc[i]
            red_pick = prematch_test_data_patch_slice['red_pick'+pick].iloc[i]
            
            # if this pick exists in the training data
            if blue_pick in prematch_data_patch_slice['blue_pick'+pick].values:
                
                # get values from the training data
                x_blue_pick_slice = x_patch_slice[prematch_data_patch_slice['blue_pick'+pick]==blue_pick]
                blue_pick_rate[i] = x_blue_pick_slice['blue_pick'+pick+'_rate'].iloc[0]
                blue_pick_ban_rate[i] = x_blue_pick_slice['blue_pick'+pick+'_ban_rate'].iloc[0]
                
            # otherwise use the mean value
            else:
                blue_pick_rate[i] = np.mean(x_patch_slice['blue_pick'+pick+'_rate'])
                blue_pick_ban_rate[i] = np.mean(x_patch_slice['blue_pick'+pick+'_ban_rate'])
                
            if red_pick in prematch_data_patch_slice['red_pick'+pick].values:
                
                # get values from the training data
                x_red_pick_slice = x_patch_slice[prematch_data_patch_slice['red_pick'+pick]==red_pick]
                red_pick_rate[i] = x_red_pick_slice['red_pick'+pick+'_rate'].iloc[0]
                red_pick_ban_rate[i] = x_red_pick_slice['red_pick'+pick+'_ban_rate'].iloc[0]
                
            # otherwise use the mean value
            else:
                red_pick_rate[i] = np.mean(x_patch_slice['red_pick'+pick+'_rate'])
                red_pick_ban_rate[i] = np.mean(x_patch_slice['red_pick'+pick+'_ban_rate'])
                
        prematch_test_data.loc[prematch_test_data['patch']==patch, 'blue_pick'+pick+'_rate'] = blue_pick_rate
        prematch_test_data.loc[prematch_test_data['patch']==patch, 'blue_pick'+pick+'_ban_rate'] = blue_pick_ban_rate
        prematch_test_data.loc[prematch_test_data['patch']==patch, 'red_pick'+pick+'_rate'] = red_pick_rate
        prematch_test_data.loc[prematch_test_data['patch']==patch, 'red_pick'+pick+'_ban_rate'] = red_pick_ban_rate
        
# copy values for the target encoded features
missing_category = []
missing_feature = []
missing_patch = []
prematch_test_data['missing'] = np.zeros(len(prematch_test_data), dtype=int)


for feature in TE_features:
    
    feature_missing_category = []
    feature_missing_feature = []
    
    for patch in unique_patch:
            
        prematch_test_data_patch_slice = prematch_test_data[prematch_test_data['patch']==patch]
        total_patch_games = len(prematch_test_data_patch_slice)
        encode_vals = np.zeros(total_patch_games)
        missing_flag = np.zeros(total_patch_games, dtype=int)
        
        # if the patch is in the training data
        if patch in prematch_data['patch'].values:
            prematch_data_patch_slice = prematch_data[prematch_data['patch']==patch]
            x_patch_slice = x[x['patch']==patch]
        
        # if it isn't use the latest patch in the training data
        else:
            train_patch = np.max(prematch_data['patch'].unique())
            prematch_data_patch_slice = prematch_data[prematch_data['patch']==train_patch]
            x_patch_slice = x[x['patch']==train_patch]
        
        # iterate through the test data
        for i in range(total_patch_games):
                
            category = prematch_test_data_patch_slice[feature].iloc[i]
            
            # if this pick exists in the training data
            if category in prematch_data_patch_slice[feature].values:
                
                # get values from the training data
                x_category_slice = x_patch_slice[prematch_data_patch_slice[feature]==category]
                encode_vals[i] = x_category_slice[feature].iloc[0]
                
            # otherwise use the mean value
            else:
                if category not in feature_missing_category:
                    feature_missing_category.append(category)
                    feature_missing_feature.append(feature)
                    missing_patch.append(patch)
                encode_vals[i] = np.mean(x_patch_slice[feature])
                missing_flag[i] = 1
                
        # replace values, and also flag rows with values that don't appear in the training data
        prematch_test_data.loc[prematch_test_data['patch']==patch, feature] = encode_vals
        prematch_test_data.loc[prematch_test_data['patch']==patch, 'missing'] = missing_flag|prematch_test_data['missing']
        missing_category = missing_category + feature_missing_category
        missing_feature = missing_feature + feature_missing_feature
        
# show what's missing
for i in range(len(missing_category)):
    print('{} not in {} : patch {}'.format(missing_category[i], missing_feature[i], missing_patch[i]))

# split X and Y
test_y = prematch_test_data['blue_win'].copy()
test_x_missing = prematch_test_data['missing'].copy()
test_x = prematch_test_data[x.columns].copy()

test_x.to_csv('../../data/06_test_x.csv', index=False)
test_y.to_csv('../../data/07_test_y.csv', index=False)
test_x_missing.to_csv('../../data/08_test_x_missing.csv', index=False)