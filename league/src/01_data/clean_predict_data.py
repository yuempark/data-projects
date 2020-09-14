import pandas as pd
pd.set_option('display.max_columns', 999)
import numpy as np

from clean_raw_data import clean_raw_data

# clean
all_predict_data = pd.read_csv('../../data/04_raw_predict_data.csv')
all_prematch_predict_data = clean_raw_data(all_predict_data)

# get rid of rows that were in the training data
prematch_data = pd.read_csv('../../data/01_data.csv')
drop_inds = []
for i in all_prematch_predict_data.index:
    if i==len(prematch_data):
        break
    if all_prematch_predict_data['gameid'][i] == prematch_data['gameid'][i]:
        drop_inds.append(i)
prematch_predict_data = all_prematch_predict_data.drop(drop_inds)
prematch_predict_data.reset_index(drop=True, inplace=True)

# save
prematch_predict_data.to_csv('../../data/05_predict_data.csv', index=False)