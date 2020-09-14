import pandas as pd
pd.set_option('display.max_columns', 999)
import numpy as np

from clean_raw_data import clean_raw_data

# clean
all_test_data = pd.read_csv('../../data/04_raw_test_data.csv')
all_prematch_test_data = clean_raw_data(all_test_data)

# get rid of rows that were in the training data
prematch_data = pd.read_csv('../../data/01_data.csv')
drop_inds = []
for i in all_prematch_test_data.index:
    if i==len(prematch_data):
        break
    if all_prematch_test_data['gameid'][i] == prematch_data['gameid'][i]:
        drop_inds.append(i)
prematch_test_data = all_prematch_test_data.drop(drop_inds)
prematch_test_data.reset_index(drop=True, inplace=True)

# save
prematch_test_data.to_csv('../../data/05_test_data.csv', index=False)