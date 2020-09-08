import pandas as pd
pd.set_option('display.max_columns', 999)
import numpy as np

all_data = pd.read_csv('../../data/00_raw_data.csv')

# extract the team data only
blue_data = all_data[all_data['playerid']==100]
blue_data.reset_index(drop=True, inplace=True)

red_data = all_data[all_data['playerid']==200]
red_data.reset_index(drop=True, inplace=True)

n_matches = len(blue_data)

# we only want prematch features
prematch_team_features = ['gameid',
                          'league',
                          'split',
                          'playoffs',
                          'date',
                          'game',
                          'patch',
                          'team',
                          'ban1',
                          'ban2',
                          'ban3',
                          'ban4',
                          'ban5',
                          'result']

prematch_blue_data = blue_data[prematch_team_features].copy()
prematch_red_data = red_data[prematch_team_features].copy()

# add player champions
for i in range(n_matches):
    
    # use the time if the gameid is null
    if pd.isnull(prematch_blue_data['gameid'][i]):
        prematch_blue_data.loc[i,'gameid'] = prematch_blue_data['date'][i]
        prematch_red_data.loc[i,'gameid'] = prematch_red_data['date'][i]
        
        blue_game_slice = all_data[(all_data['date']==prematch_blue_data['gameid'][i]) &
                                   (all_data['side']=='Blue')]
        red_game_slice = all_data[(all_data['date']==prematch_red_data['gameid'][i]) &
                                  (all_data['side']=='Red')]
        
    # otherwise just use the gameid
    else:
        blue_game_slice = all_data[(all_data['gameid']==prematch_blue_data['gameid'][i]) &
                                   (all_data['side']=='Blue')]
        red_game_slice = all_data[(all_data['gameid']==prematch_red_data['gameid'][i]) &
                                  (all_data['side']=='Red')]
        
    # add columns for players
    for j in range(5):
        prematch_blue_data.loc[i,'pick'+str(j+1)] = blue_game_slice['champion'].iloc[j]
        prematch_red_data.loc[i,'pick'+str(j+1)] = red_game_slice['champion'].iloc[j]
    
# check that the gameids match
if (prematch_blue_data['gameid']!=prematch_red_data['gameid']).sum() != 0:
    print('!!! gameid mismatch between red and blue !!!')
    
# combine blue and red team data
prematch_data = pd.DataFrame({'gameid':prematch_blue_data['gameid'],
                              'league':prematch_blue_data['league'],
                              'split':prematch_blue_data['split'],
                              'playoffs':prematch_blue_data['playoffs'],
                              'game':prematch_blue_data['game'],
                              'patch':prematch_blue_data['patch'],
                              'blue_team':prematch_blue_data['team'],
                              'red_team':prematch_red_data['team'],
                              'blue_ban1':prematch_blue_data['ban1'],
                              'blue_ban2':prematch_blue_data['ban2'],
                              'blue_ban3':prematch_blue_data['ban3'],
                              'blue_ban4':prematch_blue_data['ban4'],
                              'blue_ban5':prematch_blue_data['ban5'],
                              'red_ban1':prematch_red_data['ban1'],
                              'red_ban2':prematch_red_data['ban2'],
                              'red_ban3':prematch_red_data['ban3'],
                              'red_ban4':prematch_red_data['ban4'],
                              'red_ban5':prematch_red_data['ban5'],
                              'blue_pick1':prematch_blue_data['pick1'],
                              'blue_pick2':prematch_blue_data['pick2'],
                              'blue_pick3':prematch_blue_data['pick3'],
                              'blue_pick4':prematch_blue_data['pick4'],
                              'blue_pick5':prematch_blue_data['pick5'],
                              'red_pick1':prematch_red_data['pick1'],
                              'red_pick2':prematch_red_data['pick2'],
                              'red_pick3':prematch_red_data['pick3'],
                              'red_pick4':prematch_red_data['pick4'],
                              'red_pick5':prematch_red_data['pick5'],
                              'blue_win':prematch_blue_data['result']})

# save
prematch_data.to_csv('../../data/01_data.csv', index=False)