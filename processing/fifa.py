import numpy as np
import pandas as pd


def get_processed_data(sample):
    # loading
    raw_tb = pd.read_csv('data/fifa.csv')
    raw_tb = raw_tb[:sample]
    selected_columns = ['Age','Wage','Crossing', 'Finishing', 'BallControl','Curve','LongPassing', 'Agility','ShotPower','Stamina','LongShots','Aggression','Positioning', 'Marking']
#Crossing,Finishing,HeadingAccuracy,ShortPassing,Volleys,Dribbling,Curve,FKAccuracy,LongPassing,BallControl,Acceleration,SprintSpeed,Agility,Reactions,Balance,ShotPower,Jumping,Stamina,Strength,LongShots,Aggression,Interceptions,Positioning,Vision,Penalties,Composure,Marking,StandingTackle,SlidingTackle,GKDiving,GKHandling,GKKicking,GKPositioning,GKReflexes
    _tb = raw_tb[selected_columns]

    _tb.loc[:, 'Wage'] = pd.to_numeric(_tb.loc[:, 'Wage'].str[3:-1])
    #_tb['Wage'].astype(float)

    # values = _tb.values
    # print values.sum()

    return np.nan_to_num(_tb.values)