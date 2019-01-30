import numpy as np
import pandas as pd

raw_tb = pd.read_csv('data/fifa.csv')
raw_tb = raw_tb[:5000]
selected_columns = ['Age','Wage','Crossing', 'Finishing', 'Curve']

_tb = raw_tb[selected_columns]

# _tb.loc[:,'Wage'] = _tb.loc[:,'Wage'].str[3:-1]
# _tb.loc[:,'Wage'] = pd.to_numeric(_tb.loc[:,'Wage'])

#.fillna(0, inplace=True))
_tb.loc[:,'Wage'] = pd.to_numeric(_tb.loc[:,'Wage'].str[3:-1])
#_tb.loc[:,'Wage'] = (_tb.loc[:,'Wage']).fillna(0, inplace=True)

#_tb['Wage'].astype(float)

# values = _tb.values
# print values.sum()

print(np.nan_to_num(_tb.values))