import pandas as pd
import numpy as np
import re
import datetime
import math

# df = pd.read_csv("fights_all_pre.csv")
df = pd.read_csv("fights_all_pre.csv")



df_len = len(df.iloc[0, :]) - 1

select_cols = []

for i in range(df_len):

    if type(df.iloc[0, i+1]) is np.float64:
        if math.isnan(df.iloc[0, i+1]) == False:
            select_cols.append(i+1)
    elif type(df.iloc[0, i+1]) is np.float:
        if math.isnan(df.iloc[0, i+1]) == False:
            select_cols.append(i+1)
    # elif type(df.iloc[0, i+1]) is np.str:
    #     # print(df.iloc[0, i+1])
    #     if df.iloc[0, i+1].isdigit():
    #         select_cols.append(i+1)
        # elif re.match("\d+?-\d+?-\d+?", df.iloc[0, i+1]):
        #     select_cols.append(i + 1)


res_df=df.iloc[:, select_cols]

# print(res_df.iloc[0, :])
# print(list(res_df))
list_pop = list(res_df)
list_res = ['B_F1_Bool_Result', 'Event_Date']
list_pop.pop()

for i in range(150):
    list_pop.pop()
list_res = list_res + list_pop
# print(list_res)

# print()

print(list_res.index('Event_Date'))


