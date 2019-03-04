import pandas as pd
import numpy as np

get_weight = {
    "BW":61.2,
    "LHW":93.0,
    "WW":77.1,
    "HW":120.2,
    "LW":70.3,
    "MW":83.9,
    "FW":65.8,
    "FlyW":56.7,
    "W_FW":65.8,
    "W_SW":52.2,
    "W_FlyW":56.7,
    "W_BW":61.2
}

fm_bd_all=pd.read_csv('fights_all.csv')

df = fm_bd_all
bw = fm_bd_all['B_WClass']

result = []

add_list = []

for item in list(fm_bd_all):

    if "F2" not in item:
        print(item)