from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import griddata
import numpy as np
from numpy import array

data = [[891, 0.00161, 38, 23],
        [98.9, 0.00185, -20, -5],
        [-374, 0.0019, -10, 37],
        [-35.1, 0.0034, 5, -7],
        [50.2, 0.00345, -49, -1],
        [-392, 0.00541, -17, 23],
        [-1350, 0.00642, 38, -36],
        [618, 0.00668, -23, -27],
        [203, 0.00685, -25, -8],
        [-1630, 0.00701, 38, -43],
        [-1590, 0.00733, -33, 48],
        [767, 0.008, -41, -19],
        [246, 0.00835, -7, -35],
        [269, 0.00839, -6, -45],
        [506, 0.0109, -16, -32],
        [583, 0.0142, -13, -45],
        [-24, 0.0159, 6, -4],
        [22.5, 0.0162, -2, -11],
        [423, 0.0172, -30, -14],
        [-965, 0.019, -47, 20],
        [-973, 0.0205, -25, 39]]

data = np.array(data)
data_num = len(data)

z = data[:, 0]
x = data[:, 1]
y = data[:, 2]
t = data[:, 3]

points = data[:, 1:]
test_pts = array([[0.0205, -25, 39], [0.0172, -30, -14]])


# first method
print(griddata(points, z, test_pts))

# second data
linInter = LinearNDInterpolator(points, z)
print(linInter(test_pts))


#---------


import numpy as np
import scipy as sp
import pandas as pd
import collections
import re


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import neighbors
from sklearn.metrics import mean_squared_error

from math import sqrt
import math

from sklearn import preprocessing

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.style.use('ggplot')


#
# def create_F1_F2_cols(cols):
#     F12_cols = []
#     for x in cols:
#         F12_cols.append('F1_' + x)
#         F12_cols.append('F2_' + x)
#     return F12_cols

def get_train_test_split(df, test_params=('date_cutoff', '2018'), drop_nan=True, bool_result_only=True):
    """
    creates a test set of the last test_size percent of rows in the given dataframe.

    :param df: fm_bd type of DF. 1st col must be F1_Bool_Result, one of the cols must be 'Event_Date'
    :param test_params: ('date_cutoff', '2018'), ('pct_last_rows', 0.15)
    :param drop_nan:
    :param bool_result_only:
    :return:
    """
    # sort by date
    df = df.sort_values('Event_Date')

    if bool_result_only:
        df['B_F1_Bool_Result'] = pd.to_numeric(df['B_F1_Bool_Result'], errors='coerce')
        df['B_F1_Bool_Result'].astype('bool', inplace=True)
    if drop_nan:
        df = df.dropna()

    # print(df)

    # Create train and test splits
    if test_params[0] == 'pct_last_rows':
        total_rows = df.shape[0]
        rows_test = int(round(total_rows*test_params[1], 0))

        df_test = df.drop(columns='Event_Date').tail(rows_test)
        df_train = df.drop(columns='Event_Date').head(int(total_rows-rows_test))
    elif test_params[0] == 'date_cutoff':
        df_test = df[df['Event_Date']>=test_params[1]].drop(columns='Event_Date')
        df_train = df[df['Event_Date']<test_params[1]].drop(columns='Event_Date')


    X_train, y_train = df_train.iloc[:, 1:], df_train.iloc[:, 0]
    X_test, y_test = df_test.iloc[:, 1:], df_test.iloc[:, 0]

    # print(y_test.iloc[0])

    return X_train, X_test, y_train, y_test


def get_scaled(X_train, X_test):
    scaler = MinMaxScaler(feature_range=(-1, 1))

    # scaler = preprocessing.StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    # print(X_train_scaled)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler

def create_F1_F2_cols(col_base_list, output='both'):
    """
    generates a list of F1 and F2 columns from col_base_list. Works with 'FM_*' columns.

    :param col_base_list:
    :param output: 'both', returns both F1 and F2 columns
                    'F1', returns only F1 columns
                    'F2', returns only F2 columns
    :return:
    """
    F12_cols = []
    for x in col_base_list:
        pref = x[:3]
        if output == 'both':
            if pref =='FM_':
                F12_cols.append('FM_F1_'+ x[3:])
                F12_cols.append('FM_F2_' + x[3:])
            else:
                F12_cols.append('F1_' + x)
                F12_cols.append('F2_' + x)
        elif output =='F1':
            if pref =='FM_':
                F12_cols.append('FM_F1_'+ x[3:])
            else:
                F12_cols.append('F1_' + x)
        elif output =='F2':
            if pref =='FM_':
                F12_cols.append('FM_F2_'+ x[3:])
            else:
                F12_cols.append('F2_' + x)
    return F12_cols


def change_col_prefix(df, old_prefix, new_prefix ):
    """
    changes the column names from old_prefix to new_prefix

    :param df:
    :param old_prefix:
    :param new_prefix:
    :return: df with renamed columns
    """
    op_regex = old_prefix + '.+'
    op_cols = list(df.filter(regex=op_regex).columns)
    np_cols = [col.replace(old_prefix,new_prefix) for col in op_cols]
    rename_map = {x[0]:x[1] for x in zip(op_cols, np_cols)}
    return df.rename(columns=rename_map)


def do_F1_F2_operation (df, cols, operation):
    """
    performs an operation on the df

    :param df: source df to perform the operation on
    :param cols: columns to perform the operation on
    :param operation: 'subtract', df_F1 - df_F2
                      'over', df_F1 / df_F2
                      pct_F1, (df_F2 - df_F1)/df_F1
    :return:
    """
    df_f1 = df[create_F1_F2_cols(cols, output='F1')]
    df_f2 = df[create_F1_F2_cols(cols, output='F2')]

    if operation =='subtract':
        df_f1 = change_col_prefix(df_f1, 'F1_', 'F1mF2_')
        df_f1 = change_col_prefix(df_f1, 'FM_F1_', 'FM_F1mF2_')
        df_f2 = change_col_prefix(df_f2, 'F2_', 'F1mF2_')
        df_f2 = change_col_prefix(df_f2, 'FM_F2_', 'FM_F1mF2_')
        df_diff = df_f1 - df_f2
    elif operation == 'over':
        df_f1 = change_col_prefix(df_f1, 'F1_', 'F1oF2_')
        df_f1 = change_col_prefix(df_f1, 'FM_F1_', 'FM_F1oF2_')
        df_f2 = change_col_prefix(df_f2, 'F2_', 'F1oF2_')
        df_f2 = change_col_prefix(df_f2, 'FM_F2_', 'FM_F1oF2_')
        df_diff = df_f1/df_f2
    elif operation == 'pct_F1':
        df_f1 = change_col_prefix(df_f1, 'F1_', 'pct_F1_')
        df_f1 = change_col_prefix(df_f1, 'FM_F1_', 'pct_FM_F1_')
        df_f2 = change_col_prefix(df_f2, 'F2_', 'pct_F1_')
        df_f2 = change_col_prefix(df_f2, 'FM_F2_', 'pct_FM_F1_')
        df_diff = (df_f2 - df_f1)/df_f1

    return df_diff

output_cols = ['Event_Date']
def get_columns (df_f1_f2, col_comb):
    """

    :param df_f1_f2:
    :param col_comb: format is ( ([col_base_],'type_1'), ([col_base],'type_1') )
                               ( (['B_F1_Bool_Result'],'direct'), (['Exp', 'FM_5yr_T_SStr_L'], 'F1oF2') )
                col_base: base of the column name or column name w/o F1, F2
                          e.g. {F1_Age: Age, FM_F1_5yr_T_SStr_L: FM_5yr_T_SStr_L)

                types: direct - puts columns directly as listed
                       F1_F2 - retrieves F1_col_base, F2_col_base
                       F1mF1 - computes, then retrieves (F1_col_base - F2_col_base)
                       F1oF1 - computes, then retrieves (F1_col_base / F2_col_base)
                       pct_F1 - computes, then retrieves (F2_col_base - F1_col_base)/F1_col_base

    :return:
    """
    df = df_f1_f2.copy()
    d_cols = F1_F2_cols = F1mF2_cols = F1oF2_cols = pct_F1_cols = []
    for x in col_comb:
        cols = x[0]

        if x[1]=='direct':
            d_cols = cols

        elif x[1]=='F1_F2':
            F1_F2_cols = create_F1_F2_cols(cols)

        elif x[1] == 'F1mF2':
            _df = do_F1_F2_operation(df, cols, operation='subtract') #columns don't exist need to compute
            df = pd.merge(df, _df, left_index=True, right_index=True)
            F1mF2_cols = list(_df.columns)

        elif x[1] == 'F1oF2':
            _df = do_F1_F2_operation(df, cols, operation='over')
            df = pd.merge(df, _df, left_index=True, right_index=True)
            F1oF2_cols = list(_df.columns)

        elif x[1] == 'pct_F1':
            _df = do_F1_F2_operation(df, cols, operation='pct_F1')
            df = pd.merge(df,_df, left_index=True, right_index=True )
            pct_F1_cols = list(_df.columns)

    global output_cols

    output_cols = ['Event_Date']
    output_cols = d_cols + F1_F2_cols + F1mF2_cols + F1oF2_cols + pct_F1_cols + output_cols

    # print(output_cols)

    return df[output_cols]

#Sample
cols = ((['B_F1_Bool_Result'],'direct'),
        (['Age','Reach'],'F1_F2'),
        (['Height', 'FM_5yr_T_SStr_L', 'Open'], 'F1mF2'),
        (['Exp', 'FM_5yr_T_SStr_L', 'Close_Worst'], 'F1oF2'),
        (['cum_Win', 'FM_5yr_T_TD_L', 'Close_Best'], 'pct_F1'))


def pre_get_data(df):
    df_len = len(df.iloc[0, :]) - 1

    select_cols = []

    for i in range(df_len):

        if type(df.iloc[0, i + 1]) is np.float64:
            if math.isnan(df.iloc[0, i + 1]) == False:
                select_cols.append(i + 1)
        elif type(df.iloc[0, i + 1]) is np.float:
            if math.isnan(df.iloc[0, i + 1]) == False:
                select_cols.append(i + 1)


    res_df = df.iloc[:, select_cols]

    list_pop = list(res_df)
    list_res = ['B_F1_Bool_Result', 'Event_Date']
    list_pop.pop()

    for i in range(150):
        list_pop.pop()

    list_res = list_res + list_pop

    return df[list_res]



#Data
fm_bd_all=pd.read_csv('fights_all.csv')

fm_bd_all['F2_Open'] = 5