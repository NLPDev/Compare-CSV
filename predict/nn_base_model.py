from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers

import pandas as pd

def create_F1_F2_cols(cols):
    F12_cols = []
    for x in cols:
        F12_cols.append('F1_' + x)
        F12_cols.append('F2_' + x)
    return F12_cols


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

    return X_train, X_test, y_train, y_test


def get_random_train_test_split(bouts_data):
    # get ready for deep learning
    X, y = bouts_data.iloc[:, 1:], bouts_data.iloc[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # normalization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled


def get_scaled (X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler


def model_train(X_train_scaled,  y_train, epochs=200):
    model = Sequential()

    model.add(Dense(16, input_dim=X_train_scaled.shape[1],
                    activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

    model.fit(x=X_train_scaled, y=y_train, epochs=epochs, batch_size=64, verbose=0)
    return model


def model_eval(model, X_test_scaled, y_test, return_score=False):
    test_results = model.evaluate(x=X_test_scaled, y=y_test, verbose=0)
    if return_score:
        return test_results[1]
    else:
        return print("Test Accuracy = {}".format(test_results[1]))


def get_predictions(fm_bd, model_cols, model, scaler):
    fm_bd_pred = fm_bd[model_cols]

    X = fm_bd_pred.iloc[:, 1:].dropna()
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)

    X['Pred'] = preds

    fm_bd['F1_P_Win'] = X['Pred']
    fm_bd['F2_P_Win'] = 1 - fm_bd['F1_P_Win']

    return fm_bd

#Data
fm_bd_all=pd.read_csv('fights_all_pre.csv')

fm_bd_model = fm_bd_all.copy()



model_cols = ['B_F1_Bool_Result'] + create_F1_F2_cols(['Reach','Height','Age','Exp','Win_PCT', 'Open','Close_Best'])
model_cols_w_date = model_cols + ['Event_Date'] #needed to create a train/test split
fm_bd_model = fm_bd_model[model_cols_w_date]
fm_bd_model = fm_bd_model.dropna()

X_train, X_test, y_train, y_test = get_train_test_split(fm_bd_model)
X_train_scaled, X_test_scaled, scaler = get_scaled (X_train, X_test) #Normalizing

#Training
model = model_train(X_train_scaled, y_train, epochs=300)
model_eval(model, X_test_scaled, y_test, return_score=False)

#Generating Predictions
df_w_predicts = get_predictions(fm_bd_all[fm_bd_all['Event_Date']>='2018'], model_cols, model, scaler)