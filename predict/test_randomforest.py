import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestClassifier
import math
import matplotlib

#Visualization
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import scikitplot as skplt


matplotlib.style.use('ggplot')

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
    list_res = ['B_F1_Bool_Result', 'Event_Date', 'B_WClass']

    list_pop.pop()
    for item in list_pop:
        if "F1" in item:
            aa = item
            bb = aa.replace("F1", "F2")
            if bb in list_pop:
                cc = aa.replace("F1", "F12")
                df[cc] = df[aa] / df[bb]
                list_res.append(cc)
        elif "F2" not in item:
            list_res.append(item)


    bw = df['B_WClass']
    i = -1
    j = df.columns.get_loc('B_WClass')

    for item in bw:
        i = i + 1
        if item != item:
            df.iloc[i, j] = np.nan
        else:
            df.iloc[i, j] = get_weight[item]
            # print(type(df.iloc[i, j]))

    df['B_WClass'] = df['B_WClass'].astype(float)
    return df[list_res]

def add_weight(df, source):
    bw = source['B_WClass']
    add_list = []

    for item in bw:
        if item != item:
            add_list.append(np.nan)
            continue
        else:
            add_list.append(get_weight[item])

    df['B_Weight'] = add_list

#Data
fm_bd_all=pd.read_csv('fights_all.csv')

fm_bd_model = fm_bd_all.copy()
fm_bd_model = pre_get_data(fm_bd_model)
fm_bd_model = fm_bd_model.dropna()
fm_bd_model=fm_bd_model[~fm_bd_model.isin([np.inf, -np.inf]).any(1)]


X_train, X_test, y_train, y_test = get_train_test_split(fm_bd_model)
X_train_scaled, X_test_scaled, scaler = get_scaled (X_train, X_test) #Normalizing

rmse_val = []

from rfpimp import importances

def get_feature_imp(model,X_train, y_train, X_test, y_test, return_n_top_fetures = 10):

    model.fit(X_train,y_train)
    imp = importances(model, X_test, y_test)
    # print(imp)
    return imp.head(n=return_n_top_fetures),imp

dropdata=fm_bd_model

top_10_concat_features, all_f_imp_concat = get_feature_imp(RandomForestClassifier(max_features="sqrt",n_estimators = 700,max_depth = None,n_jobs=-1), X_train, y_train, X_test, y_test)
top_pos = top_10_concat_features.index.values
add_pos = ['F12_Height', 'F12_Age', 'F12_Open', 'F12_Close_Best']
for item in add_pos:
    if item not in top_pos:
        top_pos = np.append(top_pos, item)
# print(list(fm_bd_model))
# print(type(top_pos))
# print(top_pos)
# exit()
X_train_pos = X_train[top_pos]
X_test_pos = X_test[top_pos]
#
rfc = RandomForestClassifier(max_features="sqrt",n_estimators = 700,max_depth = None,n_jobs=-1)
rfc.fit(X_train_pos, y_train)
pred = rfc.predict(X_test_pos)
# print(accuracy_score(y_test, pred))



def rfc_model(X_train, y_train, X_test, y_test, results):
    rfc = RandomForestClassifier(max_features="sqrt", n_estimators=700, max_depth=None, n_jobs=-1)
    rfc.fit(X_train, y_train)
    Y_pred = rfc.predict(X_test)
    results['RFC'] = {}
    results['RFC']['Accuracy'] = accuracy_score(y_test, Y_pred)
    results['RFC']['cm'] = confusion_matrix(y_test, Y_pred)
    results['RFC']['f1_macro'] = f1_score(y_test, Y_pred, average='macro')
    results['RFC']['f1_class'] = f1_score(y_test, Y_pred, average=None)
    results['RFC']['pred_prob'] = rfc.predict_proba(X_test)


def plot_cm(cm, title):
    plt.figure()
    labels = ['Blue', 'Draw', 'No Contest', 'Red']
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)

    fmt = 'd'
    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    # plt.show()


def pprint_results(results, Y_test):
    for model in results.keys():
        print(f"==========RESULTS FOR {model}============")
        print(f"Accuracy of {model} = {results[model]['Accuracy']}")
        print(f"F1 Macro of {model} = {results[model]['f1_macro']}")
        print(f"F1 Each class of {model} = {results[model]['f1_class']}")
        plot_cm(results[model]['cm'], f"{model} CM")
        skplt.metrics.plot_roc(Y_test, results[model]['pred_prob'], title=f"{model} ROC curve")

results = dict()
rfc_model(X_train, y_train, X_test, y_test, results)

pprint_results(results, y_test)

plt.show()




X_train_pos = X_train[top_pos]
X_test_pos = X_test[top_pos]


prob = accuracy_score(y_test, pred)
pred_proba = rfc.predict_proba(X_test_pos)


pos = []
pos.append('FO')
pos.append('Event_Date')
pos.append('F1_Close_Best')
pos.append('F1_Close_Worst')
pos.append('F2_Close_Best')
pos.append('F2_Close_Worst')
pos.append('T_Event_URL')

get_index = X_test_pos.index.tolist()

df_rest = pd.DataFrame(fm_bd_all, columns=pos, index=get_index)


"""
Add these item to test model
"""
df_prob = X_test_pos
df_prob = df_prob.assign(FO = df_rest['FO'])
df_prob = df_prob.assign(Event_Date = df_rest['Event_Date'])
df_prob = df_prob.assign(F1_Close_Best = df_rest['F1_Close_Best'])
df_prob = df_prob.assign(F1_Close_Worst = df_rest['F1_Close_Worst'])
df_prob = df_prob.assign(F2_Close_Best = df_rest['F2_Close_Best'])
df_prob = df_prob.assign(F2_Close_Worst = df_rest['F2_Close_Worst'])
df_prob = df_prob.assign(T_Event_URL = df_rest['T_Event_URL'])


def add_win_probs(fm_bd):
    """
    adds P_Win probabilities. Currently random. Should be based on model output
    :param fm_bd_base:
    :return:
    """
    fm_bd = fm_bd.assign(F1_P_Win = np.random.uniform(0, 1, size=len(fm_bd)))
    fm_bd = fm_bd.assign(F2_P_Win=np.random.uniform(0, 1, size=len(fm_bd)))
    fm_bd['F1_P_Win'] = pred_proba[:, 0]
    fm_bd['F2_P_Win'] = pred_proba[:, 1]
    return fm_bd


def add_probable_odds(fm_bd_odds):
    """
    adds odds used to place bets. Currently uses a simple average between Best & Worst.
    TODO: make a more realistic function since Best/Worst are typically outliers.
    :param fm_bd_odds:
    :return:
    """
    fm_bd_odds['F1_P_Odds'] = (fm_bd_odds['F1_Close_Best'] + fm_bd_odds['F1_Close_Worst']) / 2
    fm_bd_odds['F2_P_Odds'] = (fm_bd_odds['F2_Close_Best'] + fm_bd_odds['F2_Close_Worst']) / 2
    return fm_bd_odds


def get_bet_on(x, p_cutoff=0.7):
    """
    binary decision on which fighter to place a bet on. Currently based on a simple cutoff.
    :param x:
    :param p_cutoff: minimum win probability to place a bet
    :return: used to modify df as .apply(get_bet_on, axis=1)
    """

    if x['F1_P_Win'] > p_cutoff:
        return 'F1'
    elif x['F2_P_Win'] > p_cutoff:
        return 'F2'
    else:
        return np.nan


def execute_strategy(event_dict, event_bank=100, strategy = 'p_balanced'):
    """
    runs a betting strategy on one event (passed as a dict).
    :param event_dict: dictionary with one event, containing all event's bouts
    :param event_bank: total amount of money to bet on the event
    :param strategy: 'even' - equal bets on all bouts, 'p_balanced' - bet size balanced to P_Win scores
    :return: event_dict with strategy results
    """
    total_bets = 0
    total_return = 0
    sum_probs = 0
    bouts_w_bets = 0

    for idx, bout in event_dict.items():
        if bout['Bet_On'] == 'F1':
            sum_probs += bout['F1_P_Win']
            bouts_w_bets += 1
        elif bout['Bet_On'] == 'F2':
            sum_probs += bout['F2_P_Win']
            bouts_w_bets += 1

    if bouts_w_bets==0:
        for idx, bout in event_dict.items():
            bout['Bet_Amount'] = bout['Bet_Return'] = bout['Bet_Net_Return'] = bout['Total_Bet_Amount'] = \
                bout['Total_Bet_Return'] = bout['Total_Net_Return'] = np.nan
        return event_dict


    even_bet_base = event_bank / bouts_w_bets
    p_balanced_bet_base = event_bank / sum_probs
    for idx, bout in event_dict.items():
        if bout['Bet_On'] == 'F1':
            if strategy == 'even': #only bet amount changes with strategy
                bet_amount = even_bet_base
            elif strategy == 'p_balanced':
                bet_amount = p_balanced_bet_base * bout['F1_P_Win']
            bet_return = bet_amount * bout['F1_P_Odds'] if bout['FO'] == 'Fighter_1' else 0
        elif bout['Bet_On'] == 'F2':
            if strategy == 'even':
                bet_amount = even_bet_base
            elif strategy == 'p_balanced':
                bet_amount = p_balanced_bet_base * bout['F2_P_Win']
            bet_return = bet_amount * bout['F2_P_Odds'] if bout['FO'] == 'Fighter_2' else 0
        else:
            bet_amount = np.nan
            bet_return = np.nan

        if not math.isnan(bet_amount):
            total_bets += bet_amount
            total_return += bet_return
            bet_net = bet_return - bet_amount
            total_net = total_return - total_bets
            bout['Bet_Amount'] = bet_amount
            bout['Bet_Return'] = bet_return
            bout['Bet_Net_Return'] = bet_net
            bout['Total_Bet_Amount'] = total_bets
            bout['Total_Bet_Return'] = total_return
            bout['Total_Net_Return'] = total_net
        else:
            bout['Bet_Amount'] = bout['Bet_Return'] = bout['Bet_Net_Return'] = bout['Total_Bet_Amount'] = \
                bout['Total_Bet_Return'] = bout['Total_Net_Return'] = np.nan
    return event_dict


def run_simulation(fm_bd_input, bank=100, bet_strategy='p_balanced', bank_strategy=0.75):
    """

    :param fm_bd_input: bout data to run simulation on. Must contain F*_W_Prob, Event_URL, F*_P_Odds
    :param bank: starting deposit
    :param bet_strategy: strategy selector for execute_strategy
    :param bank_strategy: how much of the bank is bet per event. This is to prevent wipe-outs from a single event.
    :return:
    """
    u_events = fm_bd_input[['Event_Date', 'T_Event_URL']].drop_duplicates().sort_values('Event_Date', ascending=True)

    all_events = {}
    n = 0
    for idx, row in u_events.iterrows():
        event = row['T_Event_URL']
        event_dict = fm_bd_input[fm_bd_input['T_Event_URL'] == event].to_dict('index')
        event_bank = bank * bank_strategy
        start_bank = bank
        event_dict = execute_strategy(event_dict, event_bank, bet_strategy)
        # updating bank based on Event's Net
        net = 0
        for i, x in event_dict.items():
            if not math.isnan(x['Total_Net_Return']):
                net = x['Total_Net_Return'] #the final Net for the event
        bank += net

        for i, x in event_dict.items():
            x['Bank_Start'] = start_bank
            x['Bank_End'] = bank
            # compiling all events results in one dict
            all_events[n] = x
            n += 1

    fm_bd_sim_results = pd.DataFrame(all_events).transpose()

    return fm_bd_sim_results

df = df_prob

# adding columns to execute strategy
df = add_win_probs(df) # should be replaced with model outputs

df = add_probable_odds(df) # TODO: needs to be more nuanced

df['Bet_On'] = df.apply(get_bet_on, axis=1, args=(0.7,)) # cutoff argument can be flexible

df_test = df

# df_test = df[df['Event_Date'] > '2018'].dropna(subset=['F1_Open']).sort_values('Event_Date', ascending=False) # testing on 2018's data
df_test = run_simulation(df_test)
df_test['Bank_End'].tail()

print(df_test['Bank_End'].tail())
#TODO: strategy results for different p_score cutoffs
#TODO: results aggregate stats (# events, # bets, win/loss ratio, etc)
#TODO: plots

