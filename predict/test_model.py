import pandas as pd
import numpy as np
import math

def add_random_win_probs(fm_bd):
    """
    adds P_Win probabilities. Currently random. Should be based on model output
    :param fm_bd_base:
    :return:
    """

    fm_bd['F1_P_Win'] = np.random.uniform(0, 1, size=len(fm_bd))
    fm_bd['F2_P_Win'] = 1 - fm_bd['F1_P_Win']
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


def get_bet_on(x, p_cutoff=0.6):
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

df = pd.read_csv('fights_all.csv')
# adding columns to execute strategy
df = add_random_win_probs(df) # should be replaced with model outputs
df = add_probable_odds(df) # TODO: needs to be more nuanced
df['Bet_On'] = df.apply(get_bet_on, axis=1, args=(0.7,)) # cutoff argument can be flexible

df_test = df[df['Event_Date'] > '2018'].dropna(subset=['F1_Open']).sort_values('Event_Date', ascending=False) # testing on 2018's data
df_test = run_simulation(df_test)
df_test['Bank_End'].tail()

print(df_test['Bank_End'].tail())
#TODO: strategy results for different p_score cutoffs
#TODO: results aggregate stats (# events, # bets, win/loss ratio, etc)
#TODO: plots