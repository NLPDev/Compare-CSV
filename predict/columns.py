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

    output_cols = d_cols + F1_F2_cols + F1mF2_cols + F1oF2_cols + pct_F1_cols

    return df[output_cols]

#Sample
cols = ((['B_F1_Bool_Result', 'B_WClass'],'direct'),
        (['Age','Reach'],'F1_F2'),
        (['Height', 'FM_5yr_T_SStr_L', 'Open'], 'F1mF2'),
        (['Exp', 'FM_5yr_T_SStr_L', 'Close_Worst'], 'F1oF2'),
        (['cum_Win', 'FM_5yr_T_TD_L', 'Close_Best'], 'pct_F1'))

get_columns(fm_bd_all, cols)