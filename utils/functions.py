import matplotlib.pyplot as plt
import sys
import math
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
import pytz, holidays
import datetime, pytz, holidays
from time import time, localtime, strftime, mktime, strptime, sleep
import holidays
from datetime import datetime, date, timedelta
#import yfinance as yf
from loguru import logger

phi = (1 + math.sqrt(5)) / 2
signals = []

def slev_levels_single(level, d, mean):
    # s = slev_levels(level = 1,d = 1.31417 - 1.29713, mean = 1.30600, df=df)
    # return levels
    phi = (1 + math.sqrt(5)) / 2
    s = []
    count = level * 3 + 1
    pattern = np.r_[2, 1, 2]
    # vytvoření jednotkového vektoru SLEV
    s = np.append(s, phi ** (level + 1))
    for i in range(-level, level + 1):
        s = np.append(s, phi ** (pattern + abs(i)))
    dd = np.cumsum(s / phi * d)
    a = dd - dd[count] - ((dd[count + 1] - dd[count]) / 2) + mean
    return a
def previous_working_day(check_day_, holidays=holidays.US()):
    offset = max(1, (check_day_.weekday() + 6) % 7 - 3)
    most_recent = check_day_ - timedelta(offset)
    if most_recent not in holidays:
        return most_recent
    else:
        return previous_working_day(most_recent, holidays)
def next_business_day(check_day_):
        next_day = check_day_ + timedelta(days=1)
        while next_day.weekday() in holidays.WEEKEND or next_day in holidays.US():
            next_day += timedelta(days=1)
        return next_day
def plot_heatmap_(heat_tp, heat_sl, d_min, d_max, stepd, min_value, max_value, stepm, dd, mm, Dm, Mm, result_index):
    sns.set_theme(style="darkgrid")
    fig = plt.figure(figsize=(10, 5), facecolor='w', edgecolor='k')
    sns.heatmap(heat_tp, cmap="tab20b")
    # sns.heatmap(np.rot90(heat_tp, k=1, axes=(1, 0)), cmap="tab20b")
    # plt.xticks(np.arange(0, slev_scale,1), np.flip(np.round(np.arange(d_min,d_max,stepd),4)))
    # plt.xticks(np.arange(0, slev_scale, 1), np.round(np.arange(d_min, d_max, stepd), 4))
    # plt.yticks(np.arange(0, slev_scale, 1), np.round(np.arange(min_value,max_value,stepm),4))
    plt.scatter(Mm, Dm, c="yellow")
    plt.scatter(0, 0, s=30, c="black")
    plt.text(0, 0, "0,0", fontsize=10, c="black")
    # plt.locator_params(axis='both', nbins=5)
    max_label = "Dm :" + str(np.round(Dm, 4)) + " Mm :" + str(np.round(Mm, 4))
    plt.text(Mm + .05, Dm + .05, max_label, fontsize=10, c="black")
    # sns.heatmap(heat_count, cmap="tab20b")
    plt.show()
def plot_slev(a, ts, tp, sl):
    # slev
    fig = plt.figure()
    fig.set_size_inches(10, 5)
    plt.autoscale(tight=True)
    plt.hlines(a, 2, ts["Close"].shape[0])
    plt.plot(ts["Close"].values, color='black')
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.scatter(tp, ts["Close"].values[tp], color='b', s=150)
    plt.scatter(sl, ts["Close"].values[sl], color='r', s=150)
    plt.show()
    plt.close()
def slev_signals_single(a, level, df):
    pocet_hodnot = len(df)
    signals_up = {}
    signals_down = {}
    zz = np.zeros((a.shape[0], pocet_hodnot))
    roll_back = np.roll(df, -1)  # posun zpet
    for i in range(0, a.shape[0]):
        # signals[i] = np.where(np.logical_or(np.logical_and(a[i] >= roll_back, a[i] <= Y_manual_scaled),np.logical_and(a[i] <= roll_back, a[i] >= Y_manual_scaled)), i + 1, 0)
        signals_up[i] = np.where(np.logical_and(a[i] >= roll_back, a[i] <= df), -(i + 1), 0)
        signals_down[i] = np.where(np.logical_and(a[i] <= roll_back, a[i] >= df), (i + 1), 0)

    # prevod na matrix
    for i in range(0, a.shape[0]):
        for j in range(0, pocet_hodnot):
            zz[i, j] = signals_up[i][j] + signals_down[i][j]
        # zz matrix signals h-l a level
    seq = []
    # ošetření crossu více levels najednou!!!!!
    for j in range(0, pocet_hodnot):
        # z = sum(zz[:, j])  # pouze jeden signal
        # indikace více hodnot v jednom sloupci
        if (len(zz[:, j][zz[:, j] != 0]) > 1):
            z = None
        else:
            z = sum(zz[:, j])
        seq = np.append(seq, z)

    # matice do array signálů
    index_bez_nul = np.where(seq != 0)[0]  # indexy bez 0 pro zpětnou pozici
    seq_clear = seq[seq != 0]
    ## sequence -+(level),..
    # otestovat na sinus průběru
    # TP -2  -1, 3 4, -5 -4, 6 7, -8 -7, 9 10 ...
    # SL 2 3 -3 -2,   -3 -2, 2 ,3  ...
    # neutral -2, 2,... -8,-8,...
    # NA více levels v jednom časovém úseku (např. 2, 6)
    # záporné znaménko implementovat
    # seq_clear = np.array([-8.,  8., -8.,  8., -8., -7., -6.,  6., -6.,  6.,  7., -7.,  7.,
    #       -7.,  7.,  8.])
    # test
    sl = []
    tp = []
    na = []
    # seq_clear = np.array([1,1,1,1,1,1,1,2,3,-3,-2,1,1,1,1,-2,-1,1,1,9,10,-3,-2,2,3,-8,-7,1,1,1,-5,-4])
    # tp[15. 30. 25. 19.] sl[ 7. 21.] pozice
    for n in range(1, (level * 2 + 1) * 3, 3):
        # tp = np.append(tp, np.where((seq_clear == i + 1) & (np.roll(seq_clear, -1) == i))[0])
        # tp down
        tp = np.append(tp, np.where((seq_clear == -(n + 1)) & (np.roll(seq_clear, -1) == -n))[0])
        # tp up
        tp = np.append(tp, np.where((seq_clear == n + 2) & (np.roll(seq_clear, -1) == n + 3))[0])
        # 2 3 -3 -2,   -3 -2, 2 ,3  ...
        # sl up
        sl = np.append(sl, np.where((np.roll(seq_clear, -1) == n + 2)
                                    & (np.roll(seq_clear, -2) == -(n + 2))
                                    & (np.roll(seq_clear, -3) == -(n + 1)))[0])
        # sl down
        sl = np.append(sl, np.where((np.roll(seq_clear, -1) == -(n + 1))
                                    & (np.roll(seq_clear, -2) == (n + 1))
                                    & (np.roll(seq_clear, -3) == (n + 2)))[0])
        # další sequence -4, -3  7  8  sl  4, 5, -5, -4, vyšší levely -4, -1 7,10 sl 4,7,-7,-4
        na = np.append(na, np.where(seq_clear == None)[0])
    tpr = index_bez_nul[tp.astype("int")]
    slr = index_bez_nul[sl.astype("int")]
    na = np.unique(index_bez_nul[na.astype("int")])

    return tpr, slr, na, seq, seq_clear
def slev_predict_2_HL(X, level, step, na_filter, dmin, dmax):
    # ----------------------- SLEV  parametry --------------------------------
    # pro testy MA 20
    Y_manual_scaled = X  # moving_average(Y_original,20)
    max_value = max(Y_manual_scaled)
    min_value = min(Y_manual_scaled)
    # nastavení min a max D - asi dle pips a nastavení rizika např. min 0.1 - 1 pip
    if (dmax == 0 or dmin == 0):
        d_max = (max_value - min_value) * (1 / (phi ** 2))
        d_min = (max_value - min_value) * (1 / (phi ** 6))
    else:
        d_max = dmax
        d_min = dmin
    logger.info("Dmin {} Dmax {}", d_min, d_max)
    rp = max_value - min_value
    stepm = (rp / step)
    stepd = (d_max) / step
    rr = step
    rs = step
    heat_tp = np.zeros((rr + 1, rs + 1))
    heat_sl = np.zeros((rr + 1, rs + 1))
    heat_na = np.zeros((rr + 1, rs + 1))
    # SLEV max optimal d,m
    xx = 0
    yy = 0
    for a in np.arange(d_min, d_max, stepd):
        xx = xx + 1
        yy = 0
        for b in np.arange(0, rp, stepm):  # mean
            aa = slev_levels_single(level, a, min_value + b)
            tp, sl, na, seq, seq_clear = slev_signals_single(aa, level, Y_manual_scaled)
            heat_tp[xx, yy] = (tp.shape[0])  # will count number of values
            heat_sl[xx, yy] = (sl.shape[0])
            heat_na[xx, yy] = (na.shape[0])
            yy = yy + 1
    # identifikovat max tp-sp a min na a co nejbliže Close (clustery průnik??)
    # sum_matrix = (heat_tp - heat_sl)/(heat_na+1)
    # Dm, Mm = np.where(sum_matrix == np.amax(sum_matrix[:, :]))  # pozice maxima
    # konverze výsledků  tp, sl, na do dataframe
    try:
        del df_tp, df_sl, df_na
    except:
        pass
    df_tp = pd.DataFrame(heat_tp).stack().rename_axis(['y', 'x']).reset_index(name='tp')
    df_sl = pd.DataFrame(heat_sl).stack().rename_axis(['y', 'x']).reset_index(name='sl')
    df_na = pd.DataFrame(heat_na).stack().rename_axis(['y', 'x']).reset_index(name='na')
    sl_ = df_sl["sl"]
    na_ = df_na["na"]
    df_tp = df_tp.join(sl_)
    df_tp = df_tp.join(na_)
    # lze nastavit limit pro na (risk), růuná kritéria
    # filter_best = df_tp.loc[((df_tp['tp'] - df_tp['sl']) > 0)]
    # filter_best = df_tp.loc[((df_tp['tp'] - df_tp['sl']) > 0) & (df_tp['na'] == 0)]
    if (na_filter == True):
        filter_best = df_tp.loc[((df_tp['tp'] - df_tp['sl']) > 0) & (df_tp['na'] == 0)]
    else:
        filter_best = df_tp.loc[((df_tp['tp'] - df_tp['sl']) > 0)]
    try:
        result_index = np.argmax(filter_best['tp'] - filter_best['sl'])
    except:
        pass
        print("NELZE NAJIT TP/SL HEATMAP MAX..nutno změnit parametry levels, slev_scale")
    # result_index = filter_best['y'].sub(round(Mm)).abs().idxmin()
    # y x  heat_tp[7][30]
    # Dm = (df_tp.loc[filter_best.index[0]]["y"])
    # Mm = (df_tp.loc[filter_best.index[0]]["x"])
    Dm = filter_best['y'].iloc[result_index]
    Mm = filter_best['x'].iloc[result_index]
    # Dm,Mm = np.where(sum_matrix == np.amax((heat_tp-heat_sl)/(heat_na+1))) #pozice maxima
    d = d_min + (Dm - 1) * stepd
    m = min_value + (Mm - 1) * stepm
    c = slev_levels_single(level, d, m)  # d, mean
    tp, sl, na, seq, seq_clear = slev_signals_single(c, level, X)
    # přepočet a na central  s1 s2 (hodnoty original)
    len_a = round(len(c) / 2)
    s1 = round(c[len_a - 1] * (X.max(axis=0) - X.min(axis=0)) + X.min(axis=0), 5)
    s2 = round(c[len_a] * (X.max(axis=0) - X.min(axis=0)) + X.min(axis=0), 5)
    # signaly
    #---------------------------------------------------------------------------
    return d, m, c, s1, s2, tp, sl, na, heat_tp, heat_sl, heat_na, filter_best, result_index
def slev_signals_single(a, level, df):
    pocet_hodnot = len(df)
    signals_up = {}
    signals_down = {}
    zz = np.zeros((a.shape[0], pocet_hodnot))
    roll_back = np.roll(df, -1)  # posun zpet
    for i in range(0, a.shape[0]):
        # signals[i] = np.where(np.logical_or(np.logical_and(a[i] >= roll_back, a[i] <= Y_manual_scaled),np.logical_and(a[i] <= roll_back, a[i] >= Y_manual_scaled)), i + 1, 0)
        signals_up[i] = np.where(np.logical_and(a[i] >= roll_back, a[i] <= df), -(i + 1), 0)
        signals_down[i] = np.where(np.logical_and(a[i] <= roll_back, a[i] >= df), (i + 1), 0)
    # prevod na matrix
    for i in range(0, a.shape[0]):
        for j in range(0, pocet_hodnot):
            zz[i, j] = signals_up[i][j] + signals_down[i][j]
        # zz matrix signals h-l a level
    seq = []
    # ošetření crossu více levels najednou!!!!!
    for j in range(0, pocet_hodnot):
        # z = sum(zz[:, j])  # pouze jeden signal
        # indikace více hodnot v jednom sloupci
        if (len(zz[:, j][zz[:, j] != 0]) > 1):
            z = None
        else:
            z = sum(zz[:, j])
        seq = np.append(seq, z)

    # matice do array signálů
    index_bez_nul = np.where(seq != 0)[0]  # indexy bez 0 pro zpětnou pozici
    seq_clear = seq[seq != 0]
    ## sequence -+(level),..
    # otestovat na sinus průběru
    # TP -2  -1, 3 4, -5 -4, 6 7, -8 -7, 9 10 ...
    # SL 2 3 -3 -2,   -3 -2, 2 ,3  ...
    # neutral -2, 2,... -8,-8,...
    # NA více levels v jednom časovém úseku (např. 2, 6)
    # záporné znaménko implementovat
    # seq_clear = np.array([-8.,  8., -8.,  8., -8., -7., -6.,  6., -6.,  6.,  7., -7.,  7.,
    #       -7.,  7.,  8.])
    # test
    sl = []
    tp = []
    na = []
    # seq_clear = np.array([1,1,1,1,1,1,1,2,3,-3,-2,1,1,1,1,-2,-1,1,1,9,10,-3,-2,2,3,-8,-7,1,1,1,-5,-4])
    # tp[15. 30. 25. 19.] sl[ 7. 21.] pozice
    for n in range(1, (level * 2 + 1) * 3, 3):
        # tp = np.append(tp, np.where((seq_clear == i + 1) & (np.roll(seq_clear, -1) == i))[0])
        # tp down
        tp = np.append(tp, np.where((seq_clear == -(n + 1)) & (np.roll(seq_clear, -1) == -n))[0])
        # tp up
        tp = np.append(tp, np.where((seq_clear == n + 2) & (np.roll(seq_clear, -1) == n + 3))[0])
        # 2 3 -3 -2,   -3 -2, 2 ,3  ...
        # sl up
        sl = np.append(sl, np.where((np.roll(seq_clear, -1) == n + 2)
                                    & (np.roll(seq_clear, -2) == -(n + 2))
                                    & (np.roll(seq_clear, -3) == -(n + 1)))[0])
        # sl down
        sl = np.append(sl, np.where((np.roll(seq_clear, -1) == -(n + 1))
                                    & (np.roll(seq_clear, -2) == (n + 1))
                                    & (np.roll(seq_clear, -3) == (n + 2)))[0])
        # další sequence -4, -3  7  8  sl  4, 5, -5, -4, vyšší levely -4, -1 7,10 sl 4,7,-7,-4
        na = np.append(na, np.where(seq_clear == None)[0])
    tpr = index_bez_nul[tp.astype("int")]
    slr = index_bez_nul[sl.astype("int")]
    na = np.unique(index_bez_nul[na.astype("int")])

    return tpr, slr, na, seq, seq_clear
def slev_signals_single_HL(a, level, df):
    pocet_hodnot = len(df)
    signals_up = {}
    signals_down = {}
    zz = np.zeros((a.shape[0], pocet_hodnot))
    roll_back_high = np.roll(df["High"], -1)  # posun zpet  rollback 0-high 1 2-low indey místo higj low..
    roll_back_low = np.roll(df["Low"], -1)  # posun zpet  rollback 0-high 1 2-low indey místo higj low..

    for i in range(0, a.shape[0]):
        # signals[i] = np.where(np.logical_or(np.logical_and(a[i] >= roll_back, a[i] <= Y_manual_scaled),np.logical_and(a[i] <= roll_back, a[i] >= Y_manual_scaled)), i + 1, 0)
        signals_up[i] = np.where(np.logical_and(a[i] >= roll_back_low, a[i] <= df["High"]), -(i + 1), 0)
        signals_down[i] = np.where(np.logical_and(a[i] <= roll_back_high, a[i] >= df["Low"]), (i + 1), 0)

    # prevod na matrix
    for i in range(0, a.shape[0]):
        for j in range(0, pocet_hodnot):
            zz[i, j] = signals_up[i][j] + signals_down[i][j]
        # zz matrix signals h-l a level
    seq = []
    # ošetření crossu více levels najednou!!!!!
    for j in range(0, pocet_hodnot):
        # z = sum(zz[:, j])  # pouze jeden signal
        # indikace více hodnot v jednom sloupci
        if (len(zz[:, j][zz[:, j] != 0]) > 1):
            z = None
        else:
            z = sum(zz[:, j])
        seq = np.append(seq, z)

    # matice do array signálů
    index_bez_nul = np.where(seq != 0)[0]  # indexy bez 0 pro zpětnou pozici
    seq_clear = seq[seq != 0]
    ## sequence -+(level),..
    # otestovat na sinus průběru
    # TP -2  -1, 3 4, -5 -4, 6 7, -8 -7, 9 10 ...
    # SL 2 3 -3 -2,   -3 -2, 2 ,3  ...
    # neutral -2, 2,... -8,-8,...
    # NA více levels v jednom časovém úseku (např. 2, 6)
    # záporné znaménko implementovat
    # seq_clear = np.array([-8.,  8., -8.,  8., -8., -7., -6.,  6., -6.,  6.,  7., -7.,  7.,
    #       -7.,  7.,  8.])
    # test
    sl = []
    tp = []
    na = []
    # seq_clear = np.array([1,1,1,1,1,1,1,2,3,-3,-2,1,1,1,1,-2,-1,1,1,9,10,-3,-2,2,3,-8,-7,1,1,1,-5,-4])
    # tp[15. 30. 25. 19.] sl[ 7. 21.] pozice
    for n in range(1, (level * 2 + 1) * 3, 3):
        # tp = np.append(tp, np.where((seq_clear == i + 1) & (np.roll(seq_clear, -1) == i))[0])
        # tp down
        tp = np.append(tp, np.where((seq_clear == -(n + 1)) & (np.roll(seq_clear, -1) == -n))[0])
        # tp up
        tp = np.append(tp, np.where((seq_clear == n + 2) & (np.roll(seq_clear, -1) == n + 3))[0])
        # 2 3 -3 -2,   -3 -2, 2 ,3  ...
        # sl up
        sl = np.append(sl, np.where((np.roll(seq_clear, -1) == n + 2)
                                    & (np.roll(seq_clear, -2) == -(n + 2))
                                    & (np.roll(seq_clear, -3) == -(n + 1)))[0])
        # sl down
        sl = np.append(sl, np.where((np.roll(seq_clear, -1) == -(n + 1))
                                    & (np.roll(seq_clear, -2) == (n + 1))
                                    & (np.roll(seq_clear, -3) == (n + 2)))[0])
        # další sequence -4, -3  7  8  sl  4, 5, -5, -4, vyšší levely -4, -1 7,10 sl 4,7,-7,-4
        na = np.append(na, np.where(seq_clear == None)[0])
    tpr = index_bez_nul[tp.astype("int")]
    slr = index_bez_nul[sl.astype("int")]
    na = np.unique(index_bez_nul[na.astype("int")])

    return tpr, slr, na, seq, seq_clear
def slev_predict_2(X, level, step, na_filter, dmin, dmax):
    # ----------------------- SLEV  parametry --------------------------------
    # pro testy MA 20
    Y_manual_scaled = X  # moving_average(Y_original,20)
    max_value = max(Y_manual_scaled["High"])
    min_value = min(Y_manual_scaled["Low"])
    # nastavení min a max D - asi dle pips a nastavení rizika např. min 0.1 - 1 pip
    d_min = dmin #(max_value - min_value) * (1 / (phi ** 6))
    d_max = dmax #(max_value - min_value) * (1 / (phi ** 2))
    logger.info("Dmin {} Dmax {}", d_min, d_max)
    rp = max_value - min_value
    stepm = (rp / step)
    stepd = (d_max) / step
    rr = step
    rs = step
    heat_tp = np.zeros((rr + 1, rs + 1))
    heat_sl = np.zeros((rr + 1, rs + 1))
    heat_na = np.zeros((rr + 1, rs + 1))
    # SLEV max optimal d,m
    xx = 0
    yy = 0
    for a in np.arange(d_min, d_max, stepd):
        xx = xx + 1
        yy = 0
        for b in np.arange(0, rp, stepm):  # mean
            aa = slev_levels_single(level, a, min_value + b)
            tp, sl, na, seq, seq_clear = slev_signals_single_HL(aa, level, Y_manual_scaled)
            heat_tp[xx, yy] = (tp.shape[0])  # will count number of values
            heat_sl[xx, yy] = (sl.shape[0])
            heat_na[xx, yy] = (na.shape[0])
            yy = yy + 1
    # identifikovat max tp-sp a min na a co nejbliže Close (clustery průnik??)
    # sum_matrix = (heat_tp - heat_sl)/(heat_na+1)
    # Dm, Mm = np.where(sum_matrix == np.amax(sum_matrix[:, :]))  # pozice maxima
    # konverze výsledků  tp, sl, na do dataframe
    try:
        del df_tp, df_sl, df_na
    except:
        pass
    df_tp = pd.DataFrame(heat_tp).stack().rename_axis(['y', 'x']).reset_index(name='tp')
    df_sl = pd.DataFrame(heat_sl).stack().rename_axis(['y', 'x']).reset_index(name='sl')
    df_na = pd.DataFrame(heat_na).stack().rename_axis(['y', 'x']).reset_index(name='na')
    sl_ = df_sl["sl"]
    na_ = df_na["na"]
    df_tp = df_tp.join(sl_)
    df_tp = df_tp.join(na_)
    # lze nastavit limit pro na (risk), růuná kritéria
    # filter_best = df_tp.loc[((df_tp['tp'] - df_tp['sl']) > 0)]
    # filter_best = df_tp.loc[((df_tp['tp'] - df_tp['sl']) > 0) & (df_tp['na'] == 0)]
    if (na_filter == True):
        filter_best = df_tp.loc[((df_tp['tp'] - df_tp['sl']) > 0) & (df_tp['na'] == 0)]
    else:
        filter_best = df_tp.loc[((df_tp['tp'] - df_tp['sl']) > 0)]
    try:
        result_index = np.argmax(filter_best['tp'] - filter_best['sl'])
    except:
        pass
        print("NELZE NAJIT TP/SL HEATMAP MAX..nutno změnit parametry levels, slev_scale")
    # result_index = filter_best['y'].sub(round(Mm)).abs().idxmin()
    # y x  heat_tp[7][30]
    # Dm = (df_tp.loc[filter_best.index[0]]["y"])
    # Mm = (df_tp.loc[filter_best.index[0]]["x"])
    Dm = filter_best['y'].iloc[result_index]
    Mm = filter_best['x'].iloc[result_index]
    # Dm,Mm = np.where(sum_matrix == np.amax((heat_tp-heat_sl)/(heat_na+1))) #pozice maxima
    d = d_min + (Dm - 1) * stepd
    m = min_value + (Mm - 1) * stepm
    c = slev_levels_single(level, d, m)  # d, mean
    tp, sl, na, seq, seq_clear = slev_signals_single_HL(c, level, X)
    # přepočet a na central  s1 s2 (hodnoty original)
    len_a = round(len(c) / 2)
    s1 = c[len_a - 1] * (X.max(axis=0) - X.min(axis=0)) + X.min(axis=0)
    s2 = c[len_a] * (X.max(axis=0) - X.min(axis=0)) + X.min(axis=0)
    # signaly
    #---------------------------------------------------------------------------
    return d, m, c, s1, s2, tp, sl, na, heat_tp, heat_sl, heat_na, filter_best, result_index
def SLEV_signals_prev(symbol_y, plot, ts, level, slev_scale, na_filter):
    check_day = datetime.date.today()
    # sleep_to_open()
    actual_date = check_day
    # ts = yahoo_last_period(symbol_y, "1d", "1m", "daily")
    #dd, mm, a, s1, s2, tp, sl, na, heat_tp, heat_sl, heat_na, filter_best, result_index = slev_predict_2(
    #    ts["Close"].values, level, slev_scale, na_filter)
    dd, mm, a, s1, s2, tp, sl, na, heat_tp, heat_sl, heat_na, filter_best, result_index = slev_predict_2(
        ts, level, slev_scale, na_filter,dmin,dmax)
    #nahradit
    m = len(a)
    x = range(0, m - 3, 3)
    logger.info("Previous Working Day : {}", previous_working_day(check_day))
    logger.info("Best SLEV D range: {} and Central price: {}", round(dd,4), round(mm,4))
    for n in x:
        signals.append(signal('BUY', a[n + 1], a[n + 2], a[n + 3]))
        signals.append(signal('SELL', a[n + 2], a[n + 1], a[n]))
    # for obj in signals:
    # print(obj.type, obj.sl, obj.enter, obj.exit, sep=',')
    # signals trades606
    if (plot):
        plot_heatmap_(heat_tp)
        plot_slev(a, ts, tp, sl)
    return a, tp, sl, heat_tp
def _read(bytesSize=4096):
    """
    Read socket message
    :param bytesSize:
    :return:
    """
    _receivedData = ''
    if not sockt:
        raise RuntimeError("socket connection broken")
    while True:
        char = sockt.recv(bytesSize).decode()
        _receivedData += char
        try:
            (resp, size) = json.JSONDecoder().raw_decode(_receivedData)
            if size == len(_receivedData):
                _receivedData = ''
                break
            elif size < len(_receivedData):
                _receivedData = _receivedData[size:].strip()
                break
        except ValueError as e:
            continue
    return resp
def _readObj():
    """
    Read message
    :return:
    """
    msg = _read()
    return msg
def print_stream():
    """
    Print streaming message
    :return:
    """
    api_client = APIClient()
    login_response = api_client.execute(loginCommand(userId=credentials.XTB_DEMO_ID, password=credentials.XTB_PASS_KEY))

    global sockt
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sockt = ssl.wrap_socket(sock)

    for i in range(API_MAX_CONN_TRIES):
        try:
            sockt.connect((DEFAULT_XAPI_ADDRESS, DEFUALT_XAPI_STREAMING_PORT))
        except socket.error:
            time.sleep(0.25)
            continue
        break

    ssid = login_response['streamSessionId']
    subscriptions = [
        dict(command='getBalance', streamSessionId=ssid)
    ]
    for message in subscriptions:
        msg = json.dumps(message)
        msg = msg.encode('utf-8')
        sent = 0
        while sent < len(msg):
            sent += sockt.send(msg[sent:])
            time.sleep(API_SEND_TIMEOUT / 1000)

    while True:
        msg = _readObj()
        command = msg['command']
        if command == 'balance':
            global MARGIN_FREE, BALANCE
            MARGIN_FREE = msg['data']['marginFree']
            BALANCE = msg['data']['balance']
def s_to_d_mean(a,lev):
    d = round((a[lev*3+2] - a[lev*3+1]),4)
    mean = round((a[lev*3+1] + d/2),4)
    return d, mean
def write_order_status__to_file(status, order,message):
    csvfile = open(csv_file, 'a', newline='')
    fieldnames = ['order_type', 'timestamp', 'order_ID','symbol','open_price','closed','profit','sl','tp','spread',"open_timeString","close_timeString"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    if(status == "Closed" or status == "Open"):
        writer.writerow({'order_type': status,'timestamp':datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S"),
                         'order_ID':order["order"],'symbol':order["symbol"],'open_price':order["open_price"],'closed':order["closed"],
                         'profit':order["profit"],'sl':order["sl"],'tp':order["tp"],'spread':order["spread"],
                         'open_timeString':order["open_timeString"],'close_timeString':order["close_timeString"]})
    if (status == "Failed"):
        writer.writerow({'order_type': status,'timestamp':datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
                         'order_ID':order["order"],'symbol':message})
    """
    Orders  [{'cmd': 1, 'order': 478476248, 'digits': 3, 'offset': 0, 'order2': 478476239, 'position': 478476248,
         'symbol': 'USDJPY', 'comment': '', 'customComment': 'text', 'commission': 0.0, 'storage': 0.0, 'margin_rate': 0.0, 
         'close_price': 135.37, 'open_price': 135.354, 'nominalValue': 0.0, 'profit': -2.6, 'volume': 0.01, 'sl': 135.399, 
         'tp': 135.347, 'closed': False, 'timestamp': 1677671215678, 'spread': 0, 'taxes': 0.0, 'open_time': 1677671215390, 
         'open_timeString': 'Wed Mar 01 12:46:55 CET 2023', 'close_time': None, 'close_timeString': None, 'expiration': None, 
         'expirationString': None}] 
    """

    return
def insert_order_to_SQLite(dbname, order, time_frame, a,level):
        try:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
            symbol = order["symbol"]
            order_ID = order["order"]
            order_status = order["closed"]
            profit = order["profit"]
            spread = order["spread"]
            open_price = order["open_price"]
            sl = order["sl"]
            tp = order["tp"]
            open_timeString = order["open_timeString"]
            close_timeString = order["close_timeString"]
            d, mean = s_to_d_mean(a, level)
            cxn = sqlite3.connect(dbname)
            cursor = cxn.cursor()
            print("Successfully Connected to SQLite Orders Table")
            sqlite_insert_with_param = """INSERT INTO orders
                                                      (timestamp, symbol, time_frame, order_ID, order_status , profit, spread, open_price, sl, tp, open_timeString, close_timeString, d, mean , level) 
                                                      VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?);"""
            data_tuple = (
            timestamp, symbol, time_frame, order_ID, order_status, profit, spread, open_price, sl, tp, open_timeString,
            close_timeString, d, mean, level)
            cursor.execute(sqlite_insert_with_param, data_tuple)
            cxn.commit()
            print("Record inserted successfully into Sqlite orders table ", cursor.rowcount)
            cursor.close()
        except sqlite3.Error as error:
            print("Failed to insert data into sqlite table", error)
        finally:
            if cxn:
                cxn.close()
                print("The SQLite connection is closed")
def get_a_for_TEST(price, d):
    # get last trading day data from yahoo nad calculate prev day SLEV
    global a, signals, level
    signals = []  # vynulovathodnoty jinak to bude incrementalní
    mean = price
    d = d
    phi = (1 + math.sqrt(5)) / 2
    s = []
    count = level * 3 + 1
    pattern = np.r_[2, 1, 2]
    # vytvoření jednotkového vektoru SLEV
    s = np.append(s, phi ** (level + 1))
    for i in range(-level, level + 1):
        s = np.append(s, phi ** (pattern + abs(i)))
    dd = np.cumsum(s / phi * d)
    a = dd - dd[count] - ((dd[count + 1] - dd[count]) / 2) + mean
    m = len(a)
    x = range(0, m - 3, 3)
    for n in x:
        signals.append(signal('BUY', a[n + 1], a[n + 2], a[n + 3]))
        signals.append(signal('SELL', a[n + 2], a[n + 1], a[n]))
    return a, signals  # GLOBAL signals
def is_internet():
    """
    Query internet using python
    :return:
    """
    try:
        urlopen('https://www.google.com', timeout=1)
        return True
    except urllib.error.URLError as Error:
        print(Error)
        return False
def passed_midnight(delta=1):
    today = datetime.datetime.today()
    time_ago = today - timedelta(minutes=delta)
    return today.strftime("%Y%m%d") != time_ago.strftime("%Y%m%d")
def any_data(data):
     return data
def yahoo_last_period(item_name, period, interval, perioda):  # ("USDJPY=X", "2d", "30m", "daily")
    item = [item_name]  # item = ['USDCHF=X']
    # previous period analysis
    last_period2 = yf.download(tickers=item, threads=False, interval=interval, period=period, group_by="ticker")
    last_week = last_period2.index[-1].weekday()
    last_month = last_period2.index[-1].month
    last_day = last_period2.index[-1].day
    last_year = last_period2.index[-1].year
    if (perioda == "daily"):
        last_period = last_period2[last_period2.index.date == date(last_year, last_month, last_day)]
        last_period2.drop(last_period2.tail(len(last_period)).index, inplace=True)
    if (perioda == "yearly"):
        last_period = last_period2[last_period2.index.date >= date(last_year, 1, 1)]
        last_period2.drop(last_period2.tail(len(last_period)).index, inplace=True)
    if (perioda == "monthly"):
        last_period = last_period2[last_period2.index.date >= date(last_year, 1)]
    return last_period
def yahoo_prev_day(item_name, period, interval):
    check_day = date.today()
    inter = previous_working_day(check_day).strftime("%Y-%m-%d")
    # print("Previous date: ", inter)
    try:
        df = yf.download(tickers=item_name, start=inter, interval="1m", period="1d", group_by="ticker")
        # print(df.loc[inter:inter])
    except Exception:
        print("“Error occured downloading data from yahoo")
    return df.loc[inter:inter]
def yahoo_prev_actual(item_name, period, interval):
    check_day = date.today()
    inter = previous_working_day(check_day).strftime("%Y-%m-%d")
    # print("Previous date: ", inter)
    try:
        df = yf.download(tickers=item_name, start=inter, interval="1m", period="1d", group_by="ticker")
        # print(df.loc[inter:inter])
    except Exception:
        print("“Error occured downloading data from yahoo")
    return df.loc[inter:]
def yahoo_actual_data(item_name, period, interval):
    today = date.today()
    inter = today.strftime("%Y-%m-%d")
    try:
        df = yf.download(tickers=item_name, start=inter, interval="1m", period="1d", group_by="ticker")
    except Exception:
        print("“Error occured downloading data from yahoo")
    return df.loc[inter:inter]
def to_milliseconds(days=0, hours=0, minutes=0):
    milliseconds = (days * 24 * 60 * 60 * 1000) + (hours * 60 * 60 * 1000) + (minutes * 60 * 1000)
    return milliseconds