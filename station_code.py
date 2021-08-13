import pandas as pd
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../common_functions/')
# from analytic import ReadBabyBoom
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import thinkstats2
import thinkplot
import numpy as np
# import brfss
import scipy.stats as stats
import seaborn as sns
from windrose import WindroseAxes
import windrose
import datetime
from mpl_toolkits.mplot3d import Axes3D


def season_2018():
    pass

def season_2019():
    r_dtypes = {'windsnelheid': np.float64, 'windrichting': np.float64}
    r_data = pd.read_csv('export_915096001-2.csv', ";", parse_dates=[0], header=[0], na_values='-', dtype=r_dtypes)

    # if winddirection for reference is about 20
    rwd = r_data[['datum', 'windrichting']]

    fs = 14


    n_bins = 36
    fig2, ((axd4, axd5), (axd6, axd7)) = plt.subplots(2, 2, figsize=(12, 6), sharey=True)
    rwd_winter = rwd.loc[rwd['datum'] >= '201812200000']
    rwd_winter = rwd_winter.loc[rwd_winter['datum'] < '201903200000']
    rwd_winter = rwd_winter['windrichting']
    axd4 = rwd_winter.plot.hist(ax=axd4, bins=n_bins)
    axd4.set_title('Winter', fontsize=fs)

    rwd_spring = rwd.loc[rwd['datum'] >= '201903200000']
    rwd_spring = rwd_spring.loc[rwd_spring['datum'] < '201906200000']
    rwd_spring = rwd_spring['windrichting']
    axd5 = rwd_spring.plot.hist(ax=axd5, bins=n_bins)
    axd5.set_title('Spring', fontsize=fs)

    rwd_summer = rwd.loc[rwd['datum'] >= '201906200000']
    rwd_summer = rwd_summer.loc[rwd_summer['datum'] < '201909200000']
    rwd_summer = rwd_summer['windrichting']
    axd6 = rwd_summer.plot.hist(ax=axd6, bins=n_bins)
    axd6.set_title('Summer', fontsize=fs)

    rwd_autumn = rwd.loc[rwd['datum'] >= '201909200000']
    rwd_autumn = rwd_autumn.loc[rwd_autumn['datum'] < '201912200000']
    rwd_autumn = rwd_autumn['windrichting']
    axd7 = rwd_autumn.plot.hist(ax=axd7, bins=n_bins)
    axd7.set_title('Autumn', fontsize=fs)

    plt.show()
    print('hodl')

def season_2020():
    ## Reference station:
    r_dtypes = {'windsnelheid':np.float64, 'windrichting':np.float64}
    r_data = pd.read_csv('export_915096001.csv', ";", parse_dates=[0], header=[0], na_values='-', dtype=r_dtypes)
    # print(r_data.dtypes)

    r_means = r_data.mean()
    r_std = r_data.std()
    r_var = r_data.var()
    rws = r_data['windsnelheid']
    rwd = r_data[['datum','windrichting']]

    fs = 14


    # ax1 = rws.plot.kde()
    # ax1.set_xlabel('Wind Speed [m/s]', fontsize=14)
    # plt.show()

    # ax[0] = rwd.plot.kde()
    # ax[0].set_xlabel('Wind Direction [degree]', fontsize=14)
    # ax[0].set_title('subplot 1')

    ## Neighborhood station
    n_dtypes = {'WindSpd_{Avg}':np.float64, 'WindDir_{Avg}':np.float64}
    fields = pd.read_fwf("http://weather.tudelft.nl/csv/fields.txt", header=None)
    delfshaven_data = pd.read_csv("http://weather.tudelft.nl/csv/Delfshaven.csv", names=fields[1], parse_dates=[0], na_values='NAN')

    time_range = delfshaven_data.loc[delfshaven_data['DateTime'] >= '202001010000']
    time_range = time_range.loc[time_range['DateTime'] < '202101010000']

    d_data = time_range[['WindSpd_{Avg}', 'WindDir_{Avg}']]

    dws = d_data['WindSpd_{Avg}']
    dwd = d_data['WindDir_{Avg}']


    n_bins = 36
    fig2, ((axd4, axd5), (axd6, axd7)) = plt.subplots(2,2, figsize=(10,5), sharey=True)
    rwd_winter = rwd.loc[rwd['datum'] >= '202001010000']
    rwd_winter = rwd_winter.loc[rwd_winter['datum'] < '202003200000']
    rwd_winter = rwd_winter['windrichting']
    axd4 = rwd_winter.plot.hist(ax=axd4, bins=n_bins)
    axd4.set_title('Winter', fontsize=fs)

    rwd_spring = rwd.loc[rwd['datum'] >= '202003200000']
    rwd_spring = rwd_spring.loc[rwd_spring['datum'] < '202006200000']
    rwd_spring = rwd_spring['windrichting']
    axd5 = rwd_spring.plot.hist(ax=axd5, bins=n_bins)
    axd5.set_title('Spring', fontsize=fs)

    rwd_summer = rwd.loc[rwd['datum'] >= '202006200000']
    rwd_summer = rwd_summer.loc[rwd_summer['datum'] < '202009200000']
    rwd_summer = rwd_summer['windrichting']
    axd6 = rwd_summer.plot.hist(ax=axd6, bins=n_bins)
    axd6.set_title('Summer', fontsize=fs)

    rwd_autumn = rwd.loc[rwd['datum'] >= '202009200000']
    rwd_autumn = rwd_autumn.loc[rwd_autumn['datum'] < '202012200000']
    rwd_autumn = rwd_autumn['windrichting']
    axd7 = rwd_autumn.plot.hist(ax=axd7, bins=n_bins)
    axd7.set_title('Autumn', fontsize=fs)


    fig, (ax0,ax1) = plt.subplots(2, figsize=(10,5))
    fig.suptitle('Wind Direction Histogram of Reference and Delfshaven', fontsize=16, y=0.98)
    fig.subplots_adjust(top=0.80)
    ax0 = rwd['windrichting'].plot.hist(ax=ax0, bins=25)
    # ax0 = rwd.plot.kde()
    fs = 14
    ax0.set_ylabel('Frequency', fontsize=fs)
    ax0.set_xlabel('Wind Direction [degrees]',fontsize=fs)
    ax0.tick_params(labelsize=fs)
    ax0.set_title('Reference')


    ax1 = dwd.plot.hist(ax=ax1, bins=25)
    fs = 14
    ax1.set_ylabel('Frequency', fontsize=fs)
    ax1.set_xlabel('Wind Direction [degrees]',fontsize=fs)
    ax1.tick_params(labelsize=fs)
    ax1.set_title('Delfshaven')






    n_bins = 36
    fig2, ((axd0, axd1), (axd2, axd3)) = plt.subplots(2,2, figsize=(10,5), sharey=True)


    time_range_winter = delfshaven_data.loc[delfshaven_data['DateTime'] >= '201912200000']
    time_range_winter = time_range_winter.loc[time_range_winter['DateTime'] < '202003200000']
    dwdw = time_range_winter['WindDir_{Avg}']
    axd0 = dwdw.plot.hist(ax=axd0, bins=n_bins)
    axd0.set_title('Winter', fontsize=fs)

    time_range_spring = delfshaven_data.loc[delfshaven_data['DateTime'] >= '202003200000']
    time_range_spring = time_range_spring.loc[time_range_spring['DateTime'] < '202006200000']
    dwdsp = time_range_spring['WindDir_{Avg}']
    axd1 = dwdsp.plot.hist(ax=axd1, bins=n_bins)
    axd1.set_title('Spring', fontsize=fs)

    time_range_summer = delfshaven_data.loc[delfshaven_data['DateTime'] >= '202006200000']
    time_range_summer = time_range_summer.loc[time_range_summer['DateTime'] < '202009200000']
    dwdsu = time_range_summer['WindDir_{Avg}']
    axd2 = dwdsu.plot.hist(ax=axd2, bins=n_bins)
    axd2.set_title('Summer', fontsize=fs)

    time_range_autumn = delfshaven_data.loc[delfshaven_data['DateTime'] >= '202009200000']
    time_range_autumn = time_range_autumn.loc[time_range_autumn['DateTime'] < '202012200000']
    dwda = time_range_autumn['WindDir_{Avg}']
    axd3 = dwda.plot.hist(ax=axd3, bins=n_bins)
    axd3.set_title('Autumn', fontsize=fs)




    title = fig2.suptitle("Wind Direction of Delfshaven Seasons '20", y=0.99)
    # plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.show()





    # fig, ax = plt.subplots(1,1)
    # mean, var, skew, kurt = stats.norm.stats(x, moments='mvsk')
    # ax.plot(x, stats.norm.pdf(x), 'r-', lw=5, alpha =0.6, label='norm pdf')
    # rv = stats.norm()
    # ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')
    # plt.show()
    #
    #
    # print(len(r_data))
    # r_data.dropna(how='any', inplace=True)
    # print(len(r_data))
    # # ax1 = r_data['windsnelheid'].plot.hist(bins=25)
    # ax1 = stats.gaussian_kde(r_data['windsnelheid'])
    # fs = 14
    # # ax1.set_ylabel('Frequency', fontsize=fs)
    # # ax1.set_xlabel('Wind Speed [m/s]',fontsize=fs)
    # # ax1.tick_params(labelsize=fs)
    # plt.show()








    # print(delfshaven_data['DateTime'])

def oneday_rose(rf, dh):

    # timeframes

    sdate = '202001010000'
    edate = '202001012359'

    rf_tf = (rf['datum'] > sdate) & (rf['datum'] <= edate)
    rf = rf.loc[rf_tf]

    dh_tf = (dh['DateTime'] > sdate) & (dh['DateTime'] <= edate)
    dh = dh.loc[dh_tf]

    # title string
    edateStr = datetime.datetime.strptime(edate, "%Y%m%d%H%M").strftime("%d %b %Y ")
    sdateStr = datetime.datetime.strptime(sdate, "%Y%m%d%H%M").strftime("%d %b %Y ")
    title_date = edateStr + ' - ' + sdateStr
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(title_date)

    # delfshaven windrose
    ax1 = fig.add_subplot(121, projection='windrose')
    ax1 = WindroseAxes.from_ax(ax=ax1)
    ax1.bar(dh['WindDir_{Avg}'].astype(float), dh['WindSpd_{Avg}'].astype(float), normed=True, opening=0.8, edgecolor='white')
    ax1.set_title('Delfshaven')
    ax1.set_legend()

    # reference windrose
    ax2 = fig.add_subplot(122, projection='windrose')
    ax2 = WindroseAxes.from_ax(ax=ax2)
    ax2.bar(rf['windrichting'].astype(float), rf['windsnelheid'].astype(float), normed=True, opening=0.8, edgecolor='white')
    ax2.set_title('Reference')
    ax2.set_legend()

    plt.show()

def year_rose(ol, cs, dh, rf):
    # dh_tf = (dh['DateTime'] > sdate) & (dh['DateTime'] <= edate)
    # dh_select = dh.loc[dh_tf]

    # timeframe

    sdate = '201305010000'
    edate = '201405010000'

    # '201504150000', '201512172359'
    # '201801010000', '202001012359'

    dh_tf = (dh['DateTime'] > '201504150000') & (dh['DateTime'] <= '201512172359')
    dh_select = dh.loc[dh_tf]

    ol_tf1 = (ol['DateTime'] > '201504150000') & (ol['DateTime'] <= '201512172359')
    ol_select1 = ol.loc[ol_tf1]

    ol_tf = (ol['DateTime'] > '201304221500') & (ol['DateTime'] <= '201411100930')
    ol_select = ol.loc[ol_tf]

    cs_tf = (cs['DateTime'] > '201304221500') & (cs['DateTime'] <= '201411100930')
    cs_select = cs.loc[cs_tf]

    rf_tf = (rf['datum'] > '201801010000') & (rf['datum'] <= '202101012359')
    rf_select = rf.loc[rf_tf]

    dh_tf1 = (dh['DateTime'] > '201801010000') & (dh['DateTime'] <= '202101012359')
    dh_select1 = dh.loc[dh_tf1]

    # title string
    edateStr = datetime.datetime.strptime(edate, "%Y%m%d%H%M").strftime("%d %b %Y ")
    sdateStr = datetime.datetime.strptime(sdate, "%Y%m%d%H%M").strftime("%d %b %Y ")
    title_date = sdateStr + ' - ' + edateStr
    fig = plt.figure(figsize=(8, 12))




    ax2 = fig.add_subplot(321, projection='windrose')
    ax2 = WindroseAxes.from_ax(ax=ax2)
    ax2.bar(cs_select['windrichting'].astype(float), cs_select['windsnelheid'].astype(float), bins=np.arange(0.01,8,1), normed=True, opening=0.9,
            edgecolor='white')
    ax2.set_title("a. Central Station '13-'14", {'fontsize': 10}, y=1.12)
    ax3 = fig.add_subplot(322, projection='windrose')
    ax3 = WindroseAxes.from_ax(ax=ax3)
    ax3.bar(ol_select['windrichting'].astype(float), ol_select['windsnelheid'].astype(float), bins=np.arange(0.01,8,1), normed=True, opening=0.9,
            edgecolor='white')
    ax3.title_size = 'x-small'
    ax3.set_legend(title='Windspeed [m/s]', loc='best', bbox_to_anchor=(0.95, 1, 0.2, 0.2),
                   fontsize='xx-small')
    ax3.set_title("b. Oude Leede '13-'14", {'fontsize': 10}, y=1.12)
    # ax3.set_legend()
    ax4 = fig.add_subplot(323, projection='windrose', label='Delfshaven - 2015')
    ax4 = WindroseAxes.from_ax(ax=ax4)
    ax4.bar(dh_select['windrichting'].astype(float), dh_select['windsnelheid'].astype(float), bins=np.arange(0.01,8,1), normed=True, opening=0.9,
            edgecolor='white')
    ax4.set_title("c. Delfshaven '15", {'fontsize': 10}, y=1.12)
    ax5 = fig.add_subplot(324, projection='windrose')
    ax5 = WindroseAxes.from_ax(ax=ax5)
    ax5.bar(ol_select1['windrichting'].astype(float), ol_select1['windsnelheid'].astype(float),
            bins=np.arange(0.01, 8, 1), normed=True, opening=0.9,
            edgecolor='white', label='Oude Leede - 2015')
    ax5.set_title("d. Oude Leede '15", {'fontsize': 10}, y=1.12)
    ax7 = fig.add_subplot(325, projection='windrose', label='Delfshaven - 2015')
    ax7 = WindroseAxes.from_ax(ax=ax7)
    ax7.bar(dh_select1['windrichting'].astype(float), dh_select1['windsnelheid'].astype(float),
            bins=np.arange(0.01, 8, 1), normed=True, opening=0.9,
            edgecolor='white')
    ax7.set_title("e. Delfshaven '18-'21", {'fontsize': 10}, y=1.12)
    ax6 = fig.add_subplot(326, projection='windrose')
    ax6 = WindroseAxes.from_ax(ax=ax6)
    ax6.bar(rf_select['windrichting'].astype(float), rf_select['windsnelheid'].astype(float),
            bins=np.arange(0.01, 8, 1), normed=True, opening=0.9,
            edgecolor='white', label="R'dam The Hague AP")
    ax6.set_title("f. R'dam The Hague AP '18-'21", {'fontsize': 10}, y=1.12)
    plt.show()

def winter_season2014_hist_kde(ol, cs):
    sdate = '201312200000'
    edate = '201403192359'

    wd=200
    min_wd = wd-20
    max_wd = wd+20

    ol_tf = (ol['DateTime'] > sdate) & (ol['DateTime'] <= edate)
    ol = ol.loc[ol_tf]

    cs_tf = (cs['DateTime'] > sdate) & (cs['DateTime'] <= edate)
    cs = cs.loc[cs_tf]

    # ol_ww = (ol['windrichting'] >= min_wd) & (ol['windrichting'] <= max_wd)
    # ol = ol.loc[ol_ww]
    #
    # cs_ww = (cs['windrichting'] >= min_wd) & (cs['windrichting'] <= max_wd)
    # cs = cs.loc[cs_ww]


    fig = plt.figure(figsize=(12, 6))
    ax1 = ol['windsnelheid'].plot.kde(label='Oude Leede kde')
    ax2 = cs['windsnelheid'].plot.kde(label='Central Station kde')
    ax2.set_xlabel('Wind Speed [m/s]', fontsize=12)

    fig.suptitle('Winter season 2013-2014 : Wind direction 180-220')
    plt.legend()

    fig2 = plt.figure(figsize=(12, 6))
    ax3 = ol['windsnelheid'].plot.hist(label='Oude Leede', alpha=0.5)
    ax4 = cs['windsnelheid'].plot.hist(label='Central Station', alpha=0.5)
    ax3.set_xlabel('Wind Speed [m/s]', fontsize=12)
    fig2.suptitle('Winter season 2013-2014 : Wind direction 180-220')
    ax3.axvline(x=ol['windsnelheid'].mean(), ymax=0.2, color='blue', alpha=0.5)
    ax4.axvline(x=cs['windsnelheid'].mean(), ymax=0.9, color='red', alpha=0.5)
    plt.legend()

    plt.show()



def winter_season19(rf, dh, wd):

    sdate = '201801220000'
    edate = '201903202359'

    min_wd = wd-20
    max_wd = wd+20

    # timeframes
    rf_tf = (rf['datum'] > sdate) & (rf['datum'] <= edate)
    rf = rf.loc[rf_tf]

    dh_tf = (dh['DateTime'] > sdate) & (dh['DateTime'] <= edate)
    dh = dh.loc[dh_tf]

    # windwindow min - max

    # rf_ww = (rf['windrichting'] >= min_wd) & (rf['windrichting'] <= max_wd)
    # rf = rf.loc[rf_ww]
    #
    # dh_ww = (dh['windrichting'] >= min_wd) & (dh['windrichting'] <= max_wd)
    # dh = dh.loc[dh_ww]

    # plot kde
    fig = plt.figure(figsize=(12, 6))
    ax1 = rf['windsnelheid'].plot.kde(label='Reference station kde')
    ax2 = dh['windsnelheid'].plot.kde(label='Delfshaven station kde')
    ax2.set_xlabel('Wind Speed [m/s]', fontsize=12)

    fig.suptitle('Winter season 2018-2019 : Wind direction 180-220')
    plt.legend()

    # plot hist
    fig2 = plt.figure(figsize=(12, 6))
    ax3 = rf['windsnelheid'].plot.hist(label='Reference station', alpha=0.5)
    ax4 = dh['windsnelheid'].plot.hist(label='Delfshaven station', alpha=0.5)
    ax3.set_xlabel('Wind Speed [m/s]', fontsize=12)
    fig2.suptitle('Winter season 2018-2019 : Wind direction 18-220')
    ax3.axvline(x=rf['windsnelheid'].mean(), ymax=0.2, color='blue', alpha=0.5)
    ax4.axvline(x=dh['windsnelheid'].mean(), ymax=0.9, color='red', alpha=0.5)
    plt.legend()

    plt.show()

    print('hodl')

def winter_3haverages_cs(ol, cs):
    sdate = '201312200000'
    edate = '201403192359'

    ol_tf = (ol['DateTime'] > sdate) & (ol['DateTime'] <= edate)
    ol_sl = ol.loc[ol_tf].set_index('DateTime', drop=False)

    cs_tf = (cs['DateTime'] > sdate) & (cs['DateTime'] <= edate)
    cs_sl = cs.loc[cs_tf].set_index('DateTime', drop=False)

    ol_3h = ol_sl['windsnelheid'].rolling('3H').mean()
    cs_3h = cs_sl['windsnelheid'].rolling('3H').mean()

    sns.set(rc={'figure.figsize': (10, 4)})
    # ax1 = ol['windsnelheid'].plot(label='Reference', linewidth=0.5)
    # ax2 = cs['windsnelheid'].plot(label='Delfshaven', linewidth=0.5)
    ax4 = ol_3h.plot(label='Reference 3H avg', linewidth=2)
    ax3 = cs_3h.plot(label='Delfshaven 3H avg', linewidth=2)
    ax3.set_ylabel('Windspeed [m/s]')
    ax3.set_xlabel('Time [m/d/h]')
    plt.legend()
    plt.show()
    plt.close()

def winter_3haverages_dh(rf, dh):
    sdate = '201812200000'
    edate = '201903192359'

    rf_tf = (rf['datum'] > sdate) & (rf['datum'] <= edate)
    rf_sl = rf.loc[rf_tf].set_index('datum')

    dh_tf = (dh['DateTime'] > sdate) & (dh['DateTime'] <= edate)
    dh_sl = dh.loc[dh_tf].set_index('DateTime')
    dh_resampled = dh_sl.resample('10T').mean()

    rf_2h = rf_sl['windsnelheid'].rolling('3H').mean()
    dh_2h = dh_sl['windsnelheid'].rolling('3H').mean()

    rf_cols = ['windsnelheid', 'windrichting']
    dh_cols = ['windsnelheid', 'WindDir_{Avg}']

    sns.set(rc={'figure.figsize': (10, 4)})
    # ax1 = rf_sl['windsnelheid'].plot(label='Reference', linewidth=0.5)
    # ax2 = dh_resampled['windsnelheid'].plot(label='Delfshaven', linewidth=0.5)
    ax4 = rf_2h.plot(label='Reference 3H avg', linewidth=2)
    ax3 = dh_2h.plot(label='Delfshaven 3H avg', linewidth=2)
    ax3.set_ylabel('Windspeed [m/s]')
    ax3.set_xlabel('Time [m/d/h]')
    plt.legend()
    plt.show()


def oneday_lineplot(rf, dh):
    sdates = ['202001010000', '202001080000', '202001150000']
    edates = ['202001012359', '202001082359', '202001152359']
    for sdate, edate in zip(sdates, edates):

        # timeframes
        # sdate = '202001010000'
        # edate = '202001012359'

        rf_tf = (rf['datum'] > sdate) & (rf['datum'] <= edate)
        rf_sl = rf.loc[rf_tf].set_index('datum')


        dh_tf = (dh['DateTime'] > sdate) & (dh['DateTime'] <= edate)
        dh_sl = dh.loc[dh_tf].set_index('DateTime')
        dh_resampled = dh_sl.resample('10T').mean()

        rf_2h = rf_sl['windsnelheid'].rolling('3H').mean()
        dh_2h = dh_sl['windsnelheid'].rolling('3H').mean()


        rf_cols = ['windsnelheid', 'windrichting']
        dh_cols = ['windsnelheid', 'WindDir_{Avg}']

        sns.set(rc={'figure.figsize':(10,4)})
        ax1 = rf_sl['windsnelheid'].plot(label='Reference', linewidth=0.5)
        ax2 = dh_resampled['windsnelheid'].plot(label='Delfshaven', linewidth=0.5)
        ax4 = rf_2h.plot(label='Reference 3H avg', linewidth=2)
        ax3 = dh_2h.plot(label='Delfshaven 3H avg', linewidth=2)
        ax1.set_ylabel('Windspeed [m/s]')
        ax1.set_xlabel('Time [m/d/h]')
        plt.legend()
        plt.show()
    print('hodl')


def oneweek_multiplebox(rf, dh):
    # timeframes
    sdate = '202001010000'
    edate = '202002082359'

    rf_tf = (rf['datum'] > sdate) & (rf['datum'] <= edate)
    rf['day'] = rf['datum'].dt.strftime('%A')
    rf = rf.loc[rf_tf].set_index('datum').assign(Trial=1)

    dh_tf = (dh['DateTime'] > sdate) & (dh['DateTime'] <= edate)
    dh = dh.loc[dh_tf].set_index('DateTime', drop=False)
    dh_rs = dh.resample('10T').mean()
    dh_rs['day'] = dh_rs.index.strftime('%A')
    dh_rs = dh_rs.assign(Trial=2)

    cdf = pd.concat([dh_rs[['windsnelheid', 'Trial', 'day']], rf[['windsnelheid', 'Trial', 'day']]])
    ## https://stackoverflow.com/questions/44552489/plotting-multiple-boxplots-in-seaborn

    fig, ax = plt.subplots()
    ax = sns.boxplot(x='day', y='windsnelheid', hue='Trial', data=cdf, ax=ax)
    # ax = sns.boxplot(x='day', y='WindSpd_{Avg}', data=dh_rs, ax=ax)
    ax.set_ylabel('Windspeed [m/s]')
    ax.set_xlabel('Day')
    ax.set_title('boxplot compare')

    plt.show()
    print('HODL')

def small_window_boxplot(rf, dh):
    sdate = '202001010000'
    edate = '202001012359'

    rf_tf = (rf['datum'] > sdate) & (rf['datum'] <= edate)
    rf['hour'] = rf['datum'].dt.strftime('%-H').astype(float)
    rf['slot'] = rf['hour'].apply(f)

    rf = rf.loc[rf_tf].set_index('datum').assign(Trial=1)

    dh_tf = (dh['DateTime'] > sdate) & (dh['DateTime'] <= edate)
    dh = dh.loc[dh_tf].set_index('DateTime', drop=False)
    dh_rs = dh.resample('10T').mean()
    dh_rs['hour'] = dh_rs.index.strftime('%-H').astype(float)
    dh_rs['slot'] = dh_rs['hour'].apply(f)
    dh_rs = dh_rs.assign(Trial=2)

    cdf = pd.concat([dh_rs[['windsnelheid', 'Trial', 'slot']], rf[['windsnelheid', 'Trial', 'slot']]])
    ## https://stackoverflow.com/questions/44552489/plotting-multiple-boxplots-in-seaborn

    fig, ax = plt.subplots()
    ax = sns.boxplot(x='slot', y='windsnelheid', hue='Trial', data=cdf, ax=ax)
    # ax = sns.boxplot(x='day', y='WindSpd_{Avg}', data=dh_rs, ax=ax)
    ax.set_ylabel('Windspeed [m/s]')
    ax.set_xlabel('Day')
    ax.set_title('boxplot compare')

    plt.show()
    print('HODL')

def f(x):
    if (x >= 0) and (x < 6):
        return 'Night'
    elif (x >= 6) and (x < 12 ):
        return 'Morning'
    elif (x >= 12) and (x < 18):
        return'Afternoon'
    elif (x >= 18) and (x < 24) :
        return 'Evening'


def corr_wd(rf, dh, date):

    if date == 'season':
        date_sdate = ['201912200000', '202003200000', '202006200000', '202009200000']
        date_edate = ['202003192359', '202006192359', '202009192359', '202012192359']
    if date == 'year':
        date_sdate = ['201912200000']
        date_edate = ['202012192359']
    if date == 'weeks':
        date_sdate = ['202001140000', '202004070000', '202008010000', '202009200000']
        date_edate = ['202001202359', '202004132359', '202008072359', '202009262359']
    if date == 'all':
        date_sdate = ['201812200000']
        date_edate = ['202101010000']

    min_wd = 130
    max_wd = 280

    for sdate, edate in zip(date_sdate, date_edate):

        rf_ww = (rf['windrichting'] >= min_wd) & (rf['windrichting'] <= max_wd)
        rf_wd = rf.loc[rf_ww]

        rf_tf = (rf_wd['datum'] > sdate) & (rf_wd['datum'] <= edate)
        rf_s = rf_wd.loc[rf_tf].set_index('datum')

        dh_ww = (dh['windrichting'] >= min_wd) & (dh['windrichting'] <= max_wd)
        dh_wd = dh.loc[dh_ww]

        dh_tf = (dh_wd['DateTime'] > sdate) & (dh_wd['DateTime'] <= edate)
        dh_s = dh_wd.loc[dh_tf].set_index('DateTime', drop=False)
        dh_rs = dh_s.resample('10T').mean()
        dh_rs = dh_rs.dropna(axis=0, how='all')
        cor_velocity = rf_s['windsnelheid'].corr(dh_rs['windsnelheid'])
        cor_direction = rf_s['windrichting'].corr(dh_rs['windrichting'])
        print('Date: ', sdate, edate, '| Velocity correlation: ', cor_velocity, '| Wind direction correlation: ',
              cor_direction)

    # print('banana')


def corr(rf, dh, date):

    if date == 'season':
        date_sdate = ['201912200000', '202003200000', '202006200000', '202009200000']
        date_edate = ['202003192359', '202006192359', '202009192359', '202012192359']
    if date == 'year':
        date_sdate = ['201912200000']
        date_edate = ['202012192359']
    if date == 'weeks':
        date_sdate = ['202001140000', '202004070000', '202008010000', '202009200000']
        date_edate = ['202001202359', '202004132359', '202008072359', '202009262359']
    if date == 'all':
        date_sdate = ['201812200000']
        date_edate = ['202101010000']

    # sdate = '202001010000'
    # edate = '202002082359'


    for sdate, edate in zip(date_sdate, date_edate):

        rf_tf = (rf['datum'] > sdate) & (rf['datum'] <= edate)
        rf_s = rf.loc[rf_tf].set_index('datum')

        dh_tf = (dh['DateTime'] > sdate) & (dh['DateTime'] <= edate)
        dh_s = dh.loc[dh_tf].set_index('DateTime', drop=False)
        dh_rs = dh_s.resample('10T').mean()

        cor_velocity = rf_s['windsnelheid'].corr(dh_rs['windsnelheid'])
        cor_direction = rf_s['windrichting'].corr(dh_rs['windrichting'])
        print('Date: ', sdate, edate, '| Velocity correlation: ', cor_velocity, '| Wind direction correlation: ', cor_direction)

    # print('banana')

def winter_hist(dh, ol, cs):
    sdate_2019 = '201912200000'
    edate_2019 = '202003192359'

    sdate_2013 = '201312200000'
    edate_2013 = '201403192359'

    ol_tf = (ol['DateTime'] > sdate_2013) & (ol['DateTime'] <= edate_2013)
    ol_sl = ol.loc[ol_tf].set_index('DateTime', drop=False)

    cs_tf = (cs['DateTime'] > sdate_2013) & (cs['DateTime'] <= edate_2013)
    cs_sl = cs.loc[cs_tf].set_index('DateTime', drop=False)

    dh_tf = (dh['DateTime'] > sdate_2019) & (dh['DateTime'] <= edate_2019)
    dh_sl = dh.loc[dh_tf].set_index('DateTime', drop=False)

    # fig2 = plt.figure(figsize=(12, 6))
    # ax3 = ol_sl['windsnelheid'].plot.hist(label='Oude Leede 13-14', alpha=0.5)
    # ax4 = cs_sl['windsnelheid'].plot.hist(label='Central Station 13-14', alpha=0.5)
    # ax5 = dh_sl['windsnelheid'].plot.hist(label='Delfshaven 19-20', alpha=0.5)
    # ax3.set_xlabel('Wind Speed [m/s]', fontsize=12)
    # fig2.suptitle('Winter seasons')
    # ax3.axvline(x=ol_sl['windsnelheid'].mean(), ymax=0.5, color='blue', alpha=0.5)
    # ax4.axvline(x=cs_sl['windsnelheid'].mean(), ymax=0.5, color='red', alpha=0.5)
    # ax5.axvline(x=dh_sl['windsnelheid'].mean(), ymax=0.5, color='green', alpha=0.5)
    # plt.legend()
    # plt.show()
    #
    # fig3 = plt.figure(figsize=(12, 6))
    # ax6 = ol_sl['windsnelheid'].rolling('3H').mean().plot(label='Oude Leede', linewidth=2)
    # ax7 = cs_sl['windsnelheid'].rolling('3H').mean().plot(label='Central Station', linewidth=2)
    # ax8 = dh_sl['windsnelheid'].rolling('3H').mean().plot(label='Delfshaven', linewidth=2)
    # fig2.suptitle('Winter seasons')
    # plt.legend()
    # plt.show()

    cor_velocity = ol_sl['windsnelheid'].corr(cs_sl['windsnelheid'])
    cor_direction = ol_sl['windrichting'].corr(cs_sl['windrichting'])
    print('Date: ', sdate_2013, edate_2013, '| Velocity correlation: ', cor_velocity, '| Wind direction correlation: ',
          cor_direction)

def final_comparison(dh, cs, rf, ol):

    # wind_direction = [(450 - x) % 360 for x in wind_direction_compas]

    dh.index = pd.to_datetime(dh['DateTime'])
    cs.index = pd.to_datetime(cs['DateTime'])
    ol.index = pd.to_datetime(ol['DateTime'])
    rf.index = pd.to_datetime(rf['datum'])
    print('hodl')

    dh = dh[['windsnelheid', 'windrichting', 'Tair_{Avg}']]
    ol = ol[['windsnelheid', 'windrichting', 'Tair_{Avg}']]
    cs = cs[['windsnelheid', 'windrichting', 'Tair_{Avg}']]
    rf = rf[['windsnelheid', 'windrichting']]

    # Convert wind direction from compass rose to unit circle degrees

    dh['windrichting_compass'] = [(450 - x) % 360 for x in dh['windrichting']]
    cs['windrichting_compass'] = [(450 - x) % 360 for x in cs['windrichting']]
    ol['windrichting_compass'] = [(450 - x) % 360 for x in ol['windrichting']]
    rf['windrichting_compass'] = [(450 - x) % 360 for x in rf['windrichting']]

    # query at a certain time on reference station, certain wind direction:
    WD = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    WS = [1, 2, 3, 4, 5, 6]

    ol_cs = pd.merge(left=ol, left_on='DateTime', right=cs, right_on='DateTime', suffixes=("_ol","_cs"))
    ol_dh = pd.merge(left=ol, left_on='DateTime', right=dh, right_on='DateTime', suffixes=("_ol","_dh"))

    print('hodl')

    comp_cs = pd.DataFrame(columns=['\hd{station}', '\hd{reference station}', '\hd{query WD ref}', '\hd{score}'])
    comp_cs['\hd{query WD ref}'] = WD
    comp_cs['\hd{station}'] = 'centrum'
    comp_cs['\hd{reference station}'] = 'oude leede'
    # comp_cs['\hd{score}'] = [20, 19, 17, 15, 15, 17, 19, 20, 18, 16, 16, 18]
    comp_cs['\hd{score}'] = [21, 20, 18, 16, 16, 18, 20, 21, 19, 17, 17, 19]

    for idx, row in comp_cs.iterrows():
        qwd = row['\hd{query WD ref}']
        for qws in WS:
            if qwd == 0:
                temp_ol_cs = ol_cs.loc[(ol_cs['windrichting_compass_ol'].between(355, 360) | ol_cs['windrichting_compass_ol'].between(0, 5))
                    & ol_cs['windsnelheid_ol'].between(qws - 0.5, qws + 0.5)]
            else:
                temp_ol_cs = ol_cs.loc[ol_cs['windrichting_compass_ol'].between(qwd-5, qwd+5) & ol_cs['windsnelheid_ol'].between(qws-0.5, qws+0.5)]

            comp_cs.loc[idx,'\hd{station mean WS %s ±0.5}'%qws] = round(temp_ol_cs['windsnelheid_cs'].mean(), 2)
            comp_cs.loc[idx,'\hd{ref mean WS %s ±0.5}'%qws] = round(temp_ol_cs['windsnelheid_ol'].mean(), 2)
            comp_cs.loc[idx,'\hd{\# measurements %s}'%qws] = len(temp_ol_cs.index)

    print(comp_cs.to_latex(index=False, escape=False))

    #################

    # test for smaller ws range
    comp_cs = pd.DataFrame(columns=['\hd{station}', '\hd{reference station}', '\hd{query WD ref}', '\hd{score}'])
    comp_cs['\hd{query WD ref}'] = WD
    comp_cs['\hd{station}'] = 'centrum'
    comp_cs['\hd{reference station}'] = 'oude leede'
    # comp_cs['\hd{score}'] = [20, 19, 17, 15, 15, 17, 19, 20, 18, 16, 16, 18]
    comp_cs['\hd{score}'] = [21, 20, 18, 16, 16, 18, 20, 21, 19, 17, 17, 19]
    for idx, row in comp_cs.iterrows():
        qwd = row['\hd{query WD ref}']
        for qws in WS:
            if qwd == 0:
                temp_ol_cs = ol_cs.loc[(ol_cs['windrichting_compass_ol'].between(355, 360) | ol_cs['windrichting_compass_ol'].between(0, 5))
                    & ol_cs['windsnelheid_ol'].between(qws - 0.1, qws + 0.1)]
            else:
                temp_ol_cs = ol_cs.loc[ol_cs['windrichting_compass_ol'].between(qwd-5, qwd+5) & ol_cs['windsnelheid_ol'].between(qws-0.1, qws+0.1)]

            comp_cs.loc[idx,'\hd{station mean WS %s ±0.1}'%qws] = round(temp_ol_cs['windsnelheid_cs'].mean(), 2)
            comp_cs.loc[idx,'\hd{ref mean WS %s ±0.1}'%qws] = round(temp_ol_cs['windsnelheid_ol'].mean(), 0)
            comp_cs.loc[idx,'\hd{\# measurements %s}'%qws] = len(temp_ol_cs.index)

    print(comp_cs.to_latex(index=False, escape=False))

    ##################

    comp_dh = pd.DataFrame(columns=['\hd{station}', '\hd{reference station}', '\hd{query WD ref}', '\hd{score}'])
    comp_dh['\hd{query WD ref}'] = WD
    comp_dh['\hd{station}'] = 'delfshaven'
    comp_dh['\hd{reference station}'] = 'oude leede'
    # comp_dh['\hd{score}'] = [17, 15, 14, 15, 17, 17, 16, 14, 13, 14, 16, 18]
    comp_dh['\hd{score}'] = [16, 14, 13, 14, 16, 16, 15, 13, 12, 13, 15, 17]
    for idx, row in comp_dh.iterrows():
        qwd = row['\hd{query WD ref}']
        for qws in WS:
            if qwd == 0:
                temp_ol_dh = ol_dh.loc[(ol_dh['windrichting_compass_ol'].between(355, 360) | ol_dh['windrichting_compass_ol'].between(0, 5))
                    & ol_dh['windsnelheid_ol'].between(qws - 0.5, qws + 0.5)]
            else:
                temp_ol_dh = ol_dh.loc[ol_dh['windrichting_compass_ol'].between(qwd-5, qwd+5) & ol_dh['windsnelheid_ol'].between(qws-0.5, qws+0.5)]

            comp_dh.loc[idx,'\hd{station mean WS %s ±0.5}'%qws] = round(temp_ol_dh['windsnelheid_dh'].mean(), 2)
            comp_dh.loc[idx,'\hd{ref mean WS %s ±0.5}'%qws] = round(temp_ol_dh['windsnelheid_ol'].mean(), 2)
            comp_dh.loc[idx,'\hd{\# measurements %s}'%qws] = len(temp_ol_dh.index)

    print(comp_dh.to_latex(index=False, escape=False))


    #####################


    # test for smaller ws range
    comp_dh = pd.DataFrame(columns=['\hd{station}', '\hd{reference station}', '\hd{query WD ref}', '\hd{score}'])
    comp_dh['\hd{query WD ref}'] = WD
    comp_dh['\hd{station}'] = 'delfshaven'
    comp_dh['\hd{reference station}'] = 'oude leede'
    # comp_dh['\hd{score}'] = [17, 15, 14, 15, 17, 17, 16, 14, 13, 14, 16, 18]
    comp_dh['\hd{score}'] = [16, 14, 13, 14, 16, 16, 15, 13, 12, 13, 15, 17]
    for idx, row in comp_dh.iterrows():
        qwd = row['\hd{query WD ref}']
        for qws in WS:
            if qwd == 0:
                temp_ol_dh = ol_dh.loc[(ol_dh['windrichting_compass_ol'].between(355, 360) | ol_dh['windrichting_compass_ol'].between(0, 5))
                    & ol_dh['windsnelheid_ol'].between(qws - 0.1, qws + 0.1)]
            else:
                temp_ol_dh = ol_dh.loc[ol_dh['windrichting_compass_ol'].between(qwd-5, qwd+5) & ol_dh['windsnelheid_ol'].between(qws-0.1, qws+0.1)]

            comp_dh.loc[idx,'\hd{station mean WS %s ±0.1}'%qws] = round(temp_ol_dh['windsnelheid_dh'].mean(), 2)
            comp_dh.loc[idx,'\hd{ref mean WS %s ±0.1}'%qws] = round(temp_ol_dh['windsnelheid_ol'].mean(), 2)
            comp_dh.loc[idx,'\hd{\# measurements %s}'%qws] = len(temp_ol_dh.index)


    cmap = cm.get_cmap('viridis')
    min_c = np.int(8)
    max_c = np.int(26)


    fig = plt.figure(figsize=(12,12))
    ax1 = fig.add_subplot(221, projection='3d')
    width = .66
    depth = 20

    _x = np.arange(1,7)
    _x_tp = np.arange(0.5,6.5,1)

    # _x = np.flip(_x)
    _y = np.asarray([x - 15 for x in WD])
    _c = np.asarray(comp_cs['\hd{score}'])
    _xx, _yy = np.meshgrid(_x_tp, _y)
    _cx, _cy = np.meshgrid(_x_tp, _c)
    c = _cy.ravel()
    rgba = [cmap((k - min_c) /(max_c-min_c)) for k in c]
    x, y = _xx.ravel(), _yy.ravel()
    top = comp_cs[['\hd{station mean WS 1 ±0.1}', '\hd{station mean WS 2 ±0.1}', '\hd{station mean WS 3 ±0.1}', '\hd{station mean WS 4 ±0.1}', '\hd{station mean WS 5 ±0.1}', '\hd{station mean WS 6 ±0.1}']]
    top = top.to_numpy()
    top = top.flatten()
    bottom = np.zeros_like(top)
    ax1.bar3d(x, y, bottom, width, depth, top, color=rgba, shade=True, zsort='average')
    ax1.view_init(ax1.elev, ax1.azim - 90)
    ax1.set_yticks(WD)
    ax1.set_yticklabels(list(zip(WD,_c)))
    ax1.set_xticks(list(_x))
    ax1.set_title('a. Centrum station 3D Bar plot', y=1.04, fontsize=10)
    ax1.set_zlabel('Centrum mean wind velocity [m/s]', fontsize=8)
    ax1.set_xlabel('Oude Leede mean wind velocity [m/s]', fontsize=8)
    ax1.set_ylabel('Wind direction (azimuth), score', fontsize=8, labelpad=16)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6, rotation=45)
    for ticky in ax1.yaxis.get_majorticklabels():
        ticky.set_horizontalalignment("right")
    for tickx in ax1.xaxis.get_majorticklabels():
        tickx.set_horizontalalignment("left")

    # plt.show()
    #
    # fig = plt.figure(figsize=(12,12))
    ax2 = fig.add_subplot(222, projection='3d')
    _x2 = np.arange(1,7)
    _x2_tp = np.arange(0.5,6.5,1)
    _y2 = np.asarray([x - 15 for x in WD])
    _c2 = np.asarray(comp_dh['\hd{score}'])
    _xx2, _yy2 = np.meshgrid(_x2_tp, _y2)
    _cx2, _cy2 = np.meshgrid(_x2_tp, _c2)
    c2 = _cy2.ravel()
    rgba2 = [cmap((k - min_c) /(max_c-min_c)) for k in c2]
    x2, y2 = _xx2.ravel(), _yy2.ravel()
    top2 = comp_dh[['\hd{station mean WS 1 ±0.1}', '\hd{station mean WS 2 ±0.1}', '\hd{station mean WS 3 ±0.1}', '\hd{station mean WS 4 ±0.1}', '\hd{station mean WS 5 ±0.1}', '\hd{station mean WS 6 ±0.1}']].to_numpy()
    # top2 = comp_dh[['\hd{station mean WS 1 ±0.5}', '\hd{station mean WS 2 ±0.5}', '\hd{station mean WS 3 ±0.5}', '\hd{station mean WS 4 ±0.5}', '\hd{station mean WS 5 ±0.5}', '\hd{station mean WS 6 ±0.5}']].to_numpy()
    top2 = top2.flatten()
    top2 = np.nan_to_num(top2)
    bottom2 = np.zeros_like(top2)
    ax2.bar3d(x2, y2, bottom2, width, depth, top2, color=rgba2, shade=True, zsort='average')
    ax2.view_init(ax2.elev, ax2.azim - 90)
    ax2.set_yticks(WD)
    ax2.set_yticklabels(list(zip(WD,_c2)))
    ax2.set_xticks(list(_x2))
    ax2.set_title('b. Delfshaven station 3D Bar plot', y=1.04, fontsize=10)
    ax2.set_zlabel('Delfshaven mean wind velocity [m/s]', fontsize=8)
    ax2.set_xlabel('Oude Leede mean wind velocity [m/s]', fontsize=8)
    ax2.set_ylabel('Wind direction (azimuth), score', fontsize=8, labelpad=16)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6, rotation=45)
    for ticky in ax2.yaxis.get_majorticklabels():
        ticky.set_horizontalalignment("right")
    for tickx in ax2.xaxis.get_majorticklabels():
        tickx.set_horizontalalignment("left")

    rgba3 = [cmap((k - min_c) /(max_c-min_c)) for k in _c]
    ax3 = fig.add_subplot(325)
    ax3.bar(np.asarray(WD), _c-8, color=rgba3, width=25, bottom=8)
    ax3.set_xticks(WD)
    ax3.set_yticks(range(8, 27))
    ax3.set_title('c. Centrum score', fontsize=10)
    ax3.set_xlabel('wind direction (azimuth)', fontsize=8)
    ax3.set_ylabel('score', fontsize=8)
    plt.gca().invert_xaxis()
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)

    rgba4 = [cmap((k - min_c) /(max_c-min_c)) for k in _c2]
    ax4 = fig.add_subplot(326)
    ax4.bar(np.asarray(WD), _c2-8, color=rgba4, width=25, bottom=8)
    ax4.set_xticks(WD)
    ax4.set_yticks(range(8, 27))
    ax4.set_title('d. Delfshaven score', fontsize=10)
    ax4.set_xlabel('wind direction (azimuth)', fontsize=8)
    ax4.set_ylabel('score', fontsize=8)
    plt.gca().invert_xaxis()
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.legend()


    plt.show()
    print(_c-8)
    # print(comp_dh.to_latex(index=False, escape=False))
    print(comp_dh.to_latex(na_rep="",escape=False, index=False, bold_rows=True, multirow=True, column_format="llllllllllllllllllllll"))

    # print(comp_cs.to_latex(index=False, escape=False))

    # temp_ol_dh = ol_dh.loc[ol_dh['windrichting_compass_ol'].between(325, 330) & ol_dh['windsnelheid_ol'].between(5 - 0.05,5 + 0.05)]
    # temp_ol_cs = ol_cs.between_time('06:00', '12:00').loc[ol_cs['windrichting_compass_ol'].between(325, 330) & ol_cs['windsnelheid_ol'].between(5 - 0.05, 5 + 0.05)]
    # temp_ol_cs = ol_cs.loc[ol_cs['windrichting_compass_ol'].between(325, 330) & ol_cs['windsnelheid_ol'].between(5 - 0.05,5 + 0.05)]
    # temp_ol_cs = ol_cs.between_time('06:00', '12:00').loc[ol_cs['windsnelheid_ol'].between(5 - 0.05,5 + 0.05)]
    #
    # fig2 = plt.figure(figsize=(12, 6))
    # ax10 = fig2.add_subplot(121)
    # ax10 = ol_cs.between_time('06:00', '12:00').loc[ol_cs['windrichting_compass_ol'].between(25, 30) & ol_cs['windsnelheid_ol'].between(5 - 0.05,5 + 0.05)].plot.scatter(x='windsnelheid_ol',
    #                               y='windsnelheid_cs',
    #                               c='Tair_{Avg}_ol',
    #                               colormap='gist_heat', ax=ax10)
    # ax10.set_title('06:00 - 12:00 ; WD 25 - 30')
    #
    # ax11 = fig2.add_subplot(122)
    # ax11 = ol_cs.between_time('12:00', '18:00').loc[ol_cs['windrichting_compass_ol'].between(25, 30) & ol_cs['windsnelheid_ol'].between(5 - 0.05,5 + 0.05)].plot.scatter(x='windsnelheid_ol',
    #                               y='windsnelheid_cs',
    #                               c='Tair_{Avg}_ol',
    #                               colormap='gist_heat', ax=ax11)
    # ax11.set_title('12:00 - 18:00 ; WD 25 - 30')
    #
    # plt.show()

    # print('WS - %s ±0.5' % WS[0])
    #
    # print(ol_cs_180.mean())
    # print(ol_cs_150.describe())
    # print(ol_cs_180.describe())
    # print(ol_cs_30.describe())
    # print(ol_cs_210.describe())
    # print(ol_dh_210.describe())
    # print
    # check wind directions



def function_of_wind(dh, cs, rf, ol):
    # we're going to look at a day where DH is within a certain wind window,
    # and look at what the reference stations are doing?

    def dat(start, end, data, days=7):
        period = [start, end]
        measure = data[(data.index > period[0]) &
                       (data.index < period[1])].rolling(12 * 24 * days).mean()['windsnelheid']

        mean = measure.mean()
        std = measure.std()
        print(mean)
        measure_std = (measure - mean) / std
        return measure.values, measure_std.values

    def norm(data, days=7):
        measure = data['windsnelheid']

        mean = measure.mean()
        std = measure.std()
        print(mean)
        measure_std = (measure - mean) / std
        return measure.values, measure_std.values

    def norm_avg(data, time='30T'):
        measure = data['windsnelheid'].rolling(time).mean()
        mean = measure.mean()
        std = measure.std()
        print(mean)
        measure_std = (measure - mean) / std
        return measure.values, measure_std.values

    def dif(data1, data2):
        measure_direction = data1['windrichting'] - data2['windrichting']
        measure_velocity = data1['windsnelheid'] - data2['windsnelheid']

        return measure_direction, measure_velocity

    dh.index = pd.to_datetime(dh['DateTime'])
    cs.index = pd.to_datetime(cs['DateTime'])
    ol.index = pd.to_datetime(ol['DateTime'])
    rf.index = pd.to_datetime(rf['datum'])

    cs_window = [236.25, 258.75]
    dh_window = [258.75, 281.25]

    my_dh = dh[(dh['windrichting'] >= dh_window[0]) & (dh['windrichting'] <= dh_window[1])]
    dh_day = my_dh['2015-04-15']
    ol_match_dh = ol.loc[ol.index.isin(dh_day.index),:]


    sns.set(rc={'figure.figsize': (10, 4)})
    ax1 = dh_day['windsnelheid'].plot(label='Delfshaven', linewidth=0.5)
    ax2 = ol_match_dh['windsnelheid'].plot(label='Oude Leede', linewidth=0.5)
    ax1.set_ylabel('Wind Velocity [m/s]')
    ax1.set_xlabel('Time')
    plt.legend()
    plt.show()

    my_cs = cs[(cs['windrichting'] >= cs_window[0]) & (cs['windrichting'] <= cs_window[1])]

    cs_select = cs['2014-01-05':'2014-01-05']
    ol_select = ol['2014-01-05':'2014-01-05']

    ax3 = cs_select['windrichting'].plot(label='Centrum', linewidth=0.5)
    ax4 = ol_select['windrichting'].plot(label='Oude Leede', linewidth=0.5)
    ax3.set_ylabel('Wind direction')
    ax3.set_xlabel('Time')
    plt.legend()
    plt.show()

    ax3 = cs_select['windsnelheid'].plot(label='Centrum', linewidth=0.5)
    ax4 = ol_select['windsnelheid'].plot(label='Oude Leede', linewidth=0.5)
    ax3.set_ylabel('Wind velocity [m/s]')
    ax3.set_xlabel('Time')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 4), dpi=80)
    plt.plot(norm(cs_select)[1], label='Centrum', linewidth=0.5)
    plt.plot(norm(ol_select)[1], label='Oude Leede', linewidth=0.5)
    # plt.set_ylabel('Wind velocity [m/s]')
    # plt.set_xlabel('Time')
    # plt.set_title('Normalized')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 4), dpi=80)
    plt.plot(norm_avg(cs_select)[1], label='Centrum', linewidth=0.5)
    plt.plot(norm_avg(ol_select)[1], label='Oude Leede', linewidth=0.5)
    # plt.set_ylabel('Wind velocity [m/s]')
    # plt.set_xlabel('Time')
    # plt.set_title('Normalized')
    plt.legend()
    plt.show()


    # Delfshaven
    time = ['2015-06-20','2015-06-22']
    dh_select = dh[time[0]:time[1]]
    ol_select = ol[time[0]:time[1]]

    plt.figure(figsize=(10, 4), dpi=80)
    plt.plot(norm_avg(dh_select)[0], label='Delfshaven', linewidth=0.5)
    plt.plot(norm_avg(ol_select)[0], label='Oude Leede', linewidth=0.5)
    plt.xlabel('Time')
    plt.ylabel('Z-score of wind velocity')
    plt.title('Normalized 30 min average wind velocity '+str(time))
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 4), dpi=80)
    plt.plot(dh_select['windrichting'], label='Delfshaven', linewidth=0.5)
    plt.plot(ol_select['windrichting'], label='Oude Leede', linewidth=0.5)
    plt.xlabel('Time')
    plt.ylabel('Wind direction [degree]')
    plt.title('Wind direction  '+str(time))
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 4), dpi=80)
    plt.plot(dh_select['windsnelheid'], label='Delfshaven', linewidth=0.5)
    plt.plot(ol_select['windsnelheid'], label='Oude Leede', linewidth=0.5)
    plt.plot(dif(dh_select, ol_select)[1].rolling('30T').mean(), label='Difference wind velocity 30 min rolling avg')

    plt.xlabel('Time')
    plt.ylabel('Wind velocity [m/s]')
    plt.title('Wind velocity  ' + str(time))
    plt.legend()
    plt.show()

    plt.plot(dif(dh_select, ol_select)[1].rolling('30T').mean(), label='Difference windsnelheid')
    plt.title('Difference windsnelheid '+str(time))
    plt.show()
    plt.plot(dif(dh_select, ol_select)[0], label='Difference windrichting')
    plt.plot(dh_select['windrichting'], label='Windrichting')
    plt.title('Difference windrichting '+time)
    plt.legend()
    plt.show()

    cs_select = cs['2013-12-23':'2014-04-04']
    plt.plot(norm_avg(cs, time='30d')[1])
    plt.plot(cs_select['windsnelheid'], label='cs')
    plt.show()
    plt.close()
    print('Hodl')

# reference station
rf_dtypes = {'windsnelheid': np.float64, 'windrichting': np.float64}
rf_data = pd.read_csv('/Users/wesseldejongh/PycharmProjects/ThesisCode/windystreet/export_915096001-2.csv', ";", parse_dates=[0], header=[0], na_values='-', dtype=rf_dtypes)

# delfshaven station
fields = pd.read_fwf("http://weather.tudelft.nl/csv/fields.txt", header=None)
dh_data = pd.read_csv("http://weather.tudelft.nl/csv/Delfshaven.csv", names=fields[1], parse_dates=[0], na_values='NAN')
dh_data.rename(columns={'WindSpd_{Avg}':'windsnelheid', 'WindDir_{Avg}': 'windrichting'}, inplace=True)
# print(rf_data.dtypes)

# OudeLeede station
ol_data = pd.read_csv("http://weather.tudelft.nl/csv/OudeLeede.csv", names=fields[1], parse_dates=[0], na_values='NAN')
ol_data.rename(columns={'WindSpd_{Avg}':'windsnelheid', 'WindDir_{Avg}': 'windrichting'}, inplace=True)

# centrum station
cs_data = pd.read_csv("http://weather.tudelft.nl/csv/Centrum.csv", names=fields[1], parse_dates=[0], na_values='NAN')
cs_data.rename(columns={'WindSpd_{Avg}':'windsnelheid', 'WindDir_{Avg}': 'windrichting'}, inplace=True)

# pandas print setting
pd.options.display.max_columns = None

# oneday_rose(rf_data, dh_data)
# winter_season19(rf_data, dh_data, 200)
# oneday_lineplot(rf_data, dh_data)
# oneweek_multiplebox(rf_data, dh_data)
# small_window_boxplot(rf_data, dh_data)
# print('Overall:')
# corr(rf_data, dh_data, 'season')
print('Within wind window:')
# corr_wd(rf_data, dh_data, 'season')
# season_2019()

# winter_hist(dh_data, ol_data, cs_data)
# winter_3haverages_cs(ol_data, cs_data)
# winter_3haverages_dh(rf_data, dh_data)
# year_rose(ol_data, cs_data, dh_data, rf_data)
# winter_season2014_hist_kde(ol_data, cs_data)

# function_of_wind(dh_data, cs_data, rf_data, ol_data)
final_comparison(dh_data, cs_data, rf_data, ol_data)

print('To the moon!')
