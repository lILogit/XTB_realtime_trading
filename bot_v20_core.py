

import datetime
from datetime import timedelta
import multiprocessing as mp
from dateutil.tz import *
import json
import urllib
from urllib.request import urlopen
import socket
import sqlite3
import ssl
import os
from pytz import timezone
import sys
import time
import math
from threading import Thread
import numpy as np
import csv
import pandas as pd
import pytz
from loguru import logger
#from pyqt_finance import QStockWidget
from utils.shared_data import SYMBOL

from XTBApi.api import Client, PERIOD, MODES, TRANS_TYPES
from utils.secrets import credentials
from utils.shared_data import PIP_VALUE
from utils.xAPIConnector import API_MAX_CONN_TRIES, DEFAULT_XAPI_ADDRESS, DEFUALT_XAPI_STREAMING_PORT, APIClient, \
    loginCommand, API_SEND_TIMEOUT
import utils.functions as f2

#run/install qt a pyqt z nastavení pycharm

from PyQt5.QtCore import QThread, Qt, pyqtSignal, QCoreApplication, QThreadPool, QTimer
import PyQt5.QtWidgets as qt
#import pyqtgraph as pg
from PyQt5 import QtCore, QtGui

import signal as signal_


# GLOBALS
SYMBOL = "EURUSD"
timeframe = 60
d = 0
mean = 0
dmax = 0.0035
dmin = 0.0001

sample_size = 2880  #minutes 43200 = month M1 2880 = 2 days xAPI
level = 1
slev_scale = 10
na_filter = True  # includes also NA results for SLEV optimum
only_buy = False  # only Buy or both directions
csv_file = 'orders_file.csv'

phi = (1 + math.sqrt(5)) / 2

# SQLite config
#db_path = "/home/nemo/PycharmProjects/TradeBot/mydb.db"    #Ubuntu
db_path = "/Users/jirka/PycharmProjects/MYProject/mydb.db" #mac book

local_time_zone = 'Europe/Prague'
item_name = "EURUSD"
levels = 1
time_frame = 60 #seconds
sample_size = 2880  #minutes 43200 = month M1 2880 = 2 days xAPI
scale = 10
digits = 5
min_size = 30
d_max_pct = 1 / (phi ** 2)
maximize_field = 'Equity Final [$]' #"# Trades" #'Equity Final [$]'

HIGH = 'high'
LOW = 'low'
TIMESTAMP = 'timestamp'
OPEN = 'open'
CLOSE = 'close'
VOLUME = 'vol'
CTM_STRING = 'ctm_string'


RATE_INFO = 'rateInfos'
time_format = '%d/%m/%Y %H:%M:%S'
FIXED_RISK = 0.02

BALANCE = 0.0
MARGIN_FREE = 0.0
last_trade = 0
precision = 0
last_order_type = ""
last_order_id = -1

EXPIRATION_HOURS = 5

signals = []
orders = []
a = []
a_min = 0
a_max = 0
prev_price = 0
last_price = 0
last_enter_level = -1
trigger_status = 0
open_order_id = -1  # position id
prev_order_id = -1
prev_trigger_status = -1
#open_timestamp = 0
#close_timestamp = 0
open_times = -1
close_time = -1
closed_order_status = False
open_order_status = None
#open_order_timestamp = 0
start_midnight = 0
open_obj_type = None
open_level = -1
thread_id = 0

levels = 1
signals = []
d = 51
m = 105800
#market closing time
_hour = 22
_minutes= 2
mp.set_start_method('fork')

class signal:
    def __init__(self, type, sl, enter, exit):
        self.type = type
        self.sl = sl
        self.enter = enter
        self.exit = exit
class Worker(QThread):
    def __init__(self, parent=None):

        # *********  GLOBAL VARIABLES *********
        # global a, tp, sl, heat_tp, ts_all, signals, ts

        QtCore.QThread.__init__(self, parent)
        logger.info("Init Worker instance..")
        logger.info("Installing signal handler CTRL+C...only console mode ")
        signal_.signal(signal_.SIGINT, self.ctrl_handler)
        logger.info("CTRL+C Signal handler done")

        # XTB client
        self.client = Client()
        self.client.login(credentials.XTB_DEMO_ID, credentials.XTB_PASS_KEY, mode=credentials.XTB_ACCOUNT_TYPE)
        logger.info("Client logged in")

        # QTimer end of working day, skip weekend, sunday - friday - data update
        offset = self.get_time_to_market_close(22,49)
        logger.info("Time to today market close and signals update - hours {} milliseconds {}", round(offset / 3600000, 1), offset)
        self.timer_interval = offset
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_signals)
        self.timer.setTimerType(Qt.TimerType.VeryCoarseTimer)
        self.timer.start(self.timer_interval)

        logger.info("Init Worker instance..")

        #ENTRY SIGNALS
        """
        logger.info("SLEV entry to signals array")
        a = f2.slev_levels_single(levels, d / 100000, m / 100000)
        l = len(a)
        x = range(0, l - 3, 3)
        for n in x:
            # sl, enter, exit
            signals.append(signal('BUY', a[n + 1], a[n + 2], a[n + 3]))
            signals.append(signal('SELL', a[n + 2], a[n + 1], a[n]))
        logger.info("Signals d {}  m {}", d, m)
        logger.info("Backtest entry")
        """

    def run(self):

        global signals, a, trigger_status, last_price, prev_price, last_enter_level, start_midnight
        global closed_time, open_order_status
        global open_time
        global a_min, a_max, open_order_id, prev_order_id, open_timestamp, close_timestamp, closed_order_status
        global open_obj_type, open_level,precision

        # LOGIN

        logger.info("Running trading loop...")
        self.thread = QThreadPool()
        self.thread.setMaxThreadCount(3)

        # INIT ------ TIME XTB SERVER AND MIDHIGHT
        start_str = time.strftime("%m/%d/%Y") + " 00:00:00"
        start_midnight = time.mktime(time.strptime(start_str, "%m/%d/%Y %H:%M:%S"))
        open_server_time = int(self.client.get_server_time()["time"] / 1000)
        # print("------- Midnight time today: ", time.ctime(start_midnight))  # timestamp today at 00:00:00
        # print("------- Open Server Time:     ", time.ctime(open_server_time))
        logger.info("Midnight time today: {}", time.ctime(open_server_time))
        logger.info("Open Server Time: {}", time.ctime(open_server_time))
        # CHECK IF MARKET OPEN
        if self.client.check_if_market_open([SYMBOL]):
            # print("------- Obchoduje se tento symbol..",SYMBOL)
            logger.info("Obchoduje se tento symbol..", SYMBOL)
        else:
            # print("-------Neobchoduje tento symbol..",SYMBOL)
            logger.error("Neobchoduje tento symbol: {}", SYMBOL)

        # GET actual SYMBOL tick price
        object = self.client.get_symbol(symbol=SYMBOL)
        # ROUND NUMBERS
        precision = self.client.get_symbol(symbol=SYMBOL)['precision']
        if (precision !=0): logger.info("Get Symbol precision: {}", 0.1**precision)
        else: logger.error("Can't get symbol precision")
        last_price = object['ask']
        prev_price = last_price
        # once due to first send order

        # Loop
        counter = 0
        self.keepRunning = True
        c = 0
        while self.keepRunning:
            try:
                object = self.client.get_symbol(symbol=SYMBOL)  # signal threat
                last_price = object['ask']
            except:
                # print("get price error")
                logger.error("Get price error..{}",c)

                # counter 3x if internet and error - relogin
                c = c + 1
                if c == 3 and f2.is_internet():
                    #relogin session
                    self.client = Client()
                    self.client.login(credentials.XTB_DEMO_ID, credentials.XTB_PASS_KEY, mode='demo')
                    logger.info("Client relogin: {}", self.client.status)
                    c = 0
                else:
                    logger.error("Internet disconnected.....")
                continue

            # CHECK TRIGGER ONLY IF PRICE CHANGED
            if (last_price != prev_price):
                counter = counter + 1
                logger.info("Price change: {}", object['ask'])

                # cross trigger status
                for obj in signals:
                    trigger_status = 0
                    if (last_price <= obj.enter and prev_price > obj.enter) or (
                            last_price >= obj.enter and prev_price < obj.enter):
                        trigger_status = 1
                        break

                # check open order and send order
                if trigger_status == 1:
                    logger.info("Sent order..")
                    #TEST self.send_order(obj.type, last_price+0.0005, last_price, last_price-0.0005, last_price)
                    self.send_order(obj.type, obj.sl, obj.enter, obj.exit, last_price)
                else:
                    trigger_status = 0
                time.sleep(2)
                prev_price = last_price
    def ctrl_handler(self,signal_, frm):
        logger.info("CTRL+C logout client and exit")
        self.client.logout()
        sys.exit(0)
    def get_time_to_market_close(self,hour, minute):
        now_dt = datetime.datetime.now()
        # utc_now = datetime.utcnow()
        end_day = now_dt.replace(hour=hour, minute=minute, second=0, microsecond=0)
        now_ts, utc_ts = map(time.mktime, map(datetime.datetime.timetuple, (now_dt, end_day)))
        offset = int((utc_ts - now_ts)) * 1000
        return offset
    def update_signals(self):
            logger.info("Update signals..")
            self.client.close_all_trades()
            df = self.get_last_OHLC_data_xAPI(item_name, sample_size, time_frame, local_time_zone)
            logger.info("Get last 2 days data...")
            # get last 2 days data
            ts_today, ts_prev = self.get_last_2_days_XTB_data(df)
            dm, mm = self.get_dm_range(ts_prev)
            d, m, trades = self.Backtest_optimize(ts_prev, 1, dm, mm, True)
            trades = self.Backtest_dm(ts_today, 1, d, m, True)
            #logger.info("Trades {}", trades._trades)
            logger.info("d,m entry update")
            a = slev_levels_single(levels, d / 100000, m / 100000)
            l = len(a)
            x = range(0, l - 3, 3)
            for n in x:
                # sl, enter, exit
                signals.append(signal('BUY', a[n + 1], a[n + 2], a[n + 3]))
                signals.append(signal('SELL', a[n + 2], a[n + 1], a[n]))
            logger.info("Signals d {}  m {}", d, m)
            # Sun 23:00 - Mon 00:00 - Fri 22:00 (Sun-Thu 23:00 , Fri - 22:00)
            # backtest - plot - signals
            next_day_close = next_business_day(datetime.datetime.now()).replace(hour=22, minute=30)
            logger.info("Next timer interval..{}", next_day_close)
            now_ts, utc_ts = map(time.mktime, map(datetime.datetime.timetuple, (datetime.datetime.now(), next_day_close)))
            offset = int((utc_ts - now_ts)) * 1000  # milisecond
            logger.info("Next day  interval in milisecond ..{}", offset)
            #sleep mode if weekend or holidays
            #sleep offset - 24h self.keepRunning = False untill offset, logout
            self.timer.setInterval(offset)

            # check next day close time
    def send_order(self, type, sl, enter, tp, last_price):
        global EXPIRATION_HOURS, start_midnight, open_time, thread_id
        global open_order_id, open_order_status, precision
        global open_timestamp, close_timestamp, closed_order_status
        tp = round(tp, precision)
        sl = round(sl, precision)
        # ORDER TIME EXPIRATION
        delay = f2.to_milliseconds(days=0, hours=EXPIRATION_HOURS, minutes=0)
        expiration = int(self.client.get_server_time()["time"]) + delay

        if type == "BUY":
            mode = MODES.BUY.value
            # last_price = self.client.get_symbol(symbol=SYMBOL)['bid']
        else:
            mode = MODES.SELL.value
            # last_price = self.client.get_symbol(symbol=SYMBOL)['ask']
        # TRADE ORDER
        logger.info("Order send {}, mode {}", open_order_status, mode)
        if open_order_status != mode or open_order_status == "None":  # each direction only ones
            order_id = self.client.trade_transaction(SYMBOL, mode, TRANS_TYPES.OPEN.value,
                                                     self.client.get_symbol(symbol=SYMBOL)['lotMin'], price=last_price,
                                                     sl=sl, customComment="text", tp=tp)
            open_trades = self.client.get_trades(True)  # OPEN ORDERS ONLY
            if not any(d.get('order2', 000) == order_id['order'] for d in open_trades):
                # print('--------------------- Order Error.....!!!!')
                message = self.client.trade_transaction_status(int(order_id['order']))['message']
                # print("trade status", message)
                logger.error("Order status: {}", message)
                # write_order_status__to_file("Failed", order_id, message)
            else:
                # print('--------------------- Send Order Created order w. id:', order_id, " -----------")
                logger.info("Send Order Created order w. id {}", order_id)
                logger.info("trade status {}", self.client.trade_transaction_status(int(order_id['order'])))


        # self.client.close_all_trades() #TEST
        # self.client.logout()#TEST
    def get_last_OHLC_data_xAPI(self, item_name, sample_size, time_frame, local_time_zone):
        """
        get index OHLC history based on size, timeframe and convert to local time zone
        :param item_name: index eg."EURUSD"
        :param sample_size: sample size per time frame eg. minutes 43200 = month M1, 2880 minutes = 2 days
        :param time_frame: time frame in seconds eg. 60 = minute
        :param local_time_zone: Prague
        :return: dataframe
        """
        logger.info("Item {} Size {} Frame {}", item_name, sample_size, time_frame)
        price_hl = self.client.get_lastn_candle_history(item_name, time_frame, sample_size)  # seconds interval, number records
        lst = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        # Calling DataFrame constructor on list
        df = pd.DataFrame(columns=lst)
        # get local time
        now_dt = datetime.datetime.now()
        utc_now = datetime.datetime.utcnow()
        now_ts, utc_ts = map(time.mktime, map(datetime.datetime.timetuple, (now_dt, utc_now)))
        offset = int((now_ts - utc_ts) / 3600)
        for i in price_hl:
            # time = datetime.datetime.utcfromtimestamp(i["timestamp"]).strftime('%Y-%m-%d %H:%M:%S')
            local_time = datetime.datetime.fromtimestamp(i["timestamp"], datetime.timezone.utc) + timedelta(
                hours=offset)
            local_time = local_time.strftime('%Y-%m-%d %H:%M:%S')  #
            # open, close, high, low,, volume
            new_row = {'Datetime': local_time, 'Open': i["open"], 'High': i["high"], 'Low': i["low"],
                       'Close': i["close"],
                       'Volume': i["volume"]}
            df = df._append(new_row, ignore_index=True)
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        df.set_index("Datetime", inplace=True)
        return df
    def get_last_2_days_XTB_data(self,df):
        logger.info("Login XTB...")
        # -------- GET LAST 2 DAYS DATA
        # day 0
        last_date = df.index[-1]
        last_date_b = last_date.replace(hour=0, minute=0, second=0, microsecond=0)
        last_date_e = last_date.replace(hour=23, minute=59, second=59, microsecond=0)
        ts_today = df.loc[str(last_date_b):str(last_date_e)]
        # day -1
        prev_date = df.index[-len(ts_today) - 1]
        prev_date_b = prev_date.replace(hour=0, minute=0, second=0, microsecond=0)
        prev_date_e = prev_date.replace(hour=23, minute=59, second=59, microsecond=0)
        ts_prev = df.loc[str(prev_date_b):str(prev_date_e)]
        return ts_today, ts_prev
    def get_dm_range(self,ts):
        logger.info("Backtesting Optimize scale {}", scale)
        max_price = int(math.pow(10, digits) * ts["High"].max())
        min_price = int(math.pow(10, digits) * ts["Low"].min())

        HL_distance = int((max_price - min_price))

        max_price = int(max_price + HL_distance)
        min_price = int(min_price - HL_distance)

        c_range = int((max_price - min_price) / scale)
        d_max = int((max_price - min_price) * d_max_pct)
        d_size = int((d_max - min_size) / scale)

        d_range = range(min_size, d_max, d_size)
        m_range = range(min_price, max_price, c_range)

        logger.info("DM range D={} M={}", d_range, m_range)
        return d_range, m_range
    def Backtest_optimize(self, ts, level, dm, mm, plot):
        class SignalStrategy(Strategy):
            d = 0
            m = 0

            def init(self):
                signals.clear()  # nutno čistit jinak se vedme globální hodnota
                a = slev_levels_single(level, self.d / 100000, self.m / 100000)
                a = np.round(a, 5)
                for i in a:
                    self.bid = self.I(any_data, np.repeat(i, len(self.data)), name=str(i), color='black')
                l = len(a)
                x = range(0, l - 3, 3)
                for n in x:
                    # sl, enter, exit
                    signals.append(signal('BUY', a[n + 1], a[n + 2], a[n + 3]))
                    signals.append(signal('SELL', a[n + 2], a[n + 1], a[n]))

            def next(self):
                if not self.position:
                    for obj in signals:
                        if self.data.Low[-1] < obj.enter and self.data.High[-1] > obj.enter and obj.type == "BUY":
                            self.buy(sl=obj.sl, tp=obj.exit)  # put sl tp etc.. jen up směr
                            break
                        if self.data.Low[-1] < obj.enter and self.data.High[-1] > obj.enter and obj.type == "SELL":
                            self.sell(sl=obj.sl, tp=obj.exit)  # put sl tp etc.. jen down smšr
                            break

        bt = Backtest(ts, SignalStrategy)
        stat, heatmap = bt.optimize(d=dm, m=mm, maximize=maximize_field,
                                    return_heatmap=True)  # pozor na počet digits, musí odpovídat pak /100
        if plot:
            bt.plot(filename="Optimize", plot_pl=True, show_legend=True)
            plot_heatmaps(heatmap)
            logger.info("Previous Day Backtest Optimize plot ")
            # the best trade
            logger.info(" ---- Win strategy -------")
            logger.info("Optimal D: {}", stat._strategy.d)
            logger.info("Optimal M: {}", stat._strategy.m)
            logger.info("Optimal orders: \n {}", stat.to_string())
            a = slev_levels_single(levels, stat._strategy.d / 100000, stat._strategy.m / 100000)
            logger.info("A-levels: {}", a)
            logger.info("Last Day Prediction plot ")
        return stat._strategy.d, stat._strategy.m, stat._trades
    def Backtest_dm(self,ts, level, d_, m_, plot):
        class SignalStrategy_opt(Strategy):
            d = d_
            m = m_

            def init(self):
                signals.clear()  # nutno čistit jinak se vedme globální hodnota
                a = slev_levels_single(level, self.d / 100000, self.m / 100000)
                a = np.round(a, 5)
                for i in a:
                    self.bid = self.I(any_data, np.repeat(i, len(self.data)), name=str(i), color='black')
                l = len(a)
                x = range(0, l - 3, 3)
                for n in x:
                    # sl, enter, exit
                    signals.append(signal('BUY', a[n + 1], a[n + 2], a[n + 3]))
                    signals.append(signal('SELL', a[n + 2], a[n + 1], a[n]))

            def next(self):
                if not self.position:
                    for obj in signals:
                        if self.data.Low[-1] < obj.enter and self.data.High[-1] > obj.enter and obj.type == "BUY":
                            self.buy(sl=obj.sl, tp=obj.exit)  # put sl tp etc.. jen up směr
                            break
                        if self.data.Low[-1] < obj.enter and self.data.High[-1] > obj.enter and obj.type == "SELL":
                            self.sell(sl=obj.sl, tp=obj.exit)  # put sl tp etc.. jen down smšr
                            break

        bt_org = Backtest(ts, SignalStrategy_opt)
        output = bt_org.run()
        if plot:
            logger.info("Last Day Stats {}", output)
            logger.info("Last Day Hetmap plot ")
            bt_org.plot(show_legend=True, filename="Current")
            pd.set_option('display.max_columns', None)
            print(output._trades)
        return output._trades

if __name__ == "__main__":
    # Set up logging
    #https://github.com/Delgan/loguru
    #https://loguru.readthedocs.io/en/stable/index.html
    print = logger.info
    logger.add("bot_{time}.log") #,level="INFO")
    logger.info("Bot starting.. Symbol {}  dmin/dmax {}/{} level {} scale {} na_filter {} only_buy {}", SYMBOL, np.round(dmin,4), np.round(dmax,4), level, slev_scale, na_filter,only_buy)
    logger.info("Output file {}",'orders_file.csv')

    #if afterHours(): print("Nyní se neobchoduje...!")  # if afterHours True
    app = qt.QApplication([])
    bot = Worker()
    bot.start()
    sys.exit(app.exec_())




