import pandas as pd
import numpy as np
import yfinance as yf
import os
import datetime
import time
import matplotlib.pyplot as plt
import wrds
from pandas.tseries.offsets import *


use_NYSE_brkpts = False

# WRDS login information
data_folder = 'C:/Users/Damian/Desktop/QAM/ps1/'  
id_wrds = 'bobross'  

#conn = wrds.Connection(wrds_username=id_wrds)
#conn.create_pgpass_file()
#conn.close()

# Time period
min_year = 1926
max_year = 2020

min_shrcd = 10
max_shrcd = 11
possible_exchcd = (1, 2, 3)

# Step 1: Load CRSP returns (monthly)
conn = wrds.Connection(wrds_username=id_wrds)
mcrsp_raw = conn.raw_sql("""
                      select a.permno, a.date, a.ret, b.shrcd, b.exchcd, b.ticker, a.shrout, a.prc
                      from crspq.msf as a
                      left join crspq.msenames as b
                      on a.permno=b.permno
                      and b.namedt<=a.date
                      and a.date<=b.nameendt
                      where b.shrcd between """ + str(min_shrcd) + """ and  """ + str(max_shrcd) + """
                      and a.date between '01/01/1926' and '12/31/2020'
                      and b.exchcd in """ + str(possible_exchcd) + """
                      """)
conn.close()
#mcrsp_raw.to_pickle(data_folder + 'mcrsp_raw.pkl') 

# Step 2: Load CRSP Deslisting returns (monthly)
conn = wrds.Connection(wrds_username=id_wrds)
dlret_raw = conn.raw_sql("""
                      select a.permno, a.dlstdt, a.dlret, b.ticker
                      from crspq.msedelist as a
                      left join crspq.msenames as b
                      on a.permno=b.permno
                      and b.namedt<=a.dlstdt
                      and a.dlstdt<=b.nameendt
                      where b.shrcd between """ + str(min_shrcd) + """ and  """ + str(max_shrcd) + """
                      and a.dlstdt between '01/01/""" +str(min_year)+ """' and '12/31/""" +str(max_year)+ """'
                      and b.exchcd in """ + str(possible_exchcd) + """
                      """)
conn.close()
#dlret_raw.to_pickle(data_folder + 'dlret_raw.pkl')

#Step 3: Return Cleaning
mcrsp_raw['permno'] = mcrsp_raw['permno'].astype(int)
mcrsp_raw['date'] = pd.to_datetime(mcrsp_raw['date'], format='%Y-%m-%d',errors='ignore')
mcrsp_raw = mcrsp_raw.sort_values(by=['ticker','date']).reset_index(drop=True).copy()
mcrsp_raw['me'] = mcrsp_raw['prc'].abs() * mcrsp_raw['shrout'] * 1e-6
mcrsp_raw.drop(['prc', 'shrout'], axis=1, inplace=True)

#Step 4: Delist Cleaning
dlret_raw['permno'] = dlret_raw['permno'].astype(int)
dlret_raw['dlstdt'] = pd.to_datetime(dlret_raw['dlstdt'])
dlret_raw = dlret_raw.rename(columns={"dlstdt" : "date","ticker":"dlticker"}).copy()

#Merge
mcrsp = mcrsp_raw.merge(dlret_raw,how='outer',on=['date','permno'])
mcrsp['ticker'] = np.where(mcrsp['ticker'].isnull(),mcrsp['dlticker'],mcrsp['ticker'])
mcrsp.drop('dlticker',axis=1, inplace=True)

#Adjust for delisting return
mcrsp['ret'] = np.where(mcrsp['ret'].notna() & mcrsp['dlret'].notna(),
                       (1+mcrsp['ret'])*(1+mcrsp['dlret'])-1, mcrsp['ret'])
mcrsp['ret'] = np.where(mcrsp['ret'].isna() & mcrsp['dlret'].notna(), mcrsp['dlret'], mcrsp['ret'])
mcrsp = mcrsp[mcrsp['ret'].notna()].copy()
mcrsp = mcrsp[['date','permno','ticker','ret','me']].sort_values(by=['permno','date']).reset_index(drop=True).copy()

#Create Month index
mcrsp['month_index'] = (mcrsp['date'] - datetime.datetime(1900,1,1)).dt.days // 31 + 1
mcrsp['Year'] = pd.DatetimeIndex(mcrsp['date']).year
mcrsp['Month'] = pd.DatetimeIndex(mcrsp['date']).month

Stock_lag_MV = mcrsp[['me','Year','Month']].groupby(['Year','Month'])['me'].agg(['sum','count']).reset_index()
Stock_lag_MV['Stock_lag_MV'] = Stock_lag_MV['sum'].shift()
#Stock_lag_MV = Stock_lag_MV.rename(columns={"sum" : "Stock_lag_MV"}).copy()
mcrsp_new = mcrsp.merge(Stock_lag_MV, how='outer', on=['Year','Month'])
mcrsp_new = mcrsp_new.sort_values(by=['permno','date']).reset_index(drop=True).copy()


lag_me = mcrsp_new[['permno','me']].groupby(['permno'])[['me']].shift()
lag_me = pd.concat([mcrsp_new[['date','permno']],lag_me],axis=1)


mcrsp_new = mcrsp_new.merge(lag_me, how='outer', on=['permno','date'])
mcrsp_new = mcrsp_new.rename(columns={'me_y':'lag_me'}).copy()

mcrsp_new['VWeighted_Ret'] = mcrsp_new['ret']*mcrsp_new['lag_me']/mcrsp_new['Stock_lag_MV']
mcrsp_new['EWeighted_Ret'] = mcrsp_new['ret']/mcrsp_new['count']

Monthly_CRSP_Stocks = mcrsp_new[['Year', 'Month', 'Stock_lag_MV', 'VWeighted_Ret', 'EWeighted_Ret']].groupby(
        ['Year','Month'])['VWeighted_Ret','EWeighted_Ret', 'Stock_lag_MV'].sum().reset_index()
Monthly_CRSP_Stocks = Monthly_CRSP_Stocks.rename(columns={"VWeighted_Ret" : "Stock_Vw_Ret","EWeighted_Ret" : "Stock_Ew_Ret"}).copy()
print(Monthly_CRSP_Stocks)

#Q2
FF_mkt = pd.read_csv("FF_mkt.csv")
FF_mkt['date'] = pd.to_numeric(FF_mkt['Unnamed: 0'])
FF_mkt['date'] = pd.to_datetime(FF_mkt['date'], format='%Y%m').dt.strftime('%Y-%m')
FF_mkt['Year'] = pd.DatetimeIndex(FF_mkt['date']).year
FF_mkt['Month'] = pd.DatetimeIndex(FF_mkt['date']).month

#Total 1136 observations, actual FF ex ret
FF_Exc = FF_mkt['Mkt-RF']
A_FF = pd.DataFrame([FF_Exc.mean()*12/100, FF_Exc.std()*(12**0.5)/100, 
            FF_Exc.skew(), (FF_Exc.kurt()-3), FF_Exc.mean()*(12**0.5)/FF_Exc.std()])

est = Monthly_CRSP_Stocks.merge(FF_mkt, how='right', on=['Year','Month'])

#FF's data is in %, estimated FF ex ret
VW = est['Stock_Vw_Ret']*100 - est['RF']
E_FF = pd.DataFrame([VW.mean()*12/100, VW.std()*(12**0.5)/100,
            VW.skew(), (VW.kurt()-3), VW.mean()*(12**0.5)/VW.std()])

Moments = pd.concat([E_FF, A_FF],axis=1)
Moments.columns = ['Estimated FF Market Excess Return','Actual FF Market Excess Return']
Moments.rename(index = {0:'Annualized Mean', 1:'Annualized StandardDeviation', 2:'Annualized Sharpe Ratio', 3:'Skewness', 4:'Excess Kurtosis'}, inplace=True)
print(Moments)

#Q3
correlation = VW.corr(FF_Exc).round(8)
max_difference = (VW - FF_Exc).abs().max().round(8)
print('correlation: ', correlation, '\n')
print('Max Difference: ', max_difference, '\n')
