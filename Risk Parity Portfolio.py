import pandas as pd
import numpy as np
import os
import datetime
import time
import matplotlib.pyplot as plt
import wrds
from pandas.tseries.offsets import *


os.chdir("/Users/Damian/Desktop/QAM/ps2")
mcrsp_raw = pd.read_csv("BondData.csv")
rates = pd.read_csv("RiskFreeRates.csv")
Monthly_CRSP_Stocks = pd.read_pickle("Monthly_CRSP_Stocks.pkl")

#Question 1
#Clean data
mcrsp_raw['KYCRSPID'] = mcrsp_raw['KYCRSPID'].astype(str)
mcrsp_raw['MCALDT'] = pd.to_datetime(mcrsp_raw['MCALDT'], format='%Y-%m-%d',errors='ignore')
mcrsp_raw = mcrsp_raw.sort_values(by=['KYCRSPID','MCALDT']).reset_index(drop=True).copy()
mcrsp_raw = mcrsp_raw[(mcrsp_raw['TMRETNUA'].notna() & mcrsp_raw['TMTOTOUT'].notna())].copy()\

#Create Month Index
mcrsp_raw['month_index'] = (mcrsp_raw['MCALDT'] - datetime.datetime(1900,1,1)).dt.days // 31 + 1 
mcrsp_raw['Year'] = pd.DatetimeIndex(mcrsp_raw['MCALDT']).year
mcrsp_raw['Month'] = pd.DatetimeIndex(mcrsp_raw['MCALDT']).month

#Create lagged total market value
Bond_Lag_MV = mcrsp_raw[['TMTOTOUT','Year','Month']].groupby(['Year','Month'])['TMTOTOUT'].agg(['sum','count']).reset_index()
Bond_Lag_MV['Bond_lag_MV'] = Bond_Lag_MV['sum'].shift()
Bond_Lag_MV['lag_count'] = Bond_Lag_MV['count'].shift()
Bond_new = mcrsp_raw.merge(Bond_Lag_MV, how='outer', on=['Year','Month'])
Bond_new = Bond_new.sort_values(by=['KYCRSPID','MCALDT']).reset_index(drop=True).copy()

#Lagged amount outstanding for each bond
Lag_MS = Bond_new[['KYCRSPID','TMTOTOUT']].groupby(['KYCRSPID'])[['TMTOTOUT']].shift()
Lag_MS = pd.concat([Bond_new[['KYCRSPID','MCALDT']],Lag_MS],axis=1)

Bond_new = Bond_new.merge(Lag_MS, how='outer', on=['KYCRSPID','MCALDT'])
Bond_new = Bond_new.rename(columns={'TMTOTOUT_y':'Lag_MS'}).copy()

Bond_new['Bond_Vw_Ret'] = Bond_new['TMRETNUA']*Bond_new['Lag_MS']/Bond_new['Bond_lag_MV']
Bond_new['Bond_Ew_Ret'] = Bond_new['TMRETNUA']/Bond_new['lag_count']

Bond_Data = Bond_new[['Year','Month','Bond_lag_MV','Bond_Vw_Ret','Bond_Ew_Ret']].groupby(
    ['Year','Month'])['Bond_lag_MV','Bond_Ew_Ret','Bond_Vw_Ret'].sum().reset_index()

print(Bond_Data.head())

#Question 2
#Change the date data type
rates['caldt'] = pd.to_datetime(rates['caldt'], format='%Y%m%d',errors='ignore')
rates['Year'] = pd.DatetimeIndex(rates['caldt']).year
rates['Month'] = pd.DatetimeIndex(rates['caldt']).month

Assets = Bond_Data[['Year','Month','Bond_lag_MV','Bond_Vw_Ret']].copy()
Stocks_Data = Monthly_CRSP_Stocks[['Year','Month','Stock_lag_MV','Stock_Vw_Ret']].copy()
Assets = Assets.merge(Stocks_Data, how='outer', on=['Year','Month'])
Assets = Assets.merge(rates,how='outer', on=['Year','Month'])

Assets['Stock_Excess_Vw_Ret'] = Assets['Stock_Vw_Ret'] - Assets['t30ret']
Assets['Bond_Excess_Vw_Ret'] = Assets['Bond_Vw_Ret'] - Assets['t30ret']

PS2_Q2 = Assets[['Year','Month','Stock_lag_MV','Stock_Excess_Vw_Ret','Bond_lag_MV','Bond_Excess_Vw_Ret']]
Monthly_CRSP_Universe = PS2_Q2

print(Monthly_CRSP_Universe.head())



#Question 3
#60-40
PS2_Q3 = PS2_Q2.copy()
PS2_Q3['Excess_60_40_Ret'] = 0.6*PS2_Q3['Stock_Excess_Vw_Ret'] + 0.4*PS2_Q3['Bond_Excess_Vw_Ret']

#Excess Vw Ret
#Equity Weights
PS2_Q3['Wm'] = PS2_Q3['Stock_lag_MV']/(PS2_Q3['Stock_lag_MV'] + PS2_Q3['Bond_lag_MV'])
PS2_Q3['Wb'] = PS2_Q3['Bond_lag_MV']/(PS2_Q3['Stock_lag_MV'] + PS2_Q3['Bond_lag_MV'])
PS2_Q3['Excess_Vw_Ret'] = PS2_Q3['Wm']*PS2_Q3['Stock_Excess_Vw_Ret'] + PS2_Q3['Wb']*PS2_Q3['Bond_Excess_Vw_Ret']
#Remove the first 01/1926 row 
PS2_Q3 = PS2_Q3.iloc[1:,:].copy()

#Unlevered K
PS2_Q3['mkt_vol'] = PS2_Q3['Stock_Excess_Vw_Ret'].rolling(36).std()
PS2_Q3['Stock_inverse_sigma_hat'] = 1/(PS2_Q3['mkt_vol'].shift())
PS2_Q3['bond_vol'] = PS2_Q3['Bond_Excess_Vw_Ret'].rolling(36).std()
PS2_Q3['Bond_inverse_sigma_hat'] = 1/(PS2_Q3['bond_vol'].shift())
PS2_Q3['Unlevered_k'] = 1/(PS2_Q3['Stock_inverse_sigma_hat']+PS2_Q3['Bond_inverse_sigma_hat'])
PS2_Q3['Un_mkt'] = PS2_Q3['Stock_inverse_sigma_hat']/(PS2_Q3['Stock_inverse_sigma_hat']+PS2_Q3['Bond_inverse_sigma_hat'])
PS2_Q3['Un_bond'] = PS2_Q3['Bond_inverse_sigma_hat']/(PS2_Q3['Stock_inverse_sigma_hat']+PS2_Q3['Bond_inverse_sigma_hat'])
PS2_Q3['Excess_Unlevered_RP_Ret'] = PS2_Q3['Un_mkt']*PS2_Q3['Stock_Excess_Vw_Ret'] + PS2_Q3['Un_bond']*PS2_Q3['Bond_Excess_Vw_Ret']

#Levered
target_vol = PS2_Q3['Excess_Vw_Ret'].std()
unit = PS2_Q3['Stock_inverse_sigma_hat']*PS2_Q3['Stock_Excess_Vw_Ret'] + PS2_Q3['Bond_inverse_sigma_hat']*PS2_Q3['Bond_Excess_Vw_Ret']
vol_unit = unit.std()
k = target_vol/vol_unit

PS2_Q3['Levered_k'] = np.repeat(k,PS2_Q3.shape[0])
PS2_Q3['Excess_Levered_RP_Ret'] = k*unit

Port_Rets = PS2_Q3[['Year','Month','Stock_Excess_Vw_Ret','Bond_Excess_Vw_Ret','Excess_Vw_Ret','Excess_60_40_Ret','Stock_inverse_sigma_hat','Bond_inverse_sigma_hat','Unlevered_k','Excess_Unlevered_RP_Ret','Levered_k','Excess_Levered_RP_Ret']]

print(Port_Rets.tail())

#Question4
PS2_Q4 = Port_Rets.iloc[47:,2:]
PS2_Q4.drop(['Unlevered_k','Levered_k','Stock_inverse_sigma_hat','Bond_inverse_sigma_hat'],axis=1, inplace=True)

Annualized_Mean = PS2_Q4.mean(axis=0)*12
t_stat = np.sqrt(PS2_Q4.shape[0])*PS2_Q4.mean(axis=0)/PS2_Q4.std(axis=0)
Annualized_Standard_Deviation = PS2_Q4.std(axis=0)*np.sqrt(12)
Annualized_Sharpe_Ratio = Annualized_Mean/Annualized_Standard_Deviation
Skewness = PS2_Q4.skew(axis=0)
Excess_Kurtosis = PS2_Q4.kurt(axis=0)

Table = pd.concat([Annualized_Mean,t_stat,Annualized_Standard_Deviation,Annualized_Sharpe_Ratio,Skewness,Excess_Kurtosis],axis=1)
Table.columns = ['Annualized_Mean','t_stat','Annualized_Standard_Deviation','Annualized_Sharpe_Ratio','Skewness','Excess_Kurtosis']
print(Table)