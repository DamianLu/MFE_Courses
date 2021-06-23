import os
from numpy.core.fromnumeric import size
from numpy.lib.function_base import _corrcoef_dispatcher
os.getcwd()
import pandas as pd
import numpy as np
import datetime
import wrds
import pandas_datareader
import matplotlib.pyplot as plt
from pandas.tseries.offsets import *
from scipy import stats

data_folder = 'C:/Users/Damian/Desktop/QAM/ps4/'  
id_wrds = 'bobross'  

#CRSP
min_year = 1926
max_year = 2020

min_shrcd = 10
max_shrcd = 11
possible_exchcd = (1, 2, 3)

# conn = wrds.Connection(wrds_username=id_wrds)
# crsp_raw = conn.raw_sql("""
#                       select a.permno, a.permco, a.date, a.ret, a.retx, b.exchcd, b.shrcd, a.shrout, a.prc
#                       from crspq.msf as a
#                       left join crspq.msenames as b
#                       on a.permno=b.permno
#                       and b.namedt<=a.date
#                       and a.date<=b.nameendt
#                       where b.shrcd between """ + str(min_shrcd) + """ and  """ + str(max_shrcd) + """
#                       and a.date between '01/01/1973' and '12/31/2020'
#                       and b.exchcd in """ + str(possible_exchcd) + """
#                       """)
# conn.close()
# crsp_raw
# crsp_raw.to_pickle(data_folder + 'crsp_raw.pkl')
crsp_raw = pd.read_pickle('crsp_raw.pkl')
# Load CRSP Deslisting returns
# conn = wrds.Connection(wrds_username=id_wrds)
# dlret_raw = conn.raw_sql("""
#                         select a.permno, a.dlret, a.dlstdt
#                         from crspq.msedelist as a
#                         left join crspq.msenames as b
#                         on a.permno=b.permno
#                         and b.namedt<=a.dlstdt
#                         and a.dlstdt<=b.nameendt
#                         where b.shrcd between """ + str(min_shrcd) + """ and  """ + str(max_shrcd) + """
#                         and a.dlstdt between '01/01/""" +str(min_year)+ """' and '12/31/""" +str(max_year)+ """'
#                         and b.exchcd in """ + str(possible_exchcd) + """
#                         """) 
# conn.close()
# dlret_raw
# dlret_raw.to_pickle(data_folder + 'dlret_raw.pkl')
dlret_raw = pd.read_pickle('dlret_raw.pkl')
#Return Cleaning
crsp_raw[['permco','permno','shrcd','exchcd']]=crsp_raw[['permco','permno','shrcd','exchcd']].astype(int)
crsp_raw['date'] = pd.to_datetime(crsp_raw['date'], format='%Y-%m-%d',errors='ignore')
crsp_raw['jdate'] = crsp_raw['date'] + MonthEnd(0)
crsp_raw = crsp_raw.sort_values(by=['permno','date']).reset_index(drop=True).copy()
crsp_raw['me'] = crsp_raw['prc'].abs() * crsp_raw['shrout']
#crsp_raw.drop(['prc', 'shrout'], axis=1, inplace=True)

#Delist cleaning
dlret_raw['permno'] = dlret_raw['permno'].astype(int)
dlret_raw['dlstdt'] = pd.to_datetime(dlret_raw['dlstdt'])
dlret_raw['jdate'] = dlret_raw['dlstdt'] + MonthEnd(0)

#Merge
crsp = crsp_raw.merge(dlret_raw,how='left',on=['jdate','permno'])
#Adjust for delisting return
crsp['ret'] = np.where(crsp['ret'].notna() & crsp['dlret'].notna(),
                       (1+crsp['ret'])*(1+crsp['dlret'])-1, crsp['ret'])
crsp['ret'] = np.where(crsp['ret'].isna() & crsp['dlret'].notna(), crsp['dlret'], crsp['ret'])
crsp = crsp[crsp['ret'].notna()].copy()
crsp=crsp.drop(['dlret','dlstdt','prc','shrout'], axis=1)
crsp=crsp.sort_values(by=['jdate','permco','me'])
crsp

#Get ME for each permco
crsp_come = crsp.groupby(['jdate','permco'])['me'].sum().reset_index()
crsp_maxe = crsp.groupby(['jdate','permco'])['me'].max().reset_index()
crsp_a = pd.merge(crsp, crsp_maxe, how='inner', on=['jdate','permco','me'])
crsp_a=crsp_a.drop(['me'], axis=1)
crsp_b = pd.merge(crsp_a, crsp_come, how='inner', on=['jdate','permco'])
#Choose the nearest ME
crsp_b = crsp_b.sort_values(by=['permno','jdate']).drop_duplicates()
crsp_b['year'] = crsp_b['jdate'].dt.year
crsp_b['month'] = crsp_b['jdate'].dt.month
#ME at Dec
me_yend = crsp_b[crsp_b['month']==12]
me_yend = me_yend[['permno','date','jdate','me','year']].rename(columns={'me':'dec_me'})
crsp_b['ffdate'] = crsp_b['jdate'] + MonthEnd(-6)
crsp_b['ffyear'] = crsp_b['ffdate'].dt.year
crsp_b['ffmonth'] = crsp_b['ffdate'].dt.month
crsp_b['1+retx'] = 1+crsp_b['retx']
crsp_b = crsp_b.sort_values(by=['permno','date'])

#Cumulative return
crsp_b['cumretx'] = crsp_b.groupby(['permno','ffyear'])['1+retx'].cumprod()
#lag return
crsp_b['lcumretx'] = crsp_b.groupby(['permno'])['cumretx'].shift(1)
#lag ME
crsp_b['lme'] = crsp_b.groupby(['permno'])['me'].shift(1)
crsp_b['count']=crsp_b.groupby(['permno']).cumcount()
crsp_b['lme']=np.where(crsp_b['count']==0, crsp_b['me']/crsp_b['1+retx'], crsp_b['lme'])
me_b = crsp_b[crsp_b['ffmonth']==1][['permno','ffyear', 'lme']].rename(columns={'lme':'mebase'})
crsp_c = pd.merge(crsp_b, me_b, how='left', on=['permno','ffyear'])
crsp_c['wt'] = np.where(crsp_c['ffmonth']==1, crsp_c['lme'], crsp_c['mebase']*crsp_c['lcumretx'])
me_yend['year'] = me_yend['year'] + 1
me_yend=me_yend[['permno','year','dec_me']]

crsp_cj = crsp_c[crsp_c['month']==6]

crsp_june = pd.merge(crsp_cj, me_yend, how='inner', on=['permno','year'])
crsp_june=crsp_june[['permno','date', 'jdate', 'shrcd','exchcd','ret','me','wt','cumretx','mebase','lme','dec_me']]
crsp_june=crsp_june.sort_values(by=['permno','jdate']).drop_duplicates()

########################################################################################################################
## Compustat
########################################################################################################################
# conn = wrds.Connection(wrds_username=id_wrds)
# cstat = conn.raw_sql("""
#                     select a.gvkey, a.datadate,a.seq, a.ceq, a.pstk, a.at, a.lt, a.mib,
#                     a.txditc, a.itcb, a.txdb,
#                     a.pstkrv, a.pstkl, a.fyear, 
#                     b.year1
#                     from comp.funda as a
#                     left join comp.names as b
#                     on a.gvkey = b.gvkey
#                     where indfmt='INDL'
#                     and datafmt='STD'
#                     and popsrc='D'
#                     and consol='C'
#                     and datadate between '01/01/1973' and '12/31/2020'
#                     """)
# conn.close()
# cstat.to_pickle(data_folder + 'cstat.pkl')
cstat = pd.read_pickle('cstat.pkl')
cstat['gvkey'] = cstat['gvkey'].astype(int)
cstat['datadate'] = pd.to_datetime(cstat['datadate'])
cstat['year'] = cstat['datadate'].dt.year

cstat['she'] = cstat['seq']
cstat['dt'] = cstat['txditc']
cstat['ps'] = cstat['pstkrv']


#Clean Compustat data
(cstat['she'].isna()).sum()
cstat['she'] = np.where(cstat['she'].isna(), cstat['ceq'] + cstat['pstk'], cstat['she'])
cstat['she'] = np.where(cstat['she'].isna(), cstat['at'] - cstat['lt'] - cstat['mib'], cstat['she'])
cstat['she'] = np.where(cstat['she'].isna(), cstat['at'] - cstat['lt'] - cstat['mib'], cstat['she'])
cstat['she'] = np.where(cstat['she'].isna(), cstat['at'] - cstat['lt'], cstat['she'])
# cstat['she']=np.where(cstat['she'].isnull(),0,cstat['ps'])

(cstat['dt'].isna()).sum()
cstat['dt'] = np.where(cstat['dt'].isna(), cstat['itcb'] + cstat['txdb'], cstat['dt'])
cstat['dt'] = np.where(cstat['dt'].isna() & cstat['itcb'].notna(), cstat['itcb'], cstat['dt'])
cstat['dt'] = np.where(cstat['dt'].isna() & cstat['txdb'].notna(), cstat['txdb'], cstat['dt'])
# cstat['dt']=np.where(cstat['dt'].isnull(),0,cstat['ps'])

(cstat['ps'].isna()).sum()
cstat['ps'] = np.where(cstat['ps'].isna(), cstat['pstkl'], cstat['ps'])
cstat['ps'] = np.where(cstat['ps'].isna(), cstat['pstk'], cstat['ps'])
# cstat['ps']=np.where(cstat['ps'].isnull(),0,cstat['ps'])

# conn = wrds.Connection(wrds_username=id_wrds)
# Pension = conn.raw_sql("""
#                         select gvkey, datadate, prba
#                         from comp.aco_pnfnda
#                         where indfmt='INDL'
#                         and datafmt='STD'
#                         and popsrc='D'
#                         and consol='C'
#                         and datadate between '01/01/1973' and '12/31/2020'
#                         """)
# conn.close()
# Pension.to_pickle(data_folder + 'Pension.pkl')
Pension = pd.read_pickle('Pension.pkl')
Pension['datadate'] = pd.to_datetime(Pension['datadate'])
Pension['gvkey'] = Pension['gvkey'].astype(int)
Pension['prba'] = np.where(Pension['prba'].isnull(),0,Pension['prba'])

cstat_pension = cstat.merge(Pension,how='left',on=['gvkey','datadate']) 
cstat_pension['be'] = cstat_pension['she'] - np.where(cstat_pension['ps'].isna(), 0, cstat_pension['ps']) + \
                                             np.where(cstat_pension['dt'].isna(), 0, cstat_pension['dt']) - \
                                             np.where(cstat_pension['prba'].isna(), 0, cstat_pension['prba']) 

cstat_pension=cstat_pension.sort_values(by=['gvkey','datadate'])
cstat_pension['count']=cstat_pension.groupby(['gvkey']).cumcount()

be = cstat_pension[['gvkey','datadate','year','be','count']]

########################################################################################################################
## Link
########################################################################################################################
# conn = wrds.Connection(wrds_username=id_wrds)
# crsp_cstat=conn.raw_sql("""
#                   select gvkey, lpermno as permno, lpermco as permco, linktype, linkprim, liid,
#                   linkdt, linkenddt
#                   from crspq.ccmxpf_linktable
#                   where substr(linktype,1,1)='L'
#                   and (linkprim ='C' or linkprim='P')
#                   """)
# conn.close()
# crsp_cstat.to_pickle(data_folder + 'crsp_cstat.pkl')
crsp_cstat = pd.read_pickle('crsp_cstat.pkl')
crsp_cstat['gvkey'] = crsp_cstat['gvkey'].astype(int)

crsp_cstat['linkdt']=pd.to_datetime(crsp_cstat['linkdt'])
crsp_cstat['linkenddt']=pd.to_datetime(crsp_cstat['linkenddt'])
crsp_cstat['linkenddt']=crsp_cstat['linkenddt'].fillna(pd.to_datetime('today'))

clink=pd.merge(be[['gvkey','datadate','be','count']],crsp_cstat,how='left',on=['gvkey'])
clink['yearend'] = clink['datadate'] + YearEnd(0)
clink['jdate'] = clink['yearend']+MonthEnd(6)

crcs = clink[(clink['jdate']>=clink['linkdt'])&(clink['jdate']<=clink['linkenddt'])]
crcs = crcs[['gvkey','permno','datadate','yearend', 'jdate','be', 'count']]

#Link CRSP and Compustat
final = pd.merge(crsp_june,crcs, how='inner', on=['permno', 'jdate'])
final['beme']=final['be']*1000/final['dec_me']
final.to_pickle(data_folder + 'cleaned.pkl')

########################################################################################################################
## size and book to market ratio
########################################################################################################################
# conn = wrds.Connection(wrds_username=id_wrds)
#select nyse stocks to get bins
nyse=final[(final['exchcd']==1) & (final['beme']>0) & (final['me']>0) & (final['count']>1)]
size_brkpts =  nyse.groupby(['jdate'])['me'].quantile([1/10])\
            .to_frame().reset_index().rename(columns={'me':'sizebrk1'})\
            .drop(columns=['level_1'])

for i in range(2,10):
    current_label = f'brk{i}'
    bp = nyse.groupby(['jdate'])['me'].quantile([i/10]).to_frame().reset_index()\
                       .rename(columns={'me':'size' + current_label}).drop(columns=['level_1'])
    size_brkpts = pd.merge(size_brkpts,bp,how='left', on=['jdate'])

beme_brkpts = nyse.groupby(['jdate'])['beme'].quantile([1/10])\
            .to_frame().reset_index().rename(columns={'beme':'bemebrk1'})\
            .drop(columns=['level_1'])

for i in range(2,10):
    current_label = f'brk{i}'
    bp = nyse.groupby(['jdate'])['beme'].quantile([i/10]).to_frame().reset_index()\
                       .rename(columns={'beme':'beme' + current_label}).drop(columns=['level_1'])
    beme_brkpts = pd.merge(beme_brkpts,bp,how='left', on=['jdate'])

nyse_breaks = pd.merge(size_brkpts,beme_brkpts,how='inner',on=['jdate'])
final = pd.merge(final,nyse_breaks,how='left',on=['jdate'])

def sz_bins(r):
    if r['me'] == np.nan:
        l=''
    elif r['me']<=r['sizebrk1']:
        l=1
    elif r['me']<=r['sizebrk2']:
        l=2
    elif r['me']<=r['sizebrk3']:
        l=3
    elif r['me']<=r['sizebrk4']:
        l=4
    elif r['me']<=r['sizebrk5']:
        l=5
    elif r['me']<=r['sizebrk6']:
        l=6
    elif r['me']<=r['sizebrk7']:
        l=7
    elif r['me']<=r['sizebrk8']:
        l=8
    elif r['me']<=r['sizebrk9']:
        l=9
    else:
        l=10
    return l

final['sizebin'] = np.where((final['beme']>0)&(final['me']>0)&(final['count']>=1) ,final.apply(sz_bins,axis=1),'')

def bm_bins(r):
    if r['beme'] == np.nan:
        l=''
    elif r['beme']<=r['bemebrk1']:
        l=1
    elif r['beme']<=r['bemebrk2']:
        l=2
    elif r['beme']<=r['bemebrk3']:
        l=3
    elif r['beme']<=r['bemebrk4']:
        l=4
    elif r['beme']<=r['bemebrk5']:
        l=5
    elif r['beme']<=r['bemebrk6']:
        l=6
    elif r['beme']<=r['bemebrk7']:
        l=7
    elif r['beme']<=r['bemebrk8']:
        l=8
    elif r['beme']<=r['bemebrk9']:
        l=9
    else:
        l=10
    return l

final['bmbin'] = np.where((final['beme']>0)&(final['me']>0)&(final['count']>=1) ,final.apply(bm_bins,axis=1),'')

final['pbm'] = np.where((final['beme']>0)&(final['me']>0)&(final['count']>=1), 1, 0)
final['valid_port'] = np.where((final['bmbin']!=''),1,0)

june_port = final[['permno','date', 'jdate', 'sizebin','bmbin','pbm','valid_port']]
june_port['ffyear'] = june_port['jdate'].dt.year

crsp_c = crsp_c[['date','permno','shrcd','exchcd','ret','me','wt','cumretx','ffyear','jdate']]
cc = pd.merge(crsp_c,june_port[['permno','ffyear','sizebin','bmbin','pbm','valid_port']], how='left', on=['permno','ffyear'])

portfolios = cc[(cc['wt']>0)& (cc['pbm']==1) & (cc['valid_port']==1)]

########################################################################################################################
## Portfolio Returns
########################################################################################################################
def weighted_avg(g,v,wt):
    d = g[v]
    w = g[wt]
    try:
        return (d*w).sum()/w.sum()
    except ZeroDivisionError:
        return np.nan

size_ret = portfolios.groupby(['jdate','sizebin']).apply(weighted_avg,'ret','wt').to_frame().reset_index().rename(columns={0:'Size_Ret'})
BtM_ret = portfolios.groupby(['jdate','bmbin']).apply(weighted_avg,'ret','wt').to_frame().reset_index().rename(columns={0:'BtM_Ret'})

FF1992 = pd.merge(size_ret,BtM_ret,how='inner',on=['jdate']).sort_values(by=['sizebin','bmbin'])
FF1992.rename(columns={'jdate':'date'},inplace=True)

########################################################################################################################
## Factors
########################################################################################################################
size_bks = nyse.groupby(['jdate'])['me'].median().to_frame().reset_index().rename(columns={'me':'sizebk'})
bm_bks = nyse.groupby(['jdate'])['beme'].describe(percentiles=[0.3, 0.7]).reset_index()
bm_bks = bm_bks[['jdate','30%','70%']]

factor_bks = pd.merge(size_bks,bm_bks,how='inner',on=['jdate'])
final2 = pd.read_pickle('cleaned.pkl')
final2 = pd.merge(final2,factor_bks,how='left',on=['jdate'])

def sz_bins2(r):
    if r['me'] == np.nan:
        l=''
    elif r['me']<=r['sizebk']:
        l='S'
    else:
        l='B'
    return l

def bm_bins2(r):
    if r['beme'] == np.nan:
        l=''
    elif r['beme']<=r['30%']:
        l='L'
    elif r['beme']<=r['70%']:
        l='M'
    else:
        l='H'
    return l

final2['sizebin'] = np.where((final2['beme']>0)&(final2['me']>0)&(final2['count']>=1) ,final2.apply(sz_bins2,axis=1),'')

final2['bmbin'] = np.where((final2['beme']>0)&(final2['me']>0)&(final2['count']>=1) ,final2.apply(bm_bins2,axis=1),'')

final2['pbm'] = np.where((final2['beme']>0)&(final2['me']>0)&(final2['count']>=1), 1, 0)
final2['valid_port'] = np.where((final2['bmbin']!=''),1,0)

june_port2 = final2[['permno','date', 'jdate', 'sizebin','bmbin','pbm','valid_port']]
june_port2['ffyear'] = june_port2['jdate'].dt.year

cc2 = pd.merge(crsp_c,june_port2[['permno','ffyear','sizebin','bmbin','pbm','valid_port']], how='left', on=['permno','ffyear'])

portfolios2 = cc2[(cc2['wt']>0)& (cc2['pbm']==1) & (cc2['valid_port']==1)]

vwret=portfolios2.groupby(['jdate','sizebin','bmbin']).apply(weighted_avg, 'ret','wt').to_frame().reset_index().rename(columns={0: 'vwret'})
vwret['idport']=vwret['sizebin']+vwret['bmbin']

factors=vwret.pivot(index='jdate', columns='idport', values='vwret').reset_index()

factors['H']=(factors['BH']+factors['SH'])/2
factors['L']=(factors['BL']+factors['SL'])/2
factors['HML'] = factors['H']-factors['L']

factors['B']=(factors['BL']+factors['BM']+factors['BH'])/3
factors['S']=(factors['SL']+factors['SM']+factors['SH'])/3
factors['SMB'] = factors['S']-factors['B']
FF1993=factors[['jdate','HML','SMB']].iloc[12:,:]
FF1993.rename(columns={'jdate':'date'},inplace=True)

########################################################################################################################
## Questions
########################################################################################################################
#Question1
PS4_Q1 = pd.merge(FF1992,FF1993,how='inner',on=['date'])

#Question2
pd.set_option('precision', 2)
data2 = pandas_datareader.famafrench.FamaFrenchReader('F-F_Research_Data_Factors',start='1900', end=str(datetime.datetime.now().year+1))
french = data2.read()[0] / 100 # Monthly data
french['Mkt'] = french['Mkt-RF'] + french['RF']
# Book-to-Market Portfolios
data2 = pandas_datareader.famafrench.FamaFrenchReader('Portfolios_Formed_on_BE-ME',start='1900', end=str(datetime.datetime.now().year+1))
data2 = data2.read()[0][['Lo 10', 'Dec 2', 'Dec 3', 'Dec 4', 'Dec 5', 'Dec 6', 'Dec 7', 'Dec 8', 'Dec 9', 'Hi 10']] / 100
data2.columns = 'BM01','BM02','BM03','BM04','BM05','BM06','BM07','BM08','BM09','BM10'
french = pd.merge(french,data2,how='left',on=['Date'])
data2 = pandas_datareader.famafrench.FamaFrenchReader('Portfolios_Formed_on_ME',start='1900', end=str(datetime.datetime.now().year+1))
data2 = data2.read()[0][['Lo 10', 'Dec 2', 'Dec 3', 'Dec 4', 'Dec 5', 'Dec 6', 'Dec 7', 'Dec 8', 'Dec 9', 'Hi 10']] / 100
data2.columns = 'ME01','ME02','ME03','ME04','ME05','ME06','ME07','ME08','ME09','ME10'
french = pd.merge(french,data2,how='left',on=['Date'])

# conn = wrds.Connection(wrds_username=id_wrds)
# FF3 = conn.get_table(library='ff', table='factors_monthly')
# conn.close()
# FF3.to_pickle(data_folder + 'FF3.pkl')
FF3 = pd.read_pickle('FF3.pkl')
FF3 = FF3[['date','mktrf','smb','hml','rf']]
FF3['mkt'] = FF3['mktrf'] + FF3['rf']
FF3['date']=FF3['date']+MonthEnd(0)
FF3

size_tbl=size_ret.pivot(index='jdate', columns='sizebin', values='Size_Ret').reset_index().rename(columns={'jdate':'date'})
size_tbl = pd.merge(size_tbl,FF3[['date','rf']], how='left', on=['date']).iloc[12:,:]
bm_tbl = BtM_ret.pivot(index='jdate', columns='bmbin', values='BtM_Ret').reset_index().rename(columns={'jdate':'date'})
bm_tbl = pd.merge(bm_tbl,FF3[['date','rf']], how='left', on=['date']).iloc[12:,:]

size_test = french.reset_index().iloc[:,16:]
size_test['date'] = FF3['date']
size_test = pd.merge(size_tbl,size_test,how='left',on=['date'])

PS4_Q2 = pd.DataFrame(np.zeros((4,11)))
for i in range(1,12):
    rf = size_tbl['rf']
    if (i==11):
        LS = size_tbl['1'] - size_tbl['10'] - rf
        #Annualize
        Excess_Return = LS.mean()*12
        Standard_Deviation = LS.std()*np.sqrt(12)
        SR = Excess_Return/Standard_Deviation
        SK = LS.skew()
    else:
        bin = f'{i}'
        pf = size_tbl[bin]
        Excess_Return = (pf-rf).mean()*12
        Standard_Deviation = (pf-rf).std()*np.sqrt(12)
        SR = Excess_Return/Standard_Deviation
        SK = (pf-rf).skew()
    PS4_Q2.iloc[:,i-1] = [Excess_Return,Standard_Deviation,SR,SK]

PS4_Q2.columns = ['Decile 1','Decile 2','Decile 3','Decile 4','Decile 5','Decile 6','Decile 7','Decile 8','Decile 9','Decile 10', 'Long Short']
PS4_Q2.rename(index={0:'Excess Return', 1:'Standard Deviatio',2:'Shapre Ratio',3:'Skewness'}, inplace=True)

size_comp = pd.DataFrame(np.zeros((4,11)))
for i in range(12,23):
    rf = size_test['rf']
    if (i==22):
        LS = size_test.iloc[:,12] - size_test.iloc[:,21] - rf
        #Annualize
        Excess_Return = LS.mean()*12
        Standard_Deviation = LS.std()*np.sqrt(12)
        SR = Excess_Return/Standard_Deviation
        SK = LS.skew()
    else:
        pf = size_test.iloc[:,i]
        Excess_Return = (pf-rf).mean()*12
        Standard_Deviation = (pf-rf).std()*np.sqrt(12)
        SR = Excess_Return/Standard_Deviation
        SK = (pf-rf).skew()
    size_comp.iloc[:,i-12] = [Excess_Return,Standard_Deviation,SR,SK]

size_comp.columns = ['Decile 1','Decile 2','Decile 3','Decile 4','Decile 5','Decile 6','Decile 7','Decile 8','Decile 9','Decile 10', 'Long Short']
size_comp.rename(index={0:'Excess Return', 1:'Standard Deviatio',2:'Shapre Ratio',3:'Skewness'}, inplace=True)


PS4_Q2_cor = pd.DataFrame(np.zeros(10)).transpose()
for i in range(1,11):
        my_lable = f'{i}'
        if( i == 10):
            f_lable = 'ME10'
        else:
            f_lable = f'ME0{i}'
        my_bin = size_test[my_lable]
        f_bin = size_test[f_lable]
        rho = np.corrcoef(my_bin,f_bin)[0,1].round(4)
        PS4_Q2_cor[i-1] = rho
PS4_Q2_cor.columns = ['Decile 1','Decile 2','Decile 3','Decile 4','Decile 5','Decile 6','Decile 7','Decile 8','Decile 9','Decile 10']

#Question 3
bm_test = french.reset_index().iloc[:,:16]
bm_test['date'] = FF3['date']
bm_test = pd.merge(bm_tbl,bm_test,how='left',on=['date'])
PS4_Q3 = pd.DataFrame(np.zeros((4,11)))
for i in range(1,12):
    rf = bm_tbl['rf']
    if (i==11):
        LS = bm_tbl['10'] - bm_tbl['1'] - rf
        #Annualize
        Excess_Return = LS.mean()*12
        Standard_Deviation = LS.std()*np.sqrt(12)
        SR = Excess_Return/Standard_Deviation
        SK = LS.skew()
    else:
        bin = f'{i}'
        pf = bm_tbl[bin]
        Excess_Return = (pf-rf).mean()*12
        Standard_Deviation = (pf-rf).std()*np.sqrt(12)
        SR = Excess_Return/Standard_Deviation
        SK = (pf-rf).skew()
    PS4_Q3.iloc[:,i-1] = [Excess_Return,Standard_Deviation,SR,SK]
PS4_Q3.columns = ['Decile 1','Decile 2','Decile 3','Decile 4','Decile 5','Decile 6','Decile 7','Decile 8','Decile 9','Decile 10', 'Long Short']
PS4_Q3.rename(index={0:'Excess Return', 1:'Standard Deviatio',2:'Shapre Ratio',3:'Skewness'}, inplace=True)

bm_comp = pd.DataFrame(np.zeros((4,11)))
for i in range(18,29):
    rf = bm_test['rf']
    if (i==28):
        LS = bm_test.iloc[:,18] - bm_test.iloc[:,27] - rf
        #Annualize
        Excess_Return = LS.mean()*12
        Standard_Deviation = LS.std()*np.sqrt(12)
        SR = Excess_Return/Standard_Deviation
        SK = LS.skew()
    else:
        pf = bm_test.iloc[:,i]
        Excess_Return = (pf-rf).mean()*12
        Standard_Deviation = (pf-rf).std()*np.sqrt(12)
        SR = Excess_Return/Standard_Deviation
        SK = (pf-rf).skew()
    bm_comp.iloc[:,i-18] = [Excess_Return,Standard_Deviation,SR,SK]

bm_comp.columns = ['Decile 1','Decile 2','Decile 3','Decile 4','Decile 5','Decile 6','Decile 7','Decile 8','Decile 9','Decile 10', 'Long Short']
bm_comp.rename(index={0:'Excess Return', 1:'Standard Deviatio',2:'Shapre Ratio',3:'Skewness'}, inplace=True)

PS4_Q3_cor = pd.DataFrame(np.zeros(10)).transpose()
for i in range(1,11):
        my_lable = f'{i}'
        if( i == 10):
            f_lable = 'BM10'
        else:
            f_lable = f'BM0{i}'
        my_bin = bm_test[my_lable]
        f_bin = bm_test[f_lable]
        rho = np.corrcoef(my_bin,f_bin)[0,1].round(4)
        PS4_Q3_cor[i-1] = rho
PS4_Q3_cor.columns = ['Decile 1','Decile 2','Decile 3','Decile 4','Decile 5','Decile 6','Decile 7','Decile 8','Decile 9','Decile 10']

#Question 5
FF_compare = pd.merge(FF3[['date','smb','hml','rf']],FF1993, how='inner', on=['date'])
SMB_corr = stats.pearsonr(FF_compare['smb'], FF_compare['SMB'])[0]
HML_corr = stats.pearsonr(FF_compare['hml'], FF_compare['HML'])[0]

SMB_summary = pd.DataFrame(np.zeros((4)))
HML_summary = pd.DataFrame(np.zeros((4)))

SMB_summary.iloc[0,0] = (FF_compare['SMB'] - FF_compare['rf']).mean()*12
SMB_summary.iloc[1,0] = (FF_compare['SMB'] - FF_compare['rf']).std()*np.sqrt(12)
SMB_summary.iloc[2,0] = SMB_summary.iloc[0,0]/SMB_summary.iloc[1,0]
SMB_summary.iloc[3,0] = (FF_compare['SMB'] - FF_compare['rf']).skew()
SMB_summary.rename(index={0:'Excess Return', 1:'Standard Deviatio',2:'Shapre Ratio',3:'Skewness'}, inplace=True)

HML_summary.iloc[0,0] = (FF_compare['HML'] - FF_compare['rf']).mean()*12
HML_summary.iloc[1,0] = (FF_compare['HML'] - FF_compare['rf']).std()*np.sqrt(12)
HML_summary.iloc[2,0] = SMB_summary.iloc[0,0]/SMB_summary.iloc[1,0]
HML_summary.iloc[3,0] = (FF_compare['HMLD'] - FF_compare['rf']).skew()
HML_summary.rename(index={0:'Excess Return', 1:'Standard Deviatio',2:'Shapre Ratio',3:'Skewness'}, inplace=True)

FF_compare.set_index('date',inplace=True)
plt.figure()
FF_compare[['smb','SMB']].plot()
plt.figure()
FF_compare[['hml','HML']].plot()
