import os
os.getcwd()
import pandas as pd
import numpy as np
import datetime
import wrds
import pandas_datareader
from pandas.tseries.offsets import *

data_folder = 'C:/Users/Damian/Desktop/QAM/ps4/'  
id_wrds = 'bobross'  

########################################################################################################################
## CRSP returns (data from wrds)
########################################################################################################################
conn = wrds.Connection(wrds_username=id_wrds)
crsp_raw = conn.raw_sql("""
                      select a.permno, a.permco, a.date, b.exchcd, b.siccd, b.naics,
                      a.ret, a.retx, a.shrout, a.prc
                      from crspq.msf as a
                      left join crspq.msenames as b
                      on a.permno=b.permno
                      and b.namedt<=a.date
                      and a.date<=b.nameendt
                      where b.shrcd in (10,11)
                      and b.exchcd in (1,2,3)
                      """)
conn.close()
crsp_raw

# Load CRSP Deslisting returns
conn = wrds.Connection(wrds_username=id_wrds)
dlret_raw = conn.raw_sql("""
                        select a.permno, a.permco, a.dlret, a.dlretx, a.dlstdt, 
                        b.exchcd as dlexchcd, b.siccd as dlsiccd, b.naics as dlnaics
                        from crspq.msedelist as a
                        left join crspq.msenames as b
                        on a.permno=b.permno
                        and b.namedt<=a.dlstdt
                        and a.dlstdt<=b.nameendt
                        where b.shrcd in (10,11)
                        and b.exchcd in (1,2,3)
                        """) 
conn.close()
dlret_raw

########################################################################################################################
## Compustat (data from wrds)
########################################################################################################################
conn = wrds.Connection(wrds_username=id_wrds)
cstat = conn.raw_sql("""
                    select a.gvkey, a.datadate, a.at, a.pstkl, a.txditc, a.fyear, a.ceq, a.lt, 
                    a.mib, a.itcb, a.txdb, a.pstkrv, a.seq, a.pstk, b.sic, b.year1, b.naics
                    from comp.funda as a
                    left join comp.names as b
                    on a.gvkey = b.gvkey
                    where indfmt='INDL'
                    and datafmt='STD'
                    and popsrc='D'
                    and consol='C'
                    """)
conn.close()
cstat

# Pension data
conn = wrds.Connection(wrds_username=id_wrds)
Pension = conn.raw_sql("""
                        select gvkey, datadate, prba
                        from comp.aco_pnfnda
                        where indfmt='INDL'
                        and datafmt='STD'
                        and popsrc='D'
                        and consol='C'
                        """)
conn.close()
Pension

########################################################################################################################
## CRSP-Compustat link table (data from wrds)
########################################################################################################################
conn = wrds.Connection(wrds_username=id_wrds)
crsp_cstat=conn.raw_sql("""
                  select gvkey, lpermno as permno, lpermco as permco, linktype, linkprim, liid,
                  linkdt, linkenddt
                  from crspq.ccmxpf_linktable
                  where substr(linktype,1,1)='L'
                  and (linkprim ='C' or linkprim='P')
                  """)
conn.close()
crsp_cstat

# Fama and French 3 Factors
pd.set_option('precision', 2)
data2 = pandas_datareader.famafrench.FamaFrenchReader('F-F_Research_Data_Factors',start='1900', end=str(datetime.datetime.now().year+1))
french = data2.read()[0] / 100 # Monthly data
french['Mkt'] = french['Mkt-RF'] + french['RF']

# Book-to-Market Portfolios
data2 = pandas_datareader.famafrench.FamaFrenchReader('Portfolios_Formed_on_BE-ME',start='1900', end=str(datetime.datetime.now().year+1))
data2 = data2.read()[0][['Lo 10', 'Dec 2', 'Dec 3', 'Dec 4', 'Dec 5', 'Dec 6', 'Dec 7', 'Dec 8', 'Dec 9', 'Hi 10']] / 100
data2.columns = 'BM01','BM02','BM03','BM04','BM05','BM06','BM07','BM08','BM09','BM10'
french = pd.merge(french,data2,how='left',on=['Date'])

# Size Portfolios
data2 = pandas_datareader.famafrench.FamaFrenchReader('Portfolios_Formed_on_ME',start='1900', end=str(datetime.datetime.now().year+1))
data2 = data2.read()[0][['Lo 10', 'Dec 2', 'Dec 3', 'Dec 4', 'Dec 5', 'Dec 6', 'Dec 7', 'Dec 8', 'Dec 9', 'Hi 10']] / 100
data2.columns = 'ME01','ME02','ME03','ME04','ME05','ME06','ME07','ME08','ME09','ME10'
french = pd.merge(french,data2,how='left',on=['Date'])

# 25 Book-to-Market and Size Portfolios
data2 = pandas_datareader.famafrench.FamaFrenchReader('25_Portfolios_5x5',start='1900', end=str(datetime.datetime.now().year+1))
data2 = data2.read()[0].rename(columns={"SMALL LoBM":"ME1 BM1","SMALL HiBM":"ME1 BM5","BIG LoBM":"ME5 BM1","BIG HiBM":"ME5 BM5"}) / 100
french = pd.merge(french,data2,how='left',on=['Date'])

# Changing date format and save
french = french.reset_index().rename(columns={"Date":"date"})
french['date'] = pd.DataFrame(french[['date']].values.astype('datetime64[ns]')) + MonthEnd(0)
print('Data from Ken French Website:')
print(french.columns)
french

# Here is an alternative way using wrds package (FF3 factors only...)
conn = wrds.Connection(wrds_username=id_wrds)
FF3 = conn.get_table(library='ff', table='factors_monthly')
conn.close()
FF3 = FF3[['date','mktrf','smb','hml','rf']]
FF3['mkt'] = FF3['mktrf'] + FF3['rf']
FF3['date']=FF3['date']+MonthEnd(0)
FF3
