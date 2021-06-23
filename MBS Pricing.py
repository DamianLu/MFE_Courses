import random
import math
import os
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import ncx2
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')

#Question 1
#(a)
SYt = [0.98,0.94,0.76,0.74,0.95,0.98,0.92,0.98,1.10,1.18,1.22,1.23]
#Tl
#df all 40 years of r including the first one
def tyield(df,T=30,Tl=10,t=1/12):
    rT = pd.DataFrame(np.zeros((df.shape[0],int(T/t))) )
    for i in range(int(T/t)):
        rT.iloc[:,i] = df.iloc[:,i:int(i+Tl/t)].sum(axis=1)
    
    rT = rT*t/Tl
    return rT

def MBS(T=30,yT=10,t=1/12,WAC=0.08,NA=100000,r0=0.078,k=0.6,r_bar=0.08,s=0.12,N=10000,x=0):
    n = int((T+yT)/t)+1
    r1 = pd.DataFrame(np.zeros((N,n))) #start at time 0
    r2 = pd.DataFrame(np.zeros((N,n))) #start at time 0
    Pv = pd.DataFrame(np.zeros((2*N,int(T/t)+1))) #start at time 0
    ct = pd.DataFrame(np.zeros((2*N,int(T/t))))#ith element is the cashflow at time t+1
    # CPRt = pd.DataFrame(np.zeros((2*N,(n-1))))#ith element is the CPRt at time t+1

    Z = pd.DataFrame(np.random.normal(0,1,N*(n-1)).reshape(N,n-1))
    r1.iloc[:,0] = [r0]*N
    r2.iloc[:,0] = [r0]*N
    Pv.iloc[:,0] = [NA]*(2*N)

    for i in range(1,n):
        r1.iloc[:,i] = r1.iloc[:,i-1] + k*(r_bar-r1.iloc[:,i-1])*t + s*np.sqrt(t*r1.iloc[:,i-1])*Z.iloc[:,i-1]
        r2.iloc[:,i] = r2.iloc[:,i-1] + k*(r_bar-r2.iloc[:,i-1])*t + s*np.sqrt(t*r2.iloc[:,i-1])*(-Z.iloc[:,i-1])
    r = pd.concat([r1,r2],ignore_index=True)
    r = r + x

    #Discount rates for every month 1...360
    life = r.iloc[:,1:int(T/t)+1]
    d = np.exp(-np.cumsum(life,axis=1)*t).copy()#start at t1 360 col
    #10 year treasure yield for t-1
    rt = tyield(r,T,10,1/12) #start at t1 360 col
    R = WAC
    mr = R/12
    RI = 0.28 + 0.14*np.arctan(-8.57 + 430*(R-rt)) #360 col

    for i in range(int(T/t)):
        # i = t-1 for time
        BUt = 0.3+0.7*Pv.iloc[:,i]/Pv.iloc[:,0]
        SGt = np.minimum(1,(i+1)/30)
        syt = SYt[(i+1)%12]
        CPRt = RI.iloc[:,i]*BUt*SGt*syt
        ldm = 1/(1-(1+mr)**(-T/t + i))-1
        TPPt = Pv.iloc[:,i]*mr*ldm +(Pv.iloc[:,i]-Pv.iloc[:,i]*mr*ldm)*(1-(1-CPRt)**(1/12))
        Pv.iloc[:,i+1] = Pv.iloc[:,i] - TPPt
        ct.iloc[:,i] = TPPt + Pv.iloc[:,i]*mr
    final = np.multiply(ct,d)
    Price = final.mean(axis=0).sum()
    return Price

MBS()
#(b)
ki = pd.DataFrame(np.zeros((7,2)))
ki.iloc[:,0] = np.arange(0.3,0.9,0.1)
for i in range(ki.shape[0]):
    ki.iloc[i,1] = MBS(T=30,yT=10,t=1/12,WAC=0.08,NA=100000,r0=0.078,k=ki.iloc[i,0],r_bar=0.08,s=0.12,N=10000)

plt.plot(ki.iloc[:,0],ki.iloc[:,1])

#(c)
ri = pd.DataFrame(np.zeros((7,2)))
ri.iloc[:,0] = np.arange(0.03,0.091,0.01)
for i in range(ri.shape[0]):
    ri.iloc[i,1] = MBS(T=30,yT=10,t=1/12,WAC=0.08,NA=100000,r0=0.078,k=0.6,r_bar=ri.iloc[i,0],s=0.12,N=10000)

plt.plot(ri.iloc[:,0],ri.iloc[:,1])

#2
def f(xi):
    Pi = MBS(T=30,yT=10,t=1/12,WAC=0.08,NA=100000,r0=0.078,k=0.6,r_bar=0.08,s=0.12,N=10000,x=xi)
    return (Pi - 102000)**2

res = minimize_scalar(f,bounds = (-0.1,0.1), method='bounded')
QAS = res.x

#3
P0 = 102000
y = 0.0005
P_m = MBS(x=QAS-y)
P_p = MBS(x=QAS+y)

QAS_Dur = (P_m - P_p)/(2*y*P0)
QAS_Con = (P_p+P_m-2*P0)/(2*P0*y**2)

#4
def IO(T=30,yT=10,t=1/12,WAC=0.08,NA=100000,r0=0.078,k=0.6,r_bar=0.08,s=0.12,N=10000,x=0):
    n = int((T+yT)/t)+1
    r1 = pd.DataFrame(np.zeros((N,n))) #start at time 0
    r2 = pd.DataFrame(np.zeros((N,n))) #start at time 0
    Pv = pd.DataFrame(np.zeros((2*N,int(T/t)+1))) #start at time 0
    ct = pd.DataFrame(np.zeros((2*N,int(T/t))))#ith element is the cashflow at time t+1
    # CPRt = pd.DataFrame(np.zeros((2*N,(n-1))))#ith element is the CPRt at time t+1

    Z = pd.DataFrame(np.random.normal(0,1,N*(n-1)).reshape(N,n-1))
    r1.iloc[:,0] = [r0]*N
    r2.iloc[:,0] = [r0]*N
    Pv.iloc[:,0] = [NA]*(2*N)

    for i in range(1,n):
        r1.iloc[:,i] = r1.iloc[:,i-1] + k*(r_bar-r1.iloc[:,i-1])*t + s*np.sqrt(t*r1.iloc[:,i-1])*Z.iloc[:,i-1]
        r2.iloc[:,i] = r2.iloc[:,i-1] + k*(r_bar-r2.iloc[:,i-1])*t + s*np.sqrt(t*r2.iloc[:,i-1])*(-Z.iloc[:,i-1])
    r = pd.concat([r1,r2],ignore_index=True)
    r = r + x

    #Discount rates for every month 1...360
    life = r.iloc[:,1:int(T/t)+1]
    d = np.exp(-np.cumsum(life,axis=1)*t).copy()#start at t1 360 col
    #10 year treasure yield for t-1
    rt = tyield(r,T,10,1/12) #start at t1 360 col
    R = WAC
    mr = R/12
    RI = 0.28 + 0.14*np.arctan(-8.57 + 430*(R-rt)) #360 col

    for i in range(int(T/t)):
        # i = t-1 for time
        BUt = 0.3+0.7*Pv.iloc[:,i]/Pv.iloc[:,0]
        SGt = np.minimum(1,(i+1)/30)
        syt = SYt[(i+1)%12]
        CPRt = RI.iloc[:,i]*BUt*SGt*syt
        ldm = 1/(1-(1+mr)**(-T/t + i))-1
        TPPt = Pv.iloc[:,i]*mr*ldm +(Pv.iloc[:,i]-Pv.iloc[:,i]*mr*ldm)*(1-(1-CPRt)**(1/12))
        Pv.iloc[:,i+1] = Pv.iloc[:,i] - TPPt
        ct.iloc[:,i] = Pv.iloc[:,i]*mr
    final = np.multiply(ct,d)
    Price = final.mean(axis=0).sum()
    return Price

ri_io = ri.copy()
for i in range(ri_io.shape[0]):
    ri_io.iloc[i,1] = IO(T=30,yT=10,t=1/12,WAC=0.08,NA=100000,r0=0.078,k=0.6,r_bar=ri_io.iloc[i,0],s=0.12,N=10000)

def PO(T=30,yT=10,t=1/12,WAC=0.08,NA=100000,r0=0.078,k=0.6,r_bar=0.08,s=0.12,N=10000,x=0):
    n = int((T+yT)/t)+1
    r1 = pd.DataFrame(np.zeros((N,n))) #start at time 0
    r2 = pd.DataFrame(np.zeros((N,n))) #start at time 0
    Pv = pd.DataFrame(np.zeros((2*N,int(T/t)+1))) #start at time 0
    ct = pd.DataFrame(np.zeros((2*N,int(T/t))))#ith element is the cashflow at time t+1
    # CPRt = pd.DataFrame(np.zeros((2*N,(n-1))))#ith element is the CPRt at time t+1

    Z = pd.DataFrame(np.random.normal(0,1,N*(n-1)).reshape(N,n-1))
    r1.iloc[:,0] = [r0]*N
    r2.iloc[:,0] = [r0]*N
    Pv.iloc[:,0] = [NA]*(2*N)

    for i in range(1,n):
        r1.iloc[:,i] = r1.iloc[:,i-1] + k*(r_bar-r1.iloc[:,i-1])*t + s*np.sqrt(t*r1.iloc[:,i-1])*Z.iloc[:,i-1]
        r2.iloc[:,i] = r2.iloc[:,i-1] + k*(r_bar-r2.iloc[:,i-1])*t + s*np.sqrt(t*r2.iloc[:,i-1])*(-Z.iloc[:,i-1])
    r = pd.concat([r1,r2],ignore_index=True)
    r = r + x

    #Discount rates for every month 1...360
    life = r.iloc[:,1:int(T/t)+1]
    d = np.exp(-np.cumsum(life,axis=1)*t).copy()#start at t1 360 col
    #10 year treasure yield for t-1
    rt = tyield(r,T,10,1/12) #start at t1 360 col
    R = WAC
    mr = R/12
    RI = 0.28 + 0.14*np.arctan(-8.57 + 430*(R-rt)) #360 col

    for i in range(int(T/t)):
        # i = t-1 for time
        BUt = 0.3+0.7*Pv.iloc[:,i]/Pv.iloc[:,0]
        SGt = np.minimum(1,(i+1)/30)
        syt = SYt[(i+1)%12]
        CPRt = RI.iloc[:,i]*BUt*SGt*syt
        ldm = 1/(1-(1+mr)**(-T/t + i))-1
        TPPt = Pv.iloc[:,i]*mr*ldm +(Pv.iloc[:,i]-Pv.iloc[:,i]*mr*ldm)*(1-(1-CPRt)**(1/12))
        Pv.iloc[:,i+1] = Pv.iloc[:,i] - TPPt
        ct.iloc[:,i] = TPPt
    final = np.multiply(ct,d)
    Price = final.mean(axis=0).sum()
    return Price

ri_po = ri.copy()
for i in range(ri_po.shape[0]):
    ri_po.iloc[i,1] = PO(T=30,yT=10,t=1/12,WAC=0.08,NA=100000,r0=0.078,k=0.6,r_bar=ri_po.iloc[i,0],s=0.12,N=10000)