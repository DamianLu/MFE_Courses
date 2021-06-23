import random
import math
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import ncx2
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')

#Use antithetic variates and generate short term rate paths
N = 50000
T = 0.5
t = 1/252
n = int(T/t)
r1 = pd.DataFrame(np.zeros((50000,n)))
r2 = pd.DataFrame(np.zeros((50000,n)))

r0 = 0.05
s = 0.10
k = 0.82
r_bar = 0.05

Z = pd.DataFrame(np.random.normal(0,1,N*(n-1)).reshape(N,n-1))
r1.iloc[:,0] = [r0]*N
r2.iloc[:,0] = [r0]*N

for i in range(1,n):
    r1.iloc[:,i] = r1.iloc[:,i-1] + k*(r_bar-r1.iloc[:,i-1])*t + s*np.sqrt(t)*Z.iloc[:,i-1]
    r2.iloc[:,i] = r2.iloc[:,i-1] + k*(r_bar-r2.iloc[:,i-1])*t + s*np.sqrt(t)*(-Z.iloc[:,i-1])

r = pd.concat([r1,r2],ignore_index=True)

#Question 1
#a
def v_ZBP(r0,s,k,r_bar,FV,S,N = 50000,t = 1/252):
    #Use antithetic variates and generate short term rate paths
    n = int(S/t)
    r1 = pd.DataFrame(np.zeros((N,n+1)))
    r2 = pd.DataFrame(np.zeros((N,n+1)))

    Z = pd.DataFrame(np.random.normal(0,1,N*n).reshape(N,n))
    r1.iloc[:,0] = [r0]*N
    r2.iloc[:,0] = [r0]*N

    for i in range(1,n+1):
        r1.iloc[:,i] = r1.iloc[:,i-1] + k*(r_bar-r1.iloc[:,i-1])*t + s*np.sqrt(t)*Z.iloc[:,i-1]
        r2.iloc[:,i] = r2.iloc[:,i-1] + k*(r_bar-r2.iloc[:,i-1])*t + s*np.sqrt(t)*(-Z.iloc[:,i-1])
    r = pd.concat([r1,r2],ignore_index=True)

    R = -r.sum(axis=1)*t
    BP = FV*np.exp(R).mean()
    return BP

r0 = 0.05
s = 0.10
k = 0.82
r_bar = 0.05
FV = 1000
S = 0.5
v_ZBP(r0,s,k,r_bar,FV,S,N,t)

#b
def v_CBP(r0,s,k,r_bar,c,S,Ti,N = 50000,t = 1/252):
    P = np.zeros(len(Ti))
    for i in range(len(Ti)):
        P[i] = v_ZBP(r0,s,k,r_bar,c[i],Ti[i],N,t = 1/252)
    BP = P.sum()
    return BP

c = np.append([30]*7,1030)
Ti = np.array([0.5,1,1.5,2,2.5,3,3.5,4])
S = 4
v_CBP(r0,s,k,r_bar,c,S,Ti,N,t)

#Faster Alternative
def v_CBP2(r0,s,k,r_bar,c,S,Ti,N = 50000,t = 1/252):
    n = int(S/t)
    r1 = pd.DataFrame(np.zeros((N,n+1)))
    r2 = pd.DataFrame(np.zeros((N,n+1)))

    Z = pd.DataFrame(np.random.normal(0,1,N*n).reshape(N,n))
    r1.iloc[:,0] = [r0]*N
    r2.iloc[:,0] = [r0]*N

    for i in range(1,n+1):
        r1.iloc[:,i] = r1.iloc[:,i-1] + k*(r_bar-r1.iloc[:,i-1])*t + s*np.sqrt(t)*Z.iloc[:,i-1]
        r2.iloc[:,i] = r2.iloc[:,i-1] + k*(r_bar-r2.iloc[:,i-1])*t + s*np.sqrt(t)*(-Z.iloc[:,i-1])
    r = pd.concat([r1,r2],ignore_index=True)

    P = np.zeros(len(Ti))
    for i in range(len(Ti)):
        ni = int(Ti[i]/t)
        ri = r.iloc[:,:(ni+1)]
        Ri = -ri.sum(axis=1)*t
        P[i] = c[i]*(np.exp(Ri).mean())
    BP = P.sum()
    return BP
v_CBP2(r0,s,k,r_bar,c,S,Ti,N,t)

#c
def BOP(r0,s,k,r_bar,K,FV,T,S,N = 50000, M = 100, t = 1/252):
    #Steps up to option expiration
    n = int(T/t)+1
    r1 = pd.DataFrame(np.zeros((N,n)))
    r2 = pd.DataFrame(np.zeros((N,n)))

    Z = pd.DataFrame(np.random.normal(0,1,N*(n-1)).reshape(N,n-1))
    r1.iloc[:,0] = [r0]*N
    r2.iloc[:,0] = [r0]*N

    for i in range(1,n):
        r1.iloc[:,i] = r1.iloc[:,i-1] + k*(r_bar-r1.iloc[:,i-1])*t + s*np.sqrt(t)*Z.iloc[:,i-1]
        r2.iloc[:,i] = r2.iloc[:,i-1] + k*(r_bar-r2.iloc[:,i-1])*t + s*np.sqrt(t)*(-Z.iloc[:,i-1])
    r = pd.concat([r1,r2],ignore_index=True)
    rn = r.iloc[:,n-1]

    R = -r.sum(axis=1)*t
    On = np.zeros(len(rn))
    for i in range(len(rn)):
        #At each node at P from M paths
        Pn = v_ZBP(rn[i],s,k,r_bar,FV,S-T,M,t)
        On = np.exp(R[i])*np.maximum(Pn-K,0)
    P = On.mean()
    return P

r0 = 0.05
s = 0.10
k = 0.82
r_bar = 0.05
t = 1/252
FV = 1000
S = 0.5
T = 0.25
BOP(r0,s,k,r_bar,980,1000,0.25,0.5,1000,100,t)


#d
T = 0.25
S = 4
K = 980
sig = (s/k)*(1 - np.exp(-k*(Ti-T)))*np.sqrt((1 - np.exp(-2*k*T))/(2*k))

def P(r0,s,k,r_bar,Ti,N = 50000,t = 1/252):
    n = int(S/t)
    r1 = pd.DataFrame(np.zeros((N,n+1)))
    r2 = pd.DataFrame(np.zeros((N,n+1)))

    Z = pd.DataFrame(np.random.normal(0,1,N*n).reshape(N,n))
    r1.iloc[:,0] = [r0]*N
    r2.iloc[:,0] = [r0]*N

    for i in range(1,n+1):
        r1.iloc[:,i] = r1.iloc[:,i-1] + k*(r_bar-r1.iloc[:,i-1])*t + s*np.sqrt(t)*Z.iloc[:,i-1]
        r2.iloc[:,i] = r2.iloc[:,i-1] + k*(r_bar-r2.iloc[:,i-1])*t + s*np.sqrt(t)*(-Z.iloc[:,i-1])
    r = pd.concat([r1,r2],ignore_index=True)

    P = np.zeros(len(Ti))
    for i in range(len(Ti)):
        ni = int(Ti[i]/t)
        ri = r.iloc[:,:(ni+1)]
        Ri = -ri.sum(axis=1)*t
        P[i] = (np.exp(Ri).mean())
    return P

Pi = P(r0,s,k,r_bar,Ti)

#r* optimizer
def f(x):
    Pi = P(x,s,k,r_bar,Ti)
    return ((Pi*c).sum() - K)**2

res = minimize_scalar(f,bounds = (0,1), method='bounded')

r_star = res.x

Ki = P(r_star,s,k,r_bar,Ti)

def CBOP2(r0,s,k,r_bar,K,c,T,Ti,S,N,M,t):
    P = np.zeros(len(Ti))
    On = np.zeros(N)
    for i in range(len(P)):
        P[i] = BOP(r0,s,k,r_bar,K,c[i],T,Ti[i],N,M,t)
    P.sum()

def CBOP(r0,s,k,r_bar,K,c,T,Ti,S,N,M,t):
    #Steps up to option expiration
    n = int(T/t)+1
    r1 = pd.DataFrame(np.zeros((N,n)))
    r2 = pd.DataFrame(np.zeros((N,n)))

    Z = pd.DataFrame(np.random.normal(0,1,N*(n-1)).reshape(N,n-1))
    r1.iloc[:,0] = [r0]*N
    r2.iloc[:,0] = [r0]*N

    for i in range(1,n):
        r1.iloc[:,i] = r1.iloc[:,i-1] + k*(r_bar-r1.iloc[:,i-1])*t + s*np.sqrt(t)*Z.iloc[:,i-1]
        r2.iloc[:,i] = r2.iloc[:,i-1] + k*(r_bar-r2.iloc[:,i-1])*t + s*np.sqrt(t)*(-Z.iloc[:,i-1])
    r = pd.concat([r1,r2],ignore_index=True)
    rn = r.iloc[:,n-1]

    R = -r.sum(axis=1)*t
    On = np.zeros(len(rn))
    for i in range(len(rn)):
        #At each node at P from M paths
        Pn = v_CBP2(rn[i],s,k,r_bar,c,S-T,Ti-T,M,t)
        On = np.exp(R[i])*np.maximum(Pn-K,0)
    P = On.mean()
    return P   

CBOP2(r0,s,k,r_bar,980,c,0.25,Ti,4,1000,100,t)

#e Close Form solution
PT = P(r0,s,k,r_bar,[T])[0]

di_p = (1/sig)*np.log(Pi/(Ki*PT)) + sig/2 
di_m = (1/sig)*np.log(Pi/(Ki*PT)) - sig/2 

CBOP_explicit = (c*(Pi*norm.cdf(di_p) - Ki*PT*norm.cdf(di_m))).sum()
CBOP_explicit

#2
r0 = 0.05
s = 0.12
k = 0.92
r_bar = 0.055
FV=1000
K = 980
T = 0.5
S = 1
t = 1/252
def c_ZCB(r0,s,k,r_bar,FV,S,N = 50000,t = 1/252):
    n = int(S/t)+1
    r1 = pd.DataFrame(np.zeros((N,n)))
    r2 = pd.DataFrame(np.zeros((N,n)))

    Z = pd.DataFrame(np.random.normal(0,1,N*(n-1)).reshape(N,n-1))
    r1.iloc[:,0] = [r0]*N
    r2.iloc[:,0] = [r0]*N

    for i in range(1,n):
        r1.iloc[:,i] = r1.iloc[:,i-1] + k*(r_bar-r1.iloc[:,i-1])*t + s*np.sqrt(t*r1.iloc[:,i-1])*Z.iloc[:,i-1]
        r2.iloc[:,i] = r2.iloc[:,i-1] + k*(r_bar-r2.iloc[:,i-1])*t + s*np.sqrt(t*r2.iloc[:,i-1])*(-Z.iloc[:,i-1])
    r = pd.concat([r1,r2],ignore_index=True)

    R = -(r.sum(axis=1)*t)
    BP = FV*np.exp(R).mean()
    return BP

print('Zero Coupoun Bond: ', v_ZBP(r0,s,k,r_bar,FV,S,N,t))

def c_BOP(r0,s,k,r_bar,K,FV,T,S,N, M, t = 1/252):
    #Steps up to option expiration
    n = int(T/t)+1
    r1 = pd.DataFrame(np.zeros((N,n)))
    r2 = pd.DataFrame(np.zeros((N,n)))

    Z = pd.DataFrame(np.random.normal(0,1,N*(n-1)).reshape(N,n-1))
    r1.iloc[:,0] = [r0]*N
    r2.iloc[:,0] = [r0]*N

    for i in range(1,n):
        r1.iloc[:,i] = r1.iloc[:,i-1] + k*(r_bar-r1.iloc[:,i-1])*t + s*np.sqrt(t*r1.iloc[:,i-1])*Z.iloc[:,i-1]
        r2.iloc[:,i] = r2.iloc[:,i-1] + k*(r_bar-r2.iloc[:,i-1])*t + s*np.sqrt(t*r2.iloc[:,i-1])*(-Z.iloc[:,i-1])
    r = pd.concat([r1,r2],ignore_index=True)
    rn = r.iloc[:,n-1]

    R = -r.sum(axis=1)*t
    On = np.zeros(len(rn))
    for i in range(len(rn)):
        #At each node at P from M paths
        Pn = v_ZBP(rn[i],s,k,r_bar,FV,S-T,M,t)
        On = np.exp(R[i])*np.maximum(Pn-K,0)
    P = On.mean()
    return P
c_BOP(r0,s,k,r_bar,K,FV,T,S,1000,100,t)

#b
#Bond prices at T 
def IFD(r0,s,k,r_bar,K,FV,T,S,N, M, t = 1/252):
    n = int(S/t)+1
    g = int(T/t)+1
    r1 = pd.DataFrame(np.zeros((N,n)))
    r2 = pd.DataFrame(np.zeros((N,n)))

    Z = pd.DataFrame(np.random.normal(0,1,N*(n-1)).reshape(N,n-1))
    r1.iloc[:,0] = [r0]*N
    r2.iloc[:,0] = [r0]*N

    for i in range(1,n):
        r1.iloc[:,i] = r1.iloc[:,i-1] + k*(r_bar-r1.iloc[:,i-1])*t + s*np.sqrt(t*r1.iloc[:,i-1])*Z.iloc[:,i-1]
        r2.iloc[:,i] = r2.iloc[:,i-1] + k*(r_bar-r2.iloc[:,i-1])*t + s*np.sqrt(t*r2.iloc[:,i-1])*(-Z.iloc[:,i-1])
    r = pd.concat([r1,r2],ignore_index=True)

    #Path after T
    r_T = r.iloc[:,g:n]
    R = -(r_T.sum(axis=1)*t)
    #Bond price at option expiration
    BP = FV*np.exp(R)

    #Compute CT
    CT = np.maximum(BP-K,0)
    CT[0] = r.iloc[0,g-1] - r.iloc[1,g-1]
    CT[len(BP)-1] = 0

    #Get Pu, Pm, Pd and A
    v = k*(r_bar-1)
    Pu = -0.5*t*((s**2)/0.0001 + v/0.01)
    Pm = 1 + t*(s**2)/0.0001 + r0*t 
    Pd = -0.5*t*((s**2)/0.0001 - v/0.01)
    A = pd.DataFrame(np.zeros((len(CT),len(CT))))
    #How many paths
    l = A.shape[1]
    A.iloc[0,0:2] = (1,-1)
    A.iloc[-1,-2:] = (1,-1)
    for i in range(l-2):
        zero_before = np.zeros(i)
        zero_after = np.zeros(l-i-3)
        row = np.concatenate((zero_before,[Pu,Pm,Pd],zero_after))
        A.iloc[i+1,:] = row
    Ainv = np.linalg.inv(A)

    F = pd.DataFrame(np.zeros((l,g)))
    F.iloc[:,g-1] = CT
    for i in np.arange(g-2,-1,-1):
        r_t = r.iloc[:,i:n]
        R_t = -(r_t.sum(axis=1)*t)
        #Bond price at option expiration
        BP_t = FV*np.exp(R_t)
        B = F.iloc[:,i+1]
        B[0] = r.iloc[0,i] - r.iloc[1,i]
        B[l-1] = 0
        EV = np.maximum(BP_t-K,0)
        F.iloc[:,i] = np.maximum(Ainv.dot(B),EV)

    F0 = F.iloc[:,0]/10
    return F0

IFD(r0,s,k,r_bar,K,FV,T,S,1000,100,t).mean()

#c
r0 = 0.05
s = 0.12
k = 0.92
r_bar = 0.055
FV=1000
K = 0.98
T = 0.5
S = 1
t = 1/252

h1 = np.sqrt(k**2 + 2*s**2)
h2 = (k+h1)/2
h3 = 2*k*r_bar/(s**2)

def A(t,T):
    return ((h1*np.exp(h2*(T-t)))/(h2*(np.exp(h1*(T-t))-1)+h1))**h3

def B(t,T):
    return (np.exp(h1*(T-t))-1)/(h2*(np.exp(h1*(T-t))-1)+h1)

def P(t,T,rt):
    return A(t,T)*np.exp(-B(t,T)*rt)

theta = np.sqrt(k**2 + 2*s**2)
phi = 2*theta/(s**2*(np.exp(theta*T)-1))
psi = (k+theta)/(s**2)

r_star = np.log(A(T,S)/K)/B(T,S)

CIR_c = (FV*P(0,S,r0)*ncx2.pdf(2*r_star*(phi+psi+B(T,S)), 4*k*r_bar/(s**2), (2*phi**2)*r0*np.exp(theta*T)/(phi+psi+B(T,S))) - \
        0.98*P(0,T,r0)*ncx2.pdf(2*r_star*(phi+psi), 4*k*r_bar/(s**2), (2*phi**2)*r0*np.exp(theta*T)/(phi+psi)))/100

CIR_c

#3
r0 = 0.03
x0 = 0
y0 = 0
phi0 = 0.03
rho = 0.7
a = 0.1
b = 0.3
s = 0.03
e = 0.08
S = 1
T = 0.5
N = 10000
t = 1/252
M = 1000

def G2_ZBP(r0,x0,y0,phi0,rho,a,b,s,e,FV,S,N,t = 1/252):
    n = int(S/t)
    x = pd.DataFrame(np.zeros((N,n+1)))
    y = pd.DataFrame(np.zeros((N,n+1)))
    r = pd.DataFrame(np.zeros((N,n+1)))

    Z1 = pd.DataFrame(np.random.normal(0,1,N*n).reshape(N,n))
    Z2 = pd.DataFrame(np.random.normal(0,1,N*n).reshape(N,n))
    x.iloc[:,0] = [x0]*N
    y.iloc[:,0] = [y0]*N
    r.iloc[:,0] = [r0]*N

    for i in range(1,n+1):
        x.iloc[:,i] = x.iloc[:,i-1] - a*x.iloc[:,i-1]*t + s*np.sqrt(t)*Z1.iloc[:,i-1]
        y.iloc[:,i] = y.iloc[:,i-1] - b*y.iloc[:,i-1]*t + e*np.sqrt(t)*(rho*Z1.iloc[:,i-1]+np.sqrt(1-rho**2)*Z2.iloc[:,i-1])
        r.iloc[:,i] = y.iloc[:,i] + x.iloc[:,i] + [phi0]*N

    R = -r.sum(axis=1)*t
    BP = FV*np.exp(R).mean()
    return BP

G2_ZBP(r0,x0,y0,phi0,rho,a,b,s,e,FV,S,N,t)

def G2_BOP(r0,x0,y0,phi0,rho,a,b,s,e ,K,FV,T,S,N,M,t = 1/252):
    n = int(T/t)
    x = pd.DataFrame(np.zeros((N,n+1)))
    y = pd.DataFrame(np.zeros((N,n+1)))
    r = pd.DataFrame(np.zeros((N,n+1)))

    Z1 = pd.DataFrame(np.random.normal(0,1,N*n).reshape(N,n))
    Z2 = pd.DataFrame(np.random.normal(0,1,N*n).reshape(N,n))
    x.iloc[:,0] = [x0]*N
    y.iloc[:,0] = [y0]*N
    r.iloc[:,0] = [r0]*N

    for i in range(1,n+1):
        x.iloc[:,i] = x.iloc[:,i-1] - a*x.iloc[:,i-1]*t + s*np.sqrt(t)*Z1.iloc[:,i-1]
        y.iloc[:,i] = y.iloc[:,i-1] - b*y.iloc[:,i-1]*t + e*np.sqrt(t)*(rho*Z1.iloc[:,i-1]+np.sqrt(1-rho**2)*Z2.iloc[:,i-1])
        r.iloc[:,i] = y.iloc[:,i] + x.iloc[:,i] + [phi0]*N
    
    rn = r.iloc[:,n-1]
    xn = x.iloc[:,n-1]
    yn = y.iloc[:,n-1]

    R = -r.sum(axis=1)*t
    On = np.zeros(len(rn))
    for i in range(len(rn)):
        #At each node at P from M paths
        Pn = G2_ZBP(rn[i],xn[i],yn[i],phi0,rho,a,b,s,e,FV,S-T,M,t)
        On = np.exp(R[i])*np.maximum(K-Pn,0)
    P = On.mean()
    return P

G2_BOP(r0,x0,y0,phi0,rho,a,b,s,e ,K,FV,T,S,N,M,t)

