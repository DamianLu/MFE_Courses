# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os
import pandas  as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics # make sure scikit-learn is installed
from sklearn.linear_model import ElasticNet
import warnings
warnings.filterwarnings('ignore')


# %%
os.chdir("/Users/Damian/Desktop/ML/project4")
stocks = pd.read_csv("StockRetAcct_DT.csv")


# %%
c = stocks[['lnIssue','lnProf','lnInv','lnME']]
new = c**2
new.columns = ['lnIssue2','lnProf2','lnInv2','lnME2']
ME = c.iloc[:,:3].multiply(c.iloc[:,3],axis='index')
ME.columns = ['lnIssue_ME','lnProf_ME','lnInv_ME']
chrc = pd.concat([c.reset_index(drop=True), new.reset_index(drop=True), ME.reset_index(drop=True)], axis=1)
chrc.insert(0,'year',stocks['year'])
chrc.insert(0,'FirmID',stocks['FirmID'])

#demean
#chrc.iloc[:,2:12] -= chrc.iloc[:,1:13].groupby('year').transform('mean')
mean_chrc = chrc.iloc[:,1:13].groupby('year').transform('mean')
chrc.iloc[:,2:13] -= mean_chrc
chrc.insert(13,'ones',np.repeat(1,chrc.shape[0]))

#return
rets =  np.exp(stocks[['lnAnnRet','lnRf']])
rets.columns = ['Return','Rf']
rets['R_ex'] = rets['Return']-rets['Rf']
rets.insert(0,'year',stocks['year'])
rets.insert(0,'FirmID',stocks['FirmID'])


# %%
weighted_ret = chrc.iloc[:,2:14].multiply(rets['R_ex'],axis='index')
weighted_ret.insert(0,'year',stocks['year'])

factors = weighted_ret.groupby('year').sum()
means = factors.mean(axis=0)
stds = factors.std(axis=0)

Sharpe_Ratio = means/stds

print(Sharpe_Ratio)


# %%
#Elastic Net Regression
train = np.array_split(factors.iloc[0:24,:],5)
test = factors.iloc[25:,:]
lambdas = 10**np.linspace(-1,10,50)*0.02

mse_summary = pd.DataFrame(index = lambdas)

def MSE(X,Y,model):
    y = Y.to_numpy()
    y_hat = model.predict(X)
    mse = ((y_hat-y)**2).mean()
    return mse


# %%
#To be looped for each fold
for i in range(5):
    elastic_net = ElasticNet(l1_ratio = 0.5)
    coefs = pd.DataFrame(index = lambdas, columns = train[0].columns)
    mse = pd.DataFrame(index = lambdas, columns = ['MSE'])
    sample = pd.DataFrame(np.concatenate(train[:i]+train[i+1:], axis=0))

    X = sample.cov()
    Y = sample.mean(axis=0)

    for j, l in enumerate(lambdas):
        elastic_net.alpha = l   # set the severity of the constraint
        elastic_net.fit(X, Y)
        coefs.iloc[j] = elastic_net.coef_
        mse.iloc[j] = MSE(X,Y,elastic_net)
    
    mse_summary = pd.concat([mse_summary,mse],axis=1)


# %%
minimum_mse = mse_summary.mean(axis=1).min()
optimal_lambda = mse_summary.mean(axis=1).idxmin()
#The minimum mse
print(minimum_mse,'\n')
#The lambda that gives minimum mse
print(optimal_lambda,'\n')


# %%
#iii
elastic_net = ElasticNet(alpha = optimal_lambda, l1_ratio = 0.5)
X_test = test.cov()
Y_test = test.mean(axis=0)

elastic_net.fit(X_test, Y_test)
b = elastic_net.coef_
b


# %%
portfolio_return = test.dot(b)
print(portfolio_return,'\n')

print('Mean Return: ', portfolio_return.mean(),'\n')
print('Standard Deviation of Return: ', portfolio_return.std(),'\n')

port_SR = portfolio_return.mean()/portfolio_return.std()

print('Portfolio Sharpe Ratio: ', port_SR)


# %%
rets['MEwt'] = stocks['MEwt']
rets['Wt_ret'] = rets['MEwt']*rets['R_ex']

market_ret = rets[['year','Wt_ret']].groupby('year').sum()

benchmark = market_ret.iloc[25:35,:]


# %%
#normalize
ratio = benchmark.std()/portfolio_return.std()
ratio = ratio[0]

MVE = portfolio_return * ratio

cumulative_ret = (MVE + 1).cumprod()
cumulative_benchmark = (benchmark + 1).iloc[:,0].cumprod()


# %%
plt.plot(range(2005,2015),cumulative_ret,label = "MVE")
plt.plot(range(2005,2015),cumulative_benchmark,label = "Market Equal Weighted Portfolio")
plt.legend()


