"Put all financial ratios here, no need for class I think"
from annualize import annualiseReturns, annualiseVolatility

def sharpeRatio(price, riskFreeRate, periodsPerYear):
    r = price.diff().dropna()
    rfPerPeriod = (1+riskFreeRate)**(1/periodsPerYear)-1
    excessReturn = r - rfPerPeriod
    annualiseExcessReturn = annualiseReturns(excessReturn, periodsPerYear)
    annualiseVol = annualiseVolatility(r,periodsPerYear)
    return annualiseExcessReturn/annualiseVol

def calmarRatio():
    pass
    # max drawdown

def omegaRatio(price, riskFreeRate, periodsPerYear):
    # let minimum investor threshold be risk free rate
    r = price.diff().dropna()
    rfPerPeriod = (1+riskFreeRate)**(1/periodsPerYear)-1
    excessReturn = r - rfPerPeriod
    positive = excessReturn[excessReturn > 0].sum()
    negative = excessReturn[excessReturn < 0].sum()
    return positive/ (-negative)


def sortinoRatio():
    pass

def tailRatio():
    pass

def commonSenseRatio():
    pass
