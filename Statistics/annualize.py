" Annualisation functions here, no need for class"

def annualiseReturns(r, periodsPerYear):
    compoundGrowth = (1+r).prod()
    n = r.shape[0]
    return compoundGrowth**(periodsPerYear/n)-1

def annualiseVolatility(r, periodsPerYear):
    return r.std()*(periodsPerYear**0.5)


    