import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def eval_imputation_error(df, missfun, missing_rate, type, imputation_method, n=10):
    '''
    Evaluates the imputation error of a method
        parameters:
            df: dataframe   
            missfun: function to induce missing values
            missing_rate: percentage of missing values
            type: type of missingness, can be 'single', 'patient', or 'both'
            imputation_method: method to impute missing values
            n: number of times to repeat the experiment
    '''

    mse = []
    mae = []
    missrates = []
    for i in range(n):
        df2 = missfun(df, missing_rate, type)
        df3 = imputation_method(df2)
        mse.append(mean_squared_error(df, df3))
        mae.append(mean_absolute_error(df, df3))
        missrates.append(df2.isna().sum().sum() / df2.size)
    return mse,  mae, missrates


def test_missrate(df, missfun, missrate, n=10):
    '''
    Tests the true rate of missing data for a given function
        parameters:
            df: dataframe
            missfun: function to induce missingness
            missrate: desired missing rate
            n: number of runs to find the average missing rate
    '''

    missrates = []
    for _ in range(n):
        df2 = missfun(df, missrate)
        missrates.append(df2.isna().sum().sum() / df2.size)
    return missrate, np.mean(missrates)


def get_missrates(df, cond_var=None):
    '''
    Calculates the missingness rates for each column in a dataframe
        parameters:
            df:             dataframe
            cond_var:       variable influencing the missingness
    '''
 
    if cond_var is None:
        return df.isna().mean()
    else:
        return df.groupby(cond_var).apply(lambda x: x.isna().mean())


def test_missfun(missfun, verbose=False,  runs=100, **kwargs):
    '''
    To Test if the functions to induce missingness work as intended. Shows average missing 
    rates overall and depending on the conditional variable if applicable.
        parameters:
            df:             dataframe
            missfun:        function to be tested
            runs:           how many executions of the function to average the results
            verbose:        if True, prints the results instead of returning them
            **kwargs:       additional arguments for the function to be tested
    '''

    if 'ignore_cols' in kwargs:
        print('ignore_cols', kwargs['ignore_cols'])

    overall = []
    bycondvar = []
    for _ in range(runs):
        df2 = missfun(**kwargs)
        overall.append(get_missrates(df2))
        if 'cond_var' in kwargs:
            bycondvar.append(get_missrates(df2, kwargs['cond_var']).drop(kwargs.get('cond_var'), axis=1))
    
    overall = sum(overall)/runs
    bycondvar = sum(bycondvar)/runs

    if verbose:
        print('Overall missingness rates:')
        print(overall)
        if 'cond_var' in kwargs:
            print('Missingness rates by', kwargs['cond_var'], ':')
            print(bycondvar)
        return

    return overall, bycondvar