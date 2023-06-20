import pandas as pd
import numpy as np
import multiprocessing as mp
from itertools import product, chain
from functools import partial
from tqdm.auto import tqdm

# local imports
from impfuns import ffill_median
from misscreate import induce_mcar, induce_mar, induce_mnar
from sklearn.metrics import mean_squared_error, mean_absolute_error

# load the data
df2h = pd.read_parquet('outdata/datasets/complete/data2h.parquet')
df4h = pd.read_parquet('outdata/datasets/complete/data4h.parquet')
df6h = pd.read_parquet('outdata/datasets/complete/data6h.parquet')

no_miss_cols = df2h.select_dtypes(exclude='number').columns.to_list() + ['hospital_expire_flag'] # list of all non-numeric columns + outcome

# options ##########################################################################################################
runs = 20

# take care to only add non empty lists to itertools.product as the product is always [] if any part is []
datasets =      {'2h': df2h, '4h': df4h, '6h': df6h}                         
miss_rates =    [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
miss_funs =     [induce_mnar]                      
imp_funs =      [ffill_median]
miss_types =    ['single', 'patient', 'both'] 
miss_weights =  ['equal', 'squared', 'exponential']

combinations = list(product(datasets, miss_rates, miss_funs, imp_funs, miss_types, miss_weights))

# induce_mnar(df, miss_weights, miss_rate, miss_type, ignore_cols=[])

def calculate_errors(options, run):
    result = []
    for dfkey, missing_rate, missfun, impfun, miss_type, miss_weight in options:
        df = datasets[dfkey]
        df2 = missfun(df, miss_weight, missing_rate, miss_type, no_miss_cols)
        df3 = impfun(df2, no_miss_cols)

        mse = mean_squared_error(df.drop(no_miss_cols, axis=1), df3.drop(no_miss_cols, axis=1))
        mae = mean_absolute_error(df.drop(no_miss_cols, axis=1), df3.drop(no_miss_cols, axis=1))
        rmse = np.sqrt(mse)
        missrates_exact = df2.drop(no_miss_cols, axis=1).isna().sum().sum() / df2.drop(no_miss_cols, axis=1).size
        result.append([run, miss_type, miss_weight, dfkey, missing_rate, missrates_exact, mse, mae, rmse])

    return result


if __name__ == '__main__':
    p = mp.Pool(8) # open pool with 8 cores

    func = partial(calculate_errors, combinations) # creates "preloaded" version of the function with the options parameter included
    partial_res = p.map(func,tqdm(range(runs)))
    
    p.close()
    p.join()

    res_cols = ['run','missing_pattern', 'weighting', 'dataset', 'missing_rate', 'missing_rate_exact', 'mse', 'mae', 'rmse']
    df_results = (pd.DataFrame(list(chain(*partial_res)), columns=res_cols)
                    .sort_values(by=['missing_pattern', 'dataset', 'missing_rate'])
                    .explode(['missing_rate_exact', 'mse', 'mae'])
                    .reset_index(drop=True)
                    )
    
    outfile_path = f"outdata/{miss_funs[0].__name__.split('_')[1]}_{imp_funs[0].__name__}_res{runs}.parquet"
    df_results.to_parquet(outfile_path)
