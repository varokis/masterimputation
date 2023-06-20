import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer

# Wrapper for imputation functions to be used in the eval scripts with simplified and uniform inputs (only input df and ignore cols)
# This means settings and hyperparams will be set here and remain the same through all testing 

# 1. Simple deterministic
# 1.1. Forward fill median (baseline)
def ffill_median(df, ignore_cols):
    df2 = df.copy().reset_index(drop=True)
    for col in df2.columns:
        if col not in ignore_cols and pd.isnull(df2[col][0]):
            df2.loc[df2.index[0], col] = df2[col].median()

    # df2.loc[0,] = df2.loc[0,].fillna(df2.median(numeric_only=True))
    df2 = df2.fillna(method='ffill')
    return df2

