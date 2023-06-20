import pandas as pd
import numpy as np
import types



## Missing Completely At Random ############################################################################################################
def induce_mcar(df, miss_rate, miss_type, ignore_cols=[]):
    '''
    Induces values missing completely at random into a dataframe
        parameters:
            df:             dataframe
            miss_rate:      percentage of missing values
            miss_type:      type of missingness, can be 'single', 'patient', or 'both'
            ignore_cols:    list of columns that should not have missing values (such as static and outcome vars)
    '''
    if miss_type == 'single':
        return create_mcar_single(df, miss_rate, ignore_cols)
    elif miss_type == 'patient':
        return create_mcar_patient(df, miss_rate, ignore_cols)
    elif miss_type == 'both':
        return create_mcar_both(df, miss_rate, ignore_cols)
    else:
        raise ValueError('miss_type must be "single", "patient", or "both"')


# function to create individual missing values in a dataframe
def create_mcar_single(df, miss_rate, ignore_cols):
    '''
    Induces individual missing values into a dataframe following a missing completely at random pattern
        parameters:
            df:             dataframe
            miss_rate:      percentage of missing values
            ignore_cols:    list of columns that should not have missing values (such as static and outcome vars)            
    '''

    df2 = df.copy()
    for col in df2.columns:
        if col not in ignore_cols:
            df2.loc[df2.sample(frac=miss_rate).index, col] = None
    return df2


# function to create missing values for a whole patient 
def create_mcar_patient(df, miss_rate, ignore_cols):
    '''
    Induces missing values for whole patients at a time, following a missing completely at random pattern
        parameters:
            df:             dataframe   
            miss_rate:      percentage of missing values
            ignore_cols:    list of columns that should not have missing values (such as static and outcome vars)
    '''

    df2 = df.copy()
    pats = pd.Series(df2.index.get_level_values('subject_id').unique()) # get unique patient ids
    
    for col in df2.columns:
        if col not in ignore_cols:
            misspats = pats.sample(frac=miss_rate) # select patients to be missing
            df2.loc[misspats, col] = None
    return df2


# function to create missing values for whole patients as well as individual missing values
def create_mcar_both(df, miss_rate, ignore_cols):
    ''' 
    Induces missing values for whole patients as well as individual missing values, following a missing completely at random pattern
        parameters:
            df:             dataframe
            miss_rate:      percentage of missing values
            ignore_cols:    list of columns that should not have missing values (such as static and outcome vars)
    '''

    df2 = df.copy()
    pats = pd.Series(df2.index.get_level_values('subject_id').unique()) # get unique patient ids
    
    for col in df2.columns:
        if col not in ignore_cols:
            # half of the missing values are for whole patients
            misspats = pats.sample(frac=miss_rate/2) 
            df2.loc[misspats, col] = None
            # the other half are for individual missing values
            n = int(len(df2[col]) * (miss_rate - df2[col].isna().mean())) # how many missing values to create
            idxs = np.random.choice(len(df2[col]), max(n,0), replace=False, p=df2[col].notna()/df2[col].notna().sum()) # select indexes to be missing from the non-missing values
            # using a mask to assign the missing values because 'df2[col].iloc[idxs] = None' can show a SettingWithCopyWarning 
            mask = np.zeros(len(df2[col]), dtype=bool)
            mask[idxs] = True
            df2.loc[mask, col] = None
                
    return df2


## Missing At Random #######################################################################################################################
def induce_mar(df, cond_var, miss_weights, miss_rate, miss_type, ignore_cols=[]):
    '''
    Induces values missing at random into a dataframe
        parameters:
            df:             dataframe
            cond_var:       variable influencing the missingness
            miss_weights:   weighted distribution of missingness rates depending on the cond_var (higher weigths increase likelihood of missingness)
                            Can be a list of weights (same length as number of unique values in cond_var), a function to be applied to the values of cond var 
                            or a string: 'equal'(weight is equal to the value of the cond_var), 'square' (square of weights) 
                            or 'exponential' (exponential weights)
            miss_rate:      overall percentage of missing values
            miss_type:      type of missingness, can be 'single', 'patient', or 'both'
            ignore_cols:    list of columns that should not have missing values (such as static and outcome vars)
    '''

    ### If the conditional variable is not in the ignore_cols, add it
    if cond_var not in ignore_cols:
        ignore_cols.append(cond_var)

    ### First, calculate the weights for each value of the cond_var
    df2 = calc_miss_weights(df, cond_var, miss_weights)

    ### Second, create missing values according to the desired miss_type
    if miss_type == 'single':
        return create_mar_single(df=df2, miss_rate=miss_rate, ignore_cols=ignore_cols)
    elif miss_type == 'patient':
        return  create_mar_patient(df=df2, miss_rate=miss_rate, ignore_cols=ignore_cols)
    elif miss_type == 'both':
        return create_mar_both(df=df2, miss_rate=miss_rate, ignore_cols=ignore_cols)
    else:
        raise ValueError('miss_type must be "single", "patient", or "both"')


# calculate the weights for each value of the cond_var
def calc_miss_weights(df, cond_var, miss_weights):
    '''
    Helper function to calculate the corresponding weights for each value of the cond_var
        parameters:
            df:             dataframe
            cond_var:       variable influencing the missingness
            miss_weights:   weighted distribution of missingness rates depending on the cond_var (higher weigths increase likelihood of missingness)
                            Can be a list of weights (same length as number of unique values in cond_var or interpreted per quantile), a function to be applied to the values of cond var 
                            or a string: 'equal'(weight is equal to the value of the cond_var), 'square' (square of weights) 
                            or 'exponential' (exponential weights)
    '''
    df2 = df.copy()

    vals = np.array(sorted(df[cond_var].unique()))
    # if miss_weights is a function, apply it to the values of cond_var
    if isinstance(miss_weights, types.FunctionType):
        miss_weights = miss_weights(vals)

    # if miss_weighhts is one of the predefined options, calculate the weights
    elif miss_weights == 'equal':
        miss_weights = vals
    elif miss_weights == 'squared':
        miss_weights = np.square(vals)
    elif miss_weights == 'exponential':
        miss_weights = np.exp(vals)
    
    # if miss_weights is a list, check if it has the same length as the number of unique values in cond_var
    elif isinstance(miss_weights, list):
        if len(miss_weights) != len(vals):
            # calculate the quantiles of the conditional variable and then assign the weights by quantile
            df2['quantiles'] = pd.factorize(pd.qcut(df2[cond_var], len(miss_weights), duplicates='drop'), sort=True)[0]
            vals = np.array(sorted(df2['quantiles'].unique()))
    
    else:
        raise ValueError('miss_weights must be a function, a list, or one of the predefined options ("equal", "squared", "exponential")')

    if 'quantiles' in df.columns:
        df2['missweights'] = df2['quantiles'].map(dict(zip(vals, miss_weights)))  # assign each value from cond variable to corresponding weight
        df2 = df2.drop('quantiles', axis=1)
    else:
        df2['missweights'] = df2[cond_var].map(dict(zip(vals, miss_weights)))  # assign each value from cond variable to corresponding weight


    return df2


# create individual missing values based on a condition variable
def create_mar_single(df, miss_rate, ignore_cols):
    '''
    Induces individual missing values into a dataframe following a missing at random pattern with one conditional variable
        parameters:
            df:             dataframe   
            miss_rate:      overall percentage of missing values
            ignore_cols:    list of columns that should not have missing values (such as static and outcome vars)
    '''

    # how many values should be missing to achieve the target miss rate
    n = int(np.round(len(df) * miss_rate))
    df['missweights'] = df['missweights'] / df['missweights'].sum() # normalize the weights directly because they will be used as is (set val to 0-1)


    for col in df.columns:
        if col not in ignore_cols + ['missweights']:
            idxs = np.random.choice(len(df[col]), max(n,0), replace=False, p=df['missweights']) # select indexes to be missing from the non-missing values
            # using a mask to assign the missing values because 'df[col].iloc[idxs] = None' can show a SettingWithCopyWarning 
            mask = np.zeros(len(df[col]), dtype=bool)
            mask[idxs] = True
            df.loc[mask, col] = None
                
    return df.drop('missweights', axis=1)


# create missing values for entire patients based on a conditional variable
def create_mar_patient(df, miss_rate, ignore_cols):
    '''
    Induces whole patient missing values into a dataframe following a missing at random pattern with one conditional variable
        parameters:
            df:             dataframe
            miss_rate:      overall percentage of missing values
            ignore_cols:    list of columns that should not have missing values (such as static and outcome vars)
    '''

    pats = pd.Series(df.index.get_level_values('subject_id').unique()) # get unique patient ids
    # get missprob for each patient
    pat_missprob = df.groupby('subject_id')['missweights'].first()  # using the mean in case the condition variable is not the same for all values of a patient 
    pat_missprob = pat_missprob / pat_missprob.sum()                # normalize only after we know how many patients exist
    # how many missing values do we need?
    n = int(len(pats) * miss_rate)
    
    # reproduce the approach used for the single missing values
    for col in df.columns:
        if col not in ignore_cols + ['missweights']:
            ids = np.random.choice(pats, max(n,0), replace=False, p=pat_missprob) # select subject ids to be missing
            # this only works if there is a lot of patients of roughly the same size overall --> might have to change but seems to be okay for this dataset

            # using a mask to assign the missing values because 'df[col].iloc[idxs] = None' can show a SettingWithCopyWarning 
            mask = np.zeros(len(df[col]), dtype=bool)
            # setting mask values to true for the rows that correspond to the selected patients
            mask[df.index.get_level_values('subject_id').isin(ids)] = True
            df.loc[mask, col] = None
                
    return df.drop('missweights', axis=1)


# function to create missing values for whole patients as well as individual missing values based on a conditional variable
def create_mar_both(df, miss_rate, ignore_cols):
    '''
    Induces individual missing values as well as whole missing patients into a dataframe following a missing at random pattern with one conditional variable
        parameters:
            df:             dataframe
            miss_rate:      overall percentage of missing values
            ignore_cols:    list of columns that should not have missing values (such as static and outcome vars)
    '''

    pats = pd.Series(df.index.get_level_values('subject_id').unique()) # get unique patient ids
    
    # get missprob for each patient
    pat_missprob = df.groupby('subject_id')['missweights'].mean() # using the mean in case the condition variable is not the same for all values of a patient 
    pat_missprob = pat_missprob / pat_missprob.sum()              # normalize only after we know how many patients exist
    # how many missing values do we need from full patients?
    n_pats= int((len(pats) * miss_rate) / 2) # half of all missing values should be from full patients

    for col in df.columns:
        if col not in ignore_cols + ['missweights']:
  
            ids = np.random.choice(pats, max(n_pats,0), replace=False, p=pat_missprob) # select subject ids to be missing
            # using a mask to assign the missing values because 'df[col].iloc[idxs] = None' can show a SettingWithCopyWarning 
            mask = np.zeros(len(df[col]), dtype=bool)
            # setting mask values to true for the rows that correspond to the selected patients
            mask[df.index.get_level_values('subject_id').isin(ids)] = True
            df.loc[mask, col] = None


            # the other half are for individual missing values
            #          | # of missings overall | - | # of miss vals in col already |
            n_singles = int(len(df) * miss_rate) - int(df[col].isna().sum())

            # only consider the non-missing values
            nonmiss = df[[col, 'missweights']].dropna()
            # normalize the weights to sum up to 1
            nonmiss['missweights'] = nonmiss['missweights'] / nonmiss['missweights'].sum()
            
            idxs = np.random.choice(nonmiss.index, max(n_singles,0), replace=False, p=nonmiss['missweights']) # select indexes to be missing from the non-missing values
            # using a mask to assign the missing values because 'df[col].iloc[idxs] = None' can show a SettingWithCopyWarning 
            # mask = pd.DataFrame(np.zeros(len(df[col]), dtype=bool)).set_index(df.index)
            # mask.loc[idxs] = True
            df.loc[idxs, col] = None
                
    return df.drop('missweights', axis=1)



## Missing Not At Random ####################################################################################################################
def induce_mnar(df, miss_weights, miss_rate, miss_type, ignore_cols=[]):
    '''
    Induces values missing not at random into a dataframe
        parameters:
            df:             dataframe
            miss_weights:   weighted distribution of missingness rates depending on the values of each column (higher weigths increase 
                            likelihood of missingness). Can be a function to be applied to the values of each column or a string: 
                            'equal'(weight is equal to the value), 'square' (weights are squares of the values) or 
                            'exponential' (exponential weights)
            miss_rate:      overall percentage of missing values
            miss_type:      type of missingness, can be 'single', 'patient', or 'both'
            ignore_cols:    list of columns that should not have missing values (such as static and outcome vars)
    '''

    df2 = df.copy() # so we don't change the original dataframe

    # check for non numeric cols and add them to ignore_cols if necessary 
    icols = list(set(ignore_cols + df2.select_dtypes(exclude=np.number).columns.tolist()))

    ### Create missing values according to the desired miss_type
    if miss_type == 'single':
        return create_mnar_single(df=df2, weighting=miss_weights, miss_rate=miss_rate, ignore_cols=icols)
    elif miss_type == 'patient':
        return  create_mnar_patient(df=df2, weighting=miss_weights, miss_rate=miss_rate, ignore_cols=icols)
    elif miss_type == 'both':
        return create_mnar_both(df=df2, weighting=miss_weights, miss_rate=miss_rate, ignore_cols=icols)
    else:
        raise ValueError('miss_type must be "single", "patient", or "both"')


def calc_col_miss_weights(col, miss_weights):
    '''
    Calculates the weights for each value of a column following the miss_weights parameter
        parameters:
            col:            column
            miss_weights:   weighted distribution of missingness rates depending on the values of each column (higher weigths increase 
                            likelihood of missingness). Can be a function to be applied to the values of each column or a string: 
                            'equal'(weight is equal to the value), 'square' (weights are squares of the values) or 
                            'exponential' (exponential weights)
    '''

    # if miss_weights is a function, apply it to the values of the column
    if isinstance(miss_weights, types.FunctionType):
        return miss_weights(col)
    # if miss_weighhts is one of the predefined options, calculate the weights
    elif miss_weights == 'equal':
        return col
    elif miss_weights == 'squared':
        return np.square(col)
    elif miss_weights == 'exponential':
        return np.exp(col)
    
    # if miss_weights is a list, calculate the quantiles of the variable and then assign the weights by quantile
    elif isinstance(miss_weights, list):
        tmp = pd.factorize(pd.qcut(col, len(miss_weights), duplicates='drop'), sort=True)[0]
        return pd.Series(tmp, index=col.index)
    else:
        raise ValueError('miss_weights must be a function or one of the predefined options: "equal", "squared", "exponential", "log"')
    

# create individual missing values 
def create_mnar_single(df, weighting, miss_rate, ignore_cols):
    '''
    Induces individual missing values into a dataframe following a missing not at random pattern
        parameters:
            df:             dataframe
            weighting:      weighted distribution of missingness rates depending on the values of each column (higher weigths increase 
                            likelihood of missingness). Can be a function to be applied to the values of each column or a string: 
                            'equal'(weight is equal to the value), 'square' (weights are squares of the values) or 
                            'exponential' (exponential weights)
            miss_rate:      overall percentage of missing values
            ignore_cols:    list of columns that should not have missing values (such as static and outcome vars)
    '''
    df2 = df.copy()
    # how many values should be missing to achieve the target miss rate
    n = int(np.round(len(df2) * miss_rate))
    
    for col in df2.columns:
        if col not in ignore_cols:
            df2[col] += 0.1 # to avoid 0 probabilities   
            weights = calc_col_miss_weights(df2[col], weighting) # calculate the weights for the values of the column
            weights = weights / weights.sum() 
            idxs = np.random.choice(len(df2[col]), max(n,0), replace=False, p=weights) # select indexes to be missing from the non-missing values
            mask = np.zeros(len(df2[col]), dtype=bool)           # create mask to assign the missing values
            mask[idxs] = True                                   # set the mask values to true for the selected indexes
            df2.loc[mask, col] = None                            # use the mask to induce missing values
                
    return df2


# create missing values for entire patients based on a conditional variable
def create_mnar_patient(df, weighting, miss_rate, ignore_cols):
    '''
    Induces missing values for entire patients into a dataframe following a missing not at random pattern
        parameters:
            df2:             dataframe
            weighting:      weighted distribution of missingness rates depending on the values of each column (higher weigths increase 
                            likelihood of missingness). Can be a function to be applied to the values of each column or a string: 
                            'equal'(weight is equal to the value), 'square' (weights are squares of the values) or 
                            'exponential' (exponential weights)
            miss_rate:      overall percentage of missing values
            ignore_cols:    list of columns that should not have missing values (such as static and outcome vars)
    '''
    df2 = df.copy()
    pats = pd.Series(df2.index.get_level_values('subject_id').unique()) # get unique patient ids
    n = int(len(pats) * miss_rate)                                     # how many missing values do we need?

    # reproduce the approach used for the single missing values
    for col in df2.columns:
        if col not in ignore_cols:
            df2[col] += 0.1 # to avoid 0 probabilities   
            # calculate weights for the patients
            pat_avg = df2.groupby('subject_id')[col].mean()      # get the average value of the column for each patient
            weights = calc_col_miss_weights(pat_avg, weighting) # calculate the weights for the values of each patient
            weights = weights / weights.sum()                   # normalize the weights to sum up to 1
            
            ids = np.random.choice(pats, max(n,0), replace=False, p=weights) # select subject ids to be missing
            mask = np.zeros(len(df2[col]), dtype=bool)
            mask[df2.index.get_level_values('subject_id').isin(ids)] = True   # setting mask values to true for the rows that correspond to the selected patients
            df2.loc[mask, col] = None
                
    return df2

  
# function to create missing values for whole patients as well as individual missing values based on a conditional variable
def create_mnar_both(df, weighting, miss_rate, ignore_cols):
    '''
    Induces missing values for entire patients and individual missing values into a dataframe following a missing not at random pattern
        parameters:
            df2:             dataframe
            weighting:      weighted distribution of missingness rates depending on the values of each column (higher weigths increase 
                            likelihood of missingness). Can be a function to be applied to the values of each column or a string: 
                            'equal'(weight is equal to the value), 'square' (weights are squares of the values) or 
                            'exponential' (exponential weights)
            miss_rate:      overall percentage of missing values
            ignore_cols:    list of columns that should not have missing values (such as static and outcome vars)
    '''
    df2 = df.copy()
    pats = pd.Series(df2.index.get_level_values('subject_id').unique()) # get unique patient ids
 
    # how many missing values do we need from full patients?
    n_pats= int((len(pats) * miss_rate) / 2) # half of all missing values should be from full patients

    for col in df2.columns:
        if col not in ignore_cols:
            df2[col] += 0.1 # to avoid 0 probabilities   
            pat_avg = df2.groupby('subject_id')[col].mean()      # get the average value of the column for each patient
            weights = calc_col_miss_weights(pat_avg, weighting) # calculate the weights for the values of each patient
            weights = weights / weights.sum()                   # normalize the weights to sum up to 1
            
            ids = np.random.choice(pats, max(n_pats,0), replace=False, p=weights) # select subject ids to be missing
            mask = np.zeros(len(df2[col]), dtype=bool)
            mask[df2.index.get_level_values('subject_id').isin(ids)] = True   # setting mask values to true for the rows that correspond to the selected patients
            df2.loc[mask, col] = None    

            # the other half are for individual missing values
            #          | # of missings overall | - | # of miss vals in col already |
            n_singles = int(len(df2) * miss_rate) - int(df2[col].isna().sum())

            weights = calc_col_miss_weights(df2[col].dropna(), weighting) # only consider the remaining (non missing) values
            weights = weights / weights.sum()                   # normalize the weights to sum up to 1

            idxs = np.random.choice(weights.index, max(n_singles,0), replace=False, p=weights) # select indexes to be missing from the non-missing values
            df2.loc[idxs, col] = None                      

                
    return df2
