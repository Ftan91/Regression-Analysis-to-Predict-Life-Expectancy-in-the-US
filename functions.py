import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score, KFold
from sklearn.metrics import r2_score
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant


def drop_na_columns(dataframe, list_of_columns, threshold):
    """Drop columns where number of null entries in a column exceeds a user-set percentage threshold"""
    n = dataframe.shape[0]
    to_drop = [column for column in list_of_columns if (dataframe[column].isnull().sum() / n) > threshold]
    dataframe.drop(to_drop, axis = 1, inplace = True)
    print ('Number of dropped columns: {}'.format(len(to_drop)))
    print ('\n')
    print ('Dropped columns: \n', to_drop)
    
def categorical_and_discrete_na_filler(dataframe, categorical_columns):
    """Fill empty rows with values from selected column according to current distribution percentages"""
    for column in categorical_columns:
        choice = sorted(dataframe[dataframe[column].notnull()][column].unique())
        probability = dataframe[column].value_counts(normalize = True).sort_index().values
        dataframe[column] = dataframe[column].apply(
            lambda x: np.random.choice(choice, p = probability) 
            if (pd.isnull(x)) 
            else x)
        
def continuous_na_filler(dataframe, columns, method):
    """Fill empty rows with values according to user-chosen method; mean or median"""
    if method == 'mean':
        for column in columns:
            value = np.mean(dataframe[column])
            dataframe[column].fillna(round(value, 0), inplace = True)
    elif method == 'median':
        for column in columns:
            value = np.nanmedian(dataframe[column])
            dataframe[column].fillna(round(value, 0), inplace = True)
    else:
        print ('Method not available. Please choose either mean or median, else update function for desired method.')
        
def check_outliers(dataframe, list_of_columns, lower_quantile_list, upper_quantile_list):
    """Returns a dataframe of outliers according to user provided quantiles"""
    quantile = lower_quantile_list + upper_quantile_list

    summary_dict = {}
    for col in list_of_columns:
        summary_dict[col] = []
        for i in quantile:
            summary_dict[col].append(dataframe[col].quantile(i))

    summary_df = pd.DataFrame(summary_dict)
    summary_df_final = pd.concat([pd.DataFrame(quantile, columns=['Quantile']), summary_df], axis = 1)

    return summary_df_final

def drop_values_multi(dataframe, list_of_columns, quantile):
    """Drop outliers based on quantile """
    to_drop_index = []
    quantile = quantile

    for i in list_of_columns:
        index = list(dataframe[dataframe[i] > dataframe[i].quantile(quantile)].index)
        to_drop_index = to_drop_index + index

    dataframe.drop(set(to_drop_index), axis = 0, inplace = True)
    print ('Successfully dropped rows!')
    
    
def central_limit_mean(dataset, sample_size = 50, num_simulations = 500, return_mean = False):    
    """create a distribution of means"""
    random_chosen = [np.mean(np.random.choice(dataset, size = sample_size)) for i in range(num_simulations)]
    if return_mean == False:
        return random_chosen
    else:
        return (random_chosen, round(np.mean(random_chosen), 2))
    
def CLT_violinplots(dataframe, x_axis, y_axis, sample_size = 50, num_simulations = 500):
    """create multiple violinplots in a single figure"""
    unique_list = dataframe[y_axis].unique()
    df = pd.DataFrame(None, columns = ['{} types'.format(y_axis), '{} Sample Mean'.format(x_axis)])
    
    for i in unique_list:
        CLT_data = central_limit_mean(dataframe[dataframe[y_axis] == i][x_axis], 
                                      sample_size = sample_size, 
                                      num_simulations = num_simulations)
        df = pd.concat([df, pd.DataFrame(list(zip([i] * len(CLT_data), CLT_data)), 
                                         columns = [y_axis, '{} Sample Means'.format(x_axis)])], 
                        axis = 0, 
                        ignore_index=True)
                        
    ordering = df.groupby(y_axis)['{} Sample Means'.format(x_axis)].mean().sort_values().index
        
#   plt.figure(figsize=(15,10))
    sns.set(font_scale = 1.1)
    sns.violinplot(x = '{} Sample Means'.format(x_axis), y = y_axis, data = df, palette="Set3", order = ordering)
    plt.xticks(rotation = 45)
    plt.show()
    
def sklearn_linreg_with_CV(X, y, test_size, n_splits):
    "Generates R2 score"
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = 1)
    linreg = LinearRegression().fit(X_train, y_train)
    linreg_score = round(linreg.score(X_test, y_test),2)

    cross_validation = KFold(n_splits = n_splits, shuffle = True, random_state = 1)
    linreg_cv = LinearRegression()
    cv_score = round(np.mean(cross_val_score(linreg_cv, X, y, scoring = 'r2', cv=cross_validation)),2)

    return linreg_score, cv_score


def statsmodel_regression(X_train, X_test, y_train, y_test):
    "Similar with above however uses statsmodel'
    regr = OLS(y_train, add_constant(X_train)).fit()
    predictions = regr.predict(X_test)
    r2_test = round(r2_score(y_test, predictions),2)
    return r2_test, regr


### COURTESY OF THE LEARN.CO LABS
def stepwise_selection(X, y, 
                       initial_list=[], 
                       threshold_in=0.01, 
                       threshold_out = 0.05, 
                       verbose=True):
    """ Perform a forward-backward feature selection 
    based on p-value from statsmodels.api.OLS
    Arguments:
        X - pandas.DataFrame with candidate features
        y - list-like with the target
        initial_list - list of features to start with (column names of X)
        threshold_in - include a feature if its p-value < threshold_in
        threshold_out - exclude a feature if its p-value > threshold_out
        verbose - whether to print the sequence of inclusions and exclusions
    Returns: list of selected features 
    Always set threshold_in < threshold_out to avoid infinite looping.
    See https://en.wikipedia.org/wiki/Stepwise_regression for the details
    """
    included = list(initial_list)
    while True:
        changed=False
        # forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        # backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.argmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included


def update_model(dataframe, model_name, OLS_object, R2_scores_list):
    """Updates model table"""
    dataframe.loc[dataframe.model == model_name, 'R2_train'] = R2_scores_list[0]
    dataframe.loc[dataframe.model == model_name, 'CV_R2_train'] = R2_scores_list[1]
    dataframe.loc[dataframe.model == model_name, 'AIC'] = round(OLS_object.aic,0)
    dataframe.loc[dataframe.model == model_name, 'n_features'] = len(OLS_object.pvalues)
    dataframe.loc[dataframe.model == model_name, '>0.05_pvalues'] = sum(OLS_object.pvalues > 0.05)
