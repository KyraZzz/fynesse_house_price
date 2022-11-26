# This file contains code for suporting addressing questions in the data
import datetime
from tqdm.notebook import tqdm
import warnings
import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import RidgeCV, TweedieRegressor, BayesianRidge
from termcolor import colored
import seaborn as sns
import matplotlib.pyplot as plt

from assess import get_bounding_box, get_POIs_for_list
from access import join_bbox_period, prices_coordinates_data_to_df
warnings.filterwarnings("ignore")

"""Address a particular question that arises from the data"""


def get_period_lb_ub(date, diff_lb=30, diff_ub=30):
    """ Get a date range bounded by [date - diff_lb, date + diff_ub].
    :param date: date in interest
    :param diff_lb: the lower bound of the range in difference
    :param diff_ub: the upper bound of the range in difference
    :return: the lower bound of the time period (inclusive), the upper bound of the time period (exclusive)
    """
    date_pred = datetime.datetime.strptime(date, '%Y-%m-%d').date()
    one_day = datetime.timedelta(days=1)
    time_period_lb = str(date_pred - one_day * int(diff_lb))
    time_period_ub = str(date_pred + one_day * int(diff_ub))
    return time_period_lb, time_period_ub


def get_train_dataset(conn, latitude, longitude, date, train_box_width=0.02, train_box_height=0.02, diff_lb=30, diff_ub=30, verbose=False):
    """ Construct the train dataset for the relevant time period and location.
    :param conn: connection to the database
    :param latitude: the latitude of the point
    :param longitude: the longitude of the point
    :param date: date in interest
    :param train_box_width: the width of the bounding box for the train dataset
    :param train_box_height: the height of the bounding box for the train dataset
    :param diff_lb: the lower bound of the range in difference
    :param diff_ub: the upper bound of the range in difference
    :verbose: whether to log processing information
    :return: train dataset in a DataFrame
    """
    train_north, train_south, train_west, train_east = get_bounding_box(
        latitude, longitude, train_box_width, train_box_height)
    if verbose:
        print(
            f"Bounding box for the train dataset has coordinates north: {train_north:3f}, south: {train_south:3f}, west: {train_west:3f}, east: {train_east:3f}.")
    time_period_lb, time_period_ub = get_period_lb_ub(date, diff_lb, diff_ub)
    if verbose:
        print(
            f"Time period for the train dataset ranges from {time_period_lb} to {time_period_ub}.")
    # Construct training dataset
    join_bbox_period(conn, train_north, train_south, train_west,
                     train_east, time_period_lb, time_period_ub)
    if verbose:
        print("Train dataset stored in table `prices_coordinates_data` in the database.")

    sql = """
          SELECT * from `prices_coordinates_data`;
        """
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
    except:
        raise Exception(
            "Unable to select all rows in the table `prices_coordinates_data`")

    return prices_coordinates_data_to_df(rows)


def filter_property_type(df, property_type):
    """ Only preserve houses with specified property types.
    :param df: training dataset in a DataFrame
    :param property_type: either a list of house property types (e.g., ['F', 'S', 'D', 'T']) 
                          or a single property type character
    :return: filtered training dataset in a DataFrame
    """
    if isinstance(property_type, list):
        return df[df['property_type'].isin(property_type)]
    return df[df["property_type"] == property_type]


def get_optimal_train_dataset(conn,
                              latitude,
                              longitude,
                              date,
                              property_type,
                              train_size_lb=10,
                              train_size_ub=20,
                              train_box_width=0.03,
                              train_box_height=0.03,
                              diff_lb=20,
                              diff_ub=20,
                              verbose=False):
    """ Restrict the train dataset size in range [train_size_lb, train_size_ub]
    :param conn: connection to the database
    :param latitude: the latitude of the point
    :param longitude: the longitude of the point
    :param date: date in interest
    :param property_type: a single character describing the property type of the house at the given location,
                          (e.g., a single character in ['F', 'S', 'D', 'T'])
    :param train_size_lb: the lower bound of the train dataset size
    :param train_size_ub: the upper bound of the train dataset size
    :param train_box_width: the initial bounding box width of the train dataset
    :param train_box_height: the initial bounding box height of the train dataset
    :param diff_lb: the initial lower bound of the date range in difference
    :param diff_ub: the initial upper bound of the date range in difference
    :param verbose: whether to log processing information
    :return: train dataset in a DataFrame
    """
    while True:
        df = get_train_dataset(conn, latitude, longitude, date, train_box_width,
                               train_box_height, diff_lb, diff_ub, verbose)
        df = filter_property_type(df, property_type)
        if len(df) <= train_size_ub and len(df) >= train_size_lb:
            break
        if len(df) > train_size_ub:
            if verbose:
                print(f"The train dataset has {len(df)} entries, too big.")
                print("Try narrower time range or smaller box width...")
            if diff_lb <= 10:
                train_box_width = max(0.01, train_box_width - 0.01)
            else:
                diff_lb -= 2
        elif len(df) < train_size_lb:
            if verbose:
                print(f"The train dataset has {len(df)} entries, too small...")
                print("Try wider time range...")
            diff_lb += 4
        diff_ub = diff_lb
    print(f"Optimal train dataset has {len(df)} entries.")
    df = df.reset_index(drop=True)
    return df


def merge_OSM_features(df, pois_box_width=0.005, pois_box_height=0.005):
    """Embed OSM features into the train dataset.
    :param df: the raw train dataset in a DataFrame format
    :param pois_box_width: the width of the bounding box for searching POIs
    :param pois_box_height: the height of the bounding box for searching POIs
    :return: a dataframe combined the raw train dataset with OSM features
    """
    if 'lattitude' in df.columns and 'longitude' in df.columns:
        geo_list = [(float(df.iloc[id]['lattitude']), float(
            df.iloc[id]['longitude'])) for id in range(len(df))]
        df_OSM = get_POIs_for_list(geo_list, pois_box_width, pois_box_height)
        return pd.concat((df, df_OSM), axis=1)
    return df


def one_hot_df(df, columns=None, unique_vals_columns=None):
    """ Converts qualitative data into quantitative ones via the one-hot strategy.
    :param df: a DataFrame
    :param columns: the list of qualitative data columns
    :param unique_vals_columns: the unique values for each given data columns
    :return: a DataFrame with all qualitative data columns converted into quantitative data columns
    """
    if columns is None:
        return df
    for id, col in enumerate(columns):
        if col in df.columns:
            if unique_vals_columns is None or unique_vals_columns[id] is None:
                unique_vals = df[col].unique()
            else:
                unique_vals = unique_vals_columns[id]
            for val in unique_vals:
                df[f"{col}_{val}"] = np.where(df[col] == val, 1, 0)
    return df


def preprocess_df(df, pois_box_width=0.005, pois_box_height=0.005, columns=None, unique_vals_columns=None):
    """ Preprocess the DataFrame before model training.
    :param df: a DataFrame
    :param pois_box_width: the width of the bounding box for searching POIs
    :param pois_box_height: the height of the bounding box for searching POIs
    :param columns: the list of qualitative data columns
    :param unique_vals_columns: the unique values for each given data columns
    :return: a DataFrame after being preprocessed 
             (e.g., handle missing data, merge with OSM features, 
             apply one-hot encoding, only preserve numeric columns)
    """
    # Get rid of rows with missing price data
    if 'price' in df.columns:
        df = df[df['price'] > 0]
    # Drop rows with NaN values
    df = df.dropna(axis=0)
    # Drop db_ids column
    if 'db_id' in df.columns:
        df.drop('db_id', axis=1, inplace=True)
    # Merge OSM features
    df = merge_OSM_features(df, pois_box_width, pois_box_height)
    # Apply one-hot encoding to qualitative columns
    df = one_hot_df(df, columns, unique_vals_columns)
    # Filter all numeric columns
    df = df.select_dtypes(include=np.number)
    return df


def kfold_train_test_split(df):
    """ Cross validation and train test split.
    :param df: a DataFrame
    :return: X (a DataFrame with only independent features)
    :return: y (a Series with only the dependent feature)
    :return: n_fold (number of folds)
    :return: kfold (a generator KFold object)
    """
    # Split train and test dataset
    X = df.drop('price', axis=1)
    y = df['price']
    # Cross validation for measuring model quality under a few-shot scenario
    n_fold = max(2, int(len(df) / 5))
    print(f"Split dataset into {n_fold} folds.")
    kfold = KFold(n_splits=n_fold)
    return X, y, n_fold, kfold


def visualise_price_distribution(y, log_y):
    """ Fit normal distribution onto prices and log prices and 
        plot a side-by-side diagram.
    :param y: the dependent feature (prices)
    :param log_y: log version of the dependent feature (log(prices))
    :return: None
    """
    # Fit a normal distribution onto prices and log(prices)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.distplot(y, kde=False, bins=5, fit=stats.norm, ax=ax[0])
    sns.distplot(log_y, kde=False, bins=5, fit=stats.norm, ax=ax[1])
    print("Price Skewness: %f" % y.skew())
    print("Log Price Skewness: %f" % log_y.skew())
    plt.show()


def train_kfold_eval(X, log_y, n_fold, kfold, model):
    """ Train and evluate the model with cross validation.
    :param X: a DataFrame with only independent features
    :param log_y: log version of the dependent feature (log(prices))
    :param n_fold: number of folds in cross validation
    :param kfold: a generator KFold object
    :param model: the model for curve fitting
    :return: the average root mean square error across n folds
    """
    cnt = 0
    rmse_score = 0
    fig, axes = plt.subplots(1, n_fold, figsize=(n_fold * 4, 4), sharey=True)
    # Standardise data onto unit scale
    norm_X = RobustScaler().fit_transform(X)
    for train_idx, test_idx in kfold.split(X):
        X_train, X_test = norm_X[train_idx], norm_X[test_idx]
        y_train, y_test = log_y.iloc[train_idx], log_y.iloc[test_idx]
        # Model training
        model_results = model.fit(X_train, y_train)
        y_pred = model_results.predict(X_test)
        # Plot results
        axes[cnt].scatter(np.arange(len(X_test)), np.exp(y_pred), label='pred')
        axes[cnt].scatter(np.arange(len(X_test)), np.exp(y_test), label='true')
        axes[cnt].legend()
        axes[cnt].set_xlabel("Test samples")
        axes[cnt].set_ylabel("Prices")
        # Model performance
        cnt += 1
        rmse_score += mean_squared_error(np.exp(y_test),
                                         np.exp(y_pred), squared=False)
    avg_rmse_score = rmse_score / cnt
    print(f"Average rmse score with {model}: {avg_rmse_score}")
    plt.show()
    return avg_rmse_score


def train_predict(X, log_y, X_pred, model):
    """ Train the model with the whole dataset and predict.
    :param X: a DataFrame with only independent features
    :param log_y: log version of the dependent feature (log(prices))
    :param X_pred: a DataFrame with only independent features for prediction
    :param model: the model for curve fitting
    :return: the predicted price for X_pred
    """
    # Standardise data onto unit scale
    norm_X = RobustScaler().fit_transform(X)
    norm_X_pred = RobustScaler().fit_transform(X_pred)
    # Model prediction
    model_results = model.fit(norm_X, log_y)
    y_pred = model_results.predict(norm_X_pred)
    return y_pred


def train_eval_predict(X, log_y, n_fold, kfold, X_pred, model):
    """ A pipeline consists of model training, performance evaluation and prediction making.
    :param X: a DataFrame with only independent features
    :param log_y: log version of the dependent feature (log(prices))
    :param n_fold: number of folds in cross validation
    :param kfold: a generator KFold object
    :param X_pred: a DataFrame with only independent features for prediction
    :param model: the model for curve fitting
    :return: the predicted price for X_pred
    :return: the average root mean squared error across n folds
    """
    avg_rmse_score = train_kfold_eval(X, log_y, n_fold, kfold, model)
    y_pred = train_predict(X, log_y, X_pred, model)
    return y_pred, avg_rmse_score


def predict_price(latitude, longitude, date, property_type,
                  train_size_lb=10, train_size_ub=20, train_box_width=0.03,
                  train_box_height=0.03, diff_lb=20, diff_ub=20, verbose=True):
    """ Predict the house price for a property at location with 
       a given latitude-longitude pair at a given date and with a given property type.
    :param latitude: the latitude of the point
    :param longitude: the longitude of the point
    :param date: date in interest
    :param property_type: a single character describing the property type of the house at the given location,
                            (e.g., a single character in ['F', 'S', 'D', 'T'])
    :param train_size_lb: the lower bound of the train dataset size
    :param train_size_ub: the upper bound of the train dataset size
    :param train_box_width: the initial bounding box width of the train dataset
    :param train_box_height: the initial bounding box height of the train dataset
    :param diff_lb: the initial lower bound of the date range in difference
    :param diff_ub: the initial upper bound of the date range in difference
    :param verbose: whether to log processing information
    :return: a DataFrame with each entry: ['linear_model_name', 'predict_price', 'average_RMSE']
    """
    colorList = {'red': '\033[91m', 'green': '\033[92m', 'blue': '\033[94m'}
    score_df = pd.DataFrame(
        columns=['linear_model_name', 'predict_price', 'average_RMSE'])

    try:
        print(f"{colorList['blue']}" +
              f"Get optimal train dataset with size within range: [{train_size_lb}, {train_size_ub}]...")
        df = get_optimal_train_dataset(conn, latitude, longitude, date,
                                       property_type, train_size_lb, train_size_ub,
                                       train_box_width, train_box_height,
                                       diff_lb, diff_ub, verbose)
        print(f"{colorList['green']}" + "Optimal train dataset fetched.")
    except:
        raise Exception(f"{colorList['red']}" +
                        "Unable to featch optimal train dataset.")

    try:
        print(f"{colorList['blue']}" +
              f"Preprocess dataframe (merge with OSM features)...")
        df = preprocess_df(df)
        print(f"{colorList['green']}" + f"Dataset ready for model training.")
    except:
        raise Exception(f"{colorList['red']}" +
                        "Unable to preprocess the dataframe.")

    try:
        X, y, n_fold, kfold = kfold_train_test_split(df)
        log_y = np.log(y)
        print(f"{colorList['green']}" + f"Visualise price distribution.")
        visualise_price_distribution(y, log_y)
        X_pred = get_POIs_for_list([(latitude, longitude)])
        print(f"{colorList['green']}" + f"Evalutation dataset constructed.")
        model_list = [RidgeCV(), BayesianRidge(), TweedieRegressor(power=0)]
        print(f"{colorList['blue']}" +
              "Model training, testing and evaluating ...")
        for model in model_list:
            y_pred, avg_rmse_score = train_eval_predict(
                X, log_y, n_fold, kfold, X_pred, model)
            score_df = score_df.append({"linear_model_name": str(model), "predict_price": round(
                np.exp(y_pred[0]), 3), "average_RMSE": avg_rmse_score}, ignore_index=True)
        print(f"{colorList['green']}" + "Results:")
        print(score_df)
    except:
        raise Exception(
            f"{colorList['red']}" + "Unable to train, evaluate and test the models.")

    return score_df


def predict_price_fix(latitude, longitude, date, property_type, train_box_width=0.03,
                      train_box_height=0.03, diff_lb=20, diff_ub=20, verbose=True):
    """ Predict the house price for a property at location with 
       a given latitude-longitude pair at a given date and with a given property type,
       fix the bounding box and date range.
    :param latitude: the latitude of the point
    :param longitude: the longitude of the point
    :param date: date in interest
    :param property_type: a single character describing the property type of the house at the given location,
                            (e.g., a single character in ['F', 'S', 'D', 'T'])
    :param train_box_width: the initial bounding box width of the train dataset
    :param train_box_height: the initial bounding box height of the train dataset
    :param diff_lb: the initial lower bound of the date range in difference
    :param diff_ub: the initial upper bound of the date range in difference
    :param verbose: whether to log processing information
    :return: a DataFrame with each entry: ['linear_model_name', 'predict_price', 'average_RMSE']
    """
    colorList = {'red': '\033[91m', 'green': '\033[92m', 'blue': '\033[94m'}
    score_df = pd.DataFrame(
        columns=['linear_model_name', 'predict_price', 'average_RMSE'])

    try:
        print(f"{colorList['blue']}" + f"Get train dataset ...")
        df = get_train_dataset(conn, latitude, longitude, date, train_box_width,
                               train_box_height, diff_lb, diff_ub, verbose)
        df = filter_property_type(df, property_type)
        df = df.reset_index(drop=True)
        print(f"{colorList['green']}" + "Optimal train dataset fetched.")
    except:
        raise Exception(f"{colorList['red']}" +
                        "Unable to featch optimal train dataset.")

    try:
        print(f"{colorList['blue']}" +
              f"Preprocess dataframe (merge with OSM features)...")
        df = preprocess_df(df)
        print(f"{colorList['green']}" + f"Dataset ready for model training.")
    except:
        raise Exception(f"{colorList['red']}" +
                        "Unable to preprocess the dataframe.")

    try:
        X, y, n_fold, kfold = kfold_train_test_split(df)
        log_y = np.log(y)
        print(f"{colorList['green']}" + f"Visualise price distribution.")
        visualise_price_distribution(y, log_y)
        X_pred = get_POIs_for_list([(latitude, longitude)])
        print(f"{colorList['green']}" + f"Evalutation dataset constructed.")
        model_list = [RidgeCV(), BayesianRidge(), TweedieRegressor(power=0)]
        print(f"{colorList['blue']}" +
              "Model training, testing and evaluating ...")
        for model in model_list:
            y_pred, avg_rmse_score = train_eval_predict(
                X, log_y, n_fold, kfold, X_pred, model)
            score_df = score_df.append({"linear_model_name": str(model), "predict_price": round(
                np.exp(y_pred[0]), 3), "average_RMSE": avg_rmse_score}, ignore_index=True)
        print(f"{colorList['green']}" + "Results:")
        print(score_df)
    except:
        raise Exception(
            f"{colorList['red']}" + "Unable to train, evaluate and test the models.")

    return score_df


def predict_price_relaxed_property(latitude, longitude, date, property_type,
                                   train_size_lb=10, train_size_ub=20, train_box_width=0.03,
                                   train_box_height=0.03, diff_lb=20, diff_ub=20, verbose=True):
    """ Predict the house price for a property at location with 
       a given latitude-longitude pair at a given date and with a given property type,
       relax the property type filter of the training dataset.
    :param latitude: the latitude of the point
    :param longitude: the longitude of the point
    :param date: date in interest
    :param property_type: a single character describing the property type of the house at the given location,
                            (e.g., a single character in ['F', 'S', 'D', 'T'])
    :param train_size_lb: the lower bound of the train dataset size
    :param train_size_ub: the upper bound of the train dataset size
    :param train_box_width: the initial bounding box width of the train dataset
    :param train_box_height: the initial bounding box height of the train dataset
    :param diff_lb: the initial lower bound of the date range in difference
    :param diff_ub: the initial upper bound of the date range in difference
    :param verbose: whether to log processing information
    :return: a DataFrame with each entry: ['linear_model_name', 'predict_price', 'average_RMSE']
    """
    colorList = {'red': '\033[91m', 'green': '\033[92m', 'blue': '\033[94m'}
    score_df = pd.DataFrame(
        columns=['linear_model_name', 'predict_price', 'average_RMSE'])

    try:
        print(f"{colorList['blue']}" +
              f"Get optimal train dataset with size within range: [{train_size_lb}, {train_size_ub}]...")
        property_type_all = ['F', 'S', 'D', 'T']
        df = get_optimal_train_dataset(conn, latitude, longitude, date,
                                       property_type_all, train_size_lb, train_size_ub,
                                       train_box_width, train_box_height,
                                       diff_lb, diff_ub, verbose)
        unique_property = df['property_type'].unique()
        print(f"{colorList['green']}" + "Optimal train dataset fetched.")
    except:
        raise Exception(f"{colorList['red']}" +
                        "Unable to featch optimal train dataset.")

    try:
        print(f"{colorList['blue']}" +
              f"Preprocess dataframe (merge with OSM features)...")
        df = preprocess_df(
            df, columns=['property_type'], unique_vals_columns=[unique_property])
        print(f"{colorList['green']}" + f"Dataset ready for model training.")
    except:
        raise Exception(f"{colorList['red']}" +
                        "Unable to preprocess the dataframe.")

    try:
        X, y, n_fold, kfold = kfold_train_test_split(df)
        log_y = np.log(y)
        print(f"{colorList['green']}" + f"Visualise price distribution.")
        visualise_price_distribution(y, log_y)
        X_pred = get_POIs_for_list([(latitude, longitude)])
        X_pred['property_type'] = property_type
        X_pred = preprocess_df(
            X_pred, columns=['property_type'], unique_vals_columns=[unique_property])
        print(f"{colorList['green']}" + f"Evalutation dataset constructed.")
        model_list = [RidgeCV(), BayesianRidge(), TweedieRegressor(power=0)]
        print(f"{colorList['blue']}" +
              "Model training, testing and evaluating ...")
        for model in model_list:
            y_pred, avg_rmse_score = train_eval_predict(
                X, log_y, n_fold, kfold, X_pred, model)
            score_df = score_df.append({"linear_model_name": str(model), "predict_price": round(
                np.exp(y_pred[0]), 3), "average_RMSE": avg_rmse_score}, ignore_index=True)
        print(f"{colorList['green']}" + "Results:")
        print(score_df)
    except:
        raise Exception(
            f"{colorList['red']}" + "Unable to train, evaluate and test the models.")

    return score_df
