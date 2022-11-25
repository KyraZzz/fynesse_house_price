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
        cur = conn.cursor()
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
    geo_list = [(float(df.iloc[id]['lattitude']), float(
        df.iloc[id]['longitude'])) for id in range(len(df))]
    df_OSM = get_POIs_for_list(geo_list, pois_box_width, pois_box_height)
    return pd.concat((df, df_OSM), axis=1)


def one_hot_df(df, columns=None, unique_vals_columns=None):
    """ Converts qualitative data into quantitative ones via the one-hot strategy.
    :param df: a DataFrame
    :param columns: the list of qualitative data columns
    :param unique_vals_columns: the unique values for each given data columns
    :return: a DataFrame with all qualitative data columns converted into quantitative data columns
    """
    if columns is None:
        columns = ['property_type']
    for id, col in enumerate(columns):
        if col in df.columns:
            if unique_vals_columns is None or unique_vals_columns[id] is None:
                unique_vals = df[col].unique()
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
