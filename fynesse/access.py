from .config import *

"""These are the types of import we might expect in this file
import httplib2
import oauth2
import tables
import mongodb
import sqlite"""

import yaml
from ipywidgets import interact_manual, Text, Password
import pandas as pd
import pymysql
import numpy as np
from tqdm.notebook import tqdm


def load_into_database(conn, file_name):
    """ Load a dataset from the local disk to the MariaDB database
        via a database connection.
      :param conn: a database connection
      :param file_name: the path to a local csv file
      :return: None
    """
    sql = """
            LOAD DATA LOCAL INFILE %s INTO TABLE `pp_data` 
            FIELDS TERMINATED BY ',' LINES STARTING BY '' 
            TERMINATED BY '\n'; 
         """
    cur = conn.cursor()
    try:
        cur.execute(sql, (file_name, ))
        conn.commit()
    except:
        raise Exception("Unable to load data into database.")


def dataset_downloads_uploads(year_lb=1995, year_ub=2022, save_dir="./datasets", verbose=False, conn=None):
    """ Download the uk price paid dataset (for a period) from gov.uk site 
        into a local directory, then can optionally upload the files to 
        a MariaDB database via a database connection.
      :param year_lb: the time period lower bound (in years, inclusive)
      :param year_ub: the time period upper bound (in years, inclusive)
      :param save_dir: a path on the local disk to save the csv files
      :param verbose: log processing information
      :param conn: a database connection
      :return: None
    """
    years = np.arange(year_lb, year_ub+1)
    parts = np.arange(1, 3)
    general_url = "http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com"

    with tqdm(total=len(years) * len(parts)) as pbar:
        for year in years:
            for part in parts:
                try:
                    # Try to download the yearly dataset in parts
                    url = f"{general_url}/pp-{year}-part{part}.csv"
                    save_path = f"{save_dir}/pp-{year}-part{part}.csv"
                    # Ignore bad lines and save into a dataframe
                    df = pd.read_csv(url, on_bad_lines='skip')
                    # Save the dataframe into a csv on local disk
                    df.to_csv(save_path, header=False, index=False)
                    if verbose:
                        print(
                            f"save pp-{year}-part{part}.csv into local disk successfully.")
                    if conn is not None:
                        load_into_database(conn, save_path)
                        if verbose:
                            print(
                                f"upload pp-{year}-part{part}.csv to table `pp_data` successfully.")
                    pbar.update(1)
                except Exception as e:
                    raise Exception(
                        f"dataset from year {year} failed to download or upload.")


def create_connection(user, password, host, database, port=3306):
    """ Create a database connection to the MariaDB database
        specified by the host url and database name.
    :param user: username
    :param password: password
    :param host: host url
    :param database: database
    :param port: port number
    :return: Connection object or None
    """
    conn = None
    try:
        conn = pymysql.connect(user=user,
                               passwd=password,
                               host=host,
                               port=port,
                               local_infile=1,
                               db=database
                               )
    except:
        raise Exception(f"Error connecting to the MariaDB Server: {e}")
    return conn


def data():
    # Database url
    database_details = {"url": "database-yz709-db.cgrre17yxw11.eu-west-2.rds.amazonaws.com",
                        "port": 3306}

    @interact_manual(username=Text(description="Username:"),
                     password=Password(description="Password:"))
    def write_credentials(username, password):
        with open("credentials.yaml", "w") as file:
            credentials_dict = {'username': username,
                                'password': password}
            yaml.dump(credentials_dict, file)

    # Get username and password from the yaml file for database access
    with open("credentials.yaml") as file:
        credentials = yaml.safe_load(file)
    username = credentials["username"]
    password = credentials["password"]
    url = database_details["url"]

    # Set up a database connection
    conn = create_connection(user=credentials["username"],
                             password=credentials["password"],
                             host=database_details["url"],
                             database="property_prices")
