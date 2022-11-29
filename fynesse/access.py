from .config import *

import os
import yaml
from getpass import getpass
import urllib.request as request
import pandas as pd
import pymysql
import numpy as np
from tqdm.notebook import tqdm
import zipfile
import requests
import csv


def save_credentials():
    """ Get the credentials from the user and save them into a yaml file.
    :return: None
    """
    username = input("Username:")
    password = getpass("Password:")
    with open("credentials.yaml", "w") as file:
        credentials_dict = {'username': username,
                            'password': password}
        yaml.dump(credentials_dict, file)


def connect_to_database(username, password, url):
    """ Connect to the database with the username and password authentication.
    :param username: the username for database access
    :param password: the password for database access
    :param url: the database access point url
    :return: None
    """
    %load_ext sql
    %sql mariadb+pymysql: //$username: $password @$url?local_infile = 1
    print(f"Database connected.")


def create_database_property_prices():
    """ Create a database called `property_prices`
    :return: None
    """
    %load_ext sql
    %sql SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO"
    %sql SET time_zone = "+00:00"

    %sql CREATE DATABASE IF NOT EXISTS `property_prices` DEFAULT CHARACTER SET utf8 COLLATE utf8_bin
    %sql USE `property_prices`
    print(f"Database `property_prices` created.")


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
        raise Exception(f"Error connecting to the MariaDB Server.")
    return conn


def create_database_property_prices(conn):
    """ Create a database called `property_prices`
    :param conn: a Connection object to the database
    :return: None
    """
    %load_ext sql
    %sql SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO"
    %sql SET time_zone = "+00:00"

    sql = """
        CREATE DATABASE IF NOT EXISTS `property_prices` DEFAULT CHARACTER SET utf8 COLLATE utf8_bin;
        """
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            conn.commit()
    except:
        raise Exception("Unable to create the database `property_prices`.")
    sql = "USE `property_prices`;"
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            conn.commit()
    except:
        raise Exception("Unable to use the database `property_prices`.")
    print(f"Database `property_prices` created.")


def create_table_pp_data(conn):
    """ Create a fresh database table `pp_data`.
    :param conn: a Connection object to the database
    :return: None
    """
    sql = """
        DROP TABLE IF EXISTS `pp_data`;
        """
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            conn.commit()
    except:
        raise Exception("Unable to drop the table `pp_data` if it exists.")
    sql = """
        CREATE TABLE IF NOT EXISTS `pp_data` (
          `transaction_unique_identifier` tinytext COLLATE utf8_bin NOT NULL,
          `price` int(10) unsigned NOT NULL,
          `date_of_transfer` date NOT NULL,
          `postcode` varchar(8) COLLATE utf8_bin NOT NULL,
          `property_type` varchar(1) COLLATE utf8_bin NOT NULL,
          `new_build_flag` varchar(1) COLLATE utf8_bin NOT NULL,
          `tenure_type` varchar(1) COLLATE utf8_bin NOT NULL,
          `primary_addressable_object_name` tinytext COLLATE utf8_bin NOT NULL,
          `secondary_addressable_object_name` tinytext COLLATE utf8_bin NOT NULL,
          `street` tinytext COLLATE utf8_bin NOT NULL,
          `locality` tinytext COLLATE utf8_bin NOT NULL,
          `town_city` tinytext COLLATE utf8_bin NOT NULL,
          `district` tinytext COLLATE utf8_bin NOT NULL,
          `county` tinytext COLLATE utf8_bin NOT NULL,
          `ppd_category_type` varchar(2) COLLATE utf8_bin NOT NULL,
          `record_status` varchar(2) COLLATE utf8_bin NOT NULL,
          `db_id` bigint(20) unsigned NOT NULL
        ) DEFAULT CHARSET=utf8 COLLATE=utf8_bin AUTO_INCREMENT=1 ;
    """
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            conn.commit()
    except:
        raise Exception("Unable to create the table `pp_data`.")
    print("Database table `pp_data` created.")


def add_table_pp_data_primary_key(conn):
    """ Add primary key `db_id` in the table `pp_data`.
    :param conn: a Connection object to the database
    :return: None
    """
    sql = "ALTER TABLE `pp_data` ADD PRIMARY KEY (`db_id`);"
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            conn.commit()
    except:
        raise Exception("Unable to add primary key in the table `pp_data`.")
    sql = """
        ALTER TABLE `pp_data`
        MODIFY `db_id` bigint(20) unsigned NOT NULL AUTO_INCREMENT,AUTO_INCREMENT=1;
        """
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            conn.commit()
    except:
        raise Exception(
            "Unable to set AUTO_INCREMENT policy for the primary key in the table `pp_data`.")
    print("Set up primary key `db_id` in the table `pp_data`.")


def add_table_pp_data_index(conn):
    """ Add indexes into the table `pp_data` for columns `postcode` and `pp.date_of_transfer`.
    :param conn: a Connection object
    :return: None
    """
    sql = """
        CREATE INDEX `pp.postcode` USING HASH ON `pp_data` (postcode);
        """
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            conn.commit()
    except:
        raise Exception(
            "Unable to add index for column `postcode` in the table `pp_data`.")

    sql = """
        CREATE INDEX `pp.date` USING HASH ON `pp_data` (date_of_transfer);
        """
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            conn.commit()
    except:
        raise Exception(
            "Unable to add index for column `date_of_transfer` in the table `pp_data`.")
    print("Add indexes into the table `pp_data` for columns `postcode` and `pp.date_of_transfer`.")


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
    try:
        with conn.cursor() as cur:
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


def select_top(conn, table,  n):
    """ Query n first rows of the table
    :param conn: the Connection object
    :param table: The table to query
    :param n: Number of rows to query
    :return: query results in rows
    """
    sql = """
        USE `property_prices`;
        """
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            conn.commit()
    except:
        raise Exception("Unable to use the database `property_prices`.")

    try:
        with conn.cursor() as cur:
            cur.execute(f'SELECT * FROM {table} \
                      LIMIT {n} ')
            rows = cur.fetchall()
    except:
        raise Exception(
            f"Unable to select the top {n} rows from the table {table}.")

    return rows


def pp_data_to_df(rows):
    """ Convert query results into a DataFrame format for `pp_data` table.
    :param rows: query results in rows
    :return: query results in a DataFrame
    """
    df = pd.DataFrame(rows, columns=['transaction_unique_identifier', 'price', 'date_of_transfer',
                                     'postcode', 'property_type', 'new_build_flag', 'tenure_type',
                                     'primary_addressable_object_name', 'secondary_addressable_object_name',
                                     'street', 'locality', 'town_city', 'district', 'county', 'ppd_category_type',
                                     'record_status', 'db_id'])
    df.drop("db_id", axis=1, inplace=True)
    return df


def download_postcode_data():
    """ Download and extract contents from a zip file to construct the `postcode_data` table.
    :return: None
    """
    # Download file from url
    postcode_url = 'https://www.getthedata.com/downloads/open_postcode_geo.csv.zip'
    r = requests.get(postcode_url)
    # Save zip file locally
    with open('./datasets/open_postcode_geo.csv.zip', 'wb') as outfile:
        outfile.write(r.content)
    # Unzip file
    save_path = "./datasets/"
    with zipfile.ZipFile("./datasets/open_postcode_geo.csv.zip", 'r') as zip_ref:
        zip_ref.extractall(save_path)
    print("Data for the table `postcode_data` downloaded and preprocessed.")


def create_table_postcode_data(conn):
    """ Create table `postcode_data` and config the primary key and the indexes.
    :param conn: a Connection object to the database
    :return: None
    """
    sql = "USE `property_prices`;"
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            conn.commit()
    except:
        raise Exception("Unable to use database `property_prices`.")

    sql = "DROP TABLE IF EXISTS `postcode_data`;"
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            conn.commit()
    except:
        raise Exception(
            "Unable to drop the table `postcode_data` if it exists.")

    sql = """
        CREATE TABLE IF NOT EXISTS `postcode_data` (
        `postcode` varchar(8) COLLATE utf8_bin NOT NULL,
        `status` enum('live','terminated') NOT NULL,
        `usertype` enum('small', 'large') NOT NULL,
        `easting` int unsigned,
        `northing` int unsigned,
        `positional_quality_indicator` int NOT NULL,
        `country` enum('England', 'Wales', 'Scotland', 'Northern Ireland', 'Channel Islands', 'Isle of Man') NOT NULL,
        `lattitude` decimal(11,8) NOT NULL,
        `longitude` decimal(10,8) NOT NULL,
        `postcode_no_space` tinytext COLLATE utf8_bin NOT NULL,
        `postcode_fixed_width_seven` varchar(7) COLLATE utf8_bin NOT NULL,
        `postcode_fixed_width_eight` varchar(8) COLLATE utf8_bin NOT NULL,
        `postcode_area` varchar(2) COLLATE utf8_bin NOT NULL,
        `postcode_district` varchar(4) COLLATE utf8_bin NOT NULL,
        `postcode_sector` varchar(6) COLLATE utf8_bin NOT NULL,
        `outcode` varchar(4) COLLATE utf8_bin NOT NULL,
        `incode` varchar(3)  COLLATE utf8_bin NOT NULL,
        `db_id` bigint(20) unsigned NOT NULL
      ) DEFAULT CHARSET=utf8 COLLATE=utf8_bin;
        """
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            conn.commit()
    except:
        raise Exception("Unable to create the table `postcode_data`.")

    sql = "ALTER TABLE `postcode_data` ADD PRIMARY KEY (`db_id`);"
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            conn.commit()
    except:
        raise Exception(
            "Unable to add primary key `db_id` into the table `postcode_data`.")

    sql = "ALTER TABLE `postcode_data` MODIFY `db_id` bigint(20) unsigned NOT NULL AUTO_INCREMENT,AUTO_INCREMENT=1;"
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            conn.commit()
    except:
        raise Exception(
            "Unable to set AUTO_INCREMENT policy for the primary key `db_id` of the table `postcode_data`.")

    sql = """
        CREATE INDEX `po.postcode` USING HASH
          ON `postcode_data`
            (postcode);
        """
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            conn.commit()
    except:
        raise Exception(
            "Unable to create index in the table `postcode_data` for the column `postcode`.")
    print("Create the table `postcode_data`, add the primary key and indexes.")


def load_data_table_postcode_data(conn):
    """ Load data into the table `postcode_data`.
    :param conn: a Connection object
    :return: None
    """
    sql = """
        LOAD DATA LOCAL INFILE './datasets/open_postcode_geo.csv' INTO TABLE `postcode_data`
        FIELDS TERMINATED BY ',' 
        LINES STARTING BY '' TERMINATED BY '\n';
        """
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            conn.commit()
    except:
        raise Exception("Unable to load data into the table `postcode_data`.")
    print("Data loaded into the table `postcode_data`.")


def postcode_data_to_df(rows):
    """Convert query results into a DataFrame format for `postcode_data` table.
    :param rows: query results in rows
    :return: query results in a DataFrame
    """
    return pd.DataFrame(rows, columns=['postcode', 'status', 'usertype',
                                       'easting', 'northing', 'positional_quality_indicator', 'country',
                                       'lattitude', 'longitude', 'postcode_no_space',
                                       'postcode_fixed_width_seven', 'postcode_fixed_width_eight',
                                       'postcode_area', 'postcode_district', 'postcode_sector', 'outcode',
                                       'incode', 'db_id'])


def create_prices_coordinates_data_table(conn):
    """Create a fresh table `prices_coordinates_data` in the database for storing 
       joined results from the table `pp_data` and `postcode_data`.
    :param conn: a connection to the database
    :return:None
    """
    sql = """
        USE `property_prices`;
        """
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            conn.commit()
    except:
        raise Exception("Unable to use the database `property_prices`.")

    sql = """
        DROP TABLE IF EXISTS `prices_coordinates_data`;
        """
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            conn.commit()
    except:
        raise Exception(
            "Unable to drop the table `prices_coordinates_data` if it exists.")

    sql = """
        CREATE TABLE IF NOT EXISTS `prices_coordinates_data` (
          `price` int(10) unsigned NOT NULL,
          `date_of_transfer` date NOT NULL,
          `postcode` varchar(8) COLLATE utf8_bin NOT NULL,
          `property_type` varchar(1) COLLATE utf8_bin NOT NULL,
          `new_build_flag` varchar(1) COLLATE utf8_bin NOT NULL,
          `tenure_type` varchar(1) COLLATE utf8_bin NOT NULL,
          `locality` tinytext COLLATE utf8_bin NOT NULL,
          `town_city` tinytext COLLATE utf8_bin NOT NULL,
          `district` tinytext COLLATE utf8_bin NOT NULL,
          `county` tinytext COLLATE utf8_bin NOT NULL,
          `country` enum('England', 'Wales', 'Scotland', 'Northern Ireland', 'Channel Islands', 'Isle of Man') NOT NULL,
          `latitude` decimal(11,8) NOT NULL,
          `longitude` decimal(10,8) NOT NULL,
          `db_id` bigint(20) unsigned NOT NULL
        ) DEFAULT CHARSET=utf8 COLLATE=utf8_bin AUTO_INCREMENT=1 ;
        """
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            conn.commit()
    except:
        raise Exception(
            "Unable to create the table `prices_coordinates_data`.")


def join_region_period(conn, region, time_period_lb, time_period_ub, save_dir="./datasets"):
    """Join the two tables `pp_data` and `postcode_data` for a given region and time period.
    :param conn: a Connection object to the database
    :param region: the name of the town or city in capital letters
    :param time_period_lb: the lower bound of the time period (inclusive)
    :param time_period_ub: the upper bound of the time period (exclusive)
    :return: None
    """
    sql = """
        USE `property_prices`;
        """
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            conn.commit()
    except:
        raise Exception("Unable to use the database `property_prices`.")

    sql = """SELECT pp_data.price as price, pp_data.date_of_transfer as date_of_transfer, 
                 pp_data.postcode as postcode, pp_data.property_type as property_type, 
                 pp_data.new_build_flag as new_build_flag, pp_data.tenure_type as tenure_type, 
                 pp_data.locality as locality, pp_data.town_city as town_city, 
                 pp_data.district as district, pp_data.county as county, 
                 postcode_data.country as country, postcode_data.lattitude as latitude, 
                 postcode_data.longitude as longitude 
          FROM pp_data
          INNER JOIN postcode_data
          ON pp_data.postcode = postcode_data.postcode
          WHERE pp_data.date_of_transfer >= %s 
          AND pp_data.date_of_transfer <= %s 
          AND pp_data.town_city = %s;
          """
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (time_period_lb, time_period_ub, region))
            rows = cur.fetchall()
    except:
        raise Exception(
            "Unable to join the tables `pp_data` and `postcode_data` for the given region and time period.")

    # Create a new table `prices_coordinates_data`
    create_prices_coordinates_data_table(conn)

    try:
        # Save rows into a csv file on the local disk
        save_path = f"{save_dir}/pcd-{region}-lb{time_period_lb}-ub{time_period_ub}.csv"
        fp = open(save_path, 'w')
        myFile = csv.writer(fp)
        myFile.writerows(rows)
        fp.close()
    except:
        raise Exception("Unable to save fetched rows into a local csv file.")

    # Load data into the new table
    sql = """
            LOAD DATA LOCAL INFILE %s INTO TABLE `prices_coordinates_data` 
            FIELDS TERMINATED BY ',' LINES STARTING BY '' 
            TERMINATED BY '\n'; 
         """
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (save_path, ))
            conn.commit()
    except:
        raise Exception(
            "Unable to load data into the table `prices_coordinates_data`.")


def join_bbox_period(conn, north, south, west, east, time_period_lb, time_period_ub, save_dir="./datasets", verbose=False):
    """Join the two tables `pp_data` and `postcode_data` for a given bounding box and time period.
    :param conn: a Connection object to the database
    :param north: the upper bound in latitude
    :param south: the lower bound in latitude
    :param west: the lower bound in longitude
    :param east: the upper bound in longitude
    :param time_period_lb: the lower bound of the time period (inclusive)
    :param time_period_ub: the upper bound of the time period (exclusive)
    :param save_dir: directory for saving joined table
    :return: None
    """
    sql = """
        USE `property_prices`;
        """
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            conn.commit()
    except:
        raise Exception("Unable to use the database `property_prices`.")

    sql = """SELECT pp_data.price as price, pp_data.date_of_transfer as date_of_transfer, 
                 pp_data.postcode as postcode, pp_data.property_type as property_type, 
                 pp_data.new_build_flag as new_build_flag, pp_data.tenure_type as tenure_type, 
                 pp_data.locality as locality, pp_data.town_city as town_city, 
                 pp_data.district as district, pp_data.county as county, 
                 postcode_data.country as country, postcode_data.lattitude as latitude, 
                 postcode_data.longitude as longitude 
          FROM pp_data 
          INNER JOIN postcode_data
          ON pp_data.postcode = postcode_data.postcode
          WHERE pp_data.date_of_transfer >= %s 
          AND pp_data.date_of_transfer <= %s
          AND postcode_data.lattitude <= %s 
          AND postcode_data.lattitude >= %s
          AND postcode_data.longitude >= %s
          AND postcode_data.longitude <= %s;
          """
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (time_period_lb, time_period_ub,
                        north, south, west, east))
            rows = cur.fetchall()
    except:
        raise Exception(
            "Unable to join table `pp_data` and `postcode_data` for the given bounding box and time period.")

    # Create a new table `prices_coordinates_data`
    create_prices_coordinates_data_table(conn)

    try:
        # Save rows into a csv file on the local disk
        save_path = f"{save_dir}/pcd-lb{time_period_lb}-ub{time_period_ub}.csv"
        fp = open(save_path, 'w')
        myFile = csv.writer(fp)
        myFile.writerows(rows)
        fp.close()
    except:
        raise Exception("Unable to save fetched rows into a local csv file.")

    # Load data into the new table
    sql = """
            LOAD DATA LOCAL INFILE %s INTO TABLE `prices_coordinates_data` 
            FIELDS TERMINATED BY ',' LINES STARTING BY '' 
            TERMINATED BY '\n'; 
         """
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (save_path, ))
            conn.commit()
    except:
        raise Exception("Unable to load data into `prices_coordinates_data`.")


def prices_coordinates_data_to_df(rows):
    """ Convert query results into a DataFrame format for `prices_coordinates_data` table.
    :param rows: query results in rows
    :return: query results in a DataFrame
    """
    return pd.DataFrame(rows, columns=['price', 'date_of_transfer', 'postcode',
                                       'property_type', 'new_build_flag', 'tenure_type', 'locality',
                                       'town_city', 'district', 'county',
                                       'country', 'latitude', 'longitude',
                                       'db_id'])


def data():
    # Save credentials
    save_credentials()

    # Database url
    database_details = {"url": "database-ads-yz709.cgrre17yxw11.eu-west-2.rds.amazonaws.com",
                        "port": 3306}

    # Get username and password from the yaml file for database access
    with open("credentials.yaml") as file:
        credentials = yaml.safe_load(file)
    username = credentials["username"]
    password = credentials["password"]
    url = database_details["url"]

    # Connect to the database
    connect_to_database(username, password, url)

    # Set up a database connection
    conn = create_connection(user=credentials["username"],
                             password=credentials["password"],
                             host=database_details["url"],
                             database="property_prices")

    # Create a database called `property_prices`
    create_database_property_prices(conn)

    # Create a database table `pp_data`
    create_table_pp_data(conn)

    # Set up primary key in `pp_data`
    add_table_pp_data_primary_key(conn)

    # Add indexes for columns `postcode` and `pp.date_of_transfer` in `pp_data`
    add_table_pp_data_index(conn)

    # Create table `pp_date`
    create_table_pp_data(conn)

    # Construct a folder `datasets`
    os.mkdir("./datasets")

    # Download and upload data to `pp_data`
    dataset_downloads_uploads(
        year_lb=1995, year_ub=2022, save_dir="./datasets", verbose=False, conn=conn)

    # Download and extract contents from a zip file to construct the `postcode_data` table.
    download_postcode_data()

    # Load data into the table `postcode_data`.
    load_data_table_postcode_data(conn)
