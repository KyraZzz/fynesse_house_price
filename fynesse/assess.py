from IPython.display import display
from ipywidgets import interact
import ipywidgets as widgets
from numpy.linalg import eig
from sklearn.decomposition import PCA
import seaborn as sns
from .access import config_credentials, config_price_data, get_bounding_box, get_pois, get_pois_by_key
import pandas as pd
import numpy as np
import mlai.plot as plot
import mlai
import seaborn as sns
import osmnx as ox
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('svg')


"""Place commands in this file to assess the data you have downloaded. How are missing values encoded, how are outliers encoded? What do columns represent, makes rure they are correctly labeled. How is the data indexed. Crete visualisation routines to assess the data (e.g. in bokeh). Ensure that date formats are correct and correctly timezoned."""


def plot_pois(pois, north, south, west, east, place_name="UK", ax=None):
    """ Plot a graph demonstrating the POIs.
    :param pois: POIs in a GeoPandas Dataframe
    :param north: the north coordinate of the bounding box
    :param south: the south coordinate of the bounding box
    :param west: the west coordinate of the bounding box
    :param east: the east coordinate of the bounding box
    :param place_name: the name of the place
    :return: a graph demonstrating the POIs
    """
    graph = ox.graph_from_bbox(north, south, east, west)
    # Retrieve nodes and edges
    nodes, edges = ox.graph_to_gdfs(graph)
    # Get place boundary related to the place name as a geodataframe
    area = ox.geocode_to_gdf("UK")
    if ax is None:
        fig, ax = plt.subplots(figsize=plot.big_figsize)
    # Plot the footprint
    area.plot(ax=ax, facecolor="white")
    # Plot street edges
    edges.plot(ax=ax, linewidth=1, edgecolor="dimgray")
    ax.set_xlim([west, east])
    ax.set_ylim([south, north])
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    pic_name = place_name.lower().replace(' ', '-').replace(',', '')
    ax.set_title(pic_name)
    ax.ticklabel_format(useOffset=False)

    # Plot all POIs
    pois.plot(ax=ax, color="blue", alpha=0.7, markersize=10)
    plt.tight_layout()
    mlai.write_figure(directory="./maps", filename=f"{pic_name}.svg")


def compare_places_pois_by_tag(place_name, locations, box_width, box_height, tags):
    """ Plot POIs for each tag and each place.
    :param place_name: a list of strings, each string is a common name of a location
    :param locations: a list of latitude-longitude pairs, one for each location
    :param box_width: the width of the bounding box
    :param box_height: the height of the bounding box
    :param tags: a list of POI groups
    :return: one plot per tag per place
    """
    fig, ax = plt.subplots(len(tags), len(place_name),
                           figsize=(len(place_name) * 4, len(tags) * 4))
    with tqdm(total=len(place_name) * len(tags)) as pbar:
        for i, p in enumerate(place_name):
            latitude, longitude = locations[i]
            north, south, west, east = get_bounding_box(
                latitude, longitude, box_width, box_height)
            pois = get_pois(north, south, west, east)
            for j, t in enumerate(tags):
                extract_pois = get_pois_by_key(pois, key=t)
                if extract_pois is not None:
                    plot_pois(extract_pois, north, south, west,
                              east, f"{p}-{t}", ax=ax[j][i])
                pbar.update(1)
    plt.tight_layout()
    mlai.write_figure(directory="./maps", filename="compare_places.svg")


def count_pois_by_tag(pois):
    """Count the number of POIs with relevant tags
    :param pois: POIs in a GeoPandas Dataframe
    :return: a dictionary (key: tag, value: number of POIs with the relevant tag)
    """
    # Define tags from a list of available POIs: https://wiki.openstreetmap.org/wiki/Map_features
    tags = ["aeroway", "amenity", "building", "highway", "landuse",
            "public_transport", "historic", "leisure", "shop", "tourism", "healthcare", "sport"]
    res = {k: 0 for k in tags}
    for tag in tags:
        output = get_pois_by_key(pois, tag)
        res[tag] = len(output) if output is not None else 0
    return res


def get_POIs_for_list(geo_list, box_width=0.005, box_height=0.005):
    """ Iterate through the list of latitude-longitude pairs, count the number
       of POIs in each tagged group for each location pair.
    :param geo_list: a list of latitude-longitude pairs
    :param box_width: the width of the bounding box
    :param box_height: the height of the bounding box
    :return: a Dataframe, each row contains the number of POIs in 
             each tagged group for a location pair
    """
    df = pd.DataFrame(columns=['aeroway', 'amenity', 'building', 'highway',
                               'landuse', 'public_transport', 'historic',
                               'leisure', 'shop', 'tourism', 'healthcare', 'sport'])

    for idx in tqdm(range(len(geo_list))):
        latitude, longitude = geo_list[idx]
        north, south, west, east = get_bounding_box(
            latitude, longitude, box_width, box_height)
        pois = get_pois(north, south, west, east)
        for k, v in count_pois_by_tag(pois).items():
            df.at[idx, k] = v
    return df.astype('float64')


def get_geo_list(n=10, seed=42):
    """ Get n latitude-longitude pairs.
    :param n: the number of pairs
    :param seed: the NumPy random seed, make sure to reproduce the same results
    :return: a list of n latitude-longitude pairs
    """
    np.random.seed(seed)
    # select latitude range [51, 52] N, longitude range [-0.5, 0.5] E
    lat_r = np.random.random(n)
    lon_r = np.random.random(n)
    return [[51 + i, -0.5 + j] for i, j in zip(lat_r, lon_r)]


def df_as_histogram(df):
    """ Visualise data using histograms.
    :param df: a DataFrame
    :return: None
    """
    df = df.astype('float64')
    df.hist(figsize=(12, 10))
    plt.show()


def feature_corr_heatmap(df):
    """  Visualise feature correlation with a heatmap.
    :param df: a DataFrame
    :return: None
    """
    corr = df.corr().fillna(0)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr, square=True)
    ax.set_title("Visualise correlation matrix")
    plt.show()


def df_with_PCA(corr):
    """ Find optimal number of PCA components.
    :param corr: PCA correlation matrix
    :return: None
    """
    eigVal, eigVec = eig(corr)
    pca = PCA()
    k = 3
    princ_compt = pca.fit_transform(corr)
    cum_exp_var = np.cumsum(pca.explained_variance_ratio_)
    print(
        f"With {k - 1} principal components, we can capture {round(cum_exp_var[k - 1],3)} variance.")
    print(
        f"With {k} principal components, we can capture {round(cum_exp_var[k],3)} variance.")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].bar(np.arange(len(eigVal)), eigVal)
    ax[0].get_xaxis().set_visible(False)
    ax[0].set_title("Sorted Eigenvalues")

    ax[1].plot(range(len(cum_exp_var)), cum_exp_var)
    ax[1].axvline(k, c='grey', linestyle='--',
                  label=f'{k} Principal Components')
    ax[1].set_title(
        "Explained variance against number of principal components")
    ax[1].set_xlabel("Number of principal components")
    ax[1].set_ylabel("Cumulative explained variance")
    plt.legend()
    plt.show()


def plot_PCA_2d(corr):
    """ Apply PCA with 2 components, generate a 2D plot.
    :param corr: PCA correlation matrix
    :return: None
    """
    # Apply PAC with 2 components
    pac = PCA(n_components=2)
    principal_components = pac.fit_transform(corr)
    pac_df = pd.DataFrame(data=principal_components, columns=['pc_1', 'pc_2'])
    # Visualise points in 2D
    fig = plt.figure(figsize=(6, 5))
    ax = plt.axes()
    cm = plt.cm.get_cmap('tab20').colors
    for idx, txt in enumerate(corr.columns):
        ax.scatter(pac_df['pc_1'][idx], pac_df['pc_2']
                   [idx], label=txt, color=cm[idx], s=45)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05),
              ncol=3, fancybox=True)
    ax.set_title("Visualise features in 2D")
    plt.show()


def plot_PCA_3d(corr):
    """ Apply PCA with 3 components, generate an interactive 3D plot.
    :param corr: PCA correlation matrix
    :return: None
    """
    pca = PCA(n_components=3)
    principal_components = pca.fit_transform(corr)
    pca_df = pd.DataFrame(data=principal_components,
                          columns=['pc_1', 'pc_2', 'pc_3'])

    # Interactive 3d plot
    elevation_slider = widgets.IntSlider(min=10, max=100, step=1, value=10)
    azim_slider = widgets.IntSlider(min=0, max=360, step=1, value=30)

    def plot3d(elevation, azim):
        cm = plt.cm.get_cmap('tab20').colors
        fig = plt.figure(figsize=(8, 7))
        ax = plt.axes(projection='3d')
        for idx, txt in enumerate(corr.columns):
            ax.scatter3D(pca_df['pc_1'][idx], pca_df['pc_2'][idx],
                         pca_df['pc_3'][idx], label=txt, color=cm[idx], s=45)
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, 0.98),
                  ncol=3, fancybox=True)
        ax.set_title("Visualise features in 3D")
        ax.view_init(elevation, azim)
        plt.show()

    def view(elevation, azim):
        display(plot3d(elevation, azim))

    _ = interact(view, elevation=elevation_slider, azim=azim_slider)


def data():
    """Load the data from access and ensure missing values are correctly encoded as well as indices correct, column names informative, date and times correctly formatted. Return a structured data structure such as a data frame."""
    # get a database connection
    conn = config_credentials()
    # load price data
    config_price_data(conn)


def view(geo_list):
    """Provide a view of the data that allows the user to verify some aspect of its quality."""
    df = get_POIs_for_list(geo_list, box_width=0.005, box_height=0.005)
    # Visualise data using histograms
    df_as_histogram(df)

    # Visualise feature correlation with a heatmap
    feature_corr_heatmap(df)

    # Find optimal number of PCA components
    corr = df.corr().fillna(0)
    df_with_PCA(corr)
    plot_PCA_3d(corr)
    plot_PCA_2d(corr)
