from .config import *
import osmnx as ox
import matplotlib.pyplot as plt
import mlai
import mlai.plot as plot
import numpy as np
import pandas as pd
import access

"""Place commands in this file to assess the data you have downloaded. How are missing values encoded, how are outliers encoded? What do columns represent, makes rure they are correctly labeled. How is the data indexed. Crete visualisation routines to assess the data (e.g. in bokeh). Ensure that date formats are correct and correctly timezoned."""


def get_bounding_box(latitude, longitude, box_width=0.005, box_height=0.005):
    """ Construct a bounding box for the point specified in latitude and longitude.
    :param latitude: the latitude of the point
    :param longitude: the longitude of the point
    :param box_width: the width of the bounding box
    :param box_height: the height of the bounding box
    :return: the north, south, west and east coordinates for the bounding box
    """
    north = latitude + box_height/2
    south = latitude - box_height/2
    west = longitude - box_width/2
    east = longitude + box_width/2
    return north, south, west, east


def get_pois(north, south, west, east):
    """ Retrieve the point of interests within the bounding box.
    :param north: the north coordinate of the bounding box
    :param south: the south coordinate of the bounding box
    :param west: the west coordinate of the bounding box
    :param east: the east coordinate of the bounding box
    :return: POIs in a GeoPandas Dataframe
    """

    # Define tags from a list of available POIs: https://wiki.openstreetmap.org/wiki/Map_features
    tags = {"aeroway": True,
            "amenity": True,
            "building": True,
            "highway": True,
            "landuse": True,
            "public_transport": True,
            "historic": True,
            "leisure": True,
            "shop": True,
            "tourism": True,
            "healthcare": True,
            "sport": True}

    # Retrieve POIs
    pois = ox.geometries_from_bbox(north, south, east, west, tags)

    return pois


def get_pois_by_key(pois, key):
    """ Extract relevant POIS with a key.
    :param pois: POIs in a GeoPandas Dataframe
    :param key: a column name of the POI
    :return: POIs in a GeoPandas DataFrame with entries filtered by key
    """
    if key not in pois.columns:
        return None

    pois_by_key = pois[pois[key].notnull()]
    tags = ["name",
            "geometry",
            "addr:city",
            "addr:postcode"]
    tags.append(key)

    present_keys = [tag for tag in tags if tag in pois.columns]
    return pois_by_key[present_keys]


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


def compare_places_pois_by_tag(place_names, locations, box_width, box_height, tags):
    """ Plot POIs for each tag and each place.
    :param place_names: a list of strings, each string is a common name of a location
    :param locations: a list of latitude-longitude pairs, one for each location
    :param box_width: the width of the bounding box
    :param box_height: the height of the bounding box
    :param tags: a list of POI groups
    :return: one plot per tag per place
    """
    fig, ax = plt.subplots(len(tags), len(place_names),
                           figsize=(len(place_names) * 4, len(tags) * 4))
    with tqdm(total=len(place_names) * len(tags)) as pbar:
        for i, p in enumerate(place_names):
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
    """Iterate through the list of latitude-longitude pairs, count the number
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
    """Get n latitude-longitude pairs.
    :param n: the number of pairs
    :param seed: the NumPy random seed, make sure to reproduce the same results
    :return: a list of n latitude-longitude pairs
    """
    np.random.seed(seed)
    # select latitude range [51, 52] N, longitude range [-0.5, 0.5] E
    lat_r = np.random.random(n)
    lon_r = np.random.random(n)
    return [[51 + i, -0.5 + j] for i, j in zip(lat_r, lon_r)]


def data():
    """Load the data from access and ensure missing values are correctly encoded as well as indices correct, column names informative, date and times correctly formatted. Return a structured data structure such as a data frame."""
    df = access.data()
    raise NotImplementedError


def query(data):
    """Request user input for some aspect of the data."""
    raise NotImplementedError


def view(data):
    """Provide a view of the data that allows the user to verify some aspect of its quality."""
    raise NotImplementedError


def labelled(data):
    """Provide a labelled set of data ready for supervised learning."""
    raise NotImplementedError
