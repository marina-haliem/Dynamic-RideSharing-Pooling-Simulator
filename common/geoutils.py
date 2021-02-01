# -*- coding: utf-8 -*-

import numpy as np

R = 6371000  # earth's radius in meters.

def great_circle_distance(start_lat, start_lon, end_lat, end_lon):
    """Distance in meters

    """
    start_lat, start_lon, end_lat, end_lon = map(np.deg2rad, [start_lat, start_lon, end_lat, end_lon])
    x = (end_lon - start_lon) * np.cos(0.5 * (start_lat + end_lat))
    y = end_lat - start_lat
    return R * np.sqrt(x ** 2 + y ** 2)


def bearing(start_lat, start_lon, end_lat, end_lon):
    """Bearing in radians

    """
    start_lat, start_lon, end_lat, end_lon = map(np.deg2rad, [start_lat, start_lon, end_lat, end_lon])
    del_lon = end_lon - start_lon
    num = np.sin(del_lon) * np.cos(end_lat)
    den = np.cos(start_lat) * np.sin(end_lat) \
          - np.sin(start_lat) * np.cos(end_lat) * np.cos(del_lon)
    return np.arctan2(num, den)


def end_location(start_lat, start_lon, distance_in_meter, bearing):
    """End point latitude and longitude.

    arguments
    ---------
    start_lat, start_lon: array_like
        strating point latitudes and longitudes.
    distance_in_meter: array_like
        distance from the starting point to the desired end point.
    bearing: array_like
        angle in radians with the true north.

    returns
    -------
    end_lat, end_lon: ndarray or scalar
        The desired ending position latitude and longitude.

    """

    start_lat, start_lon = map(np.deg2rad, [start_lat, start_lon])
    alpha = np.asarray(distance_in_meter) / R

    lat = np.arcsin(np.sin(start_lat) * np.cos(alpha) \
                    + np.cos(start_lat) * np.sin(alpha) * np.cos(bearing))

    num = np.sin(bearing) * np.sin(alpha) * np.cos(start_lat)
    den = np.cos(alpha) - np.sin(start_lat) * np.sin(lat)

    lon = start_lon + np.arctan2(num, den)

    return (np.rad2deg(lat), np.rad2deg(lon))

