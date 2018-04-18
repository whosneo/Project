from datetime import datetime

from geopy.distance import great_circle


def parse_dates(x):
    return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')


def geo_distance(point1, point2):
    distance = great_circle(point1, point2).meters
    return distance
