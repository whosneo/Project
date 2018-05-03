from datetime import datetime

from eviltransform import *
from geopy.distance import great_circle


def parse_dates(x):
    return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')


def geo_distance(point1, point2):
    distance = great_circle(point1, point2).meters
    return distance


def transform_save(df, file_name):
    for index, point in df.iterrows():
        current_point = df.loc[index]
        gcj_lat, gcj_lng = wgs2gcj(current_point['latitude'], current_point['longitude'])
        df.at[index, 'latitude'] = gcj_lat
        df.at[index, 'longitude'] = gcj_lng
    df.to_csv(file_name, index=False, header=False)
