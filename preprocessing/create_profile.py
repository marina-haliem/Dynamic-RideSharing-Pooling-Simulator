import numpy as np
import pandas as pd
import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from db import engine, Session
from common.time_utils import get_local_datetime
from config.settings import MIN_LAT, MIN_LON, DELTA_LAT, DELTA_LON, MAP_WIDTH, MAP_HEIGHT,\
    GLOBAL_STATE_UPDATE_CYCLE, DESTINATION_PROFILE_TEMPORAL_AGGREGATION, DESTINATION_PROFILE_SPATIAL_AGGREGATION
# from dqn_agent.settings import FLAGS

# def create_predicted_demand(source_table, profile_table, variance_rate=None):
#     query = "SELECT * FROM {}".format(source_table)
#     df = pd.read_sql(query, engine, index_col="id")
#     print("# of rows {}".format(len(df)))
#
#     df["datetime_obj"] = df.request_datetime.apply(lambda x: get_local_datetime(x))
#     df["dayofweek"] = df.datetime_obj.apply(lambda x: x.weekday())
#     df["hour"] = df.datetime_obj.apply(lambda x: x.hour)
#     df['day'] = df.datetime_obj.apply(lambda x: x.day)
#     df['month'] = df.datetime_obj.apply(lambda x: x.month)
#     df['x'] = ((df['origin_lon'] - MIN_LON) / DELTA_LON).astype(int)
#     df['y'] = ((df['origin_lat'] - MIN_LAT) / DELTA_LAT).astype(int)
#
#     df = df.groupby(['month', 'day', 'hour', 'x', 'y']).size()
#     if variance_rate:
#         df += np.maximum(-df.values, np.minimum(df.values, np.random.normal(0, np.sqrt(variance_rate * df.values))))
#     df.name = 'demand'
#     df = df.reset_index()
#
#     drop_table = """
#     DROP TABLE IF EXISTS {};
#     """.format(profile_table)
#     Session.execute(drop_table)
#     Session.commit()
#     df.to_sql(profile_table, engine, flavor=None, schema=None, if_exists='fail',
#                index=True, index_label=None, chunksize=None, dtype=None)
#
#     create_index = """
#     CREATE INDEX index_{table} ON {table} (month, day, hour);
#     """.format(table=profile_table)
#     Session.execute(create_index)
#     Session.commit()

def create_demand_profile(df, profile_table, n_weeks):
    df["dayofweek"] = df.datetime_obj.apply(lambda x: x.weekday())
    df["hour"] = df.datetime_obj.apply(lambda x: x.hour)
    df['x'] = ((df['origin_lon'] - MIN_LON) / DELTA_LON).astype(int)
    df['y'] = ((df['origin_lat'] - MIN_LAT) / DELTA_LAT).astype(int)

    profile_df = df.groupby(['dayofweek', 'hour', 'x', 'y']).size() / float(n_weeks)
    profile_df.name = 'demand'
    drop_table = """
    DROP TABLE IF EXISTS {};
    """.format(profile_table)
    Session.execute(drop_table)
    Session.commit()
    profile_df.reset_index().to_sql(profile_table, engine, schema=None, if_exists='fail',
               index=True, index_label=None, chunksize=None, dtype=None)
    create_index = """
    CREATE INDEX index_{table} ON {table} (dayofweek, hour);
    """.format(table=profile_table)
    Session.execute(create_index)
    Session.commit()


def create_od_profile(df, profile_table, n_weeks):
    hours_bin = DESTINATION_PROFILE_TEMPORAL_AGGREGATION
    n_agg = DESTINATION_PROFILE_SPATIAL_AGGREGATION
    df["dayofweek"] = df.datetime_obj.apply(lambda x: x.weekday())
    df["hours_bin"] = (df.datetime_obj.apply(lambda x: x.hour) / hours_bin).astype(int)
    df['origin_x'] = ((df['origin_lon'] - MIN_LON) / (DELTA_LON * n_agg)).astype(int)
    df['origin_y'] = ((df['origin_lat'] - MIN_LAT) / (DELTA_LAT * n_agg)).astype(int)
    df['destination_x'] = ((df['destination_lon'] - MIN_LON) / (DELTA_LON * n_agg)).astype(int)
    df['destination_y'] = ((df['destination_lat'] - MIN_LAT) / (DELTA_LAT * n_agg)).astype(int)

    od_df = df.groupby(['dayofweek', 'hours_bin', 'origin_x', 'origin_y', 'destination_x', 'destination_y']
                        ).trip_time.agg(['count', 'mean'])
    od_df = od_df.rename(columns={'count' : 'demand', 'mean' : 'trip_time'}).reset_index()
    drop_table = """
    DROP TABLE IF EXISTS {};
    """.format(profile_table)
    Session.execute(drop_table)
    Session.commit()
    od_df.to_sql(profile_table, engine, schema=None, if_exists='fail',
               index=True, index_label=None, chunksize=None, dtype=None)

    create_index = """
    CREATE INDEX index_{table} ON {table} (dayofweek, hours_bin);
    """.format(table=profile_table)
    Session.execute(create_index)
    Session.commit()


def create_latest_demand(source_table, latest_table):
    query = "SELECT * FROM {}".format(source_table)
    df = pd.read_sql(query, engine, index_col="id")
    print("# of rows {}".format(len(df)))

    unit_min = GLOBAL_STATE_UPDATE_CYCLE / 60
    df["t"] = (df.request_datetime / unit_min).astype(int) * unit_min
    df['x'] = ((df['origin_lon'] - MIN_LON) / DELTA_LON).astype(int)
    df['y'] = ((df['origin_lat'] - MIN_LAT) / DELTA_LAT).astype(int)

    latest_df = df.groupby(['t', 'x', 'y']).size()
    latest_df.name = 'demand'
    drop_table = """
    DROP TABLE IF EXISTS {};
    """.format(latest_table)
    Session.execute(drop_table)
    Session.commit()
    latest_df.reset_index().to_sql(latest_table, engine, schema=None, if_exists='fail',
               index=True, index_label=None, chunksize=None, dtype=None)
    create_index = """
    CREATE INDEX index_{table} ON {table} (t);
    """.format(table=latest_table)
    Session.execute(create_index)
    Session.commit()

def create_training_dataset(df, n_weeks):
    t_start = df.request_datetime.min()
    t_end = t_start + 3600 * 24 * 7 * n_weeks
    df = df[(df.request_datetime >= t_start) & (df.request_datetime < t_end)]
    df["datetime_obj"] = df.request_datetime.apply(lambda x: get_local_datetime(x))
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help = "path to the training data file")
    args = parser.parse_args()

    df = pd.read_csv(args.input_file, index_col='id')
    print("load {} rows".format(len(df)))

    n_weeks = 4
    df = create_training_dataset(df, n_weeks)
    print("created training dataset")

    create_demand_profile(df, "demand_profile", n_weeks)
    print("created demand_profile table")

    create_od_profile(df, "od_profile", n_weeks)
    print("created od_profile table")

    create_latest_demand("request_backlog", "demand_latest")
    print("created demand_latest table")
    # t_start = 1462075200 # 160501
    # sim_datetime = 1464753600 # 160601
