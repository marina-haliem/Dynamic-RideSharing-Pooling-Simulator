import pandas as pd
import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from simulator.services.osrm_engine import OSRMEngine
from preprocessing.preprocess_nyc_dataset import extract_bounding_box, BOUNDING_BOX

def create_snapped_trips(df, engine, batch_size=10000):
    mm_origins = []
    mm_destins = []
    for i in range(0, len(df), batch_size):
        print("n: {}".format(i))
        df_ = df.iloc[i : i + batch_size]
        origins = [(lat, lon) for lat, lon in zip(df_.origin_lat, df_.origin_lon)]
        mm_origins += [loc for loc, _ in engine.nearest_road(origins)]
        destins = [(lat, lon) for lat, lon in zip(df_.destination_lat, df_.destination_lon)]
        mm_destins += [loc for loc, _ in engine.nearest_road(destins)]

    df[['origin_lon', 'origin_lat']] = mm_origins
    df[['destination_lon', 'destination_lat']] = mm_destins
    df = extract_bounding_box(df, BOUNDING_BOX)
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="input csv file path of ride requests to be map match")
    parser.add_argument("output_file", help="output csv file path")
    args = parser.parse_args()
    engine = OSRMEngine()
    df = pd.read_csv(args.input_file, index_col='id')
    print("load {} rows".format(len(df)))
    df = create_snapped_trips(df, engine)
    print("extract {} rows".format(len(df)))
    df.to_csv(args.output_file)