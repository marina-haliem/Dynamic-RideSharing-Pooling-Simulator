import pandas as pd
import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from db import engine, Session

def create_request_backlog(input_file_path, table_name):
    df = pd.read_csv(input_file_path, index_col='id')
    print("load {} rows".format(len(df)))
    # df.index.name = 'id'
    drop_table = """
    DROP TABLE IF EXISTS {};
    """.format(table_name)
    Session.execute(drop_table)
    Session.commit()
    df.to_sql(table_name, engine, schema=None, if_exists='fail',
               index=True, index_label=None, chunksize=None, dtype=None)

    create_index = """
    CREATE INDEX index_request ON {} (request_datetime);
    """.format(table_name)
    Session.execute(create_index)
    Session.commit()

    print("complete db insert")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help = "csv data path of ride requests backlog")
    args = parser.parse_args()
    create_request_backlog(args.input_file, "request_backlog")
