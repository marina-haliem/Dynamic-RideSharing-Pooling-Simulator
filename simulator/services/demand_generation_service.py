from sqlalchemy.sql import text

from simulator.models.customer.customer import Customer
from db import Session

# import request

# query = (
#     "SELECT * FROM {table} WHERE request_datetime >= {t1} and request_datetime < {t2};"
# )


class DemandGenerator(object):
    def __init__(self, use_pattern=False):
        if use_pattern:
            self.table = "request_pattern"
        else:
            self.table = "request_backlog"

    # Generate demand from the database according to given time
    def generate(self, current_time, timestep):
        try:
            # List of requests within a certain timeframe
            t1 = current_time
            t2 = current_time + timestep

            query = text(
                f"SELECT * FROM {self.table} WHERE request_datetime >= {t1} and request_datetime < {t2};"
            )
            requests = list(Session.execute(query))
            # List of customers associated with each request
            customers = [Customer(request) for request in requests]
            # for r in requests:
            #     print("Iterating R: ", r)
            # print("Cust: ", len(customers), requests)
        except:
            Session.rollback()
            raise
        finally:
            Session.remove()
        return customers
