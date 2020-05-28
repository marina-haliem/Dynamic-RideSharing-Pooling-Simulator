from simulator.models.customer.customer import Customer
from db import Session
# import request

query = """
  SELECT *
  FROM {table}
  WHERE request_datetime >= {t1} and request_datetime < {t2};
"""


class DemandGenerator(object):
    def __init__(self, use_pattern=False):
        if use_pattern:
            self.table = "request_pattern"
        else:
            self.table = "request_backlog"


    def generate(self, current_time, timestep):
        try:
            # List of requests within a certain timeframe
            requests = list(Session.execute(query.format(table=self.table, t1=current_time, t2=current_time + timestep)))
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


