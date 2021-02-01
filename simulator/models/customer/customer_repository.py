import pandas as pd


class CustomerRepository(object):
    # State Vector for customer
    request_column_names = [
        'id',
        'request_datetime',
        'trip_time',
        'origin_lon',
        'origin_lat',
        'destination_lon',
        'destination_lat',
        'fare'
    ]

    customers = {}
    new_customers = []


    @classmethod
    def init(cls):
        cls.customers = {}
        cls.new_customers = []

    @classmethod
    # Creating customer dictionary with their associated IDs
    def update_customers(cls, customers):
        cls.new_customers = customers
        for customer in customers:
            cls.customers[customer.request.id] = customer

    @classmethod
    def get(cls, customer_id):
        return cls.customers.get(customer_id, None)

    @classmethod
    def get_all(cls):
        return list(cls.customers.values())

    @classmethod
    # Get the new requests asociated with the new customers list
    def get_new_requests(cls):
        # print("C: ", len(cls.new_customers))
        requests = [customer.get_request() for customer in cls.new_customers]
        # print(cls.request_column_names)
        #  Creating a DF with all the request columns
        df = pd.DataFrame.from_records(requests, columns=cls.request_column_names)
        # print(df.columns.values)
        return df

    @classmethod
    #  Delete a specfic customer ID
    def delete(cls, customer_id):
        cls.customers.pop(customer_id)

