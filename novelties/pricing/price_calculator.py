from common import geoutils

# def recalculate_price_per_customer(vehicle_plan, pickup_flags, pick_drop_ids):
#     cust_price_time = dict()
#     for id in pick_drop_ids:
#         if id not in cust_price_time.keys():




def calculate_price(dist, wait_time, mileage, price_per_travel_m, price_per_wait_min, gas_price, driver_base):
    # price = trip_time*self.price_per_travel_min + wait_time*self.price_per_wait_min + dist*mileage
    # print(dist, price_per_travel_m, mileage, wait_time, price_per_wait_min)
    # We can use the fare associated with each request as the base fare
    # print("Dist: ", dist, "Dist Cost: ", (dist*price_per_travel_m), "Gas Price: ", (dist*(gas_price/(mileage*1000.0))), "Base: ", price_per_travel_m, "Wait time: ", wait_time, "Wait Cost: ",(price_per_wait_min/wait_time))
    if wait_time <= 0:
        wait_time = 3600
    if wait_time > 99999:
        wait_time = 99999999
    if dist < 10:
        dist = 1000

    if((dist*price_per_travel_m) + (dist*(gas_price/(mileage*1000.0))) < (price_per_wait_min*wait_time)):
        print("ERR", )
        print("Dist: ", dist, "Dist Cost: ", (dist*price_per_travel_m), "Gas Price: ", (dist*(gas_price/(mileage*1000.0))), "Base: ", price_per_travel_m, "Wait time: ", wait_time, "Wait Cost: ",(price_per_wait_min*wait_time))

    price = (dist*price_per_travel_m) + (dist*(gas_price/(mileage*1000.0))) - (price_per_wait_min*wait_time)
    # print("Final Price: ", round(driver_base+price, 2)/100.0)
    return round(driver_base+price,2)/100.0


# SERVICE_REWARD = RIDE_REWARD*num_pass + TRIP_REWARD * trip_time - WAIT_COST * wait_time
# RIDE_REWARD = 10.0 #5,6,7,8,9,10
# TRIP_REWARD = 1.0
# WAIT_COST = 0.05
# HOP_REWARD = 3.5
# MIN_TRIPTIME = 1.0 # in meters
# ASSIGNMENT_SPEED = 50 # km/h (grand circle distance)

