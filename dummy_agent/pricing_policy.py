import numpy as np
from common.geoutils import great_circle_distance
from common import geoutils, mesh
from itertools import islice

class PricingPolicy(object):
# Vehicle utility function goes here
    def propose_price(self, vehicle, price, request):
        if len(vehicle.q_action_dict) == 0:
            # print("NOT DISPATCHED BEFORE!")
            return price
        else:
            # print(self.q_action_dict)
            r_lon, r_lat = request.origin_lon, request.origin_lat
            r_x, r_y = mesh.convert_lonlat_to_xy(r_lon, r_lat)
            # print("ID: ", self.state.id)
            # print("Request: ", r_x, r_y)
            sorted_q = {k: v for k, v in sorted(vehicle.q_action_dict.items(), key=lambda item: item[1], reverse=True)}
            # print("Sorted: ", sorted_q)
            # print("Epsilon: ", self.epsilon, len(self.q_action_dict))
            filtered_q = list(islice(sorted_q, vehicle.epsilon))
            # filtered_q = dict(filtered_q)
            # print("Filtered: ", filtered_q)
            if (r_x, r_y) in filtered_q:
                # print("Here!")
                return price

            if (r_x, r_y) in vehicle.q_action_dict.keys():
                # print("Exists!")
                # req_q = self.q_action_dict.get((r_x,r_y))
                rank = 0
                index = 0
                for (kx, ky), v in sorted_q.items():
                    # print((kx,ky), (r_x, r_y))
                    if (kx, ky) == (r_x, r_y):
                        rank = index
                    index += 1
            else:
                # print("Does not exist!")
                dist_list = {}
                for (kx,ky), v in vehicle.q_action_dict.items():
                    k_lon, k_lat = mesh.convert_xy_to_lonlat(kx, ky)
                    dist = great_circle_distance(r_lat, r_lon, k_lat, k_lon)
                    dist_list[(kx, ky)] = dist

                # print("D: ", dist_list)
                min_dist = np.min(list(dist_list.values()))
                (min_x, min_y) = list(dist_list.keys())[list(dist_list.values()).index(min_dist)]
                req_q = vehicle.q_action_dict.get((min_x, min_y))
                # print(min_dist, (min_x,min_y), req_q)

                rank = 0
                index = 0
                for (kx,ky), v in sorted_q.items():
                    if (kx,ky) == (min_x, min_y):
                        rank = index
                    index +=1
            # print("Rank: ", rank, len(self.q_action_dict))
            rank = 1 - (rank/len(vehicle.q_action_dict))
            # print("Rank_: ", rank, (self.state.driver_base_per_trip/100))
            return price + (rank*0.5*(vehicle.state.driver_base_per_trip/100))