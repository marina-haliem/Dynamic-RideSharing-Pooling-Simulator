import numpy as np
from novelties import status_codes
from common import mesh
from common.geoutils import great_circle_distance
from collections import defaultdict
from config.settings import MAP_WIDTH, MAP_HEIGHT
from simulator.services.routing_service import RoutingEngine
import pandas as pd
from simulator.models.vehicle.vehicle_repository import VehicleRepository

class MatchingPolicy(object):
    def match(self, current_time, vehicles, requests):
        return []

    def match_RS(self, current_time, vehicles, requests):
        return []

    def find_available_vehicles(self, vehicles):
        idle_vehicles = vehicles[
            (((vehicles.status == status_codes.V_IDLE) |
            (vehicles.status == status_codes.V_CRUISING)) &
            (vehicles.idle_duration > 0))
            # | (vehicles.status == status_codes.V_OCCUPIED)
        ]
        v_list = []
        for index, v in idle_vehicles.iterrows():
            if v['current_capacity'] < v['max_capacity']:
                v_list.append(v)
        return pd.DataFrame(v_list)

    # Craeting matching dictionary assciated with each vehicle ID
    def create_matching_dict(self, vehicle_id, customer_id, duration, distance):
        match_dict = {}
        match_dict["vehicle_id"] = vehicle_id
        match_dict["customer_id"] = customer_id
        match_dict["duration"] = duration
        match_dict["distance"] = distance
        return match_dict

    def find_available_vehicles_RS(self, vehicles):
        idle_vehicles = vehicles[
            (((vehicles.status == status_codes.V_IDLE) |
            (vehicles.status == status_codes.V_CRUISING)) &
            (vehicles.idle_duration > 0))
            | ((vehicles.status == status_codes.V_OCCUPIED) & vehicles.accept_new_request)
        ]
        # print(type(idle_vehicles))
        # print(idle_vehicles["status"])
        v_list = []
        v_capacity = []
        for index, v in idle_vehicles.iterrows():
            if v['current_capacity'] < v['max_capacity']:
                v_list.append(v)
                v_capacity.append(v['max_capacity'] - v['current_capacity'])
                # print(v["status"])
        return pd.DataFrame(v_list), v_capacity


class RoughMatchingPolicy(MatchingPolicy):
    def __init__(self, reject_distance=5000):
        self.reject_distance = reject_distance  # in meters

    # Matching requests to the nearest available vehicle
    def match(self, current_time, vehicles, requests):
        assignments = []
        vehicles = self.find_available_vehicles(vehicles)
        n_vehicles = len(vehicles)
        if n_vehicles == 0:
            return assignments
        # List of distances for all available vehicles to requests' origin points
        d = great_circle_distance(vehicles.lat.values, vehicles.lon.values,
                                  requests.origin_lat.values[:, None], requests.origin_lon.values[:, None])

        for ridx, request_id in enumerate(requests.index):
            vidx = d[ridx].argmin()     # Retrieving the min distance (nearest vehicle to request)
            # Check if it is within the acceptable range of travelling distance
            if d[ridx, vidx] < self.reject_distance:
                vehicle_id = vehicles.index[vidx]
                duration = d[ridx, vidx] / 8.0
                distance = d[ridx, vidx]
                assignments.append(self.create_matching_dict(vehicle_id, request_id, duration, distance))
                d[:, vidx] = float('inf')
            else:
                continue
            if len(assignments) == n_vehicles:
                return assignments
        return assignments


class GreedyMatchingPolicy(MatchingPolicy):
    def __init__(self, reject_distance=5000):
        self.reject_distance = reject_distance  # meters
        self.reject_wait_time = 15 * 60         # seconds
        self.k = 3                              # the number of mesh to aggregate
        self.unit_length = 500                  # mesh size in meters
        self.max_locations = 40                 # max number of origin/destination points
        self.routing_engine = RoutingEngine.create_engine()


    def get_coord(self, lon, lat):
        x, y = mesh.convert_lonlat_to_xy(lon, lat)
        return (int(x / self.k), int(y / self.k))

    def coord_iter(self):
        for x in range(int(MAP_WIDTH / self.k)):
            for y in range(int(MAP_HEIGHT / self.k)):
                yield (x, y)

    # Candidate Vehicle IDs from the mesh
    def find_candidates(self, coord, n_requests, V, reject_range):
        x, y = coord
        candidate_vids = V[(x, y)][:]
        for r in range(1, reject_range):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    r_2 = dx ** 2 + dy ** 2
                    if r ** 2 <= r_2 and r_2 < (r + 1) ** 2:
                        candidate_vids += V[(x + dx, y + dy)][:]
                if len(candidate_vids) > n_requests * 2:
                    break
        return candidate_vids

    # Returns list of assignments
    def assign_nearest_vehicle(self, request_ids, vehicle_ids, T, dist):
        assignments = []
        for ri, rid in enumerate(request_ids):
            if len(assignments) >= len(vehicle_ids):
                break
            # Reuturns the min distance
            vi = T[ri].argmin()
            di = dist[ri].argmin()
            tt = T[ri, vi]
            dd = dist[ri, di]
            # print("Chosen t: ", tt)
            # print("Chosen D: ", dd)
            if tt > self.reject_wait_time:
                continue
            vid = vehicle_ids[vi]

            assignments.append((vid, rid, tt, dd))
            T[:, vi] = float('inf')
        return assignments

    # Return list of candidate vehicle in order of the nearest to the request
    def filter_candidates(self, vehicles, requests):
        d = great_circle_distance(vehicles.lat.values, vehicles.lon.values,
                                  requests.origin_lat.mean(), requests.origin_lon.mean())

        within_limit_distance = d < self.reject_distance + self.unit_length * (self.k - 1)
        candidates = vehicles.index[within_limit_distance]
        d = d[within_limit_distance]
        return candidates[np.argsort(d)[:2 * len(requests) + 1]].tolist()

    def assign_nearest_vehicle_RideShare(self, request_ids, all_target_ridx, requestAssigned, vehicle_ids, vehicle_idx,
                                         T, TRR, TDD, dist, d_RR, d_DD):
        assignments = []
        # for ri, rid in enumerate(request_ids):
        # vi = T[ri,vehicle_idx].argmin()
        # tt = T[ri,vehicle_idx][vi]
        # for rid in request_ids:
        # print(len(request_ids), type(request_ids))
        # print(len(all_target_ridx), len(vehicle_ids), len(vehicle_idx))

        # for ri, rid in enumerate(request_ids):
        for rid in request_ids:
            ri = all_target_ridx[rid]
            # For this request ri, pick the min from all candidate vehicles
            vi = T[ri, vehicle_idx].argmin()
            tt = T[ri, vehicle_idx][vi]

            # print("D: ", dist, "shape: ", dist.shape)
            # print("Ri:", ri,  "Vi: ", vehicle_idx, len(vehicle_idx))
            # print("Filtered: ", dist[ri, vehicle_idx])
            # print(type(dist))
            di = dist[ri, vehicle_idx].argmin()
            dd = dist[ri, vehicle_idx][di]
            # else:
            # dd = dist

            # print(T[ri])
            # print(vehicle_idx)
            # print(T[ri,vehicle_idx])
            # print("RTT","Ri",ri,"Vi",vi,vehicle_ids,tt)
            if tt > self.reject_wait_time:
                continue

            vid = vehicle_ids[vi]
            vi = vehicle_idx[vi]

            # print("RTT","VID",vid,"Vix",vi)

            vehicle = VehicleRepository.get(vid)
            # print("vid: ", vehicle.get_id(), "tmp: ", vehicle.tmp_capacity, "Real:  ",
            #       vehicle.state.current_capacity, "Max: ", vehicle.state.max_capacity)
            if vehicle.tmp_capacity < vehicle.state.max_capacity:
                if len(requestAssigned[vid]) == 0:
                    requestAssigned[vid].append(rid)
                    vehicle.tmp_capacity += 1
                    # vehicle.state.current_capacity += 1
                    # print("1: ", dd)
                    assignments.append((vid, rid, tt, dd))
                    # print("TT",T[:, vi])
                    # print("TRR",TRR[:, ri])
                    T[:, vi] = T[:, vi] + TRR[:, ri]
                    dist[:, vi] = d_RR[:, ri]
                    # T[ri, vi] = T[ri, vi] + tt
                    # print("TTA",T[:, vi])
                else:
                    ll = list(requestAssigned[vid])
                    # print("LL",ll)
                    ll.append(rid)
                    # print("RS",requestAssigned[vid])
                    ll_x = [all_target_ridx[r] for r in ll]
                    maxdd = TDD[ll_x][:, ll_x].max()
                    # extra_d = d_DD[ll_x][:, ll_x].max()
                    # print("LLAfter",ll, "Max Time",maxdd,flush=True)
                    if maxdd <= self.reject_wait_time:
                        requestAssigned[vid].append(rid)
                        vehicle.tmp_capacity += 1
                        assignments.append((vid, rid, tt, dd))
                        # print("Nxt: ", dd)
                        # print("TT",T[:, vi])
                        T[:, vi] = T[:, vi] + TRR[:, ri]
                        dist[:, vi] = d_RR[:, ri]
                        # T[ri, vi] = T[ri, vi] + tt
                        # print("TTA",T[:, vi])
            else:
                T[:, vi] = float('inf')
                dist[:, vi] = float('inf')
        return assignments

    def match(self, current_time, vehicles, requests):
        # od_pairs = []
        match_list = []
        vehicles = self.find_available_vehicles(vehicles)
        n_vehicles = len(vehicles)
        if n_vehicles == 0:
            return match_list

        v_latlon = vehicles[["lat", "lon"]]
        V = defaultdict(list)
        vid2coord = {}
        for vid, row in v_latlon.iterrows():
            coord = self.get_coord(row.lon, row.lat)
            vid2coord[vid] = coord
            V[coord].append(vid)

        r_latlon = requests[["origin_lat", "origin_lon"]]
        R = defaultdict(list)
        for rid, row in r_latlon.iterrows():
            coord = self.get_coord(row.origin_lon, row.origin_lat)
            R[coord].append(rid)

        reject_range = int(self.reject_distance / self.unit_length / self.k) + 1
        for coord in self.coord_iter():
            if not R[coord]:
                continue

            for i in range(int(np.ceil(len(R[coord]) / self.max_locations))):
                #
                target_rids = R[coord][i * self.max_locations : (i + 1) * self.max_locations]

                candidate_vids = self.find_candidates(coord, len(target_rids), V, reject_range)
                if len(candidate_vids) == 0:
                    continue

                target_latlon = r_latlon.loc[target_rids]
                candidate_vids = self.filter_candidates(v_latlon.loc[candidate_vids], target_latlon)
                if len(candidate_vids) == 0:
                    continue
                candidate_latlon = v_latlon.loc[candidate_vids]
                T, dist = self.eta_matrix(candidate_latlon, target_latlon)

                # Calcualte Distance from the nearest vehicle's location to thr request's origin location
                # print("T.T: ", T.T)
                # print(dist_matrix)
                #T.T here .T is for Transpose
                assignments = self.assign_nearest_vehicle(target_rids, candidate_vids, T.T, dist.T)
                for vid, rid, tt, d in assignments:
                    match_list.append(self.create_matching_dict(vid, rid, tt, d))
                    # od_pairs.append((v_latlon.loc[vid], target_latlon.loc[rid]))
                    V[vid2coord[vid]].remove(vid)

        # print("Matching:")
        # print("List: ", match_list)
        # print("Pairs: ", od_pairs)
        return match_list

    def match_RS(self, current_time, vehicles, requests):
        # print("SA: Inside GreedyMatching Match ", "V:", len(vehicles), "R:", len(requests))
        commands = []
        # od_pairs = []
        requestAssigned = defaultdict(list)

        vehicles, cap_list = self.find_available_vehicles_RS(vehicles)

        n_vehicles = len(vehicles)
        # nV = 4 * len(requests)
        nV = sum(cap_list)      ######### Updated based on capacity
        # print(nV)
        if (n_vehicles > nV):
            vehicles = vehicles.iloc[range(nV), :]

        # print("SA: Inside GreedyMatching Match ", "V:", len(vehicles), "R:", len(requests))

        if n_vehicles == 0:
            return commands

        # print("Available Vehicles and grid locations")
        v_latlon = vehicles[["lat", "lon"]]
        V = defaultdict(list)
        vid2coord = {}
        all_candidate_vids = []
        # all_candidate_vidx = {}
        # vcnt = 0
        for vid, row in v_latlon.iterrows():
            # print("AAAA")
            # print(vehicles.loc[vid])
            # print(vid, row.lon, row.lat)
            coord = self.get_coord(row.lon, row.lat)
            vid2coord[vid] = coord
            # print(vid, coord)
            V[coord].append(vid)
            all_candidate_vids.append(vid)
            # all_candidate_vidx[vid] = vcnt
            # vcnt = vcnt+1

        # print("Requests and Grid locations")
        r_latlon = requests[["origin_lat", "origin_lon"]]
        R = defaultdict(list)
        all_target_rids = []
        for rid, row in r_latlon.iterrows():
            coord = self.get_coord(row.origin_lon, row.origin_lat)
            # print(rid, coord)
            R[coord].append(rid)
            all_target_rids.append(rid)
        # print("Candidate IDs: ", all_candidate_vids)
        # print("Target IDs: ", all_target_rids)
        # Dictionary of vehicle IDs and their index in the candidate list
        all_candidate_vidx = dict(zip(all_candidate_vids, range(len(all_candidate_vids))))
        # print("All Candidates: ", all_candidate_vidx, len(all_candidate_vids))
        all_target_ridx = dict(zip(all_target_rids, range(len(all_target_rids))))
        # print("All targets: ", all_target_ridx, len(all_target_rids))
        target_ridx = [all_target_ridx[r] for r in all_target_rids]

        reject_range = int(self.reject_distance / self.unit_length / self.k) + 1

        d_latlon = requests[["destination_lon", "destination_lat"]]

        candidate_latlon = v_latlon.loc[all_candidate_vids]
        # print("Cand List: ", len(candidate_latlon))
        all_target_latlon = r_latlon.loc[all_target_rids]
        # print("Target List: ", len(all_target_latlon))
        all_destination_latlon = r_latlon.loc[all_target_rids]
        T, dist = self.eta_matrix(candidate_latlon, all_target_latlon)
        # print("match_D: ", dist, dist.shape)
        TRR, Dist_RR = self.eta_matrix(all_target_latlon, all_target_latlon)
        TDD, Dist_DD = self.eta_matrix(all_destination_latlon, all_destination_latlon)

        # for vid, row in vehicles.iterrows():
        # row.earnings = 10
        # print(type(row))
        # print(vehicles[["id"]])
        # print(vid, row.earnings, row.lon, row.lat, row.status)
        for coord in self.coord_iter():
            # print("SA: Inside COORD")
            # print(dist.shape)
            # print(T.shape)
            if not R[coord]:
                # print(coord, "does not exist")
                continue
            # print(coord, "exists")
            # for i in range(int(np.ceil(len(R[coord]) / self.max_locations))):
            # target_rids = R[coord][i * self.max_locations : (i + 1) * self.max_locations]
            # for i in range(int(np.ceil(len(R[coord]) / self.max_locations))):
            target_rids = R[coord]
            # print("TR: ", target_rids)
            candidate_vids = self.find_candidates(coord, len(target_rids), V, reject_range)
            # print("Cand. Vehicles: ", len(candidate_vids))
            if len(candidate_vids) == 0:
                continue

            target_latlon = r_latlon.loc[target_rids]
            candidate_vids = self.filter_candidates(v_latlon.loc[candidate_vids], target_latlon)
            # print("Filtered Vehicles: ", len(candidate_vids))
            if len(candidate_vids) == 0:
                continue

            requestAssigned = defaultdict(list)
            # All candiaidate vehicles that can pickup this specific customer
            candidate_vidx = [all_candidate_vidx[v] for v in candidate_vids]
            assignments = self.assign_nearest_vehicle_RideShare(target_rids, all_target_ridx, requestAssigned,
                                                                candidate_vids,
                                                                candidate_vidx, T.T, TRR, TDD, dist.T, Dist_RR, Dist_DD)
            for vid, rid, tt, d in assignments:
                commands.append(self.create_matching_dict(vid, rid, tt, d))
                # od_pairs.append((v_latlon.loc[vid], target_latlon.loc[rid]))
                # vehicles[vid].current_capacity = vehicles[vid].current_capacity +1
                # print("COORD:", coord, "V ", vid, "assigned to R", rid)
                vehicle = VehicleRepository.get(vid)
                # vehicle.tmp_capacity += 1
                # vehicle.print_vehicle()
                # print(vehicle)

                # vehicles[vid].update_capacity()
                # if vid in V[vid2coord[vid]]:
                #   V[vid2coord[vid]].remove(vid)
                if vehicle.tmp_capacity >= vehicle.state.max_capacity and vid in V[vid2coord[vid]]:
                    V[vid2coord[vid]].remove(vid)

        return commands


    def eta_matrix(self, origins_array, destins_array):
        destins = [(lat, lon) for lat, lon in destins_array.values]
        origins = [(lat, lon) for lat, lon in origins_array.values]
        # origin_set = list(set(origins))
        origin_set = list(origins)
        latlon2oi = {latlon: oi for oi, latlon in enumerate(origin_set)}
        T, d = np.array(self.routing_engine.eta_many_to_many(origin_set, destins), dtype=np.float32)
        T[np.isnan(T)] = float('inf')
        d[np.isnan(d)] = float('inf')
        T = T[[latlon2oi[latlon] for latlon in origins]]
        # print("T: ", T)
        # print("D: ", d.shape)
        return [T, d]

