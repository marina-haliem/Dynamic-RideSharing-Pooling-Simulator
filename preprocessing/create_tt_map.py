import numpy as np
import argparse
import sys
import pickle
sys.path.append('/../data')
sys.path.append('C:/Users/17657/PycharmProjects/Deep_Pool')
from simulator.services.osrm_engine import OSRMEngine
from config.settings import MAP_WIDTH, MAP_HEIGHT
from common.mesh import convert_xy_to_lonlat
from common.geoutils import great_circle_distance
from simulator.settings import MAX_MOVE

state_space = [(x, y) for x in range(MAP_WIDTH) for y in range(MAP_HEIGHT)]
action_space = [(ax, ay) for ax in range(-MAX_MOVE, MAX_MOVE + 1)
                for ay in range(-MAX_MOVE, MAX_MOVE + 1)]


def create_reachable_map(engine):
    lon0, lat0 = convert_xy_to_lonlat(0, 0)
    lon1, lat1 = convert_xy_to_lonlat(1, 1)
    d_max = great_circle_distance(lat0, lon0, lat1, lon1) / 2.0

    points = []
    for x, y in state_space:
        lon, lat = convert_xy_to_lonlat(x, y)
        points.append((lat, lon))

    nearest_roads = engine.nearest_road(points)
    reachable_map = np.zeros((MAP_WIDTH, MAP_HEIGHT), dtype=np.float32)
    for (x, y), (latlon, d) in zip(state_space, nearest_roads):
        if d < d_max:
            reachable_map[x, y] = 1

    return reachable_map


def create_tt_tensor(engine, reachable_map):
    origin_destins_list = []
    for x, y in state_space:
        origin = convert_xy_to_lonlat(x, y)[::-1]
        destins = [convert_xy_to_lonlat(x + ax, y + ay)[::-1] for ax, ay in action_space]
        origin_destins_list.append((origin, destins))
    tt_list = engine.eta_one_to_many(origin_destins_list)

    a_size = MAX_MOVE * 2 + 1
    tt_tensor = np.full((MAP_WIDTH, MAP_HEIGHT, a_size, a_size), np.inf)
    for (x, y), tt in zip(state_space, tt_list):
        tt_tensor[x, y] = np.array(tt).reshape((a_size, a_size))
        for ax, ay in action_space:
            x_, y_ = x + ax, y + ay
            axi, ayi = ax + MAX_MOVE, ay + MAX_MOVE
            if x_ < 0 or x_ >= MAP_WIDTH or y_ < 0 or y_ >= MAP_HEIGHT or reachable_map[x_, y_] == 0:
                tt_tensor[x, y, axi, ayi] = float('inf')
        # if reachable_map[x, y] == 1:
        #     tt_tensor[x, y, MAX_MOVE, MAX_MOVE] = 0
    tt_tensor[np.isnan(tt_tensor)] = float('inf')
    return tt_tensor


def create_routes(engine, reachable_map):
    routes = {}
    for x, y in state_space:

        print(x, y)
        #if(x ==0 and y==14):
        #    continue
        
        origin = convert_xy_to_lonlat(x, y)[::-1]
        od_list = [(origin, convert_xy_to_lonlat(x + ax, y + ay)[::-1]) for ax, ay in action_space]
        tr_list, _ = zip(*engine.route(od_list, decode=False))
        routes[(x, y)] = {}
        for a, tr in zip(action_space, tr_list):
            routes[(x, y)][a] = tr
    return routes


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help = "data directory")
    # parser.add_argument("--route", action='store_true', help="whether compute route or not")
    args = parser.parse_args()

    engine = OSRMEngine()

    print("create reachable map")
    reachable_map = create_reachable_map(engine)
    np.save("{}/reachable_map".format(args.data_dir), reachable_map)

    print("create tt map")
    tt_tensor = create_tt_tensor(engine, reachable_map)
    np.save("{}/tt_map".format(args.data_dir), tt_tensor)

    print("create routes")
    # if args.route:
    routes = create_routes(engine, reachable_map)
    pickle.dump(routes, open("{}/routes.pkl".format(args.data_dir), "wb"))
