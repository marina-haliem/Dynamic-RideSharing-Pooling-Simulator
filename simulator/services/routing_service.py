import os
import pickle
import numpy as np
from config.settings import DATA_DIR
from common import mesh, geoutils
from .osrm_engine import OSRMEngine
from simulator.settings import FLAGS, MAX_MOVE
import polyline

class RoutingEngine(object):
    engine = None

    @classmethod
    def create_engine(cls):
        if cls.engine is None:
            if FLAGS.use_osrm:
                cls.engine = OSRMEngine()
            else:
                cls.engine = FastRoutingEngine()
        return cls.engine


class FastRoutingEngine(object):
    def __init__(self):
        self.tt_map = np.load(os.path.join(DATA_DIR, 'tt_map.npy'))
        self.routes = pickle.load(open(os.path.join(DATA_DIR, 'routes.pkl'), 'rb'))

        d = self.tt_map.copy()
        for x in range(d.shape[0]):
            origin_lon = mesh.X2lon(x)
            for y in range(d.shape[1]):
                origin_lat = mesh.Y2lat(y)
                for axi in range(d.shape[2]):
                    x_ = x + axi - MAX_MOVE
                    destin_lon = mesh.X2lon(x_)
                    for ayi in range(d.shape[3]):
                        y_ = y + ayi - MAX_MOVE
                        destin_lat = mesh.Y2lat(y_)
                        d[x, y, axi, ayi] = geoutils.great_circle_distance(
                            origin_lat, origin_lon, destin_lat, destin_lon)
        self.ref_d = d  # Distance in meters

    # (Origin - destination) pairs
    def route(self, od_pairs):
        results = []
        for (origin_lat, origin_lon), (dest_lat, dest_lon) in od_pairs:
            x, y = mesh.convert_lonlat_to_xy(origin_lon, origin_lat)
            x_, y_ = mesh.convert_lonlat_to_xy(dest_lon, dest_lat)
            ax, ay = x_ - x, y_ - y
            axi = x_ - x + MAX_MOVE
            ayi = y_ - y + MAX_MOVE
            # print(len(self.routes))
            # if ax or ay > MAX_MOVE:
                # trajectory = None
            # else:
            #     print((x, y), (ax, ay), (x_, y_))
            trajectory = polyline.decode(self.routes[(x, y)][(ax, ay)]) # Route from origin to destination
            triptime = self.tt_map[x, y, axi, ayi]
            # print(triptime)
            results.append((trajectory, triptime))
        return results

    def route_time(self, od_pairs):
        results = []
        new_od_pairs = []
        for (origin_lat, origin_lon), (dest_lat, dest_lon) in od_pairs:
            x, y = mesh.convert_lonlat_to_xy(origin_lon, origin_lat)
            x_, y_ = mesh.convert_lonlat_to_xy(dest_lon, dest_lat)
            ax, ay = x_ - x, y_ - y
            # print((x, y), (ax, ay), (x_, y_))
            # print(MAX_MOVE)
            if ax > MAX_MOVE and ay > MAX_MOVE:
                ax = MAX_MOVE
                ay = MAX_MOVE
                # print((x + ax, y + ay), (x_, y_))
                new_od_pairs.append(((x + ax, y + ay), (x_, y_)))

            elif ax < (-1 * MAX_MOVE) and ay < (-1*MAX_MOVE):
                ax = (-1*MAX_MOVE)
                ay = (-1*MAX_MOVE)
                # print((x - ax, y - ay), (x_, y_))
                new_od_pairs.append(((x + ax, y + ay), (x_, y_)))

            elif ax > MAX_MOVE and ay < (-1 * MAX_MOVE):
                ax = MAX_MOVE
                ay = (-1*MAX_MOVE)
                # print((x + ax, y - ay), (x_, y_))
                new_od_pairs.append(((x + ax, y + ay), (x_, y_)))

            elif ay > MAX_MOVE and ax < (-1 * MAX_MOVE):
                ay = MAX_MOVE
                ax = (-1*MAX_MOVE)
                # print((x - ax, y + ay), (x_, y_))
                new_od_pairs.append(((x + ax, y + ay), (x_, y_)))

            elif ax > MAX_MOVE >= ay:
                ax = MAX_MOVE
                # print(len(od_pairs))
                new_od_pairs.append(((x + ax, y_), (x_, y_)))
                # print(len(od_pairs), od_pairs)

            elif ay > MAX_MOVE >= ax:
                ay = MAX_MOVE
                # print(len(od_pairs))
                new_od_pairs.append(((x_, y + ay), (x_, y_)))
                # print(len(od_pairs), od_pairs)

            elif ax < (-1*MAX_MOVE) <= ay:
                ax = (-1*MAX_MOVE)
                # print(len(od_pairs))
                new_od_pairs.append(((x + ax, y_), (x_, y_)))
                # print(len(od_pairs), od_pairs)

            elif ay < (-1*MAX_MOVE) <= ax:
                ay = (-1*MAX_MOVE)
                # print(len(od_pairs))
                new_od_pairs.append(((x_, y + ay), (x_, y_)))
                # print(len(od_pairs), od_pairs)

            axi = ax + MAX_MOVE
            ayi = ay + MAX_MOVE
            # print((x, y), (ax, ay), (x_, y_))
            trajectory = polyline.decode(self.routes[(x, y)][(ax, ay)])  # Route from origin to destination
            triptime = self.tt_map[x, y, axi, ayi]
            # print(triptime)
            extra_res = self.extra_routes(new_od_pairs)

            for (extra_traject, extra_time) in extra_res:
                trajectory.extend(extra_traject)
                triptime += extra_time

            results.append((trajectory, triptime))

            # results.extend(extra_res)
        return results

    def extra_routes(self, od_pairs):
        res = []
        for (origin_lon, origin_lat), (dest_lon, dest_lat) in od_pairs:
            x, y = (origin_lon, origin_lat)
            x_, y_ = (dest_lon, dest_lat)
            ax, ay = x_ - x, y_ - y
            # print((x, y), (ax, ay), (x_, y_))
            # print(MAX_MOVE)
            if ax > MAX_MOVE and ay > MAX_MOVE:
                ax = MAX_MOVE
                ay = MAX_MOVE
                # print("A: ", (x + ax, y + ay), (x_, y_))
                self.extra_routes([((x + ax, y + ay), (x_, y_))])

            elif ax < (-1*MAX_MOVE) and ay < (-1*MAX_MOVE):
                ax = (-1*MAX_MOVE)
                ay = (-1*MAX_MOVE)
                # print("B: ", (x + ax, y + ay), (x_, y_))
                self.extra_routes([((x + ax, y + ay), (x_, y_))])

            elif ax > MAX_MOVE and ay < (-1*MAX_MOVE):
                ax = MAX_MOVE
                ay = (-1*MAX_MOVE)
                # print("Here: ", (x + ax, y + ay), (x_, y_))
                self.extra_routes([((x + ax, y + ay), (x_, y_))])

            elif ay > MAX_MOVE and ax < (-1*MAX_MOVE):
                ay = MAX_MOVE
                ax = (-1*MAX_MOVE)
                # print("There: ", (x + ax, y + ay), (x_, y_))
                self.extra_routes([((x + ax, y + ay), (x_, y_))])

            elif ax > MAX_MOVE >= ay:
                ax = MAX_MOVE
                # print("C: ", (x + ax, y_), (x_, y_))
                self.extra_routes([((x + ax, y_), (x_, y_))])
                # print(len(od_pairs), od_pairs)

            elif ay > MAX_MOVE >= ax:
                ay = MAX_MOVE
                # print("D: ", (x_, y + ay), (x_, y_))
                self.extra_routes([((x_, y + ay), (x_, y_))])
                # print(len(od_pairs), od_pairs)

            elif ax < (-1 * MAX_MOVE) <= ay:
                ax = (-1*MAX_MOVE)
                # print("E: ", (x + ax, y_), (x_, y_))
                self.extra_routes([((x + ax, y_), (x_, y_))])
                # print(len(od_pairs), od_pairs)

            elif ay < (-1*MAX_MOVE) <= ax:
                ay = (-1*MAX_MOVE)
                # print("F: ", (x_, y + ay), (x_, y_))
                self.extra_routes([((x_, y + ay), (x_, y_))])

            axi = ax + MAX_MOVE
            ayi = ay + MAX_MOVE
            # print((x, y), (ax, ay), (x_, y_))
            trajectory = polyline.decode(self.routes[(x, y)][(ax, ay)])  # Route from origin to destination
            triptime = self.tt_map[x, y, axi, ayi]
            # print(triptime)
            res.append((trajectory, triptime))
        return res

    # Estimating arrival (Duration) continously until we reach destination
    def eta_many_to_many(self, origins, destins, max_distance=5000, ref_speed=5.0):
        T = np.full((len(origins), len(destins)), np.inf)
        origins_lat, origins_lon = zip(*origins)
        destins_lat, destins_lon = zip(*destins)
        origins_lat, origins_lon, destins_lat, destins_lon = map(np.array, [origins_lat, origins_lon, destins_lat, destins_lon])
        origins_x, origins_y = mesh.lon2X(origins_lon), mesh.lat2Y(origins_lat)
        destins_x, destins_y = mesh.lon2X(destins_lon), mesh.lat2Y(destins_lat)
        d = geoutils.great_circle_distance(origins_lat[:, None], origins_lon[:, None],
                                           destins_lat, destins_lon)

        for i, (x, y) in enumerate(zip(origins_x, origins_y)):
            for j in np.where(d[i] < max_distance)[0]:
                axi = destins_x[j] - x + MAX_MOVE
                ayi = destins_y[j] - y + MAX_MOVE
                if 0 <= axi and axi <= 2 * MAX_MOVE and 0 <= ayi and ayi <= 2 * MAX_MOVE:
                    ref_d = self.ref_d[x, y, axi, ayi]
                    if ref_d == 0:
                        T[i, j] = d[i, j] / ref_speed
                    else:
                        T[i, j] = self.tt_map[x, y, axi, ayi] * d[i, j] / ref_d
        return [T, d]


    # def eta_matrix(self, origins_lat, origins_lon, destins_lat, destins_lon, max_distance=5000, ref_speed=5.0):
    #     T = np.full((len(origins_lat), len(destins_lat)), np.inf)
    #     origins_lat, origins_lon, destins_lat, destins_lon = map(np.array, [origins_lat, origins_lon, destins_lat, destins_lon])
    #     origins_x, origins_y = mesh.lon2X(origins_lon), mesh.lat2Y(origins_lat)
    #     destins_x, destins_y = mesh.lon2X(destins_lon), mesh.lat2Y(destins_lat)
    #     d = geoutils.great_circle_distance(origins_lat[:, None], origins_lon[:, None],
    #                                        destins_lat, destins_lon)
    #
    #     for i, j in zip(*np.where(d < max_distance)):
    #         x = origins_x[i]
    #         y = origins_y[i]
    #         axi = destins_x[j] - x + MAX_MOVE
    #         ayi = destins_y[j] - y + MAX_MOVE
    #         if 0 <= axi and axi <= 2 * MAX_MOVE and 0 <= ayi and ayi <= 2 * MAX_MOVE:
    #             ref_d = self.ref_d[x, y, axi, ayi]
    #             if ref_d == 0:
    #                 T[i, j] = d[i, j] / ref_speed
    #             else:
    #                 T[i, j] = self.tt_map[x, y, axi, ayi] * d[i, j] / ref_d
    #     return T