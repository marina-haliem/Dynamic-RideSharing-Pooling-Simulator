from pathlib import Path
import pickle
import osmnx as ox
import matplotlib.pyplot as plt

# Load the routes from the pickle file
BASE = Path(__file__).parent
with open(BASE / "data/routes.pkl", "rb") as f:
    routes = pickle.load(f)


# Create a basemap for New York City
G = ox.graph_from_place("New York City, New York, USA", network_type="drive")

print(G)

# Plot each route on the map
fig, ax = ox.plot_graph(
    G, show=False, close=False, figsize=(10, 10), edge_color="black"
)

route_nodes = [(lon, lat) for lat, lon in route]
ox.plot_route_folium(
    G,
    routes,
    route_map=None,
    route_color="red",
    route_linewidth=4,
    route_opacity=1,
)

# route_nodes = [(point["lon"], point["lat"]) for point in route["geometry"]]
# ox.plot_route_folium(
#     G,
#     route_nodes,
#     route_map=None,
#     route_color="red",
#     route_linewidth=4,
#     route_opacity=1,
# )

plt.savefig("routes.png", dpi=600)
