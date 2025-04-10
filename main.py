import csv
import matplotlib.pyplot as plt
import requests
import numpy as np
import random
from sklearn.cluster import KMeans
from python_tsp.heuristics import solve_tsp_simulated_annealing

from utils import load_places, read_distance_matrix


def plot_clusters_and_routes(csv_path, clusters, cluster_paths):
    """
    Plots the clusters and the optimal routes for each cluster on a map of Poland.
    Includes place 92 as the start and end point of each route.
    """
    place_to_coords = {}
    with open(csv_path, mode="r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            place_id = int(row['Nr'])
            cords = row['Cords'].strip('()')  
            lon, lat = map(float, cords.split(', '))
            place_to_coords[place_id] = (lon, lat)

    # Approximate boundaries of Poland
    poland_lat_min, poland_lat_max = 49.0, 54.8
    poland_lon_min, poland_lon_max = 14.0, 24.0

    plt.figure(figsize=(10, 8))
    colors = ['red', 'blue', 'green', 'purple', 'orange']

    for cluster_id, (path, color) in enumerate(zip(cluster_paths, colors)):
        # Extract coordinates for the current path
        cluster_coords = [place_to_coords[place_id] for place_id in path]
        lons, lats = zip(*cluster_coords) if cluster_coords else ([], [])

        # Plot the places and route
        plt.scatter(lons, lats, color=color, marker='o', label=f'Cluster {cluster_id + 1}')
        if len(lons) > 1:  # Ensure there's a path to plot
            plt.plot(lons, lats, color=color, linestyle=' ', linewidth=2, alpha=0.6)

    plt.xlim(poland_lon_min, poland_lon_max)
    plt.ylim(poland_lat_min, poland_lat_max)
    plt.title("Clusters and Optimal Routes in Poland")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)
    plt.legend()
    plt.show()



def divide_places_into_clusters(input_csv, seed, n_clusters=5):
    """Cluster places (excluding place 92) using KMeans."""
    places, all_coordinates = load_places(input_csv)

    places_to_cluster = [place for place in places if place['Nr'] != 92]
    coordinates_to_cluster = [all_coordinates[i] for i, place in enumerate(places) if place['Nr'] != 92]

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=seed,  # Seed for reproducibility
        init='k-means++',  # Smart initialization
        n_init=20,         # 20 runs with different seeds 
        max_iter=500,      # Maximum iterations
        tol=1e-5           # Strict convergence tolerance
    )
    kmeans.fit(coordinates_to_cluster)


    cluster_idx = 0
    for place in places:
        if place['Nr'] != 92:
            place['Cluster'] = kmeans.labels_[cluster_idx]
            cluster_idx += 1
        else:
            place['Cluster'] = None  # Mark 92 as unclustered

    # Grouping places by cluster
    clusters = {i: [] for i in range(n_clusters)}
    for place in places:
        if place['Nr'] != 92:
            clusters[place['Cluster']].append(place)

    return clusters


import folium

def plot_routes_on_map(cluster_paths, place_to_coords):
    """
    Plot real routes on a map using folium.
    :param cluster_paths: List of paths for each cluster.
    :param place_to_coords: Dictionary mapping place IDs to (lon, lat) coordinates.
    """
    map_center = [52.0, 19.0]  # Approximate center of Poland
    m = folium.Map(location=map_center, zoom_start=6)

    # colors for clusters
    colors = ['red', 'blue', 'green', 'purple', 'orange']

    for cluster_id, path in enumerate(cluster_paths):
        # Coordinates for the current path
        coords = [place_to_coords[place_id] for place_id in path]

        try:
            route_coords = get_route(coords)
        except Exception as e:
            print(f"Error fetching route for cluster {cluster_id}: {e}")
            continue

        #Adding  the route to the map
        folium.PolyLine(
            locations=route_coords,
            color=colors[cluster_id],
            weight=2.5,
            opacity=1,
            tooltip=f"Cluster {cluster_id + 1}"
        ).add_to(m)

    #Save and display the map
    m.save("routes_map.html")
    return m

def get_route(coords):
    """
    Fetch the route between coordinates using OSRM.
    :param coords: List of (lon, lat) tuples.
    :return: List of (lon, lat) tuples representing the route.
    """
    # Converting coordinates to OSRM format
    coord_str = ";".join([f"{lon},{lat}" for lon, lat in coords])
    #print(coord_str)
    url = f"http://router.project-osrm.org/route/v1/driving/{coord_str}?overview=full&geometries=geojson"

    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"OSRM request failed: {response.status_code}")

    data = response.json()
    if data.get("code") != "Ok":
        raise Exception(f"OSRM error: {data.get('message')}")

    route_coords = data["routes"][0]["geometry"]["coordinates"]
    return [(lat, lon) for lon, lat in route_coords]

def main():
    input_csv = "cords.csv"
    distances_csv = "road_distances.csv"
    start_end_id = 92
    seed = 19

    distance_matrix = read_distance_matrix(distances_csv)
    clusters = divide_places_into_clusters(input_csv, seed=seed)

    cluster_paths = []
    cluster_distances = []
    for cluster_id in clusters:
        cluster_places = clusters[cluster_id]
        if not cluster_places:
            continue

        # TSP nodes: cluster places + start/end point (92)
        cluster_nrs = [place['Nr'] for place in cluster_places]
        tsp_nrs = [start_end_id] + cluster_nrs
        tsp_indices = [nr - 1 for nr in tsp_nrs]  # Converting to 0-based indices

        # submatrix for TSP
        sub_matrix = [
            [distance_matrix[i][j] for j in tsp_indices]
            for i in tsp_indices
        ]

        #TSP
        sub_matrix = np.array(sub_matrix)  # Convert list of lists to NumPy array
        permutation, distance = solve_tsp_simulated_annealing(sub_matrix)

        # Rotate permutation to start at 0 (start_end_id's position in sub_matrix)
        try:
            start_idx = permutation.index(0)
        except ValueError:
            start_idx = 0
        rotated_permutation = permutation[start_idx:] + permutation[:start_idx]

        # Convert to place IDs and ensure the route ends at 92
        path = [tsp_nrs[i] for i in rotated_permutation]
        if path[-1] != start_end_id:
            path.append(start_end_id)

        cluster_paths.append(path)
        cluster_distances.append(distance)

    # Show distances for each cluster
    total_distance = sum(cluster_distances)
    for i, dist in enumerate(cluster_distances):
        print(f"Cluster {i} route distance: {dist} km")
    print(f"Total sum of distances: {total_distance} km")

    # Loading place coordinates
    places, _ = load_places(input_csv)
    place_to_coords = {place['Nr']: tuple(map(float, place['Cords'].strip('()').split(', '))) for place in places}

    # Plot real routes on a map
    #plot_routes_on_map(cluster_paths, place_to_coords)
    plot_clusters_and_routes(input_csv, clusters, cluster_paths)




if __name__ == '__main__':
    main()
