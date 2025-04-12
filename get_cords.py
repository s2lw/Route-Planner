import pandas as pd
import time
from geopy.geocoders import Nominatim
import csv
import matplotlib.pyplot as plt
import requests
import numpy as np
import random
from sklearn.cluster import KMeans

def get_coordinates(address):
    geolocator = Nominatim(user_agent="gis_put_zadanie")  # Use a unique user agent
    try:
        print(f"🔎 Szukam współrzędnych dla: {address}...")  
        location = geolocator.geocode(address)
        if location:
            coords = (location.longitude, location.latitude)
            print(f"Znaleziono: {coords}\n")
            return coords
        else:
            print(f"Nie znaleziono współrzędnych dla: {address}\n")
            return None
    except Exception as e:
        print(f"Błąd podczas pobierania współrzędnych dla {address}: {e}\n")
        return None
def download_cords_and_create_csv(excel_file_path, csv_file_path):
    # CSV file header
    header = ["Nr", "Adres", "Nazwa", "Województwo", "Liczba łóżek", "Cords"]


    # Load the Excel file
    df_excel = pd.read_excel(excel_file_path)

    # Check if required columns exist in the Excel file
    if 'Miejscowość' in df_excel.columns and 'Ulica' in df_excel.columns:
        # Open the CSV file in append mode and write data
        with open(csv_file_path, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(header)  # Write the header row

            # Iterate over each row in the Excel file
            for index, row in df_excel.iterrows():
                miejscowosc = row['Miejscowość']
                ulica = row['Ulica']
                address = f"{ulica[5::]}, {miejscowosc}"  # Remove 'ul. ' from the street name
                coord = get_coordinates(address)  # Get coordinates for the address

                # Print the address and coordinates (for debugging)
                # print(f"{ulica[5::]}, {miejscowosc} : {coord}\n")

                # Append a new row to the CSV file
                new_row = [index + 1, address, row['Nazwa'], row['Województwo'], row['Liczba łóżek'], coord]
                writer.writerow(new_row)

                # Sleep to avoid overloading the API (if applicable)
                time.sleep(1)
    else:
        print("The required columns 'Miejscowość' and 'Ulica' are not present in the Excel file.")


def plot_poland_coordinates(csv_path):
    """
    Reads the 'Cords' column from a CSV file and plots the coordinates on a map of Poland.

    :param csv_path: Path to the CSV file.
    """
    try:
        # Lists to store latitude and longitude values
        latitudes = []
        longitudes = []

        with open(csv_path, mode="r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)  # Read the CSV file as a dictionary
            if 'Cords' not in reader.fieldnames:
                print("Error: The 'Cords' column does not exist in the CSV file.")
                return

            # Extract latitude and longitude from the 'Cords' column
            for row in reader:
                cords = row['Cords'].strip()
                cords = cords.replace('(', '').replace(')', '')
                lon, lat = map(float, cords.split(', '))  # Split and convert to float
                print(f"lon={lon}, lat={lat}, num={row['Nr']}\n")
                latitudes.append(lat)
                longitudes.append(lon)

        # approximate boundaries of Poland
        poland_lat_min, poland_lat_max = 49.0, 54.8  # Latitude range for Poland
        poland_lon_min, poland_lon_max = 14.0, 24.0  # Longitude range for Poland

        # Plot the coordinates on a map of Poland
        plt.figure(figsize=(10, 8))
        plt.scatter(longitudes, latitudes, color='red', marker='o', label='Coordinates')

        # Set plot limits to focus on Poland
        plt.xlim(poland_lon_min, poland_lon_max)
        plt.ylim(poland_lat_min, poland_lat_max)

        # Add title and labels
        plt.title("Coordinates of Places in Poland")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.grid(True)
        plt.legend()

        # Add a background map (optional)
        # You can use a shapefile or an image of Poland as a background if needed.
        # For simplicity, we'll just use the grid and axis limits.

        plt.show()

    except FileNotFoundError:
        print(f"Error: The file '{csv_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


def calculate_road_distance(coords1, coords2):
    """
    Calculate road distance between two coordinates using OSRM's HTTP API.
    :param coords1: Tuple of (latitude, longitude) for the first place.
    :param coords2: Tuple of (latitude, longitude) for the second place.
    :return: Road distance in kilometers.
    """
    # OSRM's public server URL
    osrm_url = "http://router.project-osrm.org/route/v1/driving/"
    print("calculating road distance...")
    # Format coordinates as "lon,lat;lon,lat"
    coordinates_str = f"{coords1[1]},{coords1[0]};{coords2[1]},{coords2[0]}"

    # Make the request to OSRM
    response = requests.get(f"{osrm_url}{coordinates_str}?overview=false")

    if response.status_code == 200:
        data = response.json()
        if data['code'] == 'Ok':
            print(f"coords1={coords1}, coords2={coords2}, distance={data['routes'][0]['distance'] / 1000}")
            return data['routes'][0]['distance'] / 1000  # Convert meters to kilometers
    else:
        print(f"Error calculating distance between {coords1} and {coords2}: {response.status_code}")
    return None


def calculate_all_road_distances(input_csv, output_csv):
    """
    Calculate road distances between all pairs of places in the input CSV and write to a new CSV.
    :param input_csv: Path to the input CSV file (cords.csv).
    :param output_csv: Path to the output CSV file.
    """
    # Read coordinates from the input CSV
    places = []
    coordinates = []

    with open(input_csv, mode="r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            cords = row['Cords'].strip('()')  # Remove parentheses
            lon, lat = map(float, cords.split(', '))  # Split into longitude and latitude
            places.append(row['Nr'])  # Store place names
            coordinates.append((lat, lon))  # Store coordinates as (lat, lon)

    # Calculate road distances between all pairs of places
    num_places = len(places)
    distance_matrix = [[0] * num_places for _ in range(num_places)]  # Initialize distance matrix

    for i in range(num_places):
        for j in range(i + 1, num_places):
            print(f"i={i}, j={j}")
            distance = calculate_road_distance(coordinates[i], coordinates[j])
            if distance is not None:
                distance_matrix[i][j] = distance
                distance_matrix[j][i] = distance  # Distance matrix is symmetric

    # Write the distance matrix to the output CSV
    with open(output_csv, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        # Write header row
        writer.writerow(['From \\ To'] + places)
        # Write distance rows
        for i in range(num_places):
            writer.writerow([places[i]] + distance_matrix[i])

    print(f"Road distances have been written to {output_csv}")


def main():
    download_cords_and_create_csv('SzpitaleZakazne-Polska (5).xlsx', 'cords.csv')
    plot_poland_coordinates("cords.csv")

    input_csv = "cords.csv"  
    output_csv = "road_distances.csv"  
    calculate_all_road_distances(input_csv, output_csv)


if __name__ == "__main__":
    main()
