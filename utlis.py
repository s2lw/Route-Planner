import csv
import numpy as np

def load_places(input_csv, include_place_92=False):
    """Load places from CSV file and extract coordinates."""
    places = []
    coordinates = []

    with open(input_csv, mode="r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            place_id = int(row['Nr'])
            # Filter places based on ID range
            max_id = 92 if include_place_92 else 91
            if 1 <= place_id <= max_id:
                place_data = {
                    'Nr': place_id,
                    'Nazwa': row['Nazwa'],
                    'Adres': row['Adres'],
                    'Województwo': row['Województwo'],
                    'Liczba łóżek': int(row['Liczba łóżek']),
                    'Cords': row['Cords']
                }
                places.append(place_data)
                # Parse coordinates
                cords = row['Cords'].strip().replace('(', '').replace(')', '')
                lon, lat = map(float, cords.split(', '))
                coordinates.append([lat, lon])

    return places, np.array(coordinates)

def read_distance_matrix(csv_file):
    """Read the distance matrix from a CSV file, skipping the first row and first column."""
    with open(csv_file, newline='') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        distance_matrix = []
        for row in reader:
            # Skip the first column (row label) and convert the rest to float
            distance_matrix.append([float(x) for x in row[1:]])
    return distance_matrix