import argparse
import csv
import geopandas as gpd

# Extract waydal locations and corresponding image ids
def retrieve_locations(shp_file, csv_file):

    # Extract the polygon coordinates and image ids corresponding to the monuments
    gdf = gpd.read_file(shp_file)
    locations = gdf[gdf.geometry.type == 'Polygon'].geometry.tolist()
    ids = gdf[gdf.geometry.type == 'Polygon'].Vivid.tolist()
    num_waydals = len(locations)

    # Ensure ids are four digits long
    for i, id in enumerate(ids):
        ids[i] = str(id)
        if len(ids[i]) == 3:
            ids[i] = '0' + ids[i]
        elif len(ids[i]) == 2:
            ids[i] = '00' + ids[i]

    # Print polygons to a CSV
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for i, waydal in enumerate(locations):
            row = [coord for point in waydal.exterior.coords for coord in point]
            row.append(ids[i])
            writer.writerow(row)

    print(f"Number of waydals: {num_waydals}")

    return

if __name__=="__main__":
    parser = argparse.ArgumentParser(prog='Create waydal coordinates CSV')
    parser.add_argument('--shp_file', type=str, required=True, help="Path to .shp file")
    parser.add_argument('--csv_file', type=str, required=True, help="Path to .csv file")

    args = parser.parse_args()

    shp_file = args.shp_file
    csv_file = args.csv_file
    retrieve_locations(shp_file, csv_file)