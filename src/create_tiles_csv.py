import argparse
import csv
import cv2
import geopandas as gpd
import numpy as np
import os
import PIL
from PIL import Image
import rasterio
import re

# Extract waydal locations and corresponding image ids
def retrieve_locations(shp_file):

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

    return num_waydals, ids, locations

# Download macro images and associate them with waydal locations
def download_images(folder, ids, locations):

    # Download satellite images from Google Drive
    images = {}
    regex = r'(\d+)(?=\.tif$)'
    for image in os.listdir(folder):
        file_path = os.path.join(folder, image)
        m = re.search(regex, file_path)
        id = file_path[-8:-4:1]
        images[id] = [file_path]
        images[id].append([])
        images[id].append([])

    # Store waydal locations in dictionary with their corresponding macro images
    for i, id in enumerate(ids):
        polygon = []
        for coord in list(locations[i].exterior.coords):
            polygon.append(coord)
        if id != '1101': # There is a polygon with this id, but no corresponding macro image (!?)
            images[id][1].append(polygon)
            images[id][2].append(i)

    return images

# Check if enough of a waydal lies in the tile to consider it a positive instance
def check_waydal_proportion(waydals, x_a, y_a, ratio, tile_dim, waydal_threshold):

    has_enough = False
    total_area = 0
    covered_area = 0
    mask = np.zeros((tile_dim, tile_dim))

    # Iterate through the waydals
    for waydal in waydals:

        # Convert coordinates so they are plottable and store the area of the waydal
        points = np.array(list(waydal))
        for point in points:
            point[0] = int((point[0] - x_a) / ratio)
            point[1] = int((y_a - point[1]) / ratio)
        points = points.astype(np.int32)
        total_area += cv2.contourArea(points)

        #Clamp values if out of bounds
        for point in points:
            if point[0] < 0:
                point[0] = 0
            elif point[0] >= tile_dim:
                point[0] = tile_dim - 1
            if point[1] < 0:
                point[1] = 0
            elif point[1] >= tile_dim:
                point[1] = tile_dim - 1

        # Update the mask
        cv2.fillPoly(mask, [points], 1)

    # Determine proportion of waydals covered by this tile
    covered_area += np.sum(mask)
    if (covered_area / total_area) >= waydal_threshold:
        has_enough = True

    return has_enough

# Partition macro images into tiles for training and create one-hot encodings and/or masks
def partition_tiles(csv_file, images, image_dim, tile_dim, coord_displacement, tile_threshold = 10, waydal_threshold = 0.3):

    # Write tile information to a csv
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        PIL.Image.MAX_IMAGE_PIXELS = 383533056 * 3 * len(images) # Increase capacity due to macro image size
        ratio = coord_displacement / tile_dim # Ratio for conversion
        num_tiles = int(image_dim / tile_dim) # Per macro image

        # Count the number of waydals and the number of each type of tile
        num_pos_tiles = 0 # Number of tiles with a waydal; can differ from above due to waydals split between multiple tiles
        num_neg_tiles = 0 # Number of tiles without a waydal which are not thrown out (see below)
        num_black_tiles = 0 # Number of tiles thrown out and not used; macro images contain many black portions without satellite imagery

        # Iterate through each image tile by tile
        for k, key in enumerate(images.keys()):

            # Open the image
            img = rasterio.open(images[key][0])
            transform = img.transform
            top_left_x, top_left_y = transform * (0, 0)
            img.close()
            img = Image.open(images[key][0])
            img = img.convert('RGB')
            arr = np.array(img)
            img.close()

            # Iterate through the image top to bottom
            for i in range(num_tiles):

                # Iterate through the image left to right
                for j in range(num_tiles):

                    # -1 = black; 0 = not found; 1 = found
                    tile_type = 0

                    # Find the coordinates for the tile
                    x_1 = j * tile_dim
                    x_2 = j * tile_dim + tile_dim
                    y_1 = i * tile_dim
                    y_2 = i * tile_dim + tile_dim
                    x_a = top_left_x + j * coord_displacement
                    x_b = x_a + coord_displacement
                    y_a = top_left_y - i * coord_displacement
                    y_b = y_a - coord_displacement
                    tile = np.array(arr[y_1:y_2, x_1:x_2, :])

                    # Throw a tile which is mostly black, i.e., it does not contain satellite imagery, and exit, otherwise store it
                    if np.mean(tile) <= tile_threshold:
                        num_black_tiles += 1
                        tile_type = -1
                        writer.writerow([tile_type, images[key][0], x_1, y_1, x_2, y_2, x_a, y_a, x_b, y_b])

                    # Continue if not a black tile
                    else:

                        # Find all the waydals contained in this tile, if any; while unlikely, there could be multiple
                        waydals = []
                        polygons = []

                        # Iterate through the waydals corresponding to the macro image to which this tile belongs
                        for num, polygon in enumerate(images[key][1]):

                            # Iterate though the coordinates of the waydal
                            for coord in polygon:

                                # Determine whether the coordinate falls within this tile
                                if coord[0] >= x_a and coord[0] < x_b and coord[1] <= y_a and coord[1] > y_b:

                                    # If one coordinate falls within this tile we do not need to test the others
                                    waydals.append(images[key][2][num])
                                    polygons.append(polygon)
                                    break

                        # Update type and write to file
                        if len(waydals) > 0:
                            # Check if enough of the waydal lies in the tile to consider this a positive instance
                            has_enough = check_waydal_proportion(polygons, x_a, y_a, ratio, tile_dim, waydal_threshold)
                            if has_enough:
                                tile_type = 1
                                num_pos_tiles += 1
                                output = [tile_type, images[key][0], x_1, y_1, x_2, y_2, x_a, y_a, x_b, y_b]
                                for waydal in waydals:
                                    output.append(waydal)
                                writer.writerow(output)
                            else:
                                num_neg_tiles += 1
                                writer.writerow([tile_type, images[key][0], x_1, y_1, x_2, y_2, x_a, y_a, x_b, y_b])
                        else:
                            num_neg_tiles += 1
                            writer.writerow([tile_type, images[key][0], x_1, y_1, x_2, y_2, x_a, y_a, x_b, y_b])

    return num_pos_tiles, num_neg_tiles, num_black_tiles

if __name__=="__main__":

    parser = argparse.ArgumentParser(prog='Create waydal coordinates CSV')
    parser.add_argument('--shp_file', type=str, required=True, help="Path to .shp file")
    parser.add_argument('--csv_file', type=str, required=True, help="Path to .csv file")
    parser.add_argument('--img_folder', type=str, required=True, help="Path to image folder")

    args = parser.parse_args()

    shp_file = args.shp_file
    csv_file = args.csv_file
    folder = args.img_folder

    num_waydals, ids, locations = retrieve_locations(shp_file)

    print(f"Number of Waydals: {num_waydals}")

    images = download_images(folder, ids, locations)

    image_dim = 19584 # Image height/width in pixels
    tile_dim = 384  # Tile height/width in pixels
    meters_per_pixel = 0.5 # Meters per pixel
    coord_displacement = 0.00172334558824 # Coordinate displacement per tile height/width
    tile_threshold = 10 # Tiles whose average color is less than or equal to this will be thrown out
    waydal_threshold = 0.3

    num_pos_tiles, num_neg_tiles, num_black_tiles = partition_tiles(csv_file, images, image_dim, tile_dim, coord_displacement, tile_threshold = tile_threshold, waydal_threshold = waydal_threshold)