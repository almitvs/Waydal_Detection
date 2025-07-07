import csv
import random as rand
import matplotlib.pyplot as plt
import numpy as np
import PIL
from PIL import Image
from sklearn.model_selection import train_test_split

# Read the information from CSVs and sample negative tiles
def read_CSVs(tiles_csv, proportion):

    # Store the information
    locations = []
    pos_tiles = []
    neg_tiles = []

    # Get the tile information
    with open(tiles_csv, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[0] == '1':
                pos_tiles.append(row[1:])
            elif row[0] == '0':
                neg_tiles.append(row[1:])

    # Sample negative tiles at random
    amount = proportion * len(pos_tiles)
    neg_samples = rand.sample(neg_tiles, amount)

    return pos_tiles, neg_samples

# Get the pixel values for the tiles
def create_tiles(pos_tiles, neg_tiles):

    # Store the image tiles
    pos_tile_images = []
    neg_tile_images = []
    pos_tile_indices = []
    neg_tile_indices = []

    # Group tiles by macro image to reduce time and memory cost of opening the macro images repeatedly
    images = {}
    for i, tile in enumerate(pos_tiles):
        image = tile[0]
        if image in images:
            images[image][0].append(tile)
            images[image][2].append(i)
        else:
            images[image] = [[tile], [], [i], []]
    for i, tile in enumerate(neg_tiles):
        image = tile[0]
        if image in images:
            images[image][1].append(tile)
            images[image][3].append(i)
        else:
            images[image] = [[], [tile], [], [i]]

    # Create the tile images
    PIL.Image.MAX_IMAGE_PIXELS = 383533056 * 3 * 89 # Increase capacity due to macro image size
    for image_path in images:
        with Image.open(image_path) as image:
            img = np.array(image.convert('RGB'))
            for i, pos_tile in enumerate(images[image_path][0]):
                x_1 = int(pos_tile[1])
                y_1 = int(pos_tile[2])
                x_2 = int(pos_tile[3])
                y_2 = int(pos_tile[4])
                pos_tile_image = np.array(img[y_1:y_2, x_1:x_2, :])
                pos_tile_images.append(pos_tile_image)
                pos_tile_indices.append(images[image_path][2][i])
            for i, neg_tile in enumerate(images[image_path][1]):
                x_1 = int(neg_tile[1])
                y_1 = int(neg_tile[2])
                x_2 = int(neg_tile[3])
                y_2 = int(neg_tile[4])
                neg_tile_image = np.array(img[y_1:y_2, x_1:x_2, :])
                neg_tile_images.append(neg_tile_image)
                neg_tile_indices.append(images[image_path][3][i])

    return pos_tile_images, neg_tile_images, pos_tile_indices, neg_tile_indices

# Visualize a given tile
def visualize_tile(tile_image):
    img = Image.fromarray(tile_image)
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    return

# Split the images into train, validation, and test
def split_data(pos_tile_images, neg_tile_images, pos_tile_indices, neg_tile_indices, test_size, seed):
    train_pos_tiles, temp_pos_tiles, train_pos_indices, temp_pos_indices = train_test_split(pos_tile_images, pos_tile_indices, test_size=test_size, random_state=seed)
    valid_pos_tiles, test_pos_tiles, valid_pos_indices, test_pos_indices = train_test_split(temp_pos_tiles, temp_pos_indices, test_size=0.5, random_state=seed)
    train_neg_tiles, temp_neg_tiles, train_neg_indices, temp_neg_indices = train_test_split(neg_tile_images, neg_tile_indices, test_size=test_size, random_state=seed)
    valid_neg_tiles, test_neg_tiles, valid_neg_indices, test_neg_indices = train_test_split(temp_neg_tiles, temp_neg_indices, test_size=0.5, random_state=seed)
    return train_pos_tiles, valid_pos_tiles, test_pos_tiles, train_neg_tiles, valid_neg_tiles, test_neg_tiles, train_pos_indices, valid_pos_indices, test_pos_indices, train_neg_indices, valid_neg_indices, test_neg_indices

def sample_tiles(tiles_csv, proportion, seed):

    pos_tiles, neg_tiles = read_CSVs(tiles_csv, proportion)
    pos_tile_images, neg_tile_images, pos_tile_indices, neg_tile_indices = create_tiles(pos_tiles, neg_tiles)
    test_size = 6 * proportion
    train_pos_tiles, valid_pos_tiles, test_pos_tiles, train_neg_tiles, valid_neg_tiles, test_neg_tiles, train_pos_indices, valid_pos_indices, test_pos_indices, train_neg_indices, valid_neg_indices, test_neg_indices = split_data(pos_tile_images, neg_tile_images, pos_tile_indices, neg_tile_indices, test_size, seed)

    return train_pos_tiles, valid_pos_tiles, test_pos_tiles, train_neg_tiles, valid_neg_tiles, test_neg_tiles, train_pos_indices, valid_pos_indices, test_pos_indices, train_neg_indices, valid_neg_indices, test_neg_indices