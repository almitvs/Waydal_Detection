import csv
import random as rand
import matplotlib.pyplot as plt

# Read the information from CSVs and sample negative tiles
def read_CSVs(waydals_csv, tiles_csv, proportion):

    # Store the information
    locations = []
    pos_tiles = []
    neg_tiles = []

    # Get the waydal locations
    with open(waydals_csv, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            locations.append(row)

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

    return locations, pos_tiles, neg_samples

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

def sample_tiles():
    return