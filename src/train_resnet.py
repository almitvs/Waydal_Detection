import geopandas as gpd
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import cuda
import torchvision
from torchvision import models, transforms
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import random as rand
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix as CM
import simplekml
from sample_tiles import visualize_tile
from sample_tiles import sample_tiles

# Split the images into train, validation, and test
def split_data(pos_tile_images, neg_tile_images, pos_tile_indices, neg_tile_indices, test_size, seed):
    train_pos_tiles, temp_pos_tiles, train_pos_indices, temp_pos_indices = train_test_split(pos_tile_images, pos_tile_indices, test_size=test_size, random_state=seed)
    valid_pos_tiles, test_pos_tiles, valid_pos_indices, test_pos_indices = train_test_split(temp_pos_tiles, temp_pos_indices, test_size=0.5, random_state=seed)
    train_neg_tiles, temp_neg_tiles, train_neg_indices, temp_neg_indices = train_test_split(neg_tile_images, neg_tile_indices, test_size=test_size, random_state=seed)
    valid_neg_tiles, test_neg_tiles, valid_neg_indices, test_neg_indices = train_test_split(temp_neg_tiles, temp_neg_indices, test_size=0.5, random_state=seed)
    return train_pos_tiles, valid_pos_tiles, test_pos_tiles, train_neg_tiles, valid_neg_tiles, test_neg_tiles, train_pos_indices, valid_pos_indices, test_pos_indices, train_neg_indices, valid_neg_indices, test_neg_indices

# Structure to handle the dataset
class TileDataset(Dataset):
    def __init__(self, pos_tile_images, neg_tile_images, pos_tile_indices, neg_tile_indices, pos_transformations=None, neg_transformations=None):
        self.tiles = []
        self.values = []
        self.indices = []
        tensorize = transforms.ToTensor()

        # Apply separate transformations
        for i, tile in enumerate(pos_tile_images):
            self.tiles.append(tensorize(tile))
            self.values.append(1)
            self.indices.append(pos_tile_indices[i])
            if pos_transformations != None:
                for transform in pos_transformations:
                    self.tiles.append(transform(tensorize(tile)))
                    self.values.append(1)
                    self.indices.append(pos_tile_indices[i])
        for i, tile in enumerate(neg_tile_images):
            self.tiles.append(tensorize(tile))
            self.values.append(0)
            self.indices.append(neg_tile_indices[i])
            if neg_transformations != None:
                for transform in neg_transformations:
                    self.tiles.append(transform(tensorize(tile)))
                    self.values.append(0)
                    self.indices.append(neg_tile_indices[i])

        # Combine and shuffle
        combined = list(zip(self.tiles, self.values, self.indices))
        rand.shuffle(combined)
        self.tiles, self.values, self.indices = zip(*combined)
        self.tiles = list(self.tiles)
        self.values = list(self.values)
        self.indices = list(self.indices)

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        return self.tiles[idx], self.values[idx], self.indices[idx]
    
# Use GPU if available
def set_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

# Visualize an image in tensor format
def visualize_tensor(tile):
    arr = tile.numpy()
    arr = np.transpose(arr, (1, 2, 0))
    arr = (arr * 255).astype(np.uint8)
    visualize_tile(arr)
    return

# Create a confusion matrix to determine accuracy
def create_confusion_matrix(y, y_hat):
    y = y.cpu().numpy()
    preds = (y_hat >= 0.5).long().cpu().numpy() # 50% confidence threshold
    tn, fp, fn, tp = CM(y, preds, labels=[0, 1]).ravel()
    return tp, fp, tn, fn

# Train the model
def train_model(device, model, criterion, optimizer, num_epochs, train_dataloader, valid_dataloader):

    # Iterate though the epochs
    for epoch in range(num_epochs):

        # Train the model on the train set
        model.train()
        train_loss = 0

        for tiles, vals, _ in train_dataloader:
            tiles, vals = tiles.to(device), vals.to(device)
            #visualize_tensor((rand.choice(tiles)).cpu())
            #return
            optimizer.zero_grad()
            y_hat = model(tiles)
            loss = criterion(y_hat, vals.unsqueeze(1).float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Evaluate the model on the validation set
        model.eval()
        valid_loss = 0
        confusion_matrix = [0,0,0,0] #TP FN TN FP
        with torch.no_grad():
            for tiles, vals, _ in valid_dataloader:
                tiles, vals = tiles.to(device), vals.to(device)
                y_hat = model(tiles)
                loss = criterion(y_hat, vals.unsqueeze(1).float())
                valid_loss += loss.item()
                tp, fp, tn, fn = create_confusion_matrix(vals, y_hat)
                confusion_matrix[0] += tp
                confusion_matrix[1] += fn
                confusion_matrix[2] += tn
                confusion_matrix[3] += fp

        # Average the losses
        train_loss /= len(train_dataloader)
        valid_loss /= len(valid_dataloader)

        #scheduler.step()

        # Print out the results
        print(f"Epoch: {(epoch + 1):03d}     Train Loss: {train_loss:.3f}     Valid Loss: {valid_loss:.3f}     TP: {confusion_matrix[0]}     FN: {confusion_matrix[1]}     TN: {confusion_matrix[2]}     FP: {confusion_matrix[3]}")

    return model

def train_resnet(batch_size, learning_rate, num_epochs, seed):
    return

# Create confusion matrix and track correspondence to tile
def create_tile_confusion_matrix(y, y_hat, indices):

    locations = []
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    y = y.cpu().numpy()
    preds = (y_hat >= 0.5).long().cpu().numpy() # 50% confidence threshold
    indices = indices.cpu().numpy()


    for i, val in enumerate(y):
        pred = preds[i]
        idx = indices[i]
        if val == 1:
            if pred == 1:
                locations.append((idx, 'tp'))
                tp += 1
            else:
                locations.append((idx, 'fn'))
                fn += 1
        else:
            if pred == 1:
                locations.append((idx, 'fp'))
                fp += 1
            else:
                locations.append((idx, 'tn'))
                tn += 1

    return tp, fp, tn, fn, locations

# Get the loss and accuracy metrics for each set
def test_model(device, model, criterion, optimizer, batch_size, num_epochs, train_dataloader, valid_dataloader, test_dataloader):
    dataloaders = [train_dataloader, valid_dataloader, test_dataloader]
    lists = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]] # train valid test total
    tile_values = [[], [], []] # train valid test
    for i, dataloader in enumerate(dataloaders):
        model.eval()
        with torch.no_grad():
            for tiles, vals, indices in dataloader:
                tiles, vals, indices = tiles.to(device), vals.to(device), indices.to(device)
                y_hat = model(tiles)
                loss = criterion(y_hat, vals.unsqueeze(1).float())
                tp, fp, tn, fn, locations = create_tile_confusion_matrix(vals, y_hat, indices)
                lists[i][0] += loss.item()
                lists[i][1] += tp
                lists[i][2] += fn
                lists[i][3] += tn
                lists[i][4] += fp
                lists[3][1] += tp
                lists[3][2] += fn
                lists[3][3] += tn
                lists[3][4] += fp
                tile_values[i] += locations
        lists[i][0] /= len(dataloader)
    lists[3][0] = (lists[0][0] + lists[1][0] + lists[2][0]) / 3

    return lists[0], lists[1], lists[2], lists[3], tile_values

# Calculate accuracy metrics
def calculate_metrics(train_results, valid_results, test_results, total_results):
    results = [train_results, valid_results, test_results, total_results]
    metrics = [[], [], [], []]
    for i, result in enumerate(results):
        tp = result[1]
        fn = result[2]
        tn = result[3]
        fp = result[4]
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        metrics[i].append(accuracy)
        metrics[i].append(precision)
        metrics[i].append(recall)
    return metrics[0], metrics[1], metrics[2], metrics[3]

# Create KML file to visualize the results
def create_KML_and_Shapefile(KML_file, shape_file, tile_values, pos_tiles, neg_tiles):

    kml = simplekml.Kml()
    shp_writer = shapefile.Writer(shape_file, shapeType=shapefile.POLYGON)
    shp_writer.autoBalance = 1
    shp_writer.field('Outcome', 'C')
    shp_writer.field('Set', 'C')

    for i, split in enumerate(tile_values):
        if i == 0:
            label = 'Training'
        elif i == 1:
            label = 'Validation'
        else:
            label = 'Testing'
        for tile in split:
            row = tile[0]
            val = tile[1]
            if val == 'tp':
                outcome = 'True Positive'
                color = '3300ff00'
                coords = (pos_tiles[row][5], pos_tiles[row][6], pos_tiles[row][7], pos_tiles[row][8]) # TL & BR
            elif val == 'fn':
                outcome = 'False Negative'
                color = '330000ff'
                coords = (pos_tiles[row][5], pos_tiles[row][6], pos_tiles[row][7], pos_tiles[row][8])
            elif val == 'fp':
                outcome = 'False Positive'
                color = '3300ffff'
                coords = (neg_tiles[row][5], neg_tiles[row][6], neg_tiles[row][7], neg_tiles[row][8])
            else: #tn
                outcome = 'True Negative'
                color = '33aaaaaa'
                coords = (neg_tiles[row][5], neg_tiles[row][6], neg_tiles[row][7], neg_tiles[row][8])
            corners = [(coords[0], coords[1]), (coords[2], coords[1]), (coords[2], coords[3]), (coords[0], coords[3]), (coords[0], coords[1])] # TL, TR, BR, BL, TL

            pol = kml.newpolygon(name=f"{outcome} ({label})", description=f"Outcome: {outcome}, Set: {label}", outerboundaryis=corners)
            pol.style.polystyle.color = color
            pol.style.polystyle.fill = 1
            pol.style.polystyle.outline = 0

            shp_writer.poly([[tuple(map(float, pair)) for pair in corners]])
            shp_writer.record(outcome, label)

    kml.save(KML_file)

    return

if __name__=="__main__":
    retrieve_locations()