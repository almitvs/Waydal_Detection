import argparse
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

    parser = argparse.ArgumentParser(prog='Train ResNet')
    parser.add_argument('--csv_file', type=str, required=True, help="Path to .csv file")
    parser.add_argument('--proportion', type=int, required=True, help="num_neg_tiles = proportion * (num_pos_tiles * num_augmentations)")
    parser.add_argument('--seed', type=int, required=True, help="Random seed")

    parser.add_argument('--learning_rate', type=float, required=True, help="Learning rate")
    parser.add_argument('--batch_size', type=int, required=True, help="Batch size")
    parser.add_argument('--num_epochs', type=int, required=True, help="Number of epochs")

    args = parser.parse_args()

    csv_file = args.csv_file
    proportion = args.proportion
    seed = args.seed
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    num_epochs = args.num_epochs

    train_pos_tiles, valid_pos_tiles, test_pos_tiles, train_neg_tiles, valid_neg_tiles, test_neg_tiles, train_pos_indices, valid_pos_indices, test_pos_indices, train_neg_indices, valid_neg_indices, test_neg_indices = sample_tiles(csv_file, proportion, seed)

    pos_transformations = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation([90, 90]),
        transforms.RandomRotation([180, 180]),
        transforms.RandomRotation([270, 270]),
        #transforms.Normalize(mean=mean, std=std),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

    neg_transformations = [
        #transforms.Normalize(mean=mean, std=std),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

    train_set = TileDataset(train_pos_tiles, train_neg_tiles, train_pos_indices, train_neg_indices, pos_transformations)
    valid_set = TileDataset(valid_pos_tiles, valid_neg_tiles, valid_pos_indices, valid_neg_indices)
    test_set = TileDataset(test_pos_tiles, test_neg_tiles, test_pos_indices, test_neg_indices)

    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    device = set_device()
    print(device)

    model = torchvision.models.resnet101(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 1)
    for param in model.fc.parameters():
        param.requires_grad = True

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=learning_rate)

    model = train_model(device, model, criterion, optimizer, num_epochs, train_dataloader, valid_dataloader)

    train_set = TileDataset(train_pos_tiles, train_neg_tiles, train_pos_indices, train_neg_indices) # Remove data augmentation from the training set
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    train_results, valid_results, test_results, total_results, tile_values = test_model(device, model, criterion, optimizer, batch_size, num_epochs, train_dataloader, valid_dataloader, test_dataloader)
    print(f"Training       Loss: {train_results[0]:.3f}     TP: {train_results[1]}     FN: {train_results[2]}     TN: {train_results[3]}     FP: {train_results[4]}")
    print(f"Validation     Loss: {valid_results[0]:.3f}     TP: {valid_results[1]}      FN: {valid_results[2]}      TN: {valid_results[3]}      FP: {valid_results[4]}")
    print(f"Testing        Loss: {test_results[0]:.3f}     TP: {test_results[1]}      FN: {test_results[2]}      TN: {test_results[3]}      FP: {test_results[4]}")
    print(f"Total          Loss: {total_results[0]:.3f}     TP: {total_results[1]}     FN: {total_results[2]}     TN: {total_results[3]}     FP: {total_results[4]}")

    # Calculate accuracy metrics
    train_metrics, valid_metrics, test_metrics, total_metrics = calculate_metrics(train_results, valid_results, test_results, total_results)
    print(f"Training       Accuracy: {train_metrics[0]:.3f}     Precision: {train_metrics[1]:.3f}     Recall: {train_metrics[2]:.3f}")
    print(f"Validation     Accuracy: {valid_metrics[0]:.3f}     Precision: {valid_metrics[1]:.3f}     Recall: {valid_metrics[2]:.3f}")
    print(f"Testing        Accuracy: {test_metrics[0]:.3f}     Precision: {test_metrics[1]:.3f}     Recall: {test_metrics[2]:.3f}")
    print(f"Total          Accuracy: {total_metrics[0]:.3f}     Precision: {total_metrics[1]:.3f}     Recall: {total_metrics[2]:.3f}")

    '''
    # Create KML file to visualize the results
    KML_file = 'kml_test.kml'
    shape_file = 'shp_test.shp'
    create_KML_and_Shapefile(KML_file, shape_file, tile_values, pos_tiles, neg_tiles)
    '''