import argparse
from train_resnet import train_resnet
from train_dinov2 import train_dinov2


def train_model():
    parser = argparse.ArgumentParser(prog='Train a model to detect waydals')
    parser.add_argument('--model', type=str, required=True, help="ResNet or DINOv2")
    parser.add_argument('--bs', type=int, required=True, help="Batch size")
    parser.add_argument('--lr', type=float, required=True, help="Learning rate")
    parser.add_argument('--epochs', type=int, required=True, help="Number of epochs")
    parser.add_argument('--seed', type=int, required=True, help="Random seed")

    args = parser.parse_args()

    model = args[0]
    batch_size = args[1]
    learning_rate = args[2]
    num_epochs = args[3]
    seed = args[4]

    if model == "ResNet":
        train_resnet(batch_size, learning_rate, num_epochs, seed)

    elif model == "DINOv2":
        train_dinov2(batch_size, learning_rate, num_epochs, seed)

    return

if __name__=="__train_model__":
    train_model()