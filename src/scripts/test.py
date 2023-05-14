import os
import time
import argparse
from pathlib import Path
import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import seaborn as sn
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tqdm import tqdm
from src.classes.alex_net import AlexNet
from src.classes.baisc_cnn import CNN
from src.classes.clothes_dataset import ClothesDataset


def micro_accuracy(cm):
    return np.trace(cm) / np.sum(cm)


def macro_accuracy(cm):
    accs = np.trace(cm) / (np.trace(cm) + np.sum(cm, axis=1) + np.sum(cm, axis=0) - 2*np.diag(cm))
    return np.mean(accs)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser("Model testing")
    argparser.add_argument("--gpu", type=bool, default=True)
    argparser.add_argument("--batch-size", type=int, default=1)
    argparser.add_argument("--dataset-path", type=str, required=True)
    argparser.add_argument("--model-path", type=str, required=True)
    argparser.add_argument("--mean", type=float, nargs='+', required=True,
                           help="mean that was calculated on training set. Three values one for each channel")
    argparser.add_argument("--std", type=float, nargs='+', required=True,
                           help="std that was calculated on training set. Three values one for each channel")
    args = argparser.parse_args()

    if not os.path.exists(args.dataset_path):
        raise ValueError(f"{args.dataset_path} is not a valid path.")
    if not os.path.exists(args.model_path):
        raise ValueError(f"{args.model_path} is not a valid path.")

    if args.gpu is True:
        device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
    else:
        device = th.device("cpu")

    my_transforms = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize(mean=args.mean, std=args.std)
    ])

    test_path = Path(os.path.join(args.dataset_path, "Test"))
    test_dataset = ClothesDataset(test_path, transform=my_transforms)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    model = AlexNet(num_classes=5).to(device)
    # model = CNN(num_classes=5).to(device)

    checkpoint = th.load(args.model_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    loss_fcn = nn.CrossEntropyLoss()

    tic = time.time()
    model.eval()
    test_loss = 0
    test_correct = 0
    y_pred = []
    y_true = []
    with th.no_grad():
        for step, (inputs, labels) in enumerate(tqdm(test_dataloader)):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_fcn(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            y_true.extend(th.argmax(labels, dim=1).tolist())
            y_pred.extend(th.argmax(outputs, dim=1).tolist())
            test_correct += (th.argmax(outputs, dim=1) == th.argmax(labels, dim=1)).float().sum()

    test_loss /= len(test_dataloader.sampler)
    test_accuracy = 100. * test_correct / len(test_dataloader.sampler)
    toc = time.time()
    print('Test Loss: {:.4f} Test Accuracy: {:.2f}% Time: {:.4f}'.format(test_loss, test_accuracy, toc - tic))

    confusion_matrix = confusion_matrix(y_true, y_pred)
    print("Confusion matrix")
    print(confusion_matrix)

    print("Classification report: (NOTE THAT FOR MULTI-CLASS ONE-LABEL CLASSIFICATION:"
          " MICRO-AVG ACCURACY == MICRO-AVG PRECISION == MICRO-AVG RECALL == MICRO-AVG F1-SCORE")
    print(classification_report(y_true, y_pred, target_names=test_dataset.classes))

    print(f"Micro-avg accuracy: {micro_accuracy(confusion_matrix)}")
    print(f"Macro-avg accuracy: {macro_accuracy(confusion_matrix)}")

    df_cm = pd.DataFrame(confusion_matrix / np.sum(confusion_matrix, axis=1),
                         index=[i for i in test_dataset.classes],
                         columns=[i for i in test_dataset.classes])
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True)
    plt.show()
