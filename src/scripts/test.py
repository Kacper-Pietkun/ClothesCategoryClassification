import os
import time
import argparse
from pathlib import Path
import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
from src.classes.alex_net import AlexNet
from src.classes.baisc_cnn import CNN
from src.classes.clothes_dataset import ClothesDataset
from src.classes.metrics import Metrics


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

    num_classes = 5
    model = AlexNet(num_classes=num_classes).to(device)
    # model = CNN(num_classes=5).to(device)

    checkpoint = th.load(args.model_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    loss_fcn = nn.CrossEntropyLoss()

    tic = time.time()
    model.eval()
    test_loss = 0
    test_correct = 0
    y_true = th.empty(0).to(device)
    y_pred = th.empty(0).to(device)
    y_probas = th.empty((0, num_classes)).to(device)
    with th.no_grad():
        for step, (inputs, labels) in enumerate(tqdm(test_dataloader)):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_fcn(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            y_true = th.cat((y_true, th.argmax(labels, dim=1)))
            y_pred = th.cat((y_pred, th.argmax(outputs, dim=1)))
            y_probas = th.cat((y_probas, outputs))
            test_correct += (th.argmax(outputs, dim=1) == th.argmax(labels, dim=1)).float().sum()

    test_loss /= len(test_dataloader.sampler)
    test_accuracy = 100. * test_correct / len(test_dataloader.sampler)
    toc = time.time()
    print('Test Loss: {:.4f} Test Accuracy: {:.2f}% Time: {:.4f}'.format(test_loss, test_accuracy, toc - tic))

    metrics = Metrics()

    print("Confusion matrix")
    cm = metrics.get_confusion_matrix(y_true, y_pred)
    print(cm)

    print(metrics.get_classification_report_text(y_true, y_pred, test_dataset.classes))
    print(metrics.get_classification_report_data(y_true, y_pred, test_dataset.classes))

    print(f"Micro-avg accuracy: {metrics.get_micro_accuracy(cm)}")
    print(f"Macro-avg accuracy: {metrics.get_macro_accuracy(cm)}")

    auc_per_class = metrics.get_auc_per_class(y_true, y_probas, num_classes)
    print(f"AUC per class: {auc_per_class}")
    print(f"Average AUC: {th.mean(auc_per_class)}")

    fpr, tpr = metrics.get_fpr_tpr(y_true, y_probas, num_classes)

    metrics.plot_pretty_confusion_matrix(y_true, y_pred, test_dataset.classes)
    metrics.plot_roc_curves_torch(fpr, tpr, auc_per_class, test_dataset.classes)
