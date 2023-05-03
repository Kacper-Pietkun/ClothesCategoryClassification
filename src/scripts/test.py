import os
import time
import argparse
from pathlib import Path
import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
# from src.classes.AlexNet import AlexNet
from src.classes.baisc_cnn import CNN
from src.classes.clothes_dataset import ClothesDataset
from tqdm import tqdm


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

    # model = AlexNet(num_classes=5).to(device)
    model = CNN(num_classes=5).to(device)

    model.load_state_dict(th.load(args.model_path))
    loss_fcn = nn.CrossEntropyLoss()

    tic = time.time()
    model.eval()
    test_loss = 0
    test_correct = 0
    test_total_processed = 0
    with th.no_grad():
        for step, (inputs, labels) in enumerate(tqdm(test_dataloader)):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_fcn(outputs, labels)
            test_loss += loss.item()
            test_correct += (th.argmax(outputs, dim=1) == th.argmax(labels, dim=1)).float().sum()
            test_total_processed += len(outputs)

    test_loss /= test_total_processed
    test_accuracy = 100. * test_correct / test_total_processed
    toc = time.time()
    print('Test Loss: {:.4f} Test Accuracy: {:.2f}% Time: {:.4f}'.format(test_loss, test_accuracy, toc - tic))
