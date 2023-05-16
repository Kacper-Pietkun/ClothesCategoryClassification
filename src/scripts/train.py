import os
import numpy as np
import time
import argparse
import datetime
import pandas as pd
from pathlib import Path
import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from src.classes.alex_net import AlexNet
from src.classes.baisc_cnn import CNN
from src.classes.metrics import Metrics
from src.classes.clothes_dataset import ClothesDataset
from src.classes.best_model_saver import BestModelSaver
from tqdm import tqdm


if __name__ == "__main__":
    argparser = argparse.ArgumentParser("Model training")
    argparser.add_argument("--gpu", type=bool, default=True)
    argparser.add_argument("--num-epochs", type=int, default=100)
    argparser.add_argument("--batch-size", type=int, default=32)
    argparser.add_argument("--learning-rate", type=float, default=0.001)
    argparser.add_argument("--optimization", type=str, default="sgd")
    argparser.add_argument("--log-every", type=int, default=1)
    argparser.add_argument("--mean", type=float, nargs='+',
                           help="mean that was calculated on training set. Three values one for each channel")
    argparser.add_argument("--std", type=float, nargs='+',
                           help="std that was calculated on training set. Three values one for each channel")
    argparser.add_argument("--dataset-path", type=str, required=True)
    argparser.add_argument("--save-models", type=bool, default=False)
    argparser.add_argument("--save-model-path", type=str)
    argparser.add_argument("--training-log-folder-path", type=str, required=True)
    args = argparser.parse_args()

    if not os.path.exists(args.dataset_path):
        raise ValueError(f"{args.dataset_path} is not a valid path.")
    if os.path.isfile(args.dataset_path):
        raise ValueError(f"{args.dataset_path} is not a path to a directory.")

    if args.gpu is True:
        device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
    else:
        device = th.device("cpu")

    train_path = Path(os.path.join(args.dataset_path, "Train"))
    val_path = Path(os.path.join(args.dataset_path, "Validation"))

    # Calculate mean and std if you didn't specify it
    # When running script for the first time, you need to calculate it, after that you can save it and use in
    # next runs (it is recommended because calculating takes some time). However, remember that mean and std is
    # unique for every dataset, and you should always calculate it using training set
    if args.mean is None or args.std is None:
        my_transforms = transforms.Compose([
            transforms.Resize((227, 227)),
            transforms.ToTensor(),
        ])

        train_dataset = ClothesDataset(train_path, transform=my_transforms)

        # Mean and std are calculated for training set only!
        print("Calculating mean")
        mean = np.mean([np.mean(x.numpy(), axis=(1, 2)) for x, _ in tqdm(train_dataset)], axis=0)
        print("Calculating std")
        std = np.mean([np.std(x.numpy(), axis=(1, 2)) for x, _ in tqdm(train_dataset)], axis=0)
        print(f"mean: {mean}, std: {std}")

        updated_transforms = transforms.Compose([
            transforms.Resize((227, 227)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        train_dataset = ClothesDataset(train_path, transform=updated_transforms)
        val_dataset = ClothesDataset(val_path, transform=updated_transforms)
    else:
        updated_transforms = transforms.Compose([
            transforms.Resize((227, 227)),
            transforms.ToTensor(),
            transforms.Normalize(mean=args.mean, std=args.std)
        ])

        train_dataset = ClothesDataset(train_path, transform=updated_transforms)
        val_dataset = ClothesDataset(val_path, transform=updated_transforms)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    num_classes = 5
    model = AlexNet(num_classes=num_classes).to(device)
    # model = CNN(num_classes=num_classes).to(device)

    metrics = Metrics()

    if args.save_models is True:
        best_model_saver_accuracy = BestModelSaver(f"{args.save_model_path}_accuracy.pth")
        best_model_saver_precision = BestModelSaver(f"{args.save_model_path}_precision.pth")
        best_model_saver_recall = BestModelSaver(f"{args.save_model_path}_recall.pth")
        best_model_saver_fscore = BestModelSaver(f"{args.save_model_path}_fscore.pth")
        best_model_saver_auc = BestModelSaver(f"{args.save_model_path}_auc.pth")

    loss_fcn = nn.CrossEntropyLoss()
    if args.optimization == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    elif args.optimization == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    else:
        raise Exception(f"There is not such optimizer as {args.optimization}")

    history = []
    for epoch in range(args.num_epochs):
        tic = time.time()
        model.train()
        train_loss = 0
        train_correct = 0
        y_true_train = th.empty(0).to(device)
        y_pred_train = th.empty(0).to(device)
        y_probas_train = th.empty((0, num_classes)).to(device)
        for step, (inputs, labels) in enumerate(train_dataloader):
            tic_step = time.time()
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fcn(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            train_correct += (th.argmax(outputs, dim=1) == th.argmax(labels, dim=1)).float().sum()
            y_true_train = th.cat((y_true_train, th.argmax(labels, dim=1)))
            y_pred_train = th.cat((y_pred_train, th.argmax(outputs, dim=1)))
            y_probas_train = th.cat((y_probas_train, outputs))
            if step % args.log_every == 0 and (step != 0 or args.log_every == 1):
                batch_time = len(outputs) / (time.time() - tic_step)
                batch_acc = (th.argmax(outputs, dim=1) == th.argmax(labels, dim=1)).float().sum() / len(inputs)
                batch_loss = loss.item()
                gpu_mem_alloc = (th.cuda.max_memory_allocated() / 1000000 if th.cuda.is_available() else 0)
                print(
                    "Epoch {:05d} | Step {:05d} ({:d}/{:d}) | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MB".format(
                        epoch,
                        step,
                        (step + 1) * args.batch_size if (step + 1) * args.batch_size <= len(train_dataloader.sampler) else len(train_dataloader.sampler),
                        len(train_dataloader.sampler),
                        batch_loss,
                        batch_acc,
                        batch_time,
                        gpu_mem_alloc,
                    )
                )

        model.eval()
        val_loss = 0
        val_correct = 0
        y_true_val = th.empty(0).to(device)
        y_pred_val = th.empty(0).to(device)
        y_probas_val = th.empty((0, num_classes)).to(device)
        with th.no_grad():
            for step, (inputs, labels) in enumerate(val_dataloader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = loss_fcn(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                val_correct += (th.argmax(outputs, dim=1) == th.argmax(labels, dim=1)).float().sum()
                y_true_val = th.cat((y_true_val, th.argmax(labels, dim=1)))
                y_pred_val = th.cat((y_pred_val, th.argmax(outputs, dim=1)))
                y_probas_val = th.cat((y_probas_val, outputs))

        metrics_train = metrics.get_classification_report_data(y_true_train, y_pred_train, train_dataset.classes)
        metrics_val = metrics.get_classification_report_data(y_true_val, y_pred_val, val_dataset.classes)
        train_auc = th.mean(metrics.get_auc_per_class(y_true_train, y_probas_train, num_classes)).item()
        val_auc = th.mean(metrics.get_auc_per_class(y_true_val, y_probas_val, num_classes)).item()
        train_loss /= len(train_dataloader.sampler)
        train_accuracy = 100. * train_correct / len(train_dataloader.sampler)
        val_loss /= len(val_dataloader.sampler)
        val_accuracy = 100. * val_correct / len(val_dataloader.sampler)
        history.append({'epoch': epoch,
                        'train_accuracy': train_accuracy.item(),
                        'val_accuracy': val_accuracy.item(),
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        "train_macro_precision": metrics_train["macro_precision"],
                        "val_macro_precision": metrics_val["macro_precision"],
                        "train_macro_recall": metrics_train["macro_recall"],
                        "val_macro_recall": metrics_val["macro_recall"],
                        "train_macro_fscore": metrics_train["macro_fscore"],
                        "val_macro_fscore": metrics_val["macro_fscore"],
                        "train_avg_auc": train_auc,
                        "val_avg_auc": val_auc})
        toc = time.time()
        if args.save_models is True:
            best_model_saver_accuracy(val_accuracy, model, optimizer, epoch)
            best_model_saver_precision(metrics_val["macro_precision"], model, optimizer, epoch)
            best_model_saver_recall(metrics_val["macro_recall"], model, optimizer, epoch)
            best_model_saver_fscore(metrics_val["macro_fscore"], model, optimizer, epoch)
            best_model_saver_auc(val_auc, model, optimizer, epoch)
        print('Epoch: {} Train Loss: {:.4f} Train Accuracy: {:.2f}% Validation Loss: {:.4f} Validation Accuracy: {:.2f}% Time: {:.4f}'.format(
            epoch, train_loss, train_accuracy, val_loss, val_accuracy, toc - tic))

    now = datetime.datetime.now()
    filename = 'log_{}_{}_{}_{}_{}_{}.csv'.format(now.year, now.month, now.day, now.hour, now.minute, now.second)
    save_log_path = os.path.join(args.training_log_folder_path, filename)
    history_df = pd.DataFrame(history)
    history_df.to_csv(save_log_path, index=False)
