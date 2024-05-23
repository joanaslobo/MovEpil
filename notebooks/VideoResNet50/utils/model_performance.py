import torch
from tqdm.auto import tqdm
from sklearn.metrics import classification_report
import numpy as np


def calculate_metrics(y_true, y_pred, labels):
    print("Classification report")
    print(classification_report(y_true=y_true, y_pred=y_pred, target_names=labels))


def get_predictions(model, preprocess, dataloader, device="cpu"):
    y_pred = []
    ground_truth = []

    model.train(False)
    with tqdm(total=len(dataloader)) as pbar:
        for batch in dataloader:
            x, y = batch

            x, y = x.to(device=device), y.to(device=device)

            if preprocess is not None:
                x = preprocess(x)

            # Per batch
            logits = model(x)
            labels = logits.softmax(dim=1).argmax(dim=1).detach().cpu().numpy()

            y_pred.extend(labels)
            ground_truth.extend(y.detach().cpu().numpy())

            pbar.update(1)

    return np.array(y_pred), np.array(ground_truth)


def plot_losses(train_epoch_loss, val_epoch_loss):
    import matplotlib.pyplot as plt

    plt.title("Training vs Validation losses during training.")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.plot(train_epoch_loss, label="Training Loss")
    plt.plot(val_epoch_loss, label="Validation Loss")
    plt.legend()
    plt.show()


