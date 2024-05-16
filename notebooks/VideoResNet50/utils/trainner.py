from tqdm.auto import tqdm
import torch.utils.data
from torcheval.metrics import MulticlassAccuracy, Mean


def train_silence(model, optimizer, criterion, epochs, train_dataset, val_dataset, device, num_classes, preprocess=None):
    train_history_epoch_loss = []
    val_history_epoch_loss = []

    train_epoch_loss = Mean().to("cpu")
    val_epoch_loss = Mean().to("cpu")

    # # micro is global
    train_epoch_metric = MulticlassAccuracy(average="micro", num_classes=num_classes, k=1).to("cpu")
    val_epoch_metric = MulticlassAccuracy(average="micro", num_classes=num_classes, k=1).to("cpu")

    with tqdm(total=epochs) as epoch_p_bar:
        for epoch in range(epochs):

            model.train()
            for train_batch in train_dataset:
                x, y = train_batch

                x, y = x.to(device=device), y.to(device=device)
                train_batch_len = len(x)

                if preprocess is not None:
                    x = preprocess(x)

                y_train_pred = model(x)
                train_loss = criterion(input=y_train_pred, target=y)

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                model.eval()
                with torch.no_grad():
                    train_epoch_metric.update(input=y_train_pred.softmax(dim=1).detach().cpu(), target=y.detach().cpu())
                    train_epoch_loss.update(train_loss.detach().cpu() * train_batch_len)
                model.train()

            model.eval()
            with torch.no_grad():
                for val_batch in val_dataset:
                    x_val, y_val = val_batch

                    x_val, y_val = x_val.to(device=device), y_val.to(device=device)
                    val_batch_len = len(x_val)

                    if preprocess is not None:
                        x_val = preprocess(x_val)

                    y_val_pred = model(x_val)
                    val_loss = criterion(input=y_val_pred, target=y_val)

                    #Update metrics
                    val_epoch_metric.update(input=y_val_pred.softmax(dim=1).detach().cpu(), target=y_val.detach().cpu())
                    val_epoch_loss.update(val_loss.detach().cpu() * val_batch_len)

            train_history_epoch_loss.append(train_epoch_loss.compute().item())
            val_history_epoch_loss.append(val_epoch_loss.compute().item())

            epoch_p_bar.set_description(f"[loss: {round(train_history_epoch_loss[-1], 4)} - val_loss: {round(val_history_epoch_loss[-1], 4)} | "
                                        f"Train accuracy: {round(train_epoch_metric.compute().item(), 2)} - Val accuracy: {round(val_epoch_metric.compute().item(), 2)}]")

            #Clean metrics state at the end of the epoch
            train_epoch_loss.reset()
            val_epoch_loss.reset()
            train_epoch_metric.reset()
            val_epoch_metric.reset()

            epoch_p_bar.update(1)

    model.train(False)
    return train_history_epoch_loss, val_history_epoch_loss

