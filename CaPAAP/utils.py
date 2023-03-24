import torch
from tqdm import tqdm

from config import TRAIN_BATCH_SIZE, TEST_BATCH_SIZE


def train_model(model, criterion, train_loader, optimizer, scaler, device="cpu"):
    model.train()
    batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc="Train")

    num_correct = 0
    total_loss = 0

    for i, data in enumerate(train_loader):
        optimizer.zero_grad()

        x, y = data
        x, y = x.to(device), y.to(device)

        with torch.cuda.amp.autocast():
            pred_x, pred_y = model(x)
            loss = criterion(x, y, pred_x, pred_y)

        num_correct += (pred_y.argmax(dim=1) == y[:, y.shape[1] // 2].argmax(dim=1)).sum().item()
        total_loss += loss.item()

        batch_bar.set_postfix(
            num_correct=num_correct,
            acc="{:.04f}%".format(100 * num_correct / (TRAIN_BATCH_SIZE * (i + 1))),
            loss="{:.04f}".format(float(total_loss / (i + 1))),
            lr="{:.06f}".format(float(optimizer.param_groups[0]["lr"])),
        )

        batch_bar.update()  # Update tqdm bar

        # Another couple things you need for FP16.
        scaler.scale(loss).backward()  # This is a replacement for loss.backward()
        scaler.step(optimizer)  # This is a replacement for optimizer.step()
        scaler.update()  # This is something added just for FP16

        del x, y, loss
        torch.cuda.empty_cache()

    batch_bar.close()  # You need this to close the tqdm bar
    acc = 100 * num_correct / (TRAIN_BATCH_SIZE * len(train_loader))
    total_loss = total_loss / len(train_loader)

    return acc, total_loss


def validate_model(model, criterion, val_loader, device="cpu"):
    model.eval()
    batch_bar = tqdm(total=len(val_loader), dynamic_ncols=True, position=0, leave=False, desc="Val")

    num_correct = 0
    total_loss = 0
    vdist = 0

    for i, data in enumerate(val_loader):
        x, y = data
        x, y = x.to(device), y.to(device)

        with torch.inference_mode():
            pred_x, pred_y = model(x)
            loss = criterion(x, y, pred_x, pred_y)

        num_correct += (pred_y.argmax(dim=1) == y[:, y.shape[1] // 2].argmax(dim=1)).sum().item()
        total_loss += float(loss)

        batch_bar.set_postfix(
            acc="{:.04f}%".format(100 * num_correct / (TEST_BATCH_SIZE * (i + 1))),
            loss="{:.04f}".format(float(total_loss / (i + 1))),
            dist="{:.04f}".format(float(vdist / (i + 1))),
        )

        batch_bar.update()

        del x, y, loss
        torch.cuda.empty_cache()

    batch_bar.close()
    acc = 100 * num_correct / (TEST_BATCH_SIZE * len(val_loader))
    total_loss = total_loss / len(val_loader)

    return acc, total_loss


def save_model(model, optimizer, scheduler, epoch, path):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
        },
        path,
    )


def load_model(path, model, optimizer=None, scheduler=None):
    try:
        checkpoint = torch.load(path)
    except FileNotFoundError as e:
        print(f"Checkpoint not found: {e}")
        return

    try:
        print("Model loaded: ", model.load_state_dict(checkpoint["model_state_dict"]))
    except Exception as e:
        print(f"Model NOT loaded: {e}")

    if optimizer is not None:
        try:
            print("Optimizer loaded: ", optimizer.load_state_dict(checkpoint["optimizer_state_dict"]))
        except Exception as e:
            print(f"Optimizer NOT loaded: {e}")
    if scheduler is not None:
        try:
            print("Scheduler loaded: ", scheduler.load_state_dict(checkpoint["scheduler_state_dict"]))
        except Exception as e:
            print(f"Scheduler NOT loaded: {e}")

    epoch = checkpoint["epoch"]

    return epoch
