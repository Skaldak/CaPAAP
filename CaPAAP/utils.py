import torch
from tqdm import tqdm


def train_model(model, criterion, train_loader, optimizer, scaler, device="cpu"):
    model.train()
    batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc="Train")

    total_loss = 0

    for i, data in enumerate(train_loader):
        optimizer.zero_grad()

        x, y = data
        x, y = x.to(device), y.to(device)

        with torch.cuda.amp.autocast():
            pred_x, pred_y = model(x, y=y)
            loss = criterion(x, y, pred_x, pred_y)

        total_loss += loss.item()

        batch_bar.set_postfix(
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
    total_loss = total_loss / len(train_loader)

    return total_loss


def validate_model(model, criterion, val_loader, device="cpu"):
    model.eval()
    batch_bar = tqdm(total=len(val_loader), dynamic_ncols=True, position=0, leave=False, desc="Val")

    total_loss = 0
    vdist = 0

    for i, data in enumerate(val_loader):
        x, y = data
        x, y = x.to(device), y.to(device)

        with torch.inference_mode():
            pred_ys, pred_y, pred_x = model(x)
            loss = criterion(x, y, pred_x, pred_ys)

        total_loss += float(loss)

        batch_bar.set_postfix(
            loss="{:.04f}".format(float(total_loss / (i + 1))), dist="{:.04f}".format(float(vdist / (i + 1)))
        )

        batch_bar.update()

        del x, y, loss
        torch.cuda.empty_cache()

    batch_bar.close()
    total_loss = total_loss / len(val_loader)

    return total_loss


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
    checkpoint = torch.load(path)
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

    return model, optimizer, scheduler, epoch
