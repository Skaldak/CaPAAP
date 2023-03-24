import gc
import math
import os

import torch
from torch.utils.data import DataLoader
from torchsummaryX import summary

from config import *
from criterion import Criterion
from dataset import AcousticPhoneticDataset
from model import CapsuleNet
from utils import train_model, validate_model, save_model, load_model

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device: ", device)

if __name__ == "__main__":
    os.makedirs(CKPT_DIR, exist_ok=True)

    train_data = AcousticPhoneticDataset(split="train")
    val_data = AcousticPhoneticDataset(split="valid")

    train_loader = DataLoader(
        dataset=train_data, num_workers=4, batch_size=TRAIN_BATCH_SIZE, pin_memory=True, shuffle=True
    )
    val_loader = DataLoader(dataset=val_data, num_workers=2, batch_size=TEST_BATCH_SIZE, pin_memory=True, shuffle=False)

    print("Batch size: ", TRAIN_BATCH_SIZE, TEST_BATCH_SIZE)
    print("Train dataset samples = {}, batches = {}".format(train_data.__len__(), len(train_loader)))
    print("Val dataset samples = {}, batches = {}".format(val_data.__len__(), len(val_loader)))

    model = CapsuleNet(
        num_parameters=NUM_ACOUSTIC_PARAMETERS, num_classes=NUM_PHONEME_LOGITS, window_size=WINDOW_SIZE
    ).to(device)
    print(model)

    for data in train_loader:
        x, y = data
        summary(model, x.to(device))
        break

    criterion = Criterion()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # optimizer = torch.optim.SGD(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=LR * 0.01, factor=0.5)
    scaler = torch.cuda.amp.GradScaler()

    gc.collect()
    torch.cuda.empty_cache()

    start = 0
    end = NUM_EPOCHS
    best_valid_acc = 0.0
    load_model(os.path.join(CKPT_DIR, "checkpoint.pth"), model)

    for epoch in range(start, end):
        print("\nEpoch: {}/{}".format(epoch + 1, end))

        curr_lr = float(optimizer.param_groups[0]["lr"])

        train_acc, train_loss = train_model(model, criterion, train_loader, optimizer, scaler, device=device)
        valid_acc, valid_loss = validate_model(model, criterion, val_loader, device=device)
        scheduler.step(train_loss)

        print(
            "\nEpoch {}/{}: \nTrain Acc {:.04f}%\t Train Loss {:.04f}\t Learning Rate {:.04f}".format(
                epoch + 1, NUM_EPOCHS, train_acc, train_loss, curr_lr
            )
        )
        print("Val Acc {:.04f}%\t Val Loss {:.04f}".format(valid_acc, valid_loss))

        epoch_model_path = os.path.join(CKPT_DIR, f"checkpoint-{epoch}.pth")
        save_model(model, optimizer, scheduler, epoch, epoch_model_path)
        print("Saved epoch model")

        if valid_acc >= best_valid_acc:
            best_valid_acc = valid_acc
            best_model_path = os.path.join(CKPT_DIR, "checkpoint.pth")
            save_model(model, optimizer, scheduler, epoch, best_model_path)
            print("Saved best model")
