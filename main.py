import os

import pandas as pd
import torch
from torch.utils.data import DataLoader

from args import get_args
from dataset import knee_Xray_dataset
from model import MyModel
from trainer import train_model


def main():
    args = get_args()
    print(args)

    for fold in range(5):
        print("Fold:", fold)

        train_set = pd.read_csv(os.path.join(args.csv_dir, f"fold_{fold}_train.csv"))
        val_set = pd.read_csv(os.path.join(args.csv_dir, f"fold_{fold}_val.csv"))

        # Create datasets and loaders
        train_dataset = knee_Xray_dataset(dataset=train_set)
        val_dataset = knee_Xray_dataset(dataset=val_set)

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=torch.cuda.is_available(),
        )
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=torch.cuda.is_available(),
        )

        # Initialize the model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MyModel(backbone=args.backbone).to(device)

        # Train the model for the current fold
        train_model(model, train_loader, val_loader, args, fold)


if __name__ == "__main__":
    main()
