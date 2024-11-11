import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from utils import calculate_metrics, plot_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(model, train_loader, val_loader, args, fold):
    # Set the model to training mode
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Initialize variables to track metrics and save best model
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_balanced_accuracy": [],
        "val_balanced_accuracy": [],
        "train_roc_auc": [],
        "val_roc_auc": [],
        "train_avg_precision": [],
        "val_avg_precision": [],
    }
    best_balanced_accuracy = 0

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        model.train()
        train_loss, train_targets, train_preds, train_probs = 0.0, [], [], []

        # Training loop
        for batch in train_loader:
            inputs = batch["img"].to(device)
            targets = batch["target"].to(device)
            optimizer.zero_grad()

            # Forward pass and loss computation
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = outputs.argmax(dim=1).cpu().numpy()
            probs = F.softmax(outputs, dim=1).detach().cpu().numpy()
            train_targets.extend(targets.cpu().numpy())
            train_preds.extend(preds)
            train_probs.extend(probs)

        # Calculate training metrics
        train_loss /= len(train_loader)
        train_balanced_accuracy, train_roc_auc, train_avg_precision = calculate_metrics(
            train_targets, train_preds, train_probs
        )

        # Validation loop and metrics calculation
        val_loss, val_balanced_accuracy, val_roc_auc, val_avg_precision = (
            validate_model(model, val_loader, criterion)
        )

        # Track metrics in history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_balanced_accuracy"].append(train_balanced_accuracy)
        history["val_balanced_accuracy"].append(val_balanced_accuracy)
        history["train_roc_auc"].append(train_roc_auc)
        history["val_roc_auc"].append(val_roc_auc)
        history["train_avg_precision"].append(train_avg_precision)
        history["val_avg_precision"].append(val_avg_precision)

        # Save the best model based on validation Balanced Accuracy
        if val_balanced_accuracy > best_balanced_accuracy:
            best_balanced_accuracy = val_balanced_accuracy
            torch.save(
                model.state_dict(),
                os.path.join(args.out_dir, f"best_model_fold_{fold}.pth"),
            )

    # Plot and save metrics after training
    plot_metrics(history, args.out_dir, fold)


def validate_model(model, val_loader, criterion):
    model.eval()
    val_loss, val_targets, val_preds, val_probs = 0.0, [], [], []

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["img"].to(device)
            targets = batch["target"].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            preds = outputs.argmax(dim=1).cpu().numpy()
            probs = F.softmax(outputs, dim=1).detach().cpu().numpy()
            val_targets.extend(targets.cpu().numpy())
            val_preds.extend(preds)
            val_probs.extend(probs)

    # Calculate validation metrics
    val_loss /= len(val_loader)
    val_balanced_accuracy, val_roc_auc, val_avg_precision = calculate_metrics(
        val_targets, val_preds, val_probs
    )

    return val_loss, val_balanced_accuracy, val_roc_auc, val_avg_precision
