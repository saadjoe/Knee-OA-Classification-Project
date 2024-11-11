import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)


def calculate_metrics(targets, preds, probs):
    """Calculate Balanced Accuracy, ROC-AUC, and Average Precision for predictions."""
    balanced_acc = balanced_accuracy_score(targets, preds)
    roc_auc = roc_auc_score(targets, probs, multi_class="ovr")
    avg_precision = average_precision_score(targets, probs)
    return balanced_acc, roc_auc, avg_precision


def plot_metrics(history, out_dir, fold):
    """Generate and save training and validation metrics plots."""
    epochs = range(1, len(history["train_loss"]) + 1)

    # Plot Loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["train_loss"], label="Training Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig(f"{out_dir}/fold_{fold}_loss.png")
    plt.close()

    # Plot Balanced Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(
        epochs, history["train_balanced_accuracy"], label="Training Balanced Accuracy"
    )
    plt.plot(
        epochs, history["val_balanced_accuracy"], label="Validation Balanced Accuracy"
    )
    plt.xlabel("Epochs")
    plt.ylabel("Balanced Accuracy")
    plt.title("Training and Validation Balanced Accuracy")
    plt.legend()
    plt.savefig(f"{out_dir}/fold_{fold}_balanced_accuracy.png")
    plt.close()

    # Plot ROC-AUC
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["train_roc_auc"], label="Training ROC-AUC")
    plt.plot(epochs, history["val_roc_auc"], label="Validation ROC-AUC")
    plt.xlabel("Epochs")
    plt.ylabel("ROC-AUC")
    plt.title("Training and Validation ROC-AUC")
    plt.legend()
    plt.savefig(f"{out_dir}/fold_{fold}_roc_auc.png")
    plt.close()

    # Plot Average Precision
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["train_avg_precision"], label="Training Average Precision")
    plt.plot(epochs, history["val_avg_precision"], label="Validation Average Precision")
    plt.xlabel("Epochs")
    plt.ylabel("Average Precision")
    plt.title("Training and Validation Average Precision")
    plt.legend()
    plt.savefig(f"{out_dir}/fold_{fold}_avg_precision.png")
    plt.close()
