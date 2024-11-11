import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Knee OA Classification")
    parser.add_argument(
        "--backbone",
        type=str,
        choices=["resnet18", "resnet34", "resnet50"],
        default="resnet18",
    )
    parser.add_argument("--out_dir", "--out_dir", type=str, default="session")

    parser.add_argument("--csv_dir", "--csv_dir", default="data/CSVs")

    parser.add_argument(
        "--batch_size", "--batch_size", default=16, type=int, choices=[16, 32, 64]
    )
    parser.add_argument(
        "--lr", "--learning_rate", default=0.001, type=float, choices=[0.0001, 0.00001]
    )

    parser.add_argument("--epochs", "--epochs", default=20, type=int)

    args = parser.parse_args()

    return args
