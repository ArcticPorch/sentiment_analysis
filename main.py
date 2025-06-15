import argparse
from src import train, predict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test"], required=True)
    args = parser.parse_args()

    if args.mode == "train":
        train.train_model()
    elif args.mode == "test":
        predict.predict()
