#!/usr/bin/env python3
import gzip
import pathlib
import shutil
import urllib.request


FILES = {
    "train-images-idx3-ubyte.gz": "train-images-idx3-ubyte",
    "train-labels-idx1-ubyte.gz": "train-labels-idx1-ubyte",
    "t10k-images-idx3-ubyte.gz": "t10k-images-idx3-ubyte",
    "t10k-labels-idx1-ubyte.gz": "t10k-labels-idx1-ubyte",
}

BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"


def main():
    data_dir = pathlib.Path(__file__).resolve().parents[1] / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    for gz_name, output_name in FILES.items():
        gz_path = data_dir / gz_name
        output_path = data_dir / output_name

        if output_path.exists():
            print(f"{output_path.name} already exists")
            continue

        print(f"Downloading {gz_name}")
        urllib.request.urlretrieve(BASE_URL + gz_name, gz_path)

        print(f"Extracting {output_name}")
        with gzip.open(gz_path, "rb") as source, output_path.open("wb") as target:
            shutil.copyfileobj(source, target)

        gz_path.unlink()


if __name__ == "__main__":
    main()

