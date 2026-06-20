#!/usr/bin/env python3
from argparse import ArgumentParser
from difflib import get_close_matches
from pathlib import Path

from PIL import Image, ImageEnhance, ImageOps, ImageStat


IMAGE_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
BACKGROUND_THRESHOLD = 18
FOREGROUND_THRESHOLD = 35
MNIST_IMAGE_SIZE = 28
MNIST_DIGIT_SIZE = 22
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081
CONTRAST_FACTOR = 1.8


def border_mean(image):
    width, height = image.size
    border = max(1, min(width, height) // 10)
    regions = [
        image.crop((0, 0, width, border)),
        image.crop((0, height - border, width, height)),
        image.crop((0, 0, border, height)),
        image.crop((width - border, 0, width, height)),
    ]
    return sum(ImageStat.Stat(region).mean[0] for region in regions) / len(regions)


def should_invert(image, invert):
    # MNIST stores light digits on a dark background.
    if invert == "yes":
        return True
    if invert == "no":
        return False
    return border_mean(image) > 127


def improve_contrast(image):
    image = ImageOps.autocontrast(image, cutoff=2)
    image = ImageEnhance.Contrast(image).enhance(CONTRAST_FACTOR)
    return image.point(
        lambda pixel: 0 if pixel < BACKGROUND_THRESHOLD else min(255, int(pixel * 1.15))
    )


def center_digit_on_canvas(image):
    foreground = image.point(
        lambda pixel: 255 if pixel > FOREGROUND_THRESHOLD else 0,
        mode="L",
    )
    bbox = foreground.getbbox()

    if bbox is not None:
        image = image.crop(bbox)

    width, height = image.size
    scale = MNIST_DIGIT_SIZE / max(width, height)
    new_size = (
        max(1, round(width * scale)),
        max(1, round(height * scale)),
    )
    image = image.resize(new_size, Image.Resampling.LANCZOS)

    canvas = Image.new("L", (MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE), 0)
    left = (MNIST_IMAGE_SIZE - new_size[0]) // 2
    top = (MNIST_IMAGE_SIZE - new_size[1]) // 2
    canvas.paste(image, (left, top))
    return canvas


def preprocess_image(image_path, invert="auto"):
    image = Image.open(image_path).convert("L")
    image = ImageOps.autocontrast(image)

    if should_invert(image, invert):
        image = ImageOps.invert(image)

    image = improve_contrast(image)
    image = center_digit_on_canvas(image)
    image = improve_contrast(image)
    return image


def write_vector(image, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pixels = [((pixel / 255.0) - MNIST_MEAN) / MNIST_STD for pixel in image.getdata()]

    with output_path.open("w", encoding="utf-8") as file:
        for index, value in enumerate(pixels):
            file.write(f"{value:.8f}")
            file.write("\n" if (index + 1) % MNIST_IMAGE_SIZE == 0 else " ")


def is_image_file(path):
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def collect_images(input_path, recursive):
    if input_path.is_file():
        if not is_image_file(input_path):
            raise ValueError(f"Unsupported image extension: {input_path}")
        return [input_path]

    if not input_path.is_dir():
        message = f"Input path does not exist: {input_path}"
        parent = input_path.parent if input_path.parent != Path("") else Path(".")
        if parent.exists():
            names = [path.name for path in parent.iterdir()]
            matches = get_close_matches(input_path.name, names, n=3)
            if matches:
                suggestions = ", ".join(str(parent / match) for match in matches)
                message += f"\nDid you mean: {suggestions}?"
        raise FileNotFoundError(message)

    pattern = "**/*" if recursive else "*"
    return sorted(path for path in input_path.glob(pattern) if is_image_file(path))


def output_vector_path(image_path, input_path, output_path):
    if input_path.is_file():
        if output_path.suffix:
            return output_path
        return output_path / f"{image_path.stem}.txt"

    relative = image_path.relative_to(input_path)
    return output_path / relative.with_suffix(".txt")


def output_preview_path(image_path, input_path, preview_path):
    if preview_path is None:
        return None
    if input_path.is_file():
        if preview_path.suffix:
            return preview_path
        return preview_path / f"{image_path.stem}_28x28.png"

    relative = image_path.relative_to(input_path)
    return preview_path / relative.with_suffix(".png")


def process_one_image(image_path, vector_path, preview_path, invert):
    image = preprocess_image(image_path, invert=invert)
    write_vector(image, vector_path)

    if preview_path is not None:
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(preview_path)

    return vector_path, preview_path


def parse_args():
    parser = ArgumentParser(
        description="Preprocess digit images into 784 floats for the C++ CUDA MLP."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input image file or folder of images.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Deprecated alias for --input when processing one image.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/custom_digit.txt"),
        help="Output .txt file for one image, or output folder for a folder input.",
    )
    parser.add_argument(
        "--invert",
        choices=("auto", "yes", "no"),
        default="auto",
        help="Invert when your digit is dark on a light background.",
    )
    parser.add_argument(
        "--save-preprocessed",
        type=Path,
        default=None,
        help="Optional preview file or preview folder for the 28x28 images.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="When input is a folder, include images in nested folders.",
    )
    return parser.parse_args()


def main():
    try:
        args = parse_args()

        input_path = args.input or args.image
        if input_path is None:
            raise ValueError("pass --input IMAGE_OR_FOLDER")

        images = collect_images(input_path, recursive=args.recursive)
        if not images:
            raise ValueError(f"no supported image files found in {input_path}")

        for image_path in images:
            vector_path = output_vector_path(image_path, input_path, args.output)
            preview_path = output_preview_path(image_path, input_path, args.save_preprocessed)
            vector_path, preview_path = process_one_image(
                image_path=image_path,
                vector_path=vector_path,
                preview_path=preview_path,
                invert=args.invert,
            )

            print(f"Wrote vector: {vector_path}")
            if preview_path is not None:
                print(f"Wrote preview: {preview_path}")

        print(f"Processed {len(images)} image(s)")
    except (FileNotFoundError, OSError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from None


if __name__ == "__main__":
    main()
