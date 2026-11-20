import cv2
import numpy as np

from tools.prepare_dataset import prepare_dataset


def _write_valid_pair(directory: str, stem: str, root_path):
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:20, 8:20] = 255
    cv2.imwrite(str(root_path / "JPEGImages" / f"{stem}.jpg"), image)
    cv2.imwrite(str(root_path / "SegmentationClass" / f"{stem}.png"), mask)


def test_prepare_dataset_accepts_multiple_source_roots(tmp_path):
    root_a = tmp_path / "src_a"
    root_b = tmp_path / "src_b"
    output = tmp_path / "prepared"

    for source in (root_a, root_b):
        (source / "JPEGImages").mkdir(parents=True)
        (source / "SegmentationClass").mkdir(parents=True)

    _write_valid_pair("img1", "img1", root_a)
    _write_valid_pair("img2", "img2", root_a)
    _write_valid_pair("img3", "img3", root_b)

    summary = prepare_dataset(source_roots=[root_a, root_b], output_dir=output)

    assert summary["paired_samples"] == 3
    assert (output / "images" / "train").exists()
    assert (output / "labels" / "train").exists()
    assert (output / "masks" / "train").exists()


def test_prepare_dataset_accepts_direct_images_and_masks_root(tmp_path):
    source = tmp_path / "seg_dataset"
    output = tmp_path / "prepared"
    (source / "images").mkdir(parents=True)
    (source / "masks").mkdir(parents=True)

    for index in range(3):
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[10:22, 10:22] = 255
        cv2.imwrite(str(source / "images" / f"sample_{index}.jpg"), image)
        cv2.imwrite(str(source / "masks" / f"sample_{index}.png"), mask)

    summary = prepare_dataset(source_roots=[source], output_dir=output)

    assert summary["paired_samples"] == 3
    assert (output / "images" / "train").exists()
    assert (output / "labels" / "train").exists()
    assert (output / "masks" / "train").exists()
