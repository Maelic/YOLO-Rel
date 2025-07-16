# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import json
import os
from collections import defaultdict
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset

from ultralytics.utils import LOCAL_RANK, LOGGER, NUM_THREADS, TQDM, colorstr
from ultralytics.utils.instance import Instances
from ultralytics.utils.ops import resample_segments, segments2boxes
from ultralytics.utils.torch_utils import TORCHVISION_0_18

from .augment import (
    Compose,
    Format,
    LetterBox,
    RandomLoadText,
    classify_augmentations,
    classify_transforms,
    v8_transforms,
)
from .base import BaseDataset
from .converter import merge_multi_segment
from .utils import (
    HELP_URL,
    check_file_speeds,
    get_hash,
    img2label_paths,
    load_dataset_cache_file,
    save_dataset_cache_file,
    verify_image,
    verify_image_label,
)

# Ultralytics dataset *.cache version, >= 1.0.0 for Ultralytics YOLO models
DATASET_CACHE_VERSION = "1.0.3"


class YOLODataset(BaseDataset):
    """
    Dataset class for loading object detection and/or segmentation labels in YOLO format.

    This class supports loading data for object detection, segmentation, pose estimation, and oriented bounding box
    (OBB) tasks using the YOLO format.

    Attributes:
        use_segments (bool): Indicates if segmentation masks should be used.
        use_keypoints (bool): Indicates if keypoints should be used for pose estimation.
        use_obb (bool): Indicates if oriented bounding boxes should be used.
        data (dict): Dataset configuration dictionary.

    Methods:
        cache_labels: Cache dataset labels, check images and read shapes.
        get_labels: Return dictionary of labels for YOLO training.
        build_transforms: Build and append transforms to the list.
        close_mosaic: Set mosaic, copy_paste and mixup options to 0.0 and build transformations.
        update_labels_info: Update label format for different tasks.
        collate_fn: Collate data samples into batches.

    Examples:
        >>> dataset = YOLODataset(img_path="path/to/images", data={"names": {0: "person"}}, task="detect")
        >>> dataset.get_labels()
    """

    def __init__(self, *args, data: Optional[Dict] = None, task: str = "detect", **kwargs):
        """
        Initialize the YOLODataset.

        Args:
            data (dict, optional): Dataset configuration dictionary.
            task (str): Task type, one of 'detect', 'segment', 'pose', or 'obb'.
            *args (Any): Additional positional arguments for the parent class.
            **kwargs (Any): Additional keyword arguments for the parent class.
        """
        self.use_segments = task == "segment"
        self.use_keypoints = task == "pose"
        self.use_obb = task == "obb"
        self.data = data
        assert not (self.use_segments and self.use_keypoints), "Can not use both segments and keypoints."
        super().__init__(*args, channels=self.data["channels"], **kwargs)

    def cache_labels(self, path: Path = Path("./labels.cache")) -> Dict:
        """
        Cache dataset labels, check images and read shapes.

        Args:
            path (Path): Path where to save the cache file.

        Returns:
            (dict): Dictionary containing cached labels and related information.
        """
        x = {"labels": []}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{self.prefix}Scanning {path.parent / path.stem}..."
        total = len(self.im_files)
        nkpt, ndim = self.data.get("kpt_shape", (0, 0))
        if self.use_keypoints and (nkpt <= 0 or ndim not in {2, 3}):
            raise ValueError(
                "'kpt_shape' in data.yaml missing or incorrect. Should be a list with [number of "
                "keypoints, number of dims (2 for x,y or 3 for x,y,visible)], i.e. 'kpt_shape: [17, 3]'"
            )
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(
                func=verify_image_label,
                iterable=zip(
                    self.im_files,
                    self.label_files,
                    repeat(self.prefix),
                    repeat(self.use_keypoints),
                    repeat(len(self.data["names"])),
                    repeat(nkpt),
                    repeat(ndim),
                    repeat(self.single_cls),
                ),
            )
            pbar = TQDM(results, desc=desc, total=total)
            for im_file, lb, shape, segments, keypoint, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x["labels"].append(
                        {
                            "im_file": im_file,
                            "shape": shape,
                            "cls": lb[:, 0:1],  # n, 1
                            "bboxes": lb[:, 1:],  # n, 4
                            "segments": segments,
                            "keypoints": keypoint,
                            "normalized": True,
                            "bbox_format": "xywh",
                        }
                    )
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            pbar.close()

        if msgs:
            LOGGER.info("\n".join(msgs))
        if nf == 0:
            LOGGER.warning(f"{self.prefix}No labels found in {path}. {HELP_URL}")
        x["hash"] = get_hash(self.label_files + self.im_files)
        x["results"] = nf, nm, ne, nc, len(self.im_files)
        x["msgs"] = msgs  # warnings
        save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
        return x

    def get_labels(self) -> List[Dict]:
        """
        Return dictionary of labels for YOLO training.

        This method loads labels from disk or cache, verifies their integrity, and prepares them for training.

        Returns:
            (List[dict]): List of label dictionaries, each containing information about an image and its annotations.
        """
        self.label_files = img2label_paths(self.im_files)
        cache_path = Path(self.label_files[0]).parent.with_suffix(".cache")
        try:
            cache, exists = load_dataset_cache_file(cache_path), True  # attempt to load a *.cache file
            assert cache["version"] == DATASET_CACHE_VERSION  # matches current version
            assert cache["hash"] == get_hash(self.label_files + self.im_files)  # identical hash
        except (FileNotFoundError, AssertionError, AttributeError):
            cache, exists = self.cache_labels(cache_path), False  # run cache ops

        # Display cache
        nf, nm, ne, nc, n = cache.pop("results")  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in {-1, 0}:
            d = f"Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            TQDM(None, desc=self.prefix + d, total=n, initial=n)  # display results
            if cache["msgs"]:
                LOGGER.info("\n".join(cache["msgs"]))  # display warnings

        # Read cache
        [cache.pop(k) for k in ("hash", "version", "msgs")]  # remove items
        labels = cache["labels"]
        if not labels:
            raise RuntimeError(
                f"No valid images found in {cache_path}. Images with incorrectly formatted labels are ignored. {HELP_URL}"
            )
        self.im_files = [lb["im_file"] for lb in labels]  # update im_files

        # Check if the dataset is all boxes or all segments
        lengths = ((len(lb["cls"]), len(lb["bboxes"]), len(lb["segments"])) for lb in labels)
        len_cls, len_boxes, len_segments = (sum(x) for x in zip(*lengths))
        if len_segments and len_boxes != len_segments:
            LOGGER.warning(
                f"Box and segment counts should be equal, but got len(segments) = {len_segments}, "
                f"len(boxes) = {len_boxes}. To resolve this only boxes will be used and all segments will be removed. "
                "To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset."
            )
            for lb in labels:
                lb["segments"] = []
        if len_cls == 0:
            LOGGER.warning(f"Labels are missing or empty in {cache_path}, training may not work correctly. {HELP_URL}")
        return labels

    def build_transforms(self, hyp: Optional[Dict] = None) -> Compose:
        """
        Build and append transforms to the list.

        Args:
            hyp (dict, optional): Hyperparameters for transforms.

        Returns:
            (Compose): Composed transforms.
        """
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            hyp.cutmix = hyp.cutmix if self.augment and not self.rect else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp)
        else:
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                return_obb=self.use_obb,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
                bgr=hyp.bgr if self.augment else 0.0,  # only affect training.
            )
        )
        return transforms

    def close_mosaic(self, hyp: Dict) -> None:
        """
        Disable mosaic, copy_paste, mixup and cutmix augmentations by setting their probabilities to 0.0.

        Args:
            hyp (dict): Hyperparameters for transforms.
        """
        hyp.mosaic = 0.0
        hyp.copy_paste = 0.0
        hyp.mixup = 0.0
        hyp.cutmix = 0.0
        self.transforms = self.build_transforms(hyp)

    def update_labels_info(self, label: Dict) -> Dict:
        """
        Update label format for different tasks.

        Args:
            label (dict): Label dictionary containing bboxes, segments, keypoints, etc.

        Returns:
            (dict): Updated label dictionary with instances.

        Note:
            cls is not with bboxes now, classification and semantic segmentation need an independent cls label
            Can also support classification and semantic segmentation by adding or removing dict keys there.
        """
        bboxes = label.pop("bboxes")
        segments = label.pop("segments", [])
        keypoints = label.pop("keypoints", None)
        bbox_format = label.pop("bbox_format")
        normalized = label.pop("normalized")

        # NOTE: do NOT resample oriented boxes
        segment_resamples = 100 if self.use_obb else 1000
        if len(segments) > 0:
            # make sure segments interpolate correctly if original length is greater than segment_resamples
            max_len = max(len(s) for s in segments)
            segment_resamples = (max_len + 1) if segment_resamples < max_len else segment_resamples
            # list[np.array(segment_resamples, 2)] * num_samples
            segments = np.stack(resample_segments(segments, n=segment_resamples), axis=0)
        else:
            segments = np.zeros((0, segment_resamples, 2), dtype=np.float32)
        label["instances"] = Instances(bboxes, segments, keypoints, bbox_format=bbox_format, normalized=normalized)
        return label

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        """
        Collate data samples into batches.

        Args:
            batch (List[dict]): List of dictionaries containing sample data.

        Returns:
            (dict): Collated batch with stacked tensors.
        """
        new_batch = {}
        batch = [dict(sorted(b.items())) for b in batch]  # make sure the keys are in the same order
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k in {"img", "text_feats"}:
                value = torch.stack(value, 0)
            elif k == "visuals":
                value = torch.nn.utils.rnn.pad_sequence(value, batch_first=True)
            if k in {"masks", "keypoints", "bboxes", "cls", "segments", "obb"}:
                value = torch.cat(value, 0)
            new_batch[k] = value
        new_batch["batch_idx"] = list(new_batch["batch_idx"])
        for i in range(len(new_batch["batch_idx"])):
            new_batch["batch_idx"][i] += i  # add target image index for build_targets()
        new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)
        return new_batch


class YOLOMultiModalDataset(YOLODataset):
    """
    Dataset class for loading object detection and/or segmentation labels in YOLO format with multi-modal support.

    This class extends YOLODataset to add text information for multi-modal model training, enabling models to
    process both image and text data.

    Methods:
        update_labels_info: Add text information for multi-modal model training.
        build_transforms: Enhance data transformations with text augmentation.

    Examples:
        >>> dataset = YOLOMultiModalDataset(img_path="path/to/images", data={"names": {0: "person"}}, task="detect")
        >>> batch = next(iter(dataset))
        >>> print(batch.keys())  # Should include 'texts'
    """

    def __init__(self, *args, data: Optional[Dict] = None, task: str = "detect", **kwargs):
        """
        Initialize a YOLOMultiModalDataset.

        Args:
            data (dict, optional): Dataset configuration dictionary.
            task (str): Task type, one of 'detect', 'segment', 'pose', or 'obb'.
            *args (Any): Additional positional arguments for the parent class.
            **kwargs (Any): Additional keyword arguments for the parent class.
        """
        super().__init__(*args, data=data, task=task, **kwargs)

    def update_labels_info(self, label: Dict) -> Dict:
        """
        Add text information for multi-modal model training.

        Args:
            label (dict): Label dictionary containing bboxes, segments, keypoints, etc.

        Returns:
            (dict): Updated label dictionary with instances and texts.
        """
        labels = super().update_labels_info(label)
        # NOTE: some categories are concatenated with its synonyms by `/`.
        # NOTE: and `RandomLoadText` would randomly select one of them if there are multiple words.
        labels["texts"] = [v.split("/") for _, v in self.data["names"].items()]

        return labels

    def build_transforms(self, hyp: Optional[Dict] = None) -> Compose:
        """
        Enhance data transformations with optional text augmentation for multi-modal training.

        Args:
            hyp (dict, optional): Hyperparameters for transforms.

        Returns:
            (Compose): Composed transforms including text augmentation if applicable.
        """
        transforms = super().build_transforms(hyp)
        if self.augment:
            # NOTE: hard-coded the args for now.
            # NOTE: this implementation is different from official yoloe,
            # the strategy of selecting negative is restricted in one dataset,
            # while official pre-saved neg embeddings from all datasets at once.
            transform = RandomLoadText(
                max_samples=min(self.data["nc"], 80),
                padding=True,
                padding_value=self._get_neg_texts(self.category_freq),
            )
            transforms.insert(-1, transform)
        return transforms

    @property
    def category_names(self):
        """
        Return category names for the dataset.

        Returns:
            (Set[str]): List of class names.
        """
        names = self.data["names"].values()
        return {n.strip() for name in names for n in name.split("/")}  # category names

    @property
    def category_freq(self):
        """Return frequency of each category in the dataset."""
        texts = [v.split("/") for v in self.data["names"].values()]
        category_freq = defaultdict(int)
        for label in self.labels:
            for c in label["cls"].squeeze(-1):  # to check
                text = texts[int(c)]
                for t in text:
                    t = t.strip()
                    category_freq[t] += 1
        return category_freq

    @staticmethod
    def _get_neg_texts(category_freq: Dict, threshold: int = 100) -> List[str]:
        """Get negative text samples based on frequency threshold."""
        return [k for k, v in category_freq.items() if v >= threshold]


class GroundingDataset(YOLODataset):
    """
    Dataset class for object detection tasks using annotations from a JSON file in grounding format.

    This dataset is designed for grounding tasks where annotations are provided in a JSON file rather than
    the standard YOLO format text files.

    Attributes:
        json_file (str): Path to the JSON file containing annotations.

    Methods:
        get_img_files: Return empty list as image files are read in get_labels.
        get_labels: Load annotations from a JSON file and prepare them for training.
        build_transforms: Configure augmentations for training with optional text loading.

    Examples:
        >>> dataset = GroundingDataset(img_path="path/to/images", json_file="annotations.json", task="detect")
        >>> len(dataset)  # Number of valid images with annotations
    """

    def __init__(self, *args, task: str = "detect", json_file: str = "", **kwargs):
        """
        Initialize a GroundingDataset for object detection.

        Args:
            json_file (str): Path to the JSON file containing annotations.
            task (str): Must be 'detect' or 'segment' for GroundingDataset.
            *args (Any): Additional positional arguments for the parent class.
            **kwargs (Any): Additional keyword arguments for the parent class.
        """
        assert task in {"detect", "segment"}, "GroundingDataset currently only supports `detect` and `segment` tasks"
        self.json_file = json_file
        super().__init__(*args, task=task, data={"channels": 3}, **kwargs)

    def get_img_files(self, img_path: str) -> List:
        """
        The image files would be read in `get_labels` function, return empty list here.

        Args:
            img_path (str): Path to the directory containing images.

        Returns:
            (list): Empty list as image files are read in get_labels.
        """
        return []

    def verify_labels(self, labels: List[Dict[str, Any]]) -> None:
        """
        Verify the number of instances in the dataset matches expected counts.

        This method checks if the total number of bounding box instances in the provided
        labels matches the expected count for known datasets. It performs validation
        against a predefined set of datasets with known instance counts.

        Args:
            labels (List[Dict[str, Any]]): List of label dictionaries, where each dictionary
                contains dataset annotations. Each label dict must have a 'bboxes' key with
                a numpy array or tensor containing bounding box coordinates.

        Raises:
            AssertionError: If the actual instance count doesn't match the expected count
                for a recognized dataset.

        Note:
            For unrecognized datasets (those not in the predefined expected_counts),
            a warning is logged and verification is skipped.
        """
        expected_counts = {
            "final_mixed_train_no_coco_segm": 3662412,
            "final_mixed_train_no_coco": 3681235,
            "final_flickr_separateGT_train_segm": 638214,
            "final_flickr_separateGT_train": 640704,
        }

        instance_count = sum(label["bboxes"].shape[0] for label in labels)
        for data_name, count in expected_counts.items():
            if data_name in self.json_file:
                assert instance_count == count, f"'{self.json_file}' has {instance_count} instances, expected {count}."
                return
        LOGGER.warning(f"Skipping instance count verification for unrecognized dataset '{self.json_file}'")

    def cache_labels(self, path: Path = Path("./labels.cache")) -> Dict[str, Any]:
        """
        Load annotations from a JSON file, filter, and normalize bounding boxes for each image.

        Args:
            path (Path): Path where to save the cache file.

        Returns:
            (Dict[str, Any]): Dictionary containing cached labels and related information.
        """
        x = {"labels": []}
        LOGGER.info("Loading annotation file...")
        with open(self.json_file) as f:
            annotations = json.load(f)
        images = {f"{x['id']:d}": x for x in annotations["images"]}
        img_to_anns = defaultdict(list)
        for ann in annotations["annotations"]:
            img_to_anns[ann["image_id"]].append(ann)
        for img_id, anns in TQDM(img_to_anns.items(), desc=f"Reading annotations {self.json_file}"):
            img = images[f"{img_id:d}"]
            h, w, f = img["height"], img["width"], img["file_name"]
            im_file = Path(self.img_path) / f
            if not im_file.exists():
                continue
            self.im_files.append(str(im_file))
            bboxes = []
            segments = []
            cat2id = {}
            texts = []
            for ann in anns:
                if ann["iscrowd"]:
                    continue
                box = np.array(ann["bbox"], dtype=np.float32)
                box[:2] += box[2:] / 2
                box[[0, 2]] /= float(w)
                box[[1, 3]] /= float(h)
                if box[2] <= 0 or box[3] <= 0:
                    continue

                caption = img["caption"]
                cat_name = " ".join([caption[t[0] : t[1]] for t in ann["tokens_positive"]]).lower().strip()
                if not cat_name:
                    continue

                if cat_name not in cat2id:
                    cat2id[cat_name] = len(cat2id)
                    texts.append([cat_name])
                cls = cat2id[cat_name]  # class
                box = [cls] + box.tolist()
                if box not in bboxes:
                    bboxes.append(box)
                    if ann.get("segmentation") is not None:
                        if len(ann["segmentation"]) == 0:
                            segments.append(box)
                            continue
                        elif len(ann["segmentation"]) > 1:
                            s = merge_multi_segment(ann["segmentation"])
                            s = (np.concatenate(s, axis=0) / np.array([w, h], dtype=np.float32)).reshape(-1).tolist()
                        else:
                            s = [j for i in ann["segmentation"] for j in i]  # all segments concatenated
                            s = (
                                (np.array(s, dtype=np.float32).reshape(-1, 2) / np.array([w, h], dtype=np.float32))
                                .reshape(-1)
                                .tolist()
                            )
                        s = [cls] + s
                        segments.append(s)
            lb = np.array(bboxes, dtype=np.float32) if len(bboxes) else np.zeros((0, 5), dtype=np.float32)

            if segments:
                classes = np.array([x[0] for x in segments], dtype=np.float32)
                segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in segments]  # (cls, xy1...)
                lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
            lb = np.array(lb, dtype=np.float32)

            x["labels"].append(
                {
                    "im_file": im_file,
                    "shape": (h, w),
                    "cls": lb[:, 0:1],  # n, 1
                    "bboxes": lb[:, 1:],  # n, 4
                    "segments": segments,
                    "normalized": True,
                    "bbox_format": "xywh",
                    "texts": texts,
                }
            )
        x["hash"] = get_hash(self.json_file)
        save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
        return x

    def get_labels(self) -> List[Dict]:
        """
        Load labels from cache or generate them from JSON file.

        Returns:
            (List[dict]): List of label dictionaries, each containing information about an image and its annotations.
        """
        cache_path = Path(self.json_file).with_suffix(".cache")
        try:
            cache, _ = load_dataset_cache_file(cache_path), True  # attempt to load a *.cache file
            assert cache["version"] == DATASET_CACHE_VERSION  # matches current version
            assert cache["hash"] == get_hash(self.json_file)  # identical hash
        except (FileNotFoundError, AssertionError, AttributeError, ModuleNotFoundError):
            cache, _ = self.cache_labels(cache_path), False  # run cache ops
        [cache.pop(k) for k in ("hash", "version")]  # remove items
        labels = cache["labels"]
        self.verify_labels(labels)
        self.im_files = [str(label["im_file"]) for label in labels]
        if LOCAL_RANK in {-1, 0}:
            LOGGER.info(f"Load {self.json_file} from cache file {cache_path}")
        return labels

    def build_transforms(self, hyp: Optional[Dict] = None) -> Compose:
        """
        Configure augmentations for training with optional text loading.

        Args:
            hyp (dict, optional): Hyperparameters for transforms.

        Returns:
            (Compose): Composed transforms including text augmentation if applicable.
        """
        transforms = super().build_transforms(hyp)
        if self.augment:
            # NOTE: hard-coded the args for now.
            # NOTE: this implementation is different from official yoloe,
            # the strategy of selecting negative is restricted in one dataset,
            # while official pre-saved neg embeddings from all datasets at once.
            transform = RandomLoadText(
                max_samples=80,
                padding=True,
                padding_value=self._get_neg_texts(self.category_freq),
            )
            transforms.insert(-1, transform)
        return transforms

    @property
    def category_names(self):
        """Return unique category names from the dataset."""
        return {t.strip() for label in self.labels for text in label["texts"] for t in text}

    @property
    def category_freq(self):
        """Return frequency of each category in the dataset."""
        category_freq = defaultdict(int)
        for label in self.labels:
            for text in label["texts"]:
                for t in text:
                    t = t.strip()
                    category_freq[t] += 1
        return category_freq

    @staticmethod
    def _get_neg_texts(category_freq: Dict, threshold: int = 100) -> List[str]:
        """Get negative text samples based on frequency threshold."""
        return [k for k, v in category_freq.items() if v >= threshold]


class YOLOConcatDataset(ConcatDataset):
    """
    Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets for YOLO training, ensuring they use the same
    collation function.

    Methods:
        collate_fn: Static method that collates data samples into batches using YOLODataset's collation function.

    Examples:
        >>> dataset1 = YOLODataset(...)
        >>> dataset2 = YOLODataset(...)
        >>> combined_dataset = YOLOConcatDataset([dataset1, dataset2])
    """

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        """
        Collate data samples into batches.

        Args:
            batch (List[dict]): List of dictionaries containing sample data.

        Returns:
            (dict): Collated batch with stacked tensors.
        """
        return YOLODataset.collate_fn(batch)

    def close_mosaic(self, hyp: Dict) -> None:
        """
        Set mosaic, copy_paste and mixup options to 0.0 and build transformations.

        Args:
            hyp (dict): Hyperparameters for transforms.
        """
        for dataset in self.datasets:
            if not hasattr(dataset, "close_mosaic"):
                continue
            dataset.close_mosaic(hyp)


# TODO: support semantic segmentation
class SemanticDataset(BaseDataset):
    """Semantic Segmentation Dataset."""

    def __init__(self):
        """Initialize a SemanticDataset object."""
        super().__init__()


class ClassificationDataset:
    """
    Dataset class for image classification tasks extending torchvision ImageFolder functionality.

    This class offers functionalities like image augmentation, caching, and verification. It's designed to efficiently
    handle large datasets for training deep learning models, with optional image transformations and caching mechanisms
    to speed up training.

    Attributes:
        cache_ram (bool): Indicates if caching in RAM is enabled.
        cache_disk (bool): Indicates if caching on disk is enabled.
        samples (list): A list of tuples, each containing the path to an image, its class index, path to its .npy cache
                        file (if caching on disk), and optionally the loaded image array (if caching in RAM).
        torch_transforms (callable): PyTorch transforms to be applied to the images.
        root (str): Root directory of the dataset.
        prefix (str): Prefix for logging and cache filenames.

    Methods:
        __getitem__: Return subset of data and targets corresponding to given indices.
        __len__: Return the total number of samples in the dataset.
        verify_images: Verify all images in dataset.
    """

    def __init__(self, root: str, args, augment: bool = False, prefix: str = ""):
        """
        Initialize YOLO classification dataset with root directory, arguments, augmentations, and cache settings.

        Args:
            root (str): Path to the dataset directory where images are stored in a class-specific folder structure.
            args (Namespace): Configuration containing dataset-related settings such as image size, augmentation
                parameters, and cache settings.
            augment (bool, optional): Whether to apply augmentations to the dataset.
            prefix (str, optional): Prefix for logging and cache filenames, aiding in dataset identification.
        """
        import torchvision  # scope for faster 'import ultralytics'

        # Base class assigned as attribute rather than used as base class to allow for scoping slow torchvision import
        if TORCHVISION_0_18:  # 'allow_empty' argument first introduced in torchvision 0.18
            self.base = torchvision.datasets.ImageFolder(root=root, allow_empty=True)
        else:
            self.base = torchvision.datasets.ImageFolder(root=root)
        self.samples = self.base.samples
        self.root = self.base.root

        # Initialize attributes
        if augment and args.fraction < 1.0:  # reduce training fraction
            self.samples = self.samples[: round(len(self.samples) * args.fraction)]
        self.prefix = colorstr(f"{prefix}: ") if prefix else ""
        self.cache_ram = args.cache is True or str(args.cache).lower() == "ram"  # cache images into RAM
        if self.cache_ram:
            LOGGER.warning(
                "Classification `cache_ram` training has known memory leak in "
                "https://github.com/ultralytics/ultralytics/issues/9824, setting `cache_ram=False`."
            )
            self.cache_ram = False
        self.cache_disk = str(args.cache).lower() == "disk"  # cache images on hard drive as uncompressed *.npy files
        self.samples = self.verify_images()  # filter out bad images
        self.samples = [list(x) + [Path(x[0]).with_suffix(".npy"), None] for x in self.samples]  # file, index, npy, im
        scale = (1.0 - args.scale, 1.0)  # (0.08, 1.0)
        self.torch_transforms = (
            classify_augmentations(
                size=args.imgsz,
                scale=scale,
                hflip=args.fliplr,
                vflip=args.flipud,
                erasing=args.erasing,
                auto_augment=args.auto_augment,
                hsv_h=args.hsv_h,
                hsv_s=args.hsv_s,
                hsv_v=args.hsv_v,
            )
            if augment
            else classify_transforms(size=args.imgsz)
        )

    def __getitem__(self, i: int) -> Dict:
        """
        Return subset of data and targets corresponding to given indices.

        Args:
            i (int): Index of the sample to retrieve.

        Returns:
            (dict): Dictionary containing the image and its class index.
        """
        f, j, fn, im = self.samples[i]  # filename, index, filename.with_suffix('.npy'), image
        if self.cache_ram:
            if im is None:  # Warning: two separate if statements required here, do not combine this with previous line
                im = self.samples[i][3] = cv2.imread(f)
        elif self.cache_disk:
            if not fn.exists():  # load npy
                np.save(fn.as_posix(), cv2.imread(f), allow_pickle=False)
            im = np.load(fn)
        else:  # read image
            im = cv2.imread(f)  # BGR
        # Convert NumPy array to PIL image
        im = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        sample = self.torch_transforms(im)
        return {"img": sample, "cls": j}

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.samples)

    def verify_images(self) -> List[Tuple]:
        """
        Verify all images in dataset.

        Returns:
            (list): List of valid samples after verification.
        """
        desc = f"{self.prefix}Scanning {self.root}..."
        path = Path(self.root).with_suffix(".cache")  # *.cache file path

        try:
            check_file_speeds([file for (file, _) in self.samples[:5]], prefix=self.prefix)  # check image read speeds
            cache = load_dataset_cache_file(path)  # attempt to load a *.cache file
            assert cache["version"] == DATASET_CACHE_VERSION  # matches current version
            assert cache["hash"] == get_hash([x[0] for x in self.samples])  # identical hash
            nf, nc, n, samples = cache.pop("results")  # found, missing, empty, corrupt, total
            if LOCAL_RANK in {-1, 0}:
                d = f"{desc} {nf} images, {nc} corrupt"
                TQDM(None, desc=d, total=n, initial=n)
                if cache["msgs"]:
                    LOGGER.info("\n".join(cache["msgs"]))  # display warnings
            return samples

        except (FileNotFoundError, AssertionError, AttributeError):
            # Run scan if *.cache retrieval failed
            nf, nc, msgs, samples, x = 0, 0, [], [], {}
            with ThreadPool(NUM_THREADS) as pool:
                results = pool.imap(func=verify_image, iterable=zip(self.samples, repeat(self.prefix)))
                pbar = TQDM(results, desc=desc, total=len(self.samples))
                for sample, nf_f, nc_f, msg in pbar:
                    if nf_f:
                        samples.append(sample)
                    if msg:
                        msgs.append(msg)
                    nf += nf_f
                    nc += nc_f
                    pbar.desc = f"{desc} {nf} images, {nc} corrupt"
                pbar.close()
            if msgs:
                LOGGER.info("\n".join(msgs))
            x["hash"] = get_hash([x[0] for x in self.samples])
            x["results"] = nf, nc, len(samples), samples
            x["msgs"] = msgs  # warnings
            save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
            return samples

class YOLORelationDataset(YOLODataset):
    """
    Enhanced YOLO dataset with comprehensive support for relationship annotations between objects.
    
    This dataset extends YOLODataset to support relationship detection training with features:
    - Relationship class mapping and vocabulary
    - Negative sampling for relationship pairs
    - Support for hierarchical relationship classes
    - Integration with open-vocabulary text embeddings
    - Proper handling of object-relationship consistency during augmentation
    - Dataset statistics and class balancing utilities
    
    Attributes:
        relationships (dict): Mapping from image paths to relationship annotations
        relation_file (str): Path to relationship annotations JSON file
        relation_vocab (dict): Mapping from relation names to class IDs
        num_relation_classes (int): Number of relationship classes
        negative_sampling (bool): Whether to generate negative relationship samples
        max_relations_per_image (int): Maximum number of relations per image
        relation_stats (dict): Dataset statistics for relationship classes
        
    Examples:
        >>> dataset = YOLORelationDataset(
        ...     img_path="path/to/images",
        ...     data={"names": {0: "person", 1: "car"}, "nc": 2, "rel_names": {0: "on", 1: "in"}, "nr": 2},
        ...     task="relation",
        ...     relation_file="path/to/relations.json"
        ... )
        >>> sample = dataset[0]  # Get sample with relationships
        >>> print(sample.keys())  # Should include 'relations', 'relation_labels', etc.
    """
    
    def __init__(
        self,
        *args,
        relation_file: Optional[str] = None,
        task: str = "relation",
        negative_sampling: bool = True,
        max_relations_per_image: int = 100,
        **kwargs
    ):
        """
        Initialize YOLORelationDataset with relationship support.
        
        Args:
            relation_file (str, optional): Path to relationship annotations JSON file
            relation_vocab (Dict[str, int], optional): Mapping from relation names to IDs
            negative_sampling (bool): Whether to generate negative relationship samples
            max_relations_per_image (int): Maximum number of relations per image
            *args: Additional positional arguments for parent class
            **kwargs: Additional keyword arguments for parent class
        """
        assert task == "relation", "YOLORelationDataset only supports 'relation' task"
        super().__init__(*args, **kwargs)

        self.relation_file = relation_file
        self.negative_sampling = negative_sampling
        self.max_relations_per_image = max_relations_per_image
        self.relationships = {}  # Mapping from image path to relationship annotations
        
        # Initialize relation vocabulary from data config
        self.relation_vocab = {}
        self.id_to_relation = {}
        self._num_relation_classes = 0
        
        if self.data and 'rel_names' in self.data:
            self.relation_vocab = {v: k for k, v in self.data['rel_names'].items()}
            self.id_to_relation = self.data['rel_names']
            self._num_relation_classes = len(self.data['rel_names'])
        else:
            LOGGER.warning("No relation vocabulary found in data config")
        
        # Load relationship annotations
        if relation_file:
            self._load_relations()
        
        # Initialize statistics
        self.relation_stats = {}
        if self.relationships:
            self._compute_dataset_stats()
        
        LOGGER.info(f"YOLORelationDataset initialized with {self.num_relation_classes} relation classes")

    def _load_relations(self):
        """Load relationship annotations from a JSON file."""
        if not self.relation_file or not os.path.exists(self.relation_file):
            LOGGER.warning(f"Relation file {self.relation_file} not found.")
            return

        try:
            with open(self.relation_file, 'r') as f:
                relations_data = json.load(f)
            
            # Map relations to full image paths
            loaded_count = 0
            images_with_relations = 0
            
            for img_path in self.im_files:
                img_name = Path(img_path).name.split('.jpg')[0]  # Get just the filename
                if img_name in relations_data:
                    # Convert to numpy array
                    relations = np.array(relations_data[img_name], dtype=int)
                    self.relationships[img_path] = relations
                    loaded_count += len(relations)
                    if len(relations) > 0:
                        images_with_relations += 1
                else:
                    self.relationships[img_path] = np.zeros((0, 3), dtype=int)
                    
            LOGGER.info(f"Loaded {loaded_count} relations for {images_with_relations}/{len(self.im_files)} images")
        
        except Exception as e:
            LOGGER.error(f"Failed to load relation file: {e}")
            # Initialize with empty relations
            for img_path in self.im_files:
                self.relationships[img_path] = np.zeros((0, 3), dtype=int)

    def _compute_dataset_stats(self):
        """Compute dataset statistics for relationship classes."""
        relation_counts = {}
        total_relations = 0
        images_with_relations = 0
        
        for img_path, relations in self.relationships.items():
            if len(relations) > 0:
                images_with_relations += 1
                
            for _, _, rel_class in relations:
                if rel_class in self.id_to_relation:
                    rel_name = self.id_to_relation[rel_class]
                    relation_counts[rel_name] = relation_counts.get(rel_name, 0) + 1
                    total_relations += 1
        
        self.relation_stats = {
            "total_relations": total_relations,
            "images_with_relations": images_with_relations,
            "relation_counts": relation_counts,
            "average_relations_per_image": total_relations / len(self.im_files) if self.im_files else 0,
            "average_relations_per_image_with_relations": total_relations / images_with_relations if images_with_relations > 0 else 0
        }
        
        LOGGER.info(f"Relation stats: {total_relations} total relations, "
                   f"{self.relation_stats['average_relations_per_image']:.2f} avg per image")

    def generate_negative_samples(self, positive_relations: torch.Tensor, num_objects: int) -> torch.Tensor:
        """
        Generate negative relationship samples for balanced training.
        
        Args:
            positive_relations (torch.Tensor): Positive relations [N, 3] (subj, obj, rel)
            num_objects (int): Number of objects in the image
            
        Returns:
            torch.Tensor: Negative relations [M, 3] where M is number of negative samples
        """
        if num_objects < 2:
            return torch.zeros((0, 3), dtype=torch.long)
        
        # Create set of positive pairs
        positive_pairs = set()
        for subj, obj, rel in positive_relations:
            positive_pairs.add((subj.item(), obj.item()))
        
        # Generate all possible pairs
        all_pairs = []
        for i in range(num_objects):
            for j in range(num_objects):
                if i != j and (i, j) not in positive_pairs:
                    all_pairs.append((i, j))
        
        # Sample negative pairs (up to same number as positive relations)
        num_negatives = min(len(all_pairs), len(positive_relations) * 2)
        if num_negatives > 0:
            negative_indices = torch.randperm(len(all_pairs))[:num_negatives]
            negative_pairs = [all_pairs[i] for i in negative_indices]
            
            # Assign background class (assuming last class is background)
            background_class = self.relation_vocab.get("background", self.num_relation_classes - 1)
            negative_relations = torch.tensor([
                [subj, obj, background_class] for subj, obj in negative_pairs
            ], dtype=torch.long)
            
            return negative_relations
        
        return torch.zeros((0, 3), dtype=torch.long)

    def filter_relations_by_objects(self, relations: np.ndarray, valid_objects: np.ndarray) -> np.ndarray:
        """
        Filter relations to only include those between valid objects after augmentation.
        
        Args:
            relations (np.ndarray): Relations [N, 3] (subj, obj, rel)
            valid_objects (np.ndarray): Valid object indices after augmentation
            
        Returns:
            np.ndarray: Filtered relations with updated indices
        """
        if len(relations) == 0 or len(valid_objects) == 0:
            return np.zeros((0, 3), dtype=int)
        
        # Create mapping from old indices to new indices
        old_to_new = {}
        for new_idx, old_idx in enumerate(valid_objects):
            old_to_new[old_idx] = new_idx
        
        # Filter and remap relations
        valid_relations = []
        for subj, obj, rel in relations:
            if subj in old_to_new and obj in old_to_new:
                valid_relations.append([old_to_new[subj], old_to_new[obj], rel])
        
        return np.array(valid_relations, dtype=int) if valid_relations else np.zeros((0, 3), dtype=int)

    def create_relation_targets(self, relations: torch.Tensor, num_objects: int) -> Dict[str, torch.Tensor]:
        """
        Create relationship targets for training.
        
        Args:
            relations (torch.Tensor): Relations [N, 3] (subj, obj, rel)
            num_objects (int): Number of objects in the image
            
        Returns:
            Dict[str, torch.Tensor]: Relationship targets and metadata
        """
        if num_objects < 2:
            return {
                "relation_labels": torch.zeros((0,), dtype=torch.long),
                "object_pairs": torch.zeros((0, 2), dtype=torch.long),
                "num_relations": 0
            }
        
        # Add negative samples if enabled
        if self.negative_sampling:
            negative_relations = self.generate_negative_samples(relations, num_objects)
            all_relations = torch.cat([relations, negative_relations]) if len(relations) > 0 else negative_relations
        else:
            all_relations = relations
        
        # Handle empty relations case
        if len(all_relations) == 0:
            return {
                "relation_labels": torch.zeros((0,), dtype=torch.long),
                "object_pairs": torch.zeros((0, 2), dtype=torch.long),
                "num_relations": 0
            }
        
        # Ensure tensor is 2D
        if all_relations.dim() == 1:
            all_relations = all_relations.unsqueeze(0)
        
        # Limit number of relations per image
        if len(all_relations) > self.max_relations_per_image:
            indices = torch.randperm(len(all_relations))[:self.max_relations_per_image]
            all_relations = all_relations[indices]
        
        # Extract components
        object_pairs = all_relations[:, :2]  # [N, 2]
        relation_labels = all_relations[:, 2]  # [N]
        
        return {
            "relation_labels": relation_labels,
            "object_pairs": object_pairs,
            "num_relations": len(all_relations)
        }

    def _map_augmented_index(self, old_idx):
        """Map an object index from before augmentation to after augmentation."""
        if not hasattr(self, 'last_indices_mapping'):
            return None
        return self.last_indices_mapping.get(old_idx, None)

    def __getitem__(self, index: int) -> Dict:
        """
        Get training sample with relationship annotations.
        
        Args:
            index (int): Sample index
            
        Returns:
            Dict: Training sample with image, labels, and relationship data
        """
        # Get base sample from parent class
        result = super().__getitem__(index)
            
        # Get image path and corresponding relations
        img_path = self.im_files[index]
        relations = self.relationships.get(img_path, np.zeros((0, 3), dtype=int)).copy()
        
        # Convert to tensor immediately
        relations = torch.from_numpy(relations).long()
        
        # Handle augmentation transforms for relationships
        if self.augment and len(relations) > 0 and hasattr(self, 'last_indices_mapping'):
            # When objects are removed or reordered during augmentation,
            # we need to update the relation indices accordingly
            valid_relations = []
            for subj_idx, obj_idx, rel_class in relations:
                # Map old indices to new ones using the transformation record
                new_subj_idx = self._map_augmented_index(subj_idx.item())
                new_obj_idx = self._map_augmented_index(obj_idx.item())
                # Only keep relations where both objects are still present
                if new_subj_idx is not None and new_obj_idx is not None:
                    valid_relations.append([new_subj_idx, new_obj_idx, rel_class.item()])
            
            # Update relations with valid ones only
            relations = torch.tensor(valid_relations, dtype=torch.long) if valid_relations else torch.zeros((0, 3), dtype=torch.long)
        
        # Extract object information for relationship target creation
        if hasattr(result, 'instances') and result.instances is not None:
            num_objects = len(result.instances)
        elif 'cls' in result and result['cls'] is not None:
            num_objects = len(result['cls'])
        else:
            num_objects = 0
        
        # Create relationship targets
        relation_targets = self.create_relation_targets(relations, num_objects)
        
        # Update result with relationship data
        result.update({
            'relations': relations,
            'relation_labels': relation_targets["relation_labels"],
            'object_pairs': relation_targets["object_pairs"],
            'num_relations': relation_targets["num_relations"],
            'num_objects': num_objects
        })
        
        return result

    def get_relation_weights(self) -> np.ndarray:
        """
        Get class weights for relationship classes based on dataset statistics.
        
        Returns:
            np.ndarray: Class weights for balanced training
        """
        if not self.relation_stats.get("relation_counts"):
            return np.ones(self.num_relation_classes)
        
        # Calculate inverse frequency weights
        total_relations = self.relation_stats["total_relations"]
        weights = np.ones(self.num_relation_classes)
        
        for rel_name, count in self.relation_stats["relation_counts"].items():
            if rel_name in self.relation_vocab:
                rel_id = self.relation_vocab[rel_name]
                weights[rel_id] = total_relations / (count * self.num_relation_classes)
        
        return weights

    def export_relation_annotations(self, output_path: str):
        """
        Export relationship annotations to JSON file.
        
        Args:
            output_path (str): Path to output JSON file
        """
        annotations = {}
        
        for img_path, relations in self.relationships.items():
            img_name = Path(img_path).stem
            
            # Convert to list of dictionaries with relation names
            relation_list = []
            for subj, obj, rel_class in relations:
                relation_list.append({
                    "subject": int(subj),
                    "object": int(obj),
                    "relation": self.id_to_relation.get(rel_class, f"unknown_{rel_class}")
                })
            
            annotations[img_name] = relation_list
        
        with open(output_path, 'w') as f:
            json.dump(annotations, f, indent=2)
        
        LOGGER.info(f"Exported relationship annotations to {output_path}")

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        """
        Collate data samples into batches with relationship data.
        
        Args:
            batch (List[dict]): List of dictionaries containing sample data
            
        Returns:
            Dict: Collated batch with stacked tensors including relationship data
        """
        # Use parent class collate function for standard data
        new_batch = YOLODataset.collate_fn(batch)
        
        # Handle relationship-specific data
        if 'relations' in batch[0]:
            relations = [b['relations'] for b in batch]
            new_batch['relations'] = relations
        
        if 'relation_labels' in batch[0]:
            relation_labels = [b['relation_labels'] for b in batch]
            # Pad and concatenate relation labels
            max_relations = max(len(labels) for labels in relation_labels) if relation_labels else 0
            if max_relations > 0:
                padded_labels = []
                for labels in relation_labels:
                    if len(labels) < max_relations:
                        # Pad with background class
                        padding = np.full(max_relations - len(labels), -1, dtype=int)
                        padded_labels.append(np.concatenate([labels, padding]))
                    else:
                        padded_labels.append(labels[:max_relations])
                new_batch['relation_labels'] = torch.from_numpy(np.array(padded_labels))
            else:
                new_batch['relation_labels'] = torch.empty(0, dtype=torch.long)
        
        if 'object_pairs' in batch[0]:
            if max_relations > 0:
                object_pairs = [b['object_pairs'] for b in batch]
                # Pad and concatenate object pairs
                max_pairs = max(len(pairs) for pairs in object_pairs) if object_pairs else 0
                padded_pairs = []
                for pairs in object_pairs:
                    if len(pairs) < max_pairs:
                        padding = np.full((max_pairs - len(pairs), 2), -1, dtype=int)
                        padded_pairs.append(np.concatenate([pairs, padding], axis=0))
                    else:
                        padded_pairs.append(pairs[:max_pairs])
                new_batch['object_pairs'] = torch.from_numpy(np.array(padded_pairs))
            else:
                new_batch['object_pairs'] = torch.empty((0, 2), dtype=torch.long)
        
        if 'num_relations' in batch[0]:
            num_relations = [b['num_relations'] for b in batch]
            new_batch['num_relations'] = torch.tensor(num_relations)
        
        if 'num_objects' in batch[0]:
            num_objects = [b['num_objects'] for b in batch]
            new_batch['num_objects'] = torch.tensor(num_objects)
        
        return new_batch

    @property
    def num_relation_classes(self):
        """Return the number of relation classes."""
        return getattr(self, '_num_relation_classes', len(self.id_to_relation) if hasattr(self, 'id_to_relation') else 0)
    
    @num_relation_classes.setter
    def num_relation_classes(self, value):
        """Set the number of relation classes."""
        self._num_relation_classes = value