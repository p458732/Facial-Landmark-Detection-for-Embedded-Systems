#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import torch

from data.datasets import DATASET_REGISTRY
from data.datasets.multi_modal_img_text.base_multi_modal_img_text import (
    BaseMultiModalImgText,
)
from utils import logger
from utils.ddp_utils import is_master


@DATASET_REGISTRY.register(name="flickr", type="multi_modal_image_text")
class FlickrDataset(BaseMultiModalImgText):
    """
    Dataset loader for Flickr-30k and Flickr-8k datasets.

    For more info see:
        http://hockenmaier.cs.illinois.edu/8k-pictures.html
        https://shannon.cs.illinois.edu/DenotationGraph/

    Splits: train, val, and test
        Also known in literature as Karpathy splits
        https://cs.stanford.edu/people/karpathy/deepimagesent/

    Tracking license info:
        Captions have CC BY 3.0 license (see links above).
        Splits are under BSD License (see Github of NeuralTalk by Karpathy et. al.).
        Images are from Flickr. We do not own them and are only used for research purposes.

    Args:
        opts: command-line arguments
        is_training (Optional[bool]): A flag used to indicate training or
            validation mode. Default: True
        is_evaluation (Optional[bool]): A flag used to indicate evaluation (or
            inference) mode. Default: False
    """

    def get_dataset(self, *args, **kwargs) -> None:
        """
        The data under `self.root` is expected to consist of:

            dataset.json   # Karpathy splits + captions
            images/        # Raw images

        The metdatadata cap be downloaded from:
            https://cs.stanford.edu/people/karpathy/deepimagesent/flickr30k.zip

        Images can be obtained from:
            Flickr-8k:  http://hockenmaier.cs.illinois.edu/8k-pictures.html
            Flickr-30k: https://shannon.cs.illinois.edu/DenotationGraph/
        """
        metadata_path = os.path.join(self.root, "dataset.json")
        with open(metadata_path, "r") as metadata_file:
            metadata = json.load(metadata_file)["images"]

        split = self.mode

        self.samples = [
            {
                "image_name": sample["filename"],
                "captions": [x["raw"] for x in sample["sentences"]],
            }
            for sample in metadata
            if sample["split"] == split
        ]

        if self.is_training:
            # For training, flatten the captions by copying each image multiple times
            # This way at each epoch, each caption will be seen 1 time but each image
            # will be seen #num_captions (= 5) times.
            self.samples = [
                {"image_name": sample["image_name"], "captions": caption}
                for sample in self.samples
                for caption in sample["captions"]
            ]
            # No need for shuffling, since dataloader takes care of it

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, batch_indexes_tup: Tuple[int, int, int]) -> Dict[str, Dict]:
        """Return the sample given an index, a pre-specified height and width.

        Args:
            batch_indexes_tup: Tuple of (crop size height, crop size width, image index)

        Returns:
            A dict like:
            {
                "samples": {
                    "image": FloatTensor of shape [C, H, W] -> [B, C, H, W] after collate
                    "text": LongTensor of shape
                        if is_training: [num_tokens]  -> [B, T] after collate
                        o.w.:           [num_captions, num_tokens] -> [B, N, T] after collate
                    "padding_mask": Optional; same size as text
                    "max_seq_len": int indicating the maximum caption length (T)
                        Gets removed after collate
                },
                "targets": -1
            }
        """
        crop_size_h, crop_size_w, img_index = batch_indexes_tup
        crop_size = (crop_size_h, crop_size_w)

        transform_fn = self.get_augmentation_transforms(size=crop_size)

        sample = self.samples[img_index]

        img_path = os.path.join(self.root, "images", sample["image_name"])
        captions = sample["captions"]

        input_img = self.read_image_pil(img_path)

        img_tensor, captions_tensor, max_seq_len = self._process_img_caption(
            input_img=input_img,
            captions_str=captions,
            img_transform_fn=transform_fn,
            zero_shot=self.zero_shot_dataset is not None,
        )

        data = {
            "samples": {
                "image": img_tensor,
                "text": captions_tensor,
                "padding_mask": (captions_tensor == self.padding_index)
                if self.padding_index is not None
                else None,
                "max_seq_len": max_seq_len,
            },
            "targets": -1,
        }

        return data

    def __repr__(self) -> str:
        return "{}(\n\troot={}\n\tis_training={}\n\tsamples={}\n)".format(
            self.__class__.__name__,
            self.root,
            self.is_training,
            len(self),
        )
