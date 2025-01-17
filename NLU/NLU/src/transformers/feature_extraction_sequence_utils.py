# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Sequence feature extraction class for common feature extrcactors to preprocess sequences.
"""

from typing import Dict, List, Optional, Union

import numpy as np

from .feature_extraction_utils import BatchFeature, FeatureExtractionMixin
from .file_utils import (
    PaddingStrategy,
    TensorType,
    _is_tensorflow,
    _is_torch,
    is_tf_available,
    is_torch_available,
    to_py_obj,
)
from .utils import logging


logger = logging.get_logger(__name__)


class SequenceFeatureExtractor(FeatureExtractionMixin):
    """
    This is a general feature extraction class for speech recognition.

    Args:
        feature_size (:obj:`int`):
            The feature dimension of the extracted features.
        sampling_rate (:obj:`int`):
            The sampling rate at which the audio files should be digitalized expressed in Hertz per second (Hz).
        padding_value (:obj:`float`):
            The value that is used to fill the padding values / vectors.
    """

    def __init__(
        self, feature_size: int, sampling_rate: int, padding_value: float, **kwargs
    ):
        self.feature_size = feature_size
        self.sampling_rate = sampling_rate
        self.padding_value = padding_value

        self.padding_side = kwargs.pop("padding_side", "right")
        self.return_attention_mask = kwargs.pop("return_attention_mask", True)

        # Additional attributes without default values
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                logger.error(f"Can't set {key} with value {value} for {self}")
                raise err

    def pad(
        self,
        processed_features: Union[
            BatchFeature,
            List[BatchFeature],
            Dict[str, BatchFeature],
            Dict[str, List[BatchFeature]],
            List[Dict[str, BatchFeature]],
        ],
        padding: Union[bool, str, PaddingStrategy] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
    ) -> BatchFeature:
        """
        Pad input values / input vectors or a batch of input values / input vectors up to predefined length or to the
        max sequence length in the batch.

        Padding side (left/right) padding values are defined at the feature extractor level (with
        ``self.padding_side``, ``self.padding_value``)

        .. note::

            If the ``processed_features`` passed are dictionary of numpy arrays, PyTorch tensors or TensorFlow tensors,
            the result will use the same type unless you provide a different tensor type with ``return_tensors``. In
            the case of PyTorch tensors, you will lose the specific device of your tensors however.

        Args:
            processed_features (:class:`~transformers.BatchFeature`, list of :class:`~transformers.BatchFeature`, :obj:`Dict[str, List[float]]`, :obj:`Dict[str, List[List[float]]` or :obj:`List[Dict[str, List[float]]]`):
                Processed inputs. Can represent one input (:class:`~transformers.BatchFeature` or :obj:`Dict[str,
                List[float]]`) or a batch of input values / vectors (list of :class:`~transformers.BatchFeature`,
                `Dict[str, List[List[float]]]` or `List[Dict[str, List[float]]]`) so you can use this method during
                preprocessing as well as in a PyTorch Dataloader collate function.

                Instead of :obj:`List[float]` you can have tensors (numpy arrays, PyTorch tensors or TensorFlow
                tensors), see the note above for the return type.
            padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding
                index) among:

                * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a
                  single sequence if provided).
                * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
                  maximum acceptable input length for the model if that argument is not provided.
                * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
                  different lengths).
            max_length (:obj:`int`, `optional`):
                Maximum length of the returned list and optionally padding length (see above).
            pad_to_multiple_of (:obj:`int`, `optional`):
                If set will pad the sequence to a multiple of the provided value.

                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                >= 7.5 (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128.
            return_attention_mask (:obj:`bool`, `optional`):
                Whether to return the attention mask. If left to the default, will return the attention mask according
                to the specific feature_extractor's default.

                `What are attention masks? <../glossary.html#attention-mask>`__
            return_tensors (:obj:`str` or :class:`~transformers.file_utils.TensorType`, `optional`):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                * :obj:`'tf'`: Return TensorFlow :obj:`tf.constant` objects.
                * :obj:`'pt'`: Return PyTorch :obj:`torch.Tensor` objects.
                * :obj:`'np'`: Return Numpy :obj:`np.ndarray` objects.
        """
        # If we have a list of dicts, let's convert it in a dict of lists
        # We do this to allow using this method as a collate_fn function in PyTorch Dataloader
        if isinstance(processed_features, (list, tuple)) and isinstance(
            processed_features[0], (dict, BatchFeature)
        ):
            processed_features = {
                key: [example[key] for example in processed_features]
                for key in processed_features[0].keys()
            }

        # The model's main input name, usually `input_values`, has be passed for padding
        if self.model_input_names[0] not in processed_features:
            raise ValueError(
                "You should supply an instance of :class:`~transformers.BatchFeature` or list of :class:`~transformers.BatchFeature` to this method"
                f"that includes {self.model_input_names[0]}, but you provided {list(processed_features.keys())}"
            )

        required_input = processed_features[self.model_input_names[0]]
        return_attention_mask = (
            return_attention_mask
            if return_attention_mask is not None
            else self.return_attention_mask
        )

        if not required_input:
            if return_attention_mask:
                processed_features["attention_mask"] = []
            return processed_features

        # If we have PyTorch/TF/NumPy tensors/arrays as inputs, we cast them as python objects
        # and rebuild them afterwards if no return_tensors is specified
        # Note that we lose the specific device the tensor may be on for PyTorch

        first_element = required_input[0]
        if isinstance(first_element, (list, tuple)):
            # first_element might be an empty list/tuple in some edge cases so we grab the first non empty element.
            index = 0
            while len(required_input[index]) == 0:
                index += 1
            if index < len(required_input):
                first_element = required_input[index][0]
        # At this state, if `first_element` is still a list/tuple, it's an empty one so there is nothing to do.
        if not isinstance(first_element, (float, int, list, tuple)):
            if is_tf_available() and _is_tensorflow(first_element):
                return_tensors = "tf" if return_tensors is None else return_tensors
            elif is_torch_available() and _is_torch(first_element):
                return_tensors = "pt" if return_tensors is None else return_tensors
            elif isinstance(first_element, np.ndarray):
                return_tensors = "np" if return_tensors is None else return_tensors
            else:
                raise ValueError(
                    f"type of {first_element} unknown: {type(first_element)}. "
                    f"Should be one of a python, numpy, pytorch or tensorflow object."
                )

            for key, value in processed_features.items():
                processed_features[key] = to_py_obj(value)

        # Convert padding_strategy in PaddingStrategy
        padding_strategy, max_length, _ = self._get_padding_strategies(
            padding=padding, max_length=max_length
        )

        required_input = processed_features[self.model_input_names[0]]
        if required_input and not isinstance(required_input[0], (list, tuple)):
            processed_features = self._pad(
                processed_features,
                max_length=max_length,
                padding_strategy=padding_strategy,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
            )
            return BatchFeature(processed_features, tensor_type=return_tensors)

        batch_size = len(required_input)
        assert all(
            len(v) == batch_size for v in processed_features.values()
        ), "Some items in the output dictionary have a different batch size than others."

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = max(len(inputs) for inputs in required_input)
            padding_strategy = PaddingStrategy.MAX_LENGTH

        batch_outputs = {}
        for i in range(batch_size):
            inputs = dict((k, v[i]) for k, v in processed_features.items())
            outputs = self._pad(
                inputs,
                max_length=max_length,
                padding_strategy=padding_strategy,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
            )

            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

        return BatchFeature(batch_outputs, tensor_type=return_tensors)

    def _pad(
        self,
        processed_features: Union[Dict[str, List[float]], BatchFeature],
        max_length: Optional[int] = None,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
    ) -> dict:
        """
        Pad inputs (on left/right and up to predefined length or max length in the batch)

        Args:
            processed_features: Dictionary of input values (`List[float]`) / input vectors (`List[List[float]]`) or batch of inputs values (`List[List[int]]`) / input vectors (`List[List[List[int]]]`)
            max_length: maximum length of the returned list and optionally padding length (see below)
            padding_strategy: PaddingStrategy to use for padding.

                - PaddingStrategy.LONGEST Pad to the longest sequence in the batch
                - PaddingStrategy.MAX_LENGTH: Pad to the max length (default)
                - PaddingStrategy.DO_NOT_PAD: Do not pad
                The feature_extractor padding sides are defined in self.padding_side:

                    - 'left': pads on the left of the sequences
                    - 'right': pads on the right of the sequences
            pad_to_multiple_of: (optional) Integer if set will pad the sequence to a multiple of the provided value.
                This is especially useful to enable the use of Tensor Core on NVIDIA hardware with compute capability
                >= 7.5 (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128.
            return_attention_mask: (optional) Set to False to avoid returning attention mask (default: set to model specifics)
        """
        required_input = processed_features[self.model_input_names[0]]

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = len(required_input)

        if (
            max_length is not None
            and pad_to_multiple_of is not None
            and (max_length % pad_to_multiple_of != 0)
        ):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        needs_to_be_padded = (
            padding_strategy != PaddingStrategy.DO_NOT_PAD
            and len(required_input) != max_length
        )

        if needs_to_be_padded:
            difference = max_length - len(required_input)
            padding_vector = (
                self.feature_size * [self.padding_value]
                if self.feature_size > 1
                else self.padding_value
            )
            if self.padding_side == "right":
                if return_attention_mask:
                    processed_features["attention_mask"] = [1] * len(required_input) + [
                        0
                    ] * difference
                processed_features[self.model_input_names[0]] = required_input + [
                    padding_vector for _ in range(difference)
                ]
            elif self.padding_side == "left":
                if return_attention_mask:
                    processed_features["attention_mask"] = [0] * difference + [1] * len(
                        required_input
                    )
                processed_features[self.model_input_names[0]] = [
                    padding_vector for _ in range(difference)
                ] + required_input
            else:
                raise ValueError("Invalid padding strategy:" + str(self.padding_side))
        elif return_attention_mask and "attention_mask" not in processed_features:
            processed_features["attention_mask"] = [1] * len(required_input)

        return processed_features

    def _get_padding_strategies(
        self, padding=False, max_length=None, pad_to_multiple_of=None, **kwargs
    ):
        """
        Find the correct padding strategy
        """

        # Get padding strategy
        if padding is not False:
            if padding is True:
                padding_strategy = (
                    PaddingStrategy.LONGEST
                )  # Default to pad to the longest sequence in the batch
            elif not isinstance(padding, PaddingStrategy):
                padding_strategy = PaddingStrategy(padding)
            elif isinstance(padding, PaddingStrategy):
                padding_strategy = padding
        else:
            padding_strategy = PaddingStrategy.DO_NOT_PAD

        # Set max length if needed
        if max_length is None:
            if padding_strategy == PaddingStrategy.MAX_LENGTH:
                raise ValueError(
                    f"When setting ``padding={PaddingStrategy.MAX_LENGTH}``, make sure that"
                    f" max_length is defined"
                )

        # Test if we have a padding value
        if padding_strategy != PaddingStrategy.DO_NOT_PAD and (
            self.padding_value is None
        ):
            raise ValueError(
                "Asking to pad but the feature_extractor does not have a padding value. "
                "Please select a value to use as `padding_value`. For example: `feature_extractor.padding_value = 0.0`."
            )

        return padding_strategy, max_length, kwargs
