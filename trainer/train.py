# coding=utf-8
# NB chua: modified from https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/models/llava/modeling_llava.py#L350
# Copyright 2023 the HuggingFace Inc. team. All rights reserved.
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
# Some code adapted from Tincans-ai/gazelle - credit to Chris Hua
import numpy as np
import transformers
from transformers import AutoTokenizer, AutoProcessor, LlavaForConditionalGeneration
import torch
from gazelle import (
    GazelleConfig,
    GazelleForConditionalGeneration,
    GazelleProcessor,
)
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from transformers.tokenization_utils_base import (
    PaddingStrategy,
    PreTokenizedInput,
    TextInput,
    TruncationStrategy,
)
from transformers import (
    CONFIG_MAPPING,
    AutoModel,
    AutoModelForCausalLM,
    BatchFeature,
    PretrainedConfig,
    PreTrainedModel,
    ProcessorMixin,
    TensorType,
)
import math
from transformers import Trainer
import wandb
from transformers import TrainingArguments


class GazelleProcessor(ProcessorMixin):
    r"""
    Constructs a Gazelle processor which wraps a Gazelle image processor and a Gazelle tokenizer into a single processor.

    [`GazelleProcessor`] offers all the functionalities of [`Wav2Vec2Processor`] and [`LlamaTokenizerFast`]. See the
    [`~GazelleProcessor.__call__`] and [`~GazelleProcessor.decode`] for more information.

    Args:
        audio_processor ([`Wav2Vec2Processor`, `SeamlessM4TFeatureExtractor`], *optional*):
            The audio processor is a required input.
        tokenizer ([`LlamaTokenizerFast`], *optional*):
            The tokenizer is a required input.
    """

    attributes = ["audio_processor", "tokenizer"]
    audio_processor_class = (
        "Wav2Vec2Processor",
        "SeamlessM4TFeatureExtractor",
    )
    tokenizer_class = (
        "LlamaTokenizer",
        "LlamaTokenizerFast",
    )

    def __init__(self, audio_processor=None, tokenizer=None):
        super().__init__(audio_processor, tokenizer)

    def __call__(
        self,
        text: Union[
            TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]
        ] = None,
        audio=None,
        labels: Union[
            TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]
        ] = None,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length=None,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
        sampling_rate: int = 16000,
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to LlamaTokenizerFast's [`~LlamaTokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the image(s), this method forwards the `images` and `kwrags` arguments to
        CLIPImageProcessor's [`~CLIPImageProcessor.__call__`] if `images` is not `None`. Please refer to the doctsring
        of the above two methods for more information.

        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            audio (`np.ndarray`, `torch.Tensor`, `List[np.ndarray]`, `List[torch.Tensor]`):
                 The audio or batch of audios to be prepared. Each audio can be NumPy array or PyTorch tensor. In case of a
                NumPy array/PyTorch tensor, each audio should be of shape (C, T), where C is a number of channels, and T the
                sample length of the audio.
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `False`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding
                index) among:
                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            max_length (`int`, *optional*):
                Maximum length of the returned list and optionally padding length (see above).
            truncation (`bool`, *optional*):
                Activates truncation to cut input sequences longer than `max_length` to `max_length`.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.
            sampling_rate (`int`, *optional*, defaults to 16000):
                Sampling rate of the input audio. We expect 16kHz audio. Don't change this value unless you know what
                you are doing.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **audio_values** -- Processed audio values to be fed to a model. Returned when `audios` is not `None`.
        """
        if audio is not None and len(audio) > 0:
            audio_values = self.audio_processor(
                audio, return_tensors=return_tensors, sampling_rate=sampling_rate
            )
            audio_values = audio_values.input_values
            audio_values = audio_values.to("cuda", dtype=torch.float16)
            audio_values = wav2vec_model(audio_values).last_hidden_state
            audio_values = audio_values.cpu()

        else:
            audio_values = None

        if labels is not None:
            labels = self.tokenizer(
                labels,
                return_tensors=return_tensors,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                add_special_tokens=False,
            )["input_ids"]

        if text is not None:
            text_inputs = self.tokenizer(
                text,
                return_tensors=return_tensors,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                add_special_tokens=False,
            )
            return BatchFeature(data={**text_inputs, "audio_values": audio_values, "labels": labels})
        else:
            return BatchFeature(data={"audio_values": audio_values})

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.batch_decode with CLIP->Llama
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.decode with CLIP->Llama
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.model_input_names
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        audio_processor_input_names = self.audio_processor_class.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + audio_processor_input_names))


class GazelleDataCollator:
    def __init__(self, proz):
        self.proz = proz

    def __call__(self, examples):
        texts = []
        audios = []
        labels = []
        max_len = 0
        # print(examples)
        for example in examples:
            # print(example)
            text = example["input"]
            audio = example["audio"]
            label = example["text"]
            if len(audio) > max_len:
                max_len = len(audio)
            audio = np.array(audio)
            texts.append(text)
            audios.append(audio)
            labels.append(label)
        temp_audio_array = []
        for audioz in audios:
            audioz = np.pad(
                audioz, (0, max_len-audioz.shape[0]), mode='constant', constant_values=0)
            temp_audio_array.append(audioz)
        audios = np.array(temp_audio_array)
        batch = self.proz(texts, audios, labels, "max_length",
                          return_tensors="pt", max_length=256)
        return batch


def freeze(model):
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the multi_modal_projector parameters
    for param in model.multi_modal_projector.linear_1.parameters():
        param.requires_grad = True
    for param in model.multi_modal_projector.linear_2.parameters():
        param.requires_grad = True


if __name__ == "__main__":
    model_id = "tincans-ai/gazelle-v0.2"
    audio_processor = transformers.Wav2Vec2Processor.from_pretrained(
        "facebook/wav2vec2-base-960h"
    )
    config = GazelleConfig.from_pretrained(model_id)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    device = "cpu"
    dtype = torch.float32
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
        print(f"Using {device} device")
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
        print(f"Using {device} device")
    torch.device(device)
    model = GazelleForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
    ).to(device, dtype=dtype)
    wav2vec_model = AutoModel.from_pretrained("facebook/wav2vec2-base-960h")
    wav2vec_model = wav2vec_model.to("cuda", dtype=torch.float16)
    proz = GazelleProcessor(audio_processor, tokenizer)
    dc = GazelleDataCollator(proz)
    model = freeze(model)
    arges = TrainingArguments(output_dir="./", per_device_train_batch_size=4,
                              remove_unused_columns=False, num_train_epochs=3, logging_steps=10)
    trainer = Trainer(model=model, args=arges,
                      train_dataset=dataset, data_collator=dc)
    trainer.train()
