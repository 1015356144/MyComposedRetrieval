from itertools import repeat
from typing import Optional
from torch.jit import isinstance

import logging
from dataclasses import dataclass
from transformers import ProcessorMixin, AutoProcessor, AutoTokenizer
from src.arguments import DataArguments, ModelArguments, TrainingArguments
import torch
from qwen_vl_utils import smart_resize

from src.model.processor import LLAVA_NEXT, QWEN2_VL, QWEN2_5_VL, \
    QWEN2_VL_TOKENSELECTION, QWEN2_5_VL_TOKENSELECTION, PHI3V, process_vlm_inputs_fns
from PIL import Image
import io
from src.utils import print_rank, print_master
import os
import psutil
from datetime import datetime


logger = logging.getLogger(__name__)


PHI_IMAGE_TOKEN_MAX_INPUT_ID = int(1e9)
LLAVA_IMAGE_TOKEN_ID = 32000


def split_and_process_vlm_inputs(model_input: dict, chunk_size: int):
    '''
    通用批切分器：对每个键，若为 Tensor 则按 dim=0 用 chunk_size 拆分，
    否则视为序列按同样步长切片；保持键集合不变并重组为 [{arg_key: chunk_dict}, ...]。
    适用于所有键都与样本维对齐的输入；若图像是按全局图片数堆叠，请改用专门的 split_vlm_inputs。
    '''
    assert len(model_input) == 1
    arg_key = list(model_input.keys())[0]
    arg_val = model_input[arg_key]

    keys = list(arg_val.keys())
    chunked_tensors = []
    for k in keys:
        if isinstance(arg_val[k], torch.Tensor):
            chunked_tensor = arg_val[k].split(chunk_size, dim=0)
        else:
            chunked_tensor = [arg_val[k][i: i + chunk_size] for i in list(range(0, len(arg_val[k]), chunk_size))]
        chunked_tensors.append(chunked_tensor)
    chunked_arg_val = [dict(zip(kk, tt)) for kk, tt in zip(repeat(keys), zip(*chunked_tensors))]
    chunked_inputs = [{arg_key: c} for c in chunked_arg_val]

    return chunked_inputs


def split_vlm_inputs(model_input: dict, chunk_size: int):
    '''
    将多模态批输入按 batch 维以 chunk_size 切分：文本张量(input_ids/attention_mask)等长切，
    图像张量(pixel_values/image_sizes)按每个 chunk 内“含图样本数”切分并对无图 chunk 省略图像键，
    保证图文对齐后返回 [{arg_key: chunk_dict}, ...] 的列表。
    '''
    assert len(model_input) == 1
    arg_key = list(model_input.keys())[0]
    arg_val = model_input[arg_key]
    keys = list(arg_val.keys())

    # for input_ids and attention_mask, split directly
    chunked_tensors = [arg_val[k].split(chunk_size, dim=0) for k in ["input_ids", "attention_mask"]]

    # for pixel_values and image_sizes, need to split based on the position of images
    input_ids = arg_val["input_ids"]
    positions = torch.nonzero((input_ids < 0) & (input_ids > -PHI_IMAGE_TOKEN_MAX_INPUT_ID), as_tuple=True)
    row_contain_image = torch.unique(positions[0])  # indicates which row in input_ids contain images
    num_chunks = len(chunked_tensors[0])
    chunk_image_count = []
    for chunk_idx in range(num_chunks):
        chunk_image_count.append(torch.sum(
            (row_contain_image >= chunk_idx * chunk_size) & (row_contain_image < (chunk_idx + 1) * chunk_size)).item())
    if "pixel_values" in keys:
        pixel_values = arg_val["pixel_values"]
        image_sizes = arg_val["image_sizes"]
        chunked_tensors.append(torch.split(pixel_values, chunk_image_count))
        chunked_tensors.append(torch.split(image_sizes, chunk_image_count))

    chunked_arg_val = []
    for kk, tt in zip(repeat(keys), zip(*chunked_tensors)):
        if "pixel_values" in keys and tt[2].numel() == 0:  # this chunk doesn't contain image
            chunked_arg_val.append(dict(zip(kk[:2], tt[:2])))
        else:
            chunked_arg_val.append(dict(zip(kk, tt)))

    return [{arg_key: c} for c in chunked_arg_val]


def get_dense_rep(x):
    """
    Get either qry_reps or tgt_reps.
    """
    if x["qry_reps"] is None:
        return x["tgt_reps"]
    else:
        return x["qry_reps"]


@dataclass
class TrainTextImageDataCollator:
    data_args: DataArguments
    model_args: ModelArguments
    processor: ProcessorMixin

    def __call__(self, examples):
        """
        :param examples: qry, qry_image, pos_text, pos_image
        """
        qry_inputs = self._get_batch_inputs(examples, "query_text", "query_image")
        pos_inputs = self._get_batch_inputs(examples, "pos_text", "pos_image")
        neg_inputs = self._get_batch_inputs(examples, "neg_text", "neg_image")
        return qry_inputs, pos_inputs

    def _get_batch_inputs(self, examples, text_keyname, image_keyname):
        texts, images = [], []
        for example in examples:
            # @ruimeng filter invalid data examples here will lead to fail to sync across devices (unequal batch size)
            # use dummy input for now
            if example is None or not example:
                text, image = '  ', None
            text, image = example[text_keyname], example[image_keyname]
            if type(text) == list:
                if len(text) == 0 or len(image) == 0:
                    text, image = '  ', None
                else:
                    text, image = text[0], image[0]
            texts.append(text)
            images.append(image)
        inputs = {'text': texts, 'image': images}
        return inputs


@dataclass
class MultimodalDataCollator:
    processor: ProcessorMixin
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments
    batch_size: Optional[int] = None

    def __post_init__(self):
        # 统计计数器
        self.long_text_count = 0
        self.total_text_count = 0

    def _get_batch_inputs(self, batch, text_keyname, image_keyname):
        texts, visual_inputs = [], []
        
        for batch_idx, example in enumerate(batch):
            if example is None or not example:
                text, visual_input = '  ', None
            else:
                text, raw_images = example[text_keyname], example[image_keyname]
                
                # 🔥 更积极的文本长度限制 - 专门针对增强数据的长文本问题
                if text and len(text) > 1000:  # 降低到1000字符（约250 tokens）
                    self.long_text_count += 1
                    original_len = len(text)
                    text = text[:1000]  # 截断到1000字符
                    if self.long_text_count <= 10:  # 增加警告显示数量以便观察
                        print_rank(f"⚠️  Long text truncated: {original_len} -> {len(text)} chars")
                
                self.total_text_count += 1
                
                if type(raw_images) == dict:
                    visual_input = []
                    assert 'resolutions' in raw_images, "we need len(raw_images['resolutions']) to determine the number of images, set it a list of None of for cases that no resizing is needed"
                    num_images = len(raw_images['resolutions'])
                    
                    for image_idx in range(num_images):
                        bytes_data = raw_images['bytes'][image_idx] if 'bytes' in raw_images else None
                        path = raw_images['paths'][image_idx] if 'paths' in raw_images else None
                        image_resolution = raw_images['resolutions'][image_idx] if 'resolutions' in raw_images else None
                        
                        image = self._load_image(bytes_data, path)
                        
                        if not self.data_args.resize_use_processor and image is not None and image_resolution:
                            image = image.resize(image_resolution)
                            
                        if image is not None and (self.data_args.image_decay_factor is not None and image_resolution is None):
                            assert image_resolution is None, "image_resolution is conflicting with image_decay_factor"
                            assert self.model_args.model_backbone in [QWEN2_VL, QWEN2_5_VL, QWEN2_VL_TOKENSELECTION, QWEN2_5_VL_TOKENSELECTION], "image_decay_factor is only supported for Qwen models"
                            max_pixels = max(self.data_args.resize_min_pixels, self.data_args.resize_max_pixels * self.data_args.image_decay_factor ** (num_images - image_idx))
                            width, height = image.size
                            resized_height, resized_width = smart_resize(
                                height, width,
                                min_pixels=self.data_args.resize_min_pixels,
                                max_pixels=max_pixels,
                            )
                            image = image.resize((resized_width, resized_height))
                            
                        visual_input.append(image)
                else:
                    visual_input = None  
            texts.append(text)
            visual_inputs.append(visual_input)
    
        inputs = {'text': texts, 'images': visual_inputs}
        return inputs

    def _load_image(self, bytes_data, path):
        """还原原始的简单图像加载逻辑"""
        try:
            if bytes_data is None and path is None:
                return None
            elif bytes_data is not None:
                image = Image.open(io.BytesIO(bytes_data))
                return image.convert("RGB")
            elif path is not None:
                with Image.open(path) as img:
                    return img.convert("RGB")
        except Exception as e:
            print_rank(f"❌ Failed to load image: {e}")
            return None

    def __call__(self, examples):
        """保持原有逻辑不变"""
        try:
            qry_inputs = self._get_batch_inputs(examples, "query_text", "query_image")
            pos_inputs = self._get_batch_inputs(examples, "pos_text", "pos_image")
            neg_inputs = self._get_batch_inputs(examples, "neg_text", "neg_image")
            
            bs = len(qry_inputs['text'])
            assert bs > 0, 'An empty batch'
            
            if self.batch_size is not None and bs < self.batch_size:
                raise RuntimeError(f"Expect batch size {self.batch_size}, but got batch size of {bs}")
                
            process_fn = process_vlm_inputs_fns[self.training_args.model_backbone]
            processed_qry_inputs = process_fn(qry_inputs, processor=self.processor, max_length=self.data_args.max_len)
            processed_pos_inputs = process_fn(pos_inputs, processor=self.processor, max_length=self.data_args.max_len)
            
            processed_qry_inputs['text'] = [e['query_text'] for e in examples]
            processed_pos_inputs['text'] = [e['pos_text'] for e in examples]
            processed_qry_inputs['global_dataset_name'] = [e['global_dataset_name'] for e in examples]
            processed_pos_inputs['global_dataset_name'] = [e['global_dataset_name'] for e in examples]

            # 🔥 定期报告文本截断统计
            if self.total_text_count % 1000 == 0:
                print_rank(f"📊 Text stats - Total: {self.total_text_count}, Truncated: {self.long_text_count} ({self.long_text_count/self.total_text_count*100:.1f}%)")
                
            # >>> 新增：透传 sample_ids（用于跨 rank 首批对比）
            sid_list = []
            for e in examples:
                sid = e.get("sample_ids", None)
                if isinstance(sid, torch.Tensor):
                    sid_list.append(sid.cpu())
                elif sid is None:
                    sid_list.append(torch.tensor(-1))
                else:
                    sid_list.append(torch.tensor(int(sid)))
            processed_qry_inputs["sample_ids"] = torch.stack(sid_list)
            return processed_qry_inputs, processed_pos_inputs
            
        except Exception as e:
            print_rank(f"❌ Collator error: {e}")
            print_rank(f"📊 Stats - Total texts: {self.total_text_count}, Truncated: {self.long_text_count}")
            raise
