import logging

import PIL
from transformers.image_utils import ChannelDimension

from src.model.baseline_backbone.colpali import ColPaliProcessor

logger = logging.getLogger(__name__)

import torch
import numpy as np
from src.utils import print_master

from src.model.baseline_backbone.llava_next import LlavaNextForConditionalGeneration
from src.model.baseline_backbone.phi3_v.modeling_phi3_v import Phi3VForCausalLM
from src.model.vlm_backbone.qwen2_vl import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from src.model.vlm_backbone.qwen2_vl_tokenselection import \
    Qwen2VLForConditionalGeneration as Qwen2VLTokenSelectionForConditionalGeneration, \
    Qwen2VLProcessor as Qwen2VLTokenSelectionProcessor
from src.model.baseline_backbone.internvideo2.modeling_internvideo2 import InternVideo2_Stage2
from src.model.vlm_backbone.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from src.model.vlm_backbone.qwen2_5_vl_tokenselection import \
    Qwen2_5_VLForConditionalGeneration as Qwen2_5_VL_TokenSelectionForConditionalGeneration


PHI_IMAGE_TOKEN_MAX_INPUT_ID = int(1e9)
LLAVA_IMAGE_TOKEN_ID = 32000

PHI3V = 'phi3_v'
LLAVA_NEXT = 'llava_next'
QWEN2_VL = 'qwen2_vl'
QWEN2_VL_TOKENSELECTION = 'qwen2_vl'
QWEN2_5_VL = 'qwen2_5_vl'
QWEN2_5_VL_TOKENSELECTION = 'qwen2_5_vl_tokenselection'
INTERNVIDEO2 = 'internvideo2'
GME = 'gme'  # QWEN2-VL
LamRA = 'lamra'  # QWEN2-VL
LamRA_QWEN2_5 = 'lamra_qwen25'  # QWEN2.5-VL
COLPALI = 'colpali'  # PaliGemma-3B
E5_V = 'e5_v'  # Llava_next
MODEL2BACKBONE = {  # keys are from hf_config.model_type or manually added if not provided
    'phi3_v': PHI3V,
    'llava_next': LLAVA_NEXT,
    'qwen2_vl': QWEN2_VL,
    'qwen2_vl_tokenselection': QWEN2_VL,
    'qwen2_5_vl': QWEN2_5_VL,
    'qwen2_vl_tokenselection': QWEN2_VL_TOKENSELECTION,
    'qwen2_5_vl_tokenselection': QWEN2_5_VL_TOKENSELECTION,
    'internvideo2': INTERNVIDEO2,
    'gme': GME, 
    'lamra': LamRA,
    'lamra_qwen25': LamRA,
    'colpali': COLPALI,
    'e5_v': E5_V,
}
SUPPORTED_MODELS = set(MODEL2BACKBONE.keys())

VLM_IMAGE_TOKENS = {
    PHI3V: "<|image_1|>",
    LLAVA_NEXT: "<image>",
    QWEN2_VL: "<|image_pad|>",
    QWEN2_5_VL: "<|image_pad|>",
    QWEN2_VL_TOKENSELECTION: "<|image_pad|>",
    QWEN2_5_VL_TOKENSELECTION: "<|image_pad|>",
    GME: "<|image_pad|>",
    LamRA: "<|image_pad|>",
    LamRA_QWEN2_5: "<|image_pad|>",
    INTERNVIDEO2: "",
    COLPALI: "",
    E5_V: "<image>",
}

VLM_VIDEO_TOKENS = {
    LLAVA_NEXT: "<image>",
    QWEN2_VL: "<|video_pad|>",
    QWEN2_5_VL: "<|video_pad|>",
    QWEN2_VL_TOKENSELECTION: "<|video_pad|>",
    QWEN2_5_VL_TOKENSELECTION: "<|video_pad|>",
    GME: "<|video_pad|>",
    LamRA: "<|video_pad|>",
    LamRA_QWEN2_5: "<|video_pad|>",
    INTERNVIDEO2: "",
    COLPALI: "",
    E5_V: "<image>",
}

backbone2model = {
    PHI3V: Phi3VForCausalLM,
    LLAVA_NEXT: LlavaNextForConditionalGeneration,
    QWEN2_VL: Qwen2VLForConditionalGeneration,
    QWEN2_5_VL: Qwen2_5_VLForConditionalGeneration,
    QWEN2_VL_TOKENSELECTION: Qwen2VLTokenSelectionForConditionalGeneration,
    QWEN2_5_VL_TOKENSELECTION: Qwen2_5_VL_TokenSelectionForConditionalGeneration,
    INTERNVIDEO2: InternVideo2_Stage2,
    E5_V: LlavaNextForConditionalGeneration,
}


def load_processor(model_args, data_args=None):
    """
    Load processor based on VLM backbone.
    Note: due to this change, https://github.com/huggingface/transformers/commit/9215cc62d4366072aacafa4e44028c1ca187167b#diff-6505546ec5a9ab74b2ce6511681dd31194eb91e9fa3ce26282e487a5e61f9356L1102
    """
    model_name_or_path = model_args.checkpoint_path if model_args.checkpoint_path else model_args.model_name
    print_master(f'Loading processor from: {model_name_or_path}')
    if model_args.model_backbone == PHI3V:
        from src.model.baseline_backbone.phi3_v.processing_phi3_v import Phi3VProcessor
        processor = Phi3VProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            num_crops=model_args.num_crops
        )
        processor.tokenizer.padding_side = "right"
    elif model_args.model_backbone == LLAVA_NEXT:
        from src.model.baseline_backbone.llava_next import LlavaNextProcessor
        processor = LlavaNextProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
        )
    elif model_args.model_backbone in [QWEN2_VL, GME, LamRA]:
        from src.model.vlm_backbone.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor
        from src.model.vlm_backbone.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
        from src.model.vlm_backbone.qwen2_vl.tokenization_qwen2_fast import Qwen2TokenizerFast

        # 1) 读取配置
        do_resize = True
        min_pixels = 56*56          # 默认最小面积：与官方一致（3136）
        max_pixels = 28*28*1280     # 默认最大面积：与官方一致（1003520）
        if data_args is not None:
            do_resize = data_args.resize_use_processor
            if getattr(data_args, "resize_min_pixels", None) is not None:
                min_pixels = int(data_args.resize_min_pixels)
            if getattr(data_args, "resize_max_pixels", None) is not None:
                max_pixels = int(data_args.resize_max_pixels)

        # 2) 载入
        image_processor = Qwen2VLImageProcessor.from_pretrained(model_name_or_path)

        # 3) 统一设置：属性 + size 两处都覆盖，且只用官方认可的两个键
        if hasattr(image_processor, "do_resize"):
            image_processor.do_resize = do_resize
        if hasattr(image_processor, "min_pixels"):
            image_processor.min_pixels = min_pixels
        if hasattr(image_processor, "max_pixels"):
            image_processor.max_pixels = max_pixels

        # # 注意：size 的键是“面积”，不是边长
        # image_processor.size = {
        #     "shortest_edge": min_pixels,   # 最小面积
        #     "longest_edge":  max_pixels    # 最大面积
        # }

        tokenizer = Qwen2TokenizerFast.from_pretrained(model_name_or_path)
        processor = Qwen2VLProcessor.from_pretrained(
            model_name_or_path,
            image_processor=image_processor,
            tokenizer=tokenizer
        )
        ip = processor.image_processor
        print_master(
            f"[Qwen2VL] do_resize={getattr(ip,'do_resize',None)}, "
            f"size={getattr(ip,'size',None)}, "
            f"min_pixels={getattr(ip,'min_pixels',None)}, "
            f"max_pixels={getattr(ip,'max_pixels',None)}"
        )
    elif model_args.model_backbone == QWEN2_VL_TOKENSELECTION:
        from src.model.vlm_backbone.qwen2_vl_tokenselection.processing_qwen2_vl import Qwen2VLProcessor
        from src.model.vlm_backbone.qwen2_vl_tokenselection.image_processing_qwen2_vl import Qwen2VLImageProcessor
        from src.model.vlm_backbone.qwen2_vl_tokenselection.tokenization_qwen2_fast import Qwen2TokenizerFast
        image_processor = Qwen2VLImageProcessor.from_pretrained(model_name_or_path)
        if data_args is not None:
            image_processor.do_resize = data_args.resize_use_processor
            image_processor.min_pixels = data_args.resize_min_pixels
            image_processor.max_pixels = data_args.resize_max_pixels
        tokenizer = Qwen2TokenizerFast.from_pretrained(model_name_or_path)
        processor = Qwen2VLProcessor.from_pretrained(
            model_name_or_path,
            image_processor=image_processor, tokenizer=tokenizer,
            uigraph_use=model_args.uigraph_use,
            uigraph_diff=model_args.uigraph_diff,  uigraph_rand=model_args.uigraph_rand,
            uimask_ratio=model_args.uimask_ratio, uimask_rand=model_args.uimask_rand
        )
    elif model_args.model_backbone in [QWEN2_5_VL, LamRA_QWEN2_5]:
        from src.model.vlm_backbone.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor
        from src.model.vlm_backbone.qwen2_5_vl.image_processing_qwen2_5_vl import Qwen2_5_VLImageProcessor
        from src.model.vlm_backbone.qwen2_vl.tokenization_qwen2_fast import Qwen2TokenizerFast

        # 读取用户参数（允许 None -> 使用模型默认）
        user_do_resize = True
        user_min_pixels = None
        user_max_pixels = None
        if data_args is not None:
            user_do_resize = data_args.resize_use_processor
            user_min_pixels = getattr(data_args, 'resize_min_pixels', None)
            user_max_pixels = getattr(data_args, 'resize_max_pixels', None)

        # 先按模型默认加载（会读取 preprocessor_config.json）
        image_processor = Qwen2_5_VLImageProcessor.from_pretrained(model_name_or_path)

        # 采集模型默认值
        default_min = getattr(image_processor, 'min_pixels', None)
        default_max = getattr(image_processor, 'max_pixels', None)

        # 计算最终值（用户优先）
        final_min = int(user_min_pixels) if user_min_pixels is not None else default_min
        final_max = int(user_max_pixels) if user_max_pixels is not None else default_max

        # 覆盖属性
        if hasattr(image_processor, 'do_resize'):
            image_processor.do_resize = user_do_resize
        if hasattr(image_processor, 'min_pixels') and final_min is not None:
            image_processor.min_pixels = final_min
        if hasattr(image_processor, 'max_pixels') and final_max is not None:
            image_processor.max_pixels = final_max

        # 同步 size 字典（不同版本字段可能不同，做存在性检查）
        if hasattr(image_processor, 'size') and isinstance(image_processor.size, dict):
            if final_min is not None:
                image_processor.size['min_pixels'] = final_min
                if 'shortest_edge' in image_processor.size:
                    image_processor.size['shortest_edge'] = final_min
            if final_max is not None:
                image_processor.size['max_pixels'] = final_max
                if 'longest_edge' in image_processor.size:
                    image_processor.size['longest_edge'] = final_max

        tokenizer = Qwen2TokenizerFast.from_pretrained(model_name_or_path)
        processor = Qwen2_5_VLProcessor.from_pretrained(
            model_name_or_path,
            image_processor=image_processor,
            tokenizer=tokenizer
        )
        ip = processor.image_processor
        print_master(
            f"[Qwen2_5_VL] do_resize={getattr(ip,'do_resize',None)}, size={getattr(ip,'size',None)}, "
            f"min_pixels={getattr(ip,'min_pixels',None)}, max_pixels={getattr(ip,'max_pixels',None)} "
            f"(user_min={user_min_pixels}, user_max={user_max_pixels}, default_min={default_min}, default_max={default_max})"
        )
    elif model_args.model_backbone == QWEN2_5_VL_TOKENSELECTION:
        # TODO: qwen2.5 token selection not working yet
        from src.model.vlm_backbone.qwen2_5_vl_tokenselection.processing_qwen2_5_vl import Qwen2_5_VLProcessor
        from src.model.vlm_backbone.qwen2_5_vl_tokenselection.image_processing_qwen2_5_vl import Qwen2_5_VLImageProcessor
        from src.model.vlm_backbone.qwen2_vl_tokenselection.tokenization_qwen2_fast import Qwen2TokenizerFast

        user_do_resize = True
        user_min_pixels = None
        user_max_pixels = None
        if data_args is not None:
            user_do_resize = data_args.resize_use_processor
            user_min_pixels = getattr(data_args, 'resize_min_pixels', None)
            user_max_pixels = getattr(data_args, 'resize_max_pixels', None)

        image_processor = Qwen2_5_VLImageProcessor.from_pretrained(model_name_or_path)

        default_min = getattr(image_processor, 'min_pixels', None)
        default_max = getattr(image_processor, 'max_pixels', None)
        final_min = int(user_min_pixels) if user_min_pixels is not None else default_min
        final_max = int(user_max_pixels) if user_max_pixels is not None else default_max

        if hasattr(image_processor, 'do_resize'):
            image_processor.do_resize = user_do_resize
        if hasattr(image_processor, 'min_pixels') and final_min is not None:
            image_processor.min_pixels = final_min
        if hasattr(image_processor, 'max_pixels') and final_max is not None:
            image_processor.max_pixels = final_max
        if hasattr(image_processor, 'size') and isinstance(image_processor.size, dict):
            if final_min is not None:
                image_processor.size['min_pixels'] = final_min
                if 'shortest_edge' in image_processor.size:
                    image_processor.size['shortest_edge'] = final_min
            if final_max is not None:
                image_processor.size['max_pixels'] = final_max
                if 'longest_edge' in image_processor.size:
                    image_processor.size['longest_edge'] = final_max

        tokenizer = Qwen2TokenizerFast.from_pretrained(model_name_or_path)
        processor = Qwen2_5_VLProcessor.from_pretrained(
            model_name_or_path,
            image_processor=image_processor,
            tokenizer=tokenizer,
            uigraph_use=model_args.uigraph_use,
            uigraph_diff=model_args.uigraph_diff,
            uigraph_rand=model_args.uigraph_rand,
            uimask_ratio=model_args.uimask_ratio,
            uimask_rand=model_args.uimask_rand
        )
        ip = processor.image_processor
        print_master(
            f"[Qwen2_5_VL_TS] do_resize={getattr(ip,'do_resize',None)}, size={getattr(ip,'size',None)}, "
            f"min_pixels={getattr(ip,'min_pixels',None)}, max_pixels={getattr(ip,'max_pixels',None)} "
            f"(user_min={user_min_pixels}, user_max={user_max_pixels}, default_min={default_min}, default_max={default_max})"
        )
    elif model_args.model_backbone == INTERNVIDEO2:
        return None
    elif model_args.model_backbone == COLPALI:
        from transformers import AutoProcessor
        processor = ColPaliProcessor.from_pretrained(model_args.model_name)
    else:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(
            model_args.processor_name if model_args.processor_name else model_args.model_name,
            trust_remote_code=True,
        )
    return processor


def get_backbone_name(hf_config, model_type=None):
    if model_type is not None:
        setattr(hf_config, 'model_type', model_type)
    assert hf_config.model_type in SUPPORTED_MODELS, f"Unknown backbone name {hf_config.model_type}.Supported models are {SUPPORTED_MODELS}"
    return MODEL2BACKBONE[hf_config.model_type]


def Llava_NEXT_process_fn(model_inputs: dict, processor, max_length=None):
    # TODO: NOT FINISHED YET!
    input_ids, pixel_values, image_sizes = [], [], []
    texts, visual_inputs = model_inputs['text'], model_inputs['images']
    image_exists = False
    # 1. iterate each pair and process (since processors do not support batch processing)
    for text, images in zip(texts, visual_inputs):
        # in theory, each batch item should contain a list of frames, but we still check for exceptions here
        # if no images as input (not likely to happen in mmeb pro cases)
        if images is None or (type(images)==list and any(i is None for i in images)):
            inputs = processor(images=None, text=text, return_tensors="np", max_length=max_length, truncation=True)
            input_id = inputs["input_ids"].squeeze().tolist()
            if isinstance(input_id, int):
                # in case of empty string, only BOS is included
                input_id = [input_id]
            input_ids.append(input_id)
            pixel_values.append(None)
            image_sizes.append(None)
        else:
            image_exists = True
            # in theory, valid images should be a list of frames
            assert isinstance(images, list), f"images should be a list, but got {type(images)}"
            inputs = processor(images=images, text=text, return_tensors="np", max_length=max_length, truncation=True)
            input_ids.append(inputs["input_ids"].squeeze().tolist())
            pixel_values.append(inputs['pixel_values'])
            image_sizes.append(inputs['image_sizes'])

    # 2. padding inputs
    batch_encoding = processor.tokenizer.pad({'input_ids': input_ids}, return_tensors="pt")
    input_ids, attention_mask = batch_encoding['input_ids'], batch_encoding['attention_mask']
    inputs = {
        'input_ids': input_ids.long(),
        'attention_mask': attention_mask,
        # 'texts': texts,
        # 'images': visual_inputs,
    }
    image_exists = any([p is not None for p in pixel_values])
    if image_exists:
        pixel_values = torch.from_numpy(np.array(pixel_values)).float()
        pixel_values_shape = pixel_values.shape
        pixel_values = pixel_values.reshape(pixel_values_shape[0] * pixel_values_shape[1], *pixel_values_shape[2:])
        image_sizes = torch.tensor(np.array(image_sizes)).long()
        image_sizes_shape = image_sizes.shape
        image_sizes = image_sizes.reshape(image_sizes_shape[0] * image_sizes_shape[1], *image_sizes_shape[2:])
        inputs['pixel_values'] = torch.from_numpy(np.array(pixel_values)).float()
        inputs['image_sizes'] = torch.tensor(np.array(image_sizes)).long()
    else:
        inputs['pixel_values'] = torch.zeros(input_ids.shape[0], 1)
        inputs['image_sizes'] = torch.ones(input_ids.shape[0], 1)

    return inputs


def Phi3V_process_fn(model_inputs: dict, processor, max_length=None):
    input_ids, pixel_values, image_sizes, image_grid_thw = [], [], [], []
    texts, images = model_inputs['text'], model_inputs['images']
    image_exists = False
    # 1. iterate each pair and process (since processors do not support batch processing)
    for text, image in zip(texts, images):
        if image is None:
            inputs = processor(text, None, return_tensors="np", max_length=max_length, truncation=True)
            input_id = inputs["input_ids"].squeeze().tolist()
            if isinstance(input_id, int):
                # in case of empty string, only BOS is included
                input_id = [input_id]
            input_ids.append(input_id)
            pixel_values.append(None)
            image_sizes.append(None)
            image_grid_thw.append(None)
        else:
            image_exists = True
            inputs = processor(text=text, images=[image], return_tensors="np", max_length=max_length, truncation=True)
            input_ids.append(inputs["input_ids"].squeeze().tolist())
            pixel_values.append(inputs['pixel_values'])
            if 'image_sizes' in inputs:
                image_sizes.append(inputs['image_sizes'])
            if 'image_grid_thw' in inputs:
                image_grid_thw.append(inputs['image_grid_thw'])

    # 2. padding inputs
    batch_encoding = processor.tokenizer.pad({'input_ids': input_ids}, return_tensors="pt")
    input_ids, attention_mask = batch_encoding['input_ids'], batch_encoding['attention_mask']
    inputs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'texts': texts,
        'images': images,
    }
    # 3. special postcare for mixed batch (examples w/ and w/o images in the same batch)
    if image_exists:
        # add them to inputs
        inputs['pixel_values'] = pixel_values
        inputs['image_sizes'] = image_sizes
    else:
        inputs['pixel_values'] = torch.zeros(input_ids.shape[0], 1)
        inputs['image_sizes'] = torch.ones(input_ids.shape[0], 1)

    return inputs


# ==== mRoPE 对齐检查工具 ====
import torch

# def _mrope_align_check_batch(batch_inputs, tokenizer, tag=""):
#     """
#     在进入模型前做最终检查：
#     - 逐样本统计：文本中 <|vision_start|> 后紧跟 <|image_pad|> 的次数
#     - 与 batch_inputs['image_grid_thw'][i] 的行数对比
#     """
#     ids_batch = batch_inputs["input_ids"]          # torch.LongTensor [B, L]
#     mask_batch = batch_inputs["attention_mask"]    # torch.LongTensor [B, L]
#     grids_list = batch_inputs.get("image_grid_thw", None)  # list[ndarray/torch] or None

#     vs = tokenizer.convert_tokens_to_ids("<|vision_start|>")
#     vi = tokenizer.convert_tokens_to_ids("<|image_pad|>")

#     B = ids_batch.size(0)
#     for i in range(B):
#         ids_eff = ids_batch[i][mask_batch[i] == 1]
#         pos_vs = (ids_eff == vs).nonzero(as_tuple=False).squeeze(-1)
#         vision_next = ids_eff[pos_vs + 1] if pos_vs.numel() > 0 else ids_eff.new_empty(0)
#         image_nums = int((vision_next == vi).sum().item())

#         grid_rows = 0
#         if grids_list is not None and grids_list[i] is not None:
#             g = grids_list[i]
#             try:
#                 grid_rows = int(g.shape[0])
#             except Exception:
#                 # 兼容 list[(3,), (3,), ...] 的奇怪形态
#                 grid_rows = len(g)

#         if image_nums != grid_rows:
#             # 打印可读文本，助定位是哪条样本出了问题
#             text_preview = tokenizer.decode(ids_eff.tolist(), skip_special_tokens=False)
#             raise AssertionError(
#                 f"[mRoPE mismatch{(' '+tag) if tag else ''}] "
#                 f"sample#{i}: text has {image_nums} <|vision_start|><|image_pad|> segments, "
#                 f"but image_grid_thw has {grid_rows} rows.\n"
#                 f"Decoded text (trimmed): {text_preview[:400]}"
#             )


def Qwen2_VL_process_fn(model_inputs: dict, processor: Qwen2VLProcessor, max_length=None):
    # TODO: set separate max_len for text/visual inputs, currently max_length is only applied to text-only data
    input_ids, pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw = [], [], [], [], []
    texts, visual_inputs = model_inputs['text'], model_inputs['images']
    image_exists = False
    vlm_image_token, vlm_video_token = VLM_IMAGE_TOKENS[QWEN2_VL], VLM_VIDEO_TOKENS[QWEN2_VL]

    # 1. iterate each pair and process, since processors do not support processing for mixed batch (contains data w/ and w/o visual inputs)
    for text, images in zip(texts, visual_inputs):
        if images is None or (type(images)==list and any(i is None for i in images)):
            # all images must be valid
            inputs = processor(text=[text], images=None, return_tensors="np", max_length=max_length, truncation=True)
            input_id = inputs["input_ids"].squeeze().tolist()
            if isinstance(input_id, int):
                # in case of empty string, only BOS is included
                input_id = [input_id]
            input_ids.append(input_id)
            pixel_values.append(None)
            image_grid_thw.append(None)
            pixel_values_videos.append(None)
            video_grid_thw.append(None)
        else:
            try:
                if vlm_image_token in text:
                    if isinstance(images, PIL.Image.Image):
                        # images is a single image
                        images = [images]
                    for iid, image in enumerate(images):
                        # rare case in MMEB eval: resize to 28*28 if either w or h is smaller than 28
                        if image.size[0] < 28 or image.size[1] < 28:
                            image = image.resize((56, 56))
                            images[iid] = image
                    inputs = processor(
                        text=[text], 
                        images=images, 
                        return_tensors="np", 
                        max_length=None, 
                        truncation=False, 
                        input_data_format=ChannelDimension.LAST,
                    )
                elif vlm_video_token in text:
                    # TODO: check text/video data validity
                    inputs = processor(
                        text=[text], 
                        videos=[images], 
                        return_tensors="np", 
                        max_length=None, 
                        truncation=False, 
                        input_data_format=ChannelDimension.LAST,
                    )
                else:
                    raise NotImplementedError(f"No visual token found ({vlm_image_token} or {vlm_video_token}) in the text: {text}")
            except Exception as e:
                print(f"Error in Qwen2_VL_process_fn: {e}")
                for i, img in enumerate(images):
                    print(f"Image {i}: {type(img)}, size: {getattr(img, 'size', 'unknown')}")
                raise e
            input_ids.append(inputs["input_ids"].squeeze().tolist())
            if 'pixel_values' in inputs:
                pixel_values.append(inputs['pixel_values'])
                image_grid_thw.append(inputs['image_grid_thw'])
                pixel_values_videos.append(None)
                video_grid_thw.append(None)
            else:
                pixel_values.append(None)
                image_grid_thw.append(None)
                pixel_values_videos.append(inputs['pixel_values_videos'])
                video_grid_thw.append(inputs['video_grid_thw'])

    # 2. padding inputs
    batch_encoding = processor.tokenizer.pad({'input_ids': input_ids}, return_tensors="pt")
    input_ids, attention_mask = batch_encoding['input_ids'], batch_encoding['attention_mask']
    # manually enforce long type due to:
    # (1) [rank7]: RuntimeError: Expected tensor for argument #1 'indices' to have one of the following scalar types: Long, Int; but got torch.cuda.FloatTensor instead (while checking arguments for embedding)
    # (2) [rank7]:   File "/fsx/home/ruimeng/project/VLM2Vec/src/model.py", line 45, in _pooling
    #     [rank7]:     reps = last_hidden_state[
    #     [rank7]: IndexError: tensors used as indices must be long, int, byte or bool tensors
    inputs = {
        'input_ids': input_ids.long(),
        'attention_mask': attention_mask.long(), 
        'texts': texts,
        'images': visual_inputs,
    }
    inputs['pixel_values'] = pixel_values
    inputs['image_grid_thw'] = image_grid_thw
    inputs['pixel_values_videos'] = pixel_values_videos
    inputs['video_grid_thw'] = video_grid_thw

    # === 修复开始：把 list[ (1,3) ] → (N,3) ===
    def _stack_grids(grids_list):
        elems = []
        for g in grids_list:
            if g is None:
                continue
            a = np.asarray(g)
            a = a.reshape(-1, 3)           # (3,) / (1,3) / (k,3) 都标准化为 (k,3)
            elems.append(a)
        if len(elems) == 0:
            return None
        cat = np.concatenate(elems, axis=0)  # (N,3)
        return torch.as_tensor(cat, dtype=torch.long)

    inputs['image_grid_thw'] = _stack_grids(image_grid_thw)
    inputs['video_grid_thw'] = _stack_grids(video_grid_thw)

    # _mrope_align_check_batch(inputs, processor.tokenizer, tag="Qwen2_VL_process_fn")
    return inputs

def Gme_process_fn(model_inputs: dict, processor: Qwen2VLProcessor, max_length=None):
    inputs = {
        'texts': model_inputs['text'],
        'images': model_inputs['images'],
    }
    return inputs


def Qwen2_VL_TokenSelection_process_fn(model_inputs: dict, processor: Qwen2VLTokenSelectionProcessor, max_length=None):
    # TODO: set separate max_len for text/visual inputs, currently max_length is only applied to text-only data
    input_ids, pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw = [], [], [], [], []
    patch_pos, select_mask = [], []
    texts, visual_inputs = model_inputs['text'], model_inputs['images']
    image_exists = False
    # 1. iterate each pair and process (since processors do not support batch processing)
    for text, images in zip(texts, visual_inputs):
        if images is None or (type(images)==list and any(i is None for i in images)):
            # all images must be valid
            inputs = processor(text=[text], images=None, return_tensors="np", max_length=max_length, truncation=True)
            input_id = inputs["input_ids"].squeeze().tolist()
            if isinstance(input_id, int):
                # in case of empty string, only BOS is included
                input_id = [input_id]
            input_ids.append(input_id)
            pixel_values.append(None)
            image_grid_thw.append(None)
            patch_pos.append(None)
            select_mask.append(None)
            pixel_values_videos.append(None)
            video_grid_thw.append(None)
        else:
            image_exists = True
            # TODO only
            # handling multi-image data from videos, cannot deal with mixed image + video data
            if VLM_IMAGE_TOKENS[QWEN2_VL] in text:
                inputs = processor(text=[text], images=[images], return_tensors="np", max_length=None, truncation=False, input_data_format=ChannelDimension.LAST)
            elif VLM_VIDEO_TOKENS[QWEN2_VL] in text:
                assert len(images) > 1, f"Video data must have more than 1 frame, got {len(images)}"
                inputs = processor(text=[text], videos=[images], return_tensors="np", max_length=None, truncation=False, input_data_format=ChannelDimension.LAST)
            else:
                raise NotImplementedError(f"Unsupported visual token in text: {text}")
            input_ids.append(inputs["input_ids"].squeeze().tolist())
            if 'pixel_values' in inputs:
                pixel_values.append(inputs['pixel_values'])
                image_grid_thw.append(inputs['image_grid_thw'])
                pixel_values_videos.append(None)
                video_grid_thw.append(None)
                if 'patch_pos' in inputs:
                    patch_pos.append(inputs['patch_pos'])
                if 'select_mask' in inputs:
                    select_mask.append(inputs['select_mask'])
            else:
                pixel_values.append(None)
                image_grid_thw.append(None)
                patch_pos.append(None)
                select_mask.append(None)
                pixel_values_videos.append(inputs['pixel_values_videos'])
                video_grid_thw.append(inputs['video_grid_thw'])

    # 2. padding inputs
    batch_encoding = processor.tokenizer.pad({'input_ids': input_ids}, return_tensors="pt")
    input_ids, attention_mask = batch_encoding['input_ids'], batch_encoding['attention_mask']

    if image_exists:
        if patch_pos:
            patch_pos_shape_for_padding = list(v.shape for v in patch_pos if v is not None)[0]
            key_tmp = [torch.from_numpy(v) if v is not None else (torch.zeros(patch_pos_shape_for_padding) - 1) for v in patch_pos]
            max_length = input_ids.size(1)
            padded_key = [torch.nn.functional.pad(pos, (0, max_length - pos.size(1)), value=-1) for pos in key_tmp]
            patch_pos = torch.cat(padded_key, dim=0)
        if select_mask:
            select_mask_shape_for_padding = list(v.shape for v in select_mask if v is not None)[0]
            key_tmp = [torch.from_numpy(v) if v is not None else torch.ones(select_mask_shape_for_padding).bool() for v in select_mask]
            max_length = input_ids.size(1)
            padded_key = [torch.nn.functional.pad(pos, (0, max_length - pos.size(1)), value=True) for pos in key_tmp]
            select_mask = torch.cat(padded_key, dim=0)

    # manually enforce long type due to:
    # (1) [rank7]: RuntimeError: Expected tensor for argument #1 'indices' to have one of the following scalar types: Long, Int; but got torch.cuda.FloatTensor instead (while checking arguments for embedding)
    # (2) [rank7]:   File "/fsx/home/ruimeng/project/VLM2Vec/src/model.py", line 45, in _pooling
    #     [rank7]:     reps = last_hidden_state[
    #     [rank7]: IndexError: tensors used as indices must be long, int, byte or bool tensors
    inputs = {
        'input_ids': input_ids.long(),
        'attention_mask': attention_mask.long()
    }
    inputs['pixel_values'] = pixel_values
    inputs['image_grid_thw'] = image_grid_thw
    inputs['pixel_values_videos'] = pixel_values_videos
    inputs['video_grid_thw'] = video_grid_thw
    inputs['patch_pos'] = patch_pos
    inputs['select_mask'] = select_mask

    return inputs


def InternVL_process_fn(model_inputs: dict, processor, max_length=None):
    # TODO not working yet
    input_ids, pixel_values, image_sizes, image_grid_thw = [], [], [], []
    texts, images = model_inputs['text'], model_inputs['images']
    image_exists = False
    # 1. iterate each pair and process (since processors do not support batch processing)
    for text, image in zip(texts, images):
        if image is None:
            inputs = processor(text, None, return_tensors="np", max_length=max_length, truncation=True)
            input_id = inputs["input_ids"].squeeze().tolist()
            if isinstance(input_id, int):
                # in case of empty string, only BOS is included
                input_id = [input_id]
            input_ids.append(input_id)
            pixel_values.append(None)
            image_sizes.append(None)
            image_grid_thw.append(None)
        else:
            image_exists = True
            inputs = processor(text=text, images=[image], return_tensors="np", max_length=max_length, truncation=True)
            input_ids.append(inputs["input_ids"].squeeze().tolist())
            pixel_values.append(inputs['pixel_values'])
            if 'image_sizes' in inputs:
                image_sizes.append(inputs['image_sizes'])
            if 'image_grid_thw' in inputs:
                image_grid_thw.append(inputs['image_grid_thw'])

    # 2. padding inputs
    batch_encoding = processor.tokenizer.pad({'input_ids': input_ids}, return_tensors="pt")
    input_ids, attention_mask = batch_encoding['input_ids'], batch_encoding['attention_mask']
    inputs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'texts': texts,
        'images': images,
    }
    # 3. special postcare for mixed batch (examples w/ and w/o images in the same batch)
    if image_exists:
        # add them to inputs
        inputs['pixel_values'] = pixel_values
        inputs['image_sizes'] = image_sizes
    else:
        inputs['pixel_values'] = torch.zeros(input_ids.shape[0], 1)
        inputs['image_sizes'] = torch.ones(input_ids.shape[0], 1)

    return inputs


def ColPali_process_fn(model_inputs: dict, processor, max_length=None):
    texts, images = model_inputs['text'], model_inputs['images']
    if images is None or all(i is None for i in images):
        inputs = processor.process_queries(texts)
    else:
        inputs = processor.process_images(images)
    return inputs


def InternVideo2_process_fn(model_inputs: dict, processor, max_length=None):
    if all(x is None for x in model_inputs["images"]):
        # Text side
        from src.model.baseline_backbone.internvideo2.modeling_internvideo2 import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
        inputs = tokenizer(
            model_inputs["text"],
            padding="max_length",
            truncation=True,
            max_length=40,
            return_tensors="pt")
    else:
        # Video side
        from torchvision import transforms
        preprocess = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            transforms.Resize((224, 224)),  # Resize to 224x224
            transforms.ToTensor(),  # Convert from PIL image to tensor (C, H, W)
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean
                                 std=[0.229, 0.224, 0.225])  # ImageNet std
        ])
        frame_list = model_inputs["images"]
        # to make image inputs be exact 4 frames
        # Case 1: frame_list is flat (not a list of lists), e.g., [PIL, PIL, ...]
        if type(frame_list[0]) is not list:
            frame_list = [[img.copy() for _ in range(4)] for img in frame_list]
        # Case 2: frame_list is already a list of lists, ensure each has exactly 4 images
        elif type(frame_list[0]) is list and len(frame_list[0]) != 4:
            new_list = []
            for frames in frame_list:
                if len(frames) < 4:
                    frames = frames + [frames[-1].copy() for _ in range(4 - len(frames))]
                elif len(frames) > 4:
                    # Sample 4 indices uniformly across the sequence
                    indices = np.linspace(0, len(frames) - 1, num=4, dtype=int)
                    frames = [frames[i] for i in indices]
                new_list.append(frames)
            frame_list = new_list
        pixel_values = [
            torch.stack([preprocess(img) for img in frames], dim=0)  # (num_frames, C, H, W)
            for frames in frame_list
        ]

        pixel_values = torch.stack(pixel_values, dim=0)  # (B, num_frames, C, H, W)
        inputs = {'pixel_values': pixel_values}

    return inputs


def e5_v_prompt_template(text, add_video_token, add_image_token):
    llama3_template = '<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n \n'
    if text is not None and add_video_token is False and add_image_token is False:  # only text
        prompt = llama3_template.format('{}\nSummary above sentence in one word: '.format(text))
    if text is None and add_video_token:  # only video
        prompt = llama3_template.format('<image>\nSummary above video in one word: ')
    if text is None and add_image_token:  # only image
        prompt = llama3_template.format('<image>\nSummary above image in one word: ')
    if text is not None and add_video_token:  # video + text
        prompt = llama3_template.format('<image>\n{}\nSummary above video and text in one word: '.format(text))
    if text is not None and add_image_token:
        prompt = llama3_template.format('<image>\n{}\nSummary above image and text in one word: '.format(text))

    return prompt


def qwen_retrieval_prompt_template(text, add_video_token, add_image_token):
    """Qwen 系列用于检索的 prompt 模板。
    注意：此处无法得知实际图片数量，因此当存在图片/视频时，各添加一个占位片段。
    若需要精确重复次数，可在调用处传入组合好的文本（包含多次占位符）。
    """
    system_prompt = ("You are a multimodal retrieval encoder (not a conversational assistant). "
                     "Your sole function is to map any input—image(s), text, or image(s) plus text—into a "
                     "compact embedding for nearest-neighbor retrieval in a shared space.")
    input_str = ""
    if add_image_token:
        # input_str += "<|vision_start|><|image_pad|><|vision_end|>"
        input_str += "<|image_pad|>"
    if add_video_token:
        input_str += "<|vision_start|><|video_pad|><|vision_end|>"
    if text:
        input_str = input_str + text
    msg = f'<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{input_str}<|im_end|>\n<|im_start|>assistant\n<|endoftext|>'
    return msg

PROMPT_TEMPLATE_DICT = {
    "e5_v": e5_v_prompt_template,
    QWEN2_VL: qwen_retrieval_prompt_template,
    QWEN2_5_VL: qwen_retrieval_prompt_template,
    QWEN2_VL_TOKENSELECTION: qwen_retrieval_prompt_template,
    QWEN2_5_VL_TOKENSELECTION: qwen_retrieval_prompt_template,
}


def process_input_text(instruction, model_backbone, text=None, add_video_token=False, add_image_token=False):
    # Formulate input text based on text, special token and instruction.
    # TBD: Reorganize the hard-code part for baselines such as internvideo2
    if model_backbone == "internvideo2":
        return text
    elif model_backbone in [QWEN2_VL, QWEN2_5_VL, QWEN2_VL_TOKENSELECTION, QWEN2_5_VL_TOKENSELECTION]:
        # 组合 instruction 与文本，按用户要求将其接在视觉占位符之后
        combined_text = instruction if instruction else ""
        if text:
            combined_text = (combined_text + " " + text) if combined_text else text
        return PROMPT_TEMPLATE_DICT[model_backbone](combined_text, add_video_token, add_image_token)
    elif model_backbone in [GME, LamRA, LamRA_QWEN2_5]:
        if text:
            return instruction + " " + text # GME and LamRA do not need special tokens
        else:
            return instruction + " "
    elif model_backbone == E5_V:
        # e5_v 模板沿用原逻辑
        return PROMPT_TEMPLATE_DICT[model_backbone](text, add_video_token, add_image_token)

    prompt = instruction
    if text:
        prompt = prompt + " " + text
    if add_video_token:
        video_token = VLM_VIDEO_TOKENS[model_backbone]
        prompt = video_token + " " + prompt
    if add_image_token:
        image_token = VLM_IMAGE_TOKENS[model_backbone]
        # image_token = "<|vision_start|>"+VLM_IMAGE_TOKENS[model_backbone]+"<|vision_end|>"
        prompt = image_token + " " + prompt

    return prompt


process_vlm_inputs_fns = {
    PHI3V: Phi3V_process_fn,
    LLAVA_NEXT: Llava_NEXT_process_fn,
    QWEN2_VL: Qwen2_VL_process_fn,
    QWEN2_5_VL: Qwen2_VL_process_fn,
    QWEN2_VL_TOKENSELECTION: Qwen2_VL_TokenSelection_process_fn,
    QWEN2_5_VL_TOKENSELECTION: Qwen2_VL_TokenSelection_process_fn,
    INTERNVIDEO2: InternVideo2_process_fn,
    GME: Gme_process_fn,
    LamRA: Gme_process_fn,
    LamRA_QWEN2_5: Gme_process_fn,
    COLPALI: ColPali_process_fn,
    E5_V: Llava_NEXT_process_fn,
}
