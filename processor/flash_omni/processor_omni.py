import requests, re, ast, ujson, os, sys, fire, glob, random, time, tempfile, math, subprocess, threading
import numpy as np
import io
import torch
from torch.utils.data import default_collate
import soundfile as sf
from typing import *
from dataclasses import dataclass, field
import transformers
from transformers.modeling_outputs import ModelOutput
from transformers.audio_utils import mel_filter_bank, spectrogram, window_function
from functools import lru_cache
from io import BytesIO
from PIL import Image
import concurrent.futures as cf
from transformers.image_transforms import resize, center_crop, get_resize_output_image_size
from transformers.image_utils import PILImageResampling
from PIL import Image, ImageOps
from PIL import ImageFile
torch.set_num_threads(1)  # 限制torch的线程数 否则可能会卡住
os.environ["TOKENIZERS_PARALLELISM"] = "false"
ImageFile.LOAD_TRUNCATED_IMAGES = True
import base64
from decord import VideoReader, cpu
import cv2
import av
import imagesize
from multiprocessing import Pool
from cairosvg import svg2png
import hashlib
# import mssapi
import logging
import librosa
from types import SimpleNamespace
# mssapi.log.setLevel(logging.ERROR)
pycat_logger = logging.getLogger('pycat.cat')
pycat_logger.setLevel(logging.CRITICAL)

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
VIDEO_TOTAL_PIXELS = 24576 * 28 * 28
FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768

def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:
    1. Both dimensions (height and width) are divisible by 'factor'.
    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def split_text(text, match_regex):
    matches = list(re.finditer(match_regex, text))
    # 初始化结果列表
    result = []
    match_flag_list = []
    # 上一个匹配的结束位置
    last_end = 0
    # 遍历所有匹配项
    for match in matches:
        # 添加匹配项之前的部分
        if text[last_end:match.start()]:
            result.append(text[last_end:match.start()])
            match_flag_list.append(False)
        # 添加匹配项
        result.append(match.group(0))
        match_flag_list.append(True)
        # 更新上一个匹配的结束位置
        last_end = match.end()
    # 添加最后一个匹配项之后的部分
    if text[last_end:]:
        result.append(text[last_end:])
        match_flag_list.append(False)
    return result, match_flag_list


def read_video(image_path, max_frame_number, decode_way):
    if decode_way=='1fps':
        try:
            # print(image_path)
            # vr = VideoReader(image_path, ctx=cpu(0), buffer_size=100<<20) # 调节缓冲区大小，否则多线程解码视频容易缓存区溢出
            vr = VideoReader(image_path, ctx=cpu(0), num_threads=1) # 禁用多线程解码视频，防止缓存区溢出
            total_frame_num = len(vr)
            fps = round(vr.get_avg_fps())
            frame_idx = [i for i in range(0, len(vr), fps)]
            frames = vr.get_batch(frame_idx).asnumpy()
            cnt = len(frames)
            frame_times = range(cnt)
        except Exception as e:
            # print(image_path)
            print('error is', e)
            # raise
            return None
    elif decode_way=='key':
        try: 
            with av.open(image_path) as container:                         
                stream = container.streams.video[0]
                stream.codec_context.skip_frame = 'NONKEY'
                frames = []
                frame_times = []
                fps = int(stream.average_rate)
                cnt = 0
                for frame in container.decode(stream): # 关键帧存成image patch
                    image = np.array(frame.to_image())
                    frames.append(image)
                    frame_time = int(frame.time)
                    frame_times.append(frame_time)
                    cnt += 1
        except Exception as e:
            print('error is', e)
            return None
    if frames is None or len(frames)==0:
        # print("读出的帧为None", "文件路径", image_path)
        # raise
        return None
    if len(frames)>max_frame_number and max_frame_number>0:
        # 生成14个均匀间隔的索引
        indices = np.linspace(0, len(frames) - 1, max_frame_number, dtype=int)
        # 根据索引获取对应元素
        frames = frames[indices]
        frame_times = frame_times[indices]
    return frames, frame_times


class OmniImageProcessor:
    def __init__(self, config, **kwargs):
        self.config = config  # visual_config
        self.min_pixels = self.config.min_pixels if hasattr(self.config, 'min_pixels') else 56 * 56
        self.max_pixels = self.config.max_pixels if hasattr(self.config, 'max_pixels') else 28 * 28 * 1280
        self.patch_size = self.config.patch_size if hasattr(self.config, 'patch_size') else 14
        self.temporal_patch_size = self.config.temporal_patch_size if hasattr(self.config, 'temporal_patch_size') else 2
        self.merge_size = self.config.merge_size if hasattr(self.config, 'merge_size') else 2
        self.spatial_merge_size = self.config.spatial_merge_size if hasattr(self.config, 'spatial_merge_size') else 2
        self.fixed_image_size = self.config.image_size if hasattr(self.config, 'image_size') else None
        
    def _center_crop_resize(self, image, output_size):
        # output_size is int 
        # Center crop the image to a square before resizing
        width, height = image.size
        min_side = min(width, height)
        left = (width - min_side) // 2
        top = (height - min_side) // 2
        right = left + min_side
        bottom = top + min_side
        image = image.crop((left, top, right, bottom))
        image = image.resize((output_size,output_size), PILImageResampling.BICUBIC)
        return image
    
    def image_transform(self, strseq, return_mm_data = True, fix_res:int=-1):
        image = None
        if isinstance(strseq, str):
            if return_mm_data:
                image = Image.open(strseq).convert("RGB") 
        elif isinstance(strseq, Image.Image):
            image = strseq
        else:
            try:
                image = Image.open(BytesIO(strseq)).convert("RGB")
            except:
                image = Image.open(BytesIO(svg2png(bytestring=strseq))).convert("RGB") # interleaved有的是矢量图，需要转换
            
        image=image.convert("RGB")  # 这一步首先将图像转换为 RGB 格式，确保图像有三个通道（R、G、B）。然后使用 np.array() 将其转换为 NumPy 数组，方便后续处理。
        # 用于固定分辨率生成
        # TODO: 对于OCR数据怎么办?
        if fix_res>0:
            image = self._center_crop_resize(image, output_size=fix_res)
        image = np.array(image)
        image_org_size = image.shape[:2]  # 这里保存了图像的原始大小（高度和宽度），image.shape 返回图像的形状 (高度, 宽度, 通道数)，而 image.shape[:2] 提取了前两个值，即原始的高度和宽度。这个信息可以用于后续的对比或其他处理。
        
        # resize, crop, scale, normalize
        # 输出一个新的尺寸，这个尺寸通常是 (宽度, 高度) 格式，用于后续的图像调整操作，如缩放或裁剪。
        resized_height, resized_width = smart_resize(
            image_org_size[0], image_org_size[1],
            factor=self.patch_size * self.spatial_merge_size,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        output_size = (resized_height, resized_width)
        
        # 使用 resize 函数将图像调整到 output_size 大小。PILImageResampling.BICUBIC 指定使用双三次插值法来进行图像缩放，这种方法通常能够提供较好的图像质量。
        # image: 输入的图像数据，可以是 NumPy 数组或 PIL 图像对象；output_size: 目标大小，通常是一个二元组 (宽度, 高度)。这个尺寸可以是图像的绝对大小，也可以是相对于原始图像的比例；
        # resample: 可选的重采样方法，通常用于确定如何插值像素。例如，PILImageResampling.BICUBIC 表示使用双三次插值法，这是一种平滑的插值方法，常用于图像缩放。
        image = resize(image, output_size, PILImageResampling.BICUBIC)
        img = image.transpose(2, 0, 1)
        # 对图像进行归一化和标准化处理
        image = (img / 255.0 - np.array(self.config.image_mean)[:, np.newaxis, np.newaxis]) / np.array(self.config.image_std)[:,np.newaxis,np.newaxis]
        # 处理成patch
        patches = image[np.newaxis, :]
        if patches.shape[0] == 1:
            patches = np.tile(patches, (self.temporal_patch_size, 1, 1, 1))
        channel = patches.shape[1]
        grid_t = patches.shape[0] // self.temporal_patch_size
        grid_h, grid_w = resized_height // self.patch_size, resized_width // self.patch_size

        patches = patches.reshape(
            grid_t,
            self.temporal_patch_size,
            channel,
            grid_h // self.spatial_merge_size,
            self.spatial_merge_size,
            self.patch_size,
            grid_w // self.spatial_merge_size,
            self.spatial_merge_size,
            self.patch_size,
        )
        patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        flatten_patches = patches.reshape(
            grid_t * grid_h * grid_w, channel * self.temporal_patch_size * self.patch_size * self.patch_size
        )
        
        return flatten_patches, image_org_size, (grid_t, grid_h, grid_w)

'''
    follow Transfomers最新版本设计，video不再复用ImageProcessor的逻辑，而是重新设计
        logger.warning(
            "`Qwen2VLImageProcessor` works only with image inputs and doesn't process videos anymore. "
            "This is a deprecated behavior and will be removed in v5.0. "
            "Your videos should be forwarded to `Qwen2VLVideoProcessor`. "
        )
'''
class OmniVideoProcessor:
    def __init__(self, config, **kwargs):
        self.config = config  # visual_config
        self.min_pixels = self.config.min_pixels if hasattr(self.config, 'min_pixels') else 56 * 56
        self.max_pixels = self.config.max_pixels if hasattr(self.config, 'max_pixels') else 28 * 28 * 768
        self.patch_size = self.config.patch_size if hasattr(self.config, 'patch_size') else 14
        self.temporal_patch_size = self.config.temporal_patch_size if hasattr(self.config, 'temporal_patch_size') else 2
        self.merge_size = self.config.merge_size if hasattr(self.config, 'merge_size') else 2
        self.spatial_merge_size = self.config.spatial_merge_size if hasattr(self.config, 'spatial_merge_size') else 2
        self.split_video = self.config.split_video if hasattr(self.config, 'split_video') else False
        self.time_type = self.config.time_type if hasattr(self.config, 'time_type') else "qwen2_vl"
        assert self.time_type in ['qwen2_vl', '3DRope'], "only support time type in ['qwen2_vl', '3DRope']"

    def extracted_images_transform(self, path_list, return_mm_data = True):
        extracted_images = []
        for strseq in path_list:
            image = None
            if isinstance(strseq, str):
                if return_mm_data:
                    max_attempts = 3
                    attempts = 0
                    while attempts < max_attempts:
                        try:
                            image = Image.open(strseq).convert("RGB") 
                            break
                        except Exception as e:
                            attempts += 1
                            if attempts == max_attempts:
                                raise  # 达到最大重试次数，抛出原始异常
                            time.sleep(0.1)  # 可选：短暂延迟后重试，避免立即重试
            else:
                try:
                    image = Image.open(BytesIO(strseq)).convert("RGB")
                except:
                    image = Image.open(BytesIO(svg2png(bytestring=strseq))).convert("RGB") # interleaved有的是矢量图，需要转换
                
            image = np.array(image.convert("RGB"))  # 这一步首先将图像转换为 RGB 格式，确保图像有三个通道（R、G、B）。然后使用 np.array() 将其转换为 NumPy 数组，方便后续处理。
            image_org_size = image.shape[:2]  # 这里保存了图像的原始大小（高度和宽度），image.shape 返回图像的形状 (高度, 宽度, 通道数)，而 image.shape[:2] 提取了前两个值，即原始的高度和宽度。这个信息可以用于后续的对比或其他处理。
            
            # resize, crop, scale, normalize
            # 输出一个新的尺寸，这个尺寸通常是 (宽度, 高度) 格式，用于后续的图像调整操作，如缩放或裁剪。
            resized_height, resized_width = smart_resize(
                image_org_size[0], image_org_size[1],
                factor=self.patch_size * self.spatial_merge_size,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
            )
            output_size = (resized_height, resized_width)
        
            # 使用 resize 函数将图像调整到 output_size 大小。PILImageResampling.BICUBIC 指定使用双三次插值法来进行图像缩放，这种方法通常能够提供较好的图像质量。
            # image: 输入的图像数据，可以是 NumPy 数组或 PIL 图像对象；output_size: 目标大小，通常是一个二元组 (宽度, 高度)。这个尺寸可以是图像的绝对大小，也可以是相对于原始图像的比例；
            # resample: 可选的重采样方法，通常用于确定如何插值像素。例如，PILImageResampling.BICUBIC 表示使用双三次插值法，这是一种平滑的插值方法，常用于图像缩放。
            image = resize(image, output_size, PILImageResampling.BICUBIC)
            img = image.transpose(2, 0, 1)
            # 对图像进行归一化和标准化处理
            image = (img / 255.0 - np.array(self.config.image_mean)[:, np.newaxis, np.newaxis]) / np.array(self.config.image_std)[:,np.newaxis,np.newaxis]
            extracted_images.append(image)
        # 处理成patch
        patches = np.asarray(extracted_images)

        # Check that videos have `num_frames` divisible by `temporal_patch_size`
        if patches.shape[0] % self.temporal_patch_size != 0:
            repeats = np.tile(patches[-1:], (self.temporal_patch_size-1, 1, 1, 1)) # repeat
            patches = np.concatenate((patches, repeats), axis=0)

        try:
            channel = patches.shape[1]
        except:
            print("无channle, patches.shape:", patches.shape)
            raise
        grid_t = patches.shape[0] // self.temporal_patch_size
        grid_h, grid_w = resized_height // self.patch_size, resized_width // self.patch_size
        patches = patches.reshape(
            grid_t,
            self.temporal_patch_size,
            channel,
            grid_h // self.spatial_merge_size,
            self.spatial_merge_size,
            self.patch_size,
            grid_w // self.spatial_merge_size,
            self.spatial_merge_size,
            self.patch_size,
        )
        patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        flatten_patches = patches.reshape(
            grid_t * grid_h * grid_w, channel * self.temporal_patch_size * self.patch_size * self.patch_size
        ) # 通过邻域合并的方式来减少token量，spatial_merge_size为2，空间域上减少四倍token量

        return flatten_patches, image_org_size, (grid_t, grid_h, grid_w)

class OmniAudioProcessor:
    # 包含基本的音频特征抽取模块 + 输入数据解析模块
    def __init__(
        self,
        config,  # audio processor config
        **kwargs
    ):
        # torchaudio 2.10+ removed list_audio_backends(), using librosa/soundfile as backend
        # soundfile is available as the audio backend
        self.config = config
        self.mel_filters = mel_filter_bank(
            num_frequency_bins=1 + self.config.n_fft // 2,
            num_mel_filters=self.config.num_mel_bins,
            min_frequency=0.0,
            max_frequency=self.config.sampling_rate / 2.0,
            sampling_rate=self.config.sampling_rate,
            norm="slaney",
            mel_scale="slaney",
        )
        self.window = torch.hann_window(self.config.n_fft)
        
    @staticmethod
    def dynamic_range_compression(x, C=1, clip_val=1e-6):
        return torch.log(torch.clamp(x, min=clip_val) * C)

    @staticmethod
    def zero_mean_unit_var_norm(x):
        return (x - x.mean()) / torch.sqrt(x.var() + 1e-8)

    def load_audio_waveform(self, uri, metadata=None, waveform_tensor=None, return_tensors=True, do_normalize=False):
        if metadata is None or waveform_tensor is None:
            # 使用 librosa 统一处理所有音频格式（包括 mp3, wav, flac 等）
            # librosa.load 返回的已经是归一化的 float32 数据
            waveform_np, sample_rate = librosa.load(uri, sr=None, mono=False)
            
            # 转换为 tensor，确保维度为 (channels, samples)
            if waveform_np.ndim == 1:
                waveform_tensor = torch.from_numpy(waveform_np).unsqueeze(0)
            else:
                waveform_tensor = torch.from_numpy(waveform_np)
            
            # 获取音频元信息
            try:
                sf_info = sf.info(uri)
                metadata = SimpleNamespace(
                    sample_rate=sample_rate,
                    num_frames=waveform_tensor.shape[1],
                    num_channels=waveform_tensor.shape[0],
                    bits_per_sample=getattr(sf_info, 'bits_per_sample', 16),
                    encoding=getattr(sf_info, 'subtype', 'PCM_F')
                )
            except Exception:
                # 如果 soundfile.info 失败，使用 librosa 提供的信息
                metadata = SimpleNamespace(
                    sample_rate=sample_rate,
                    num_frames=waveform_tensor.shape[1],
                    num_channels=waveform_tensor.shape[0],
                    bits_per_sample=16,
                    encoding='PCM_F'
                )
        
        assert(metadata.num_channels <= 2), "acoustic file with {} channels.".format(metadata.num_channels)  # whisper only accept mono channel audio
            
        if self.config.sampling_rate != metadata.sample_rate:
            # 使用 torch.functional 进行重采样
            waveform_tensor = torch.nn.functional.interpolate(
                waveform_tensor.unsqueeze(0), 
                size=int(waveform_tensor.shape[1] * self.config.sampling_rate / metadata.sample_rate),
                mode='linear',
                align_corners=False
            ).squeeze(0)

        # downmix to mono channel https://trac.ffmpeg.org/wiki/AudioChannelManipulation
        if metadata.num_channels > 1:
            waveform_tensor = torch.mean(waveform_tensor, dim=0, keepdim=True)

        # normalized to zero mean (Qwen Audio没有处理 但Whisper官方实现)
        if do_normalize:
            waveform_tensor = self.zero_mean_unit_var_norm(waveform_tensor)

        if return_tensors:  # (channels, samples)
            return waveform_tensor
        else:
            return waveform_tensor.numpy() 

    def split_with_overlap(self, waveform):  # 如果长度超过最大长度限制 分割为带overlap的多段
        channels, wave_samples = waveform.shape
        max_audio_samples = self.config.max_audio_seconds * self.config.sampling_rate
        if wave_samples <= max_audio_samples or self.config.split_overlap < 0:
            return [waveform]  # 没有超出最大长度or截断逻辑 统一返回list
        
        split_waveform, start = [], 0
        while start < wave_samples:  # 统一按秒数对齐overlap
            if start > int(self.config.sampling_rate * self.config.split_overlap):
                start -= int(self.config.sampling_rate * self.config.split_overlap)  # 0表示没有overlap，>0 overlap对应秒数
            end = min(start + max_audio_samples, wave_samples)
            if end - start>= self.config.n_fft: # 保证至少有一帧数据
                split_waveform.append(waveform[:, start:end])  # 注意这里可能会切割出特别短的片段 需要在预处理判断并丢弃
            start = end
        return split_waveform

    @classmethod        
    def inference_output_length(cls, config, input_length):
        # for whisper + bridge
        kernel_size = config.kernel_size
        stride_size = config.stride_size
        avg_pooler = config.avg_pooler
        encoder_length = (input_length + 2 * (kernel_size // 2) - kernel_size) // 1 + 1  # conv layer1 with pad=1
        encoder_length = (encoder_length + 2 * (kernel_size // 2) - kernel_size) // stride_size + 1  # conv layer2 with pad=1
        if avg_pooler > 1:
            bridge_length = encoder_length // avg_pooler
        return encoder_length, bridge_length

    def extract_fbank_features(self, waveform):
        # ref: https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/feature_extraction_whisper.py
        channels, wave_samples = waveform.shape
        assert(wave_samples >= self.config.n_fft)
        valid_frame_nums = min(self.config.max_audio_seconds * self.config.sampling_rate // self.config.hop_length, wave_samples // self.config.hop_length + 1)
        if wave_samples < self.config.max_audio_seconds * self.config.sampling_rate:
            waveform = torch.nn.functional.pad(waveform, (0, self.config.max_audio_seconds * self.config.sampling_rate - wave_samples), "constant", 0)
        else:
            waveform = waveform[:, :self.config.max_audio_seconds * self.config.sampling_rate]

        # window = torch.hann_window(self.config.n_fft)
        stft = torch.stft(waveform, self.config.n_fft, self.config.hop_length, window=self.window, return_complex=True)  # fft, len(wave) // n_fft // 2 + 1
        magnitudes = stft[..., :-1].abs() ** 2

        mel_filters = torch.from_numpy(self.mel_filters).type(torch.float32)
        mel_spec = mel_filters.T @ magnitudes
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        if waveform.dim() == 2:
            max_val = log_spec.max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]
            log_spec = torch.maximum(log_spec, max_val - 8.0)
        else:
            log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0

        log_spec = log_spec[0].numpy()  # (channel, filters, samples) -> (filters, samples)
        log_spec[:, valid_frame_nums:] = 0.0  # pad0

        return log_spec, valid_frame_nums

    def data_augment(self, feature: np.array, input_length, training=True):
        # reference https://arxiv.org/pdf/1904.08779
        def mask_start_indices(input_length, mask_length, min_masks, mask_prob):
            num_masked_span = int(mask_prob * input_length / mask_length + random.random())
            num_masked_span = max(num_masked_span, min_masks)
            start_indices = list(range(input_length - mask_length))
            random.shuffle(start_indices)
            start_indices = start_indices[:num_masked_span]
            return start_indices

        if not training or (self.config.mask_time_prob <= 0 and self.config.mask_feature_prob <= 0):
            return feature
        if input_length < self.config.mask_time_length * self.config.mask_time_min_masks + 1:
            return feature
        if self.config.num_mel_bins < self.config.mask_feature_length * self.config.mask_feature_min_masks + 1: 
            return feature
        
        if self.config.mask_time_prob > 0:
            start_indices = mask_start_indices(input_length, self.config.mask_time_length, self.config.mask_time_min_masks, self.config.mask_time_prob) 
            for start_idx in start_indices:
                feature[:, start_idx: start_idx + self.config.mask_time_length] = 0.0
        if self.config.mask_feature_prob > 0:
            start_indices = mask_start_indices(self.config.num_mel_bins, self.config.mask_feature_length, self.config.mask_feature_min_masks, self.config.mask_feature_prob) 
            for start_idx in start_indices:
                feature[start_idx: start_idx + self.config.mask_feature_length, :] = 0.0

        return feature

@dataclass
class OmniProcessorOutput(ModelOutput):  
    input_ids: Optional[Union[List, torch.Tensor]] = None
    labels: Optional[Union[List, torch.Tensor]] = None
    attention_mask: Optional[Union[List, torch.Tensor]] = None
    position_ids: Optional[Union[List, torch.Tensor]] = None
    seqlens: Optional[Union[List, torch.Tensor]] = None  # 需要配合Omni Modeling使用
    # audio fields
    audios: Optional[Union[List, torch.Tensor]] = None
    encoder_length: Optional[Union[List, torch.Tensor]] = None
    bridge_length: Optional[Union[List, torch.Tensor]] = None
    audiotext_ids: Optional[Union[List, torch.Tensor]] = None
    # image fields
    images: Optional[Union[List, torch.Tensor]] = None
    patch_nums: Optional[Union[List, torch.Tensor]] = None
    images_size: Optional[Union[List, torch.Tensor]] = None
    crop_size: Optional[Union[List, torch.Tensor]] = None
    images_grid: Optional[Union[List, torch.Tensor]] = None
    # video fields
    videos: Optional[Union[List, torch.Tensor]] = None
    videos_patch_nums: Optional[Union[List, torch.Tensor]] = None
    videos_size: Optional[Union[List, torch.Tensor]] = None
    videos_crop_size: Optional[Union[List, torch.Tensor]] = None
    videos_grid: Optional[Union[List, torch.Tensor]] = None
    # processor fields
    raw_text: Optional[str] = None
    index: Optional[int] = None

    def concatenate(self, other):  # 仅限list使用
        def concat_one(a, b):
            if a is None and b is None:
                return None
            elif a is None and b is not None:
                return b 
            elif a is not None and b is None: 
                return a 
            else: 
                return a + b
        return OmniProcessorOutput(
            input_ids=concat_one(self.input_ids, other.input_ids),
            labels=concat_one(self.labels, other.labels),
            audios=concat_one(self.audios, other.audios),
            encoder_length=concat_one(self.encoder_length, other.encoder_length),
            bridge_length=concat_one(self.bridge_length, other.bridge_length), 
            audiotext_ids=concat_one(self.audiotext_ids, other.audiotext_ids),
            images=concat_one(self.images, other.images),
            images_grid=concat_one(self.images_grid, other.images_grid),
            patch_nums=concat_one(self.patch_nums, other.patch_nums),

            videos=concat_one(self.videos, other.videos),
            videos_grid=concat_one(self.videos_grid, other.videos_grid),
            videos_patch_nums=concat_one(self.videos_patch_nums, other.videos_patch_nums),

            position_ids=concat_one(self.position_ids, other.position_ids),
            seqlens=concat_one(self.seqlens, other.seqlens),
            images_size=concat_one(self.images_size, other.images_size),
            videos_size=concat_one(self.videos_size, other.videos_size),
            index = self.index # concat保持index不变
        )

class OmniMMProcessor(object):
    def __init__(self,
                tokenizer: transformers.PreTrainedTokenizer,
                config,
                training,
                relative_path=None,
                default_client='',
                parallel=None,
                **kwargs, 
    ):
        self.tokenizer = tokenizer
        self.config = config
        self.audio_processor = OmniAudioProcessor(config.audio_config)
        # self.audio_processor = None
        self.visual_processor = None
        if hasattr(config, "visual_config"):
            self.visual_processor = OmniImageProcessor(config.visual_config)
        self.video_processor = None
        if hasattr(config, "video_config"):
            self.video_processor = OmniVideoProcessor(config.video_config)
        self.training = training
        self.relative_path = relative_path
        self.default_client = default_client
        self.s3_client = None  # lazy初始化 防止进程间传递S3Client(host=default_client)
        self.parallel = parallel
        # audio tag
        self.audio_start_tag = None
        self.audio_end_tag = None
        self.audio_pad_tag = None
        self.audio_delim_tag = None
        self.audiotext_start_tag = None
        self.audiotext_end_tag = None
        self.audiotext_pad_tag = None
        self.audiogen_start_tag = None
        self.audiogen_end_tag = None
        if hasattr(self.config, "audio_config"):
            self.audio_start_tag = self.tokenizer.convert_ids_to_tokens(self.config.audio_config.audio_start_token_id)
            self.audio_end_tag = self.tokenizer.convert_ids_to_tokens(self.config.audio_config.audio_end_token_id)
            self.audio_pad_tag = self.tokenizer.convert_ids_to_tokens(self.config.audio_config.audio_pad_token_id)
            self.audio_delim_tag = self.tokenizer.convert_ids_to_tokens(self.config.audio_config.audio_delim_token_id)
            self.audiotext_start_tag = self.tokenizer.convert_ids_to_tokens(self.config.audio_config.audiotext_start_token_id)
            self.audiotext_end_tag = self.tokenizer.convert_ids_to_tokens(self.config.audio_config.audiotext_end_token_id)
            self.audiotext_pad_tag = self.tokenizer.convert_ids_to_tokens(self.config.audio_config.audiotext_pad_token_id)
            self.audiogen_start_tag = self.tokenizer.convert_ids_to_tokens(self.config.audio_config.audiogen_start_token_id)
            self.audiogen_end_tag = self.tokenizer.convert_ids_to_tokens(self.config.audio_config.audiogen_end_token_id)
            # self.audio_text_delay = self.config.audio_config.audio_text_delay
        # image tag
        self.image_start_tag = None
        self.image_end_tag = None
        self.image_pad_tag = None
        self.video_start_tag = None
        self.video_end_tag = None
        # videoframe tag只是为了兼容图片帧作为输入的情况，没有token id，在抽取视频帧的时候，会将这个替换成image tag的start、end
        self.videoframe_start_tag = '<videoframe_start_omni>'
        self.videoframe_end_tag = '<videoframe_end_omni>'
        if hasattr(self.config, "visual_config") and self.config.visual_config.enable:
            # special token for start_tag
            self.image_start_tag = self.tokenizer.convert_ids_to_tokens(self.config.visual_config.image_start_token_id)
            # special token for end_tag
            self.image_end_tag = self.tokenizer.convert_ids_to_tokens(self.config.visual_config.image_end_token_id)
            # special token for pad_tag
            self.image_pad_tag = self.tokenizer.convert_ids_to_tokens(self.config.visual_config.image_pad_token_id)
            self.image_line_tag = self.tokenizer.convert_ids_to_tokens(self.config.visual_config.image_line_token_id)
            self.image_delimiter_tag = self.tokenizer.convert_ids_to_tokens(self.config.visual_config.image_delimiter_token_id) 
        if hasattr(self.config, "video_config") and  self.config.video_config.enable:
            self.video_start_tag = self.tokenizer.convert_ids_to_tokens(self.config.video_config.video_start_token_id)
            self.video_end_tag = self.tokenizer.convert_ids_to_tokens(self.config.video_config.video_end_token_id)
            # self.image_start_tag = self.tokenizer.convert_ids_to_tokens(self.config.video_config.image_start_token_id)
            # self.image_end_tag = self.tokenizer.convert_ids_to_tokens(self.config.video_config.image_end_token_id)
            # self.image_pad_tag = self.tokenizer.convert_ids_to_tokens(self.config.video_config.image_pad_token_id)
            self.video_place_tag = self.tokenizer.convert_ids_to_tokens(self.config.video_config.video_place_token_id)
            
            self.frame_pattern = getattr(self.config.video_config, 'frame_pattern', '<frame>')

    def _get_audio(self, audio_info):
        # if self.s3_client is None:
        #     self.s3_client = S3Client(host=self.default_client)
        try:
            audio_info = ujson.loads(audio_info)
            if 'path' in audio_info.keys():
                audio_uri, metadata, waveform_tensors = None, None, None
                # audio_info['path'] = audio_info['path'].lstrip('/') 
                # audio_uri = os.path.join(self.relative_path, audio_info['path'])
                audio_uri = audio_info['path']
                if not os.path.exists(audio_uri):  # 本地找不到到数据
                    print("WARNNING!!!没找到本地音频数据")
                    audio_uri = self.s3_client('llm-mm-audio', audio_uri)
                    metadata, waveform_tensors = self.s3_client.read_audio_bytes(audio_uri)

                waveforms = self.audio_processor.load_audio_waveform(audio_uri, metadata, waveform_tensors, True)
                waveforms = self.audio_processor.split_with_overlap(waveforms)  # 分割逻辑

                ret = OmniProcessorOutput()  # 默认初始化 audios字段为None
                for i, waveform in enumerate(waveforms):
                    audio, input_length = self.audio_processor.extract_fbank_features(waveform)
                    audio = self.audio_processor.data_augment(audio, input_length, self.training)
                    encoder_length, bridge_length = self.audio_processor.inference_output_length(self.config.audio_config, input_length)
                    if bridge_length <= 0:  # 过滤极端短数据 1. 如果len(waveforms)==1 ret=None; 2. len(waveforms)>1 则说明最后一段太短被抛弃
                        continue
                    current_ret = OmniProcessorOutput(
                        audios=[audio], 
                        encoder_length=[encoder_length], 
                        bridge_length=[bridge_length],
                        )
                    if ret.audios is None:
                        ret = current_ret
                    else:
                        ret = ret.concatenate(current_ret)  # 拼接多个切片
                return ret
            else:
                raise ValueError("can not find path in audio_info") 
        except Exception as e:
            print("**** get audio error: {}......, info: {} *****".format(str(e)[:256], str(audio_info)))
        return OmniProcessorOutput()

    def _get_pure_audio_curriculum_loss(self, audio_info, audio_label, min_seconds=3.0, max_seconds=20.0):
        # mask开头至少x秒+随机y秒的音频，减轻 副语言信息 对LLM的负向
        # reference https://arxiv.org/pdf/2412.17048 Why Do Speech Language Models Fail to Generate Semantically Coherent Outputs? A Modality Evolving Perspective
        min_tokens = min_seconds * self.config.audio_config.sampling_rate // self.config.audio_config.hop_length + 1
        _, min_tokens = self.audio_processor.inference_output_length(self.config.audio_config, min_tokens)
        max_tokens = max_seconds * self.config.audio_config.sampling_rate // self.config.audio_config.hop_length
        _, max_tokens = self.audio_processor.inference_output_length(self.config.audio_config, max_tokens)
        min_tokens, max_tokens = int(min_tokens), int(max_tokens) 
        try:
            audio_info = ujson.loads(audio_info)
            if not audio_info.get('curriculum', False):  # 数据不开启
                # print("NOT apply curriculum loss:\n{}".format(str(audio_info)))
                return audio_label
            if min_tokens >= len(audio_label):
                return audio_label  # 太短不开启（理论上不存在这类数据）
            masked_tokens = min(random.randint(min_tokens, max_tokens), len(audio_label) // 2)
            masked_tokens = max(min_tokens, masked_tokens)
            for mi in range(masked_tokens):
                if mi > 0:
                    audio_label[mi] = -100
            # print("DO apply curriculum loss:\n{}\nmin={},max={},masked={}\n{}".format(str(audio_info), min_tokens, max_tokens, masked_tokens, audio_label), flush=True)
        except Exception as e:
            print("*** _get_pure_audio_curriculum_loss error:={}".format(str(e)))
            pass
        return audio_label

    def _get_image(self, image_info):
        
        # 如果有类似<fix_res_1024>的标志符号的话,提取其中的1024数字,并将其替换为''
        # TODO(pengyuqi): 这里上游直接传入的是 image_data，没有字符串，所以这里提取标志的方法需要在上游做
        fix_res_number = -1
        # fix_res_pattern = r"<fixres_(\d+)>"
        # match = re.search(fix_res_pattern, image_info)
        # fix_res_number = int(match.group(1)) if match else -1
        # image_info = re.sub(fix_res_pattern, "", image_info)

        try:
            image_feat, org_size, image_list = self.visual_processor.image_transform(image_info, fix_res=fix_res_number)
            merge_length = self.visual_processor.merge_size**2
            patch_nums = np.array(image_list).prod() // merge_length
            
            if org_size[0] * org_size[1] > 16**2:  # 极端小的图过滤
                return OmniProcessorOutput(
                        images=[image_feat],
                        patch_nums=[patch_nums],
                        crop_size=[image_list],
                        images_size= [org_size],
                        images_grid=[image_list]
                        )
            else:
                print("**** image too small: {}, info: {} *****".format(str(org_size), str(image_info)))
                return OmniProcessorOutput()
           
        except Exception as e:
            print("**** get image error: {}......, info: {} *****".format(str(e)[:256], str(image_info)))
        return OmniProcessorOutput()
    
    def _get_video_frame(self, video_frame_infos):
        try:
            # print("in _get_video_frame with: ", video_frame_infos)
            if isinstance(video_frame_infos, str):
                pattern = r'\{.*?\}'
                matches = re.findall(pattern, video_frame_infos)
                if len(matches) == 0:
                    # print("没有匹配, raw video_frame_infos:", video_frame_infos)
                    # raise
                    return OmniProcessorOutput()
                
                # image_list 
                video_sampled_frames = []
                for match in matches:
                    video_frame_info = ujson.loads(match)
                    
                    if 'local' in video_frame_info.keys():
                        video_sampled_frames.append(video_frame_info['local'])
                    elif 'path' in video_frame_info.keys() and os.path.exists(video_frame_info['path']):
                        video_sampled_frames.append(video_frame_info['path'])
                    else:
                        raise ValueError("can not find any path in video_info")
            elif isinstance(video_frame_infos, list):
                video_sampled_frames = video_frame_infos
            else:
                raise
            video_feat, org_size, video_list = self.video_processor.extracted_images_transform(video_sampled_frames)
            merge_length = self.video_processor.merge_size**2
            patch_nums = np.array(video_list).prod() // merge_length
            
            if org_size[0] * org_size[1] > 16**2:  # 极端小的图过滤
                return  OmniProcessorOutput(
                            videos=[video_feat],
                            videos_patch_nums=[patch_nums],
                            videos_crop_size=[video_list],
                            videos_size= [org_size],
                            videos_grid=[video_list]
                        )
            else:
                print("**** video too small: {}, info: {} *****".format(str(org_size), str(video_frame_infos)))
                return OmniProcessorOutput()
           
        except Exception as e:
            print("**** get video error: {}, info: {} *****".format(str(e)[:256], str(video_frame_infos)))
            raise
        return OmniProcessorOutput()

    def _get_video_time(self, video_frame_infos):
        try:
            if isinstance(video_frame_infos, str):
                pattern = r'at second\[.*?\]'
                matches = re.findall(pattern, video_frame_infos)
                if len(matches) == 0:
                    return []
                
                # image_list 
                video_time = []
                for match in matches:
                    video_time.append(match)
                return video_time
        except Exception as e:
            print("**** get video time error: {}, info: {} *****".format(str(e)[:256], str(video_frame_infos)))
            raise
        return []

    # 读取视频
    def _get_vision_obj_byte(self, source, path):
        vision_obj_byte = None
        if source == "local":
            if os.path.exists(path):
                vision_obj_byte = open(path, "rb").read()
            else:
                vision_obj_byte = None
        if source == "base64":
            vision_obj_byte = base64.b64decode(path)
        if source == "url":
            vision_obj_byte = requests.get(url=path).content
        return vision_obj_byte
    
    # 将视频切分为帧，保存至子目录中，本质上是在将视频抽帧-》处理成图像模态的数据组织形式
    def _split_video_to_frames(self, video_info, max_frame_number=-1, decode_way="1fps"):
        if decode_way=='1fps':
            frame_suffix = f'_frames'
        elif decode_way=='key':
            frame_suffix = f'_keyframes'
        else:
            raise ValueError('unvalid decode way!!!')
        
        server = "local"
        if 'local' in video_info.keys():
            # 本地路径
            video_path = video_info['local']
            # 帧保存本地路径
            frame_path = video_path[:video_path.rfind('.')] + frame_suffix
            mm_obj_byte = self._get_vision_obj_byte('local', video_path)
        elif 'base64' in video_info.keys():
            md5 = hashlib.md5(video_info['base64'].encode('utf-8')).hexdigest()
            if self.relative_path is not None: 
                video_path = os.path.join(self.relative_path, md5)
            else:
                video_path = os.path.join(os.getcwd(), md5)
            frame_path = video_path + frame_suffix
            mm_obj_byte = self._get_vision_obj_byte('base64', video_info['base64'])
        elif 'url' in video_info.keys():
            md5 = hashlib.md5(video_info['url'].encode('utf-8')).hexdigest()
            if self.relative_path is not None: 
                video_path = os.path.join(self.relative_path, md5)
            else:
                video_path = os.path.join(os.getcwd(), md5)
            frame_path = video_path + frame_suffix
            mm_obj_byte = self._get_vision_obj_byte('url', video_info['url'])
        else:
            raise ValueError('unvalid video server !!!')
            return ""
        
        if mm_obj_byte is None: # 未读取到视频文件
            return ""
        if not os.path.exists(frame_path) or len(os.listdir(frame_path))==0:
            # 保存帧
            os.makedirs(frame_path, exist_ok=True)
            try:
                result = read_video(io.BytesIO(mm_obj_byte), max_frame_number=-1, decode_way=decode_way)
                if result is None:
                    # Handle None return explicitly
                    return ""
                
                frames, frame_times = result
                for frame_idx, frame in enumerate(frames):
                    output_filename = os.path.join(frame_path, f"{frame_times[frame_idx]}.jpg")
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(output_filename, frame)
            except Exception as e:
                # Add error logging for debugging
                print(f"Error processing video: {str(e)}")
                return ""  # Fail gracefully

        frame_paths = os.listdir(frame_path)
        
        # 选取帧
        frame_times = [int(filename.split('/')[-1].replace('.jpg', '')) for filename in frame_paths if filename.endswith('.jpg')] # 文件名对应秒数
        frame_times.sort() #从小到大排序
        frame_number = len(frame_times)
        
        # 确保采样帧数能被temporal_patch_size整除
        sample_num = min(frame_number, max_frame_number)
        sample_num = floor_by_factor(sample_num, self.video_processor.temporal_patch_size)
        # 防止sample_num变成0
        if sample_num == 0:
            sample_num = min(frame_number, max_frame_number)

        indices = np.linspace(0, frame_number - 1, sample_num, dtype=int)
        # 拼接模式
        replace_str = ""
        for frame_idx, idx in enumerate(indices):
            frame_time = frame_times[idx]  # frame_time表示帧对应的时间 单位为s 同时也是存储的文件名
            frame_dict = {"local": os.path.join(frame_path, f'{frame_time}.jpg')}
            frame_str = self.frame_pattern.format(frame_idx) if '{}' in self.frame_pattern else self.frame_pattern  # {}对应的是第几张图片
            frame_str = frame_str.replace('<TIMEIDX>', "at second["+str(frame_time)+"]") 
            frame_str = frame_str.replace('<TIMESTAMP>', time.strftime("%H:%M:%S", time.gmtime(frame_time)))  # TIMESTAMP对应的是时间戳
            frame_str = frame_str.replace('<frame>', f'{self.image_start_tag}{ujson.dumps(frame_dict)}{self.image_end_tag}')
            replace_str += frame_str
        return replace_str
    
    def sample_frame(self,frames_str,max_frame = 32):
        def uniform_sample(lst, num_samples):
            frame_number = len(lst)
            max_frame_number = num_samples
            sample_num = min(frame_number, max_frame_number)
            sample_num = floor_by_factor(sample_num, self.video_processor.temporal_patch_size)
            # 防止sample_num变成0
            if sample_num == 0:
                sample_num = min(frame_number, max_frame_number)

            interval = len(lst) / sample_num
            samples = [lst[int(i * interval)] for i in range(sample_num)]
            return samples
        p = rf'({self.image_start_tag}.*?{self.image_end_tag})'
        frames_str_split = re.split(p,frames_str)
        frame_idxs = [idx for idx in range(len(frames_str_split)) if self.image_start_tag in frames_str_split[idx]]
        sample_frame_idxs = set(uniform_sample(frame_idxs, max_frame))
        ans = ''.join([item for idx,item in enumerate(frames_str_split) if idx in sample_frame_idxs or self.image_start_tag not in frames_str_split[idx]])

        start_tag = re.escape(self.image_start_tag)
        end_tag = re.escape(self.image_end_tag)
        
        # 构建正则表达式模式
        pattern = rf'{start_tag}(.*?){end_tag}'
        
        # 使用findall获取所有匹配的文件路径
        paths = re.findall(pattern, ans, re.DOTALL)
        return paths # 返回的是图片的路径的list，不包含特殊起始和终止token

    def _get_video_frame_str(self, video_info):
        try:
            if self.videoframe_start_tag in video_info:#如果是以视频帧的形式表示一个视频，则替换成image tag
                frames_str = video_info
                frames_str = frames_str.replace(self.videoframe_start_tag,self.image_start_tag).replace(self.videoframe_end_tag,self.image_end_tag)
                # print("抽帧之前的frames_str: ", frames_str)
                # raise
                return self.sample_frame(frames_str, max_frame = self.config.video_config.max_frame_num)
            video_info = ujson.loads(video_info)
            # 完成视频抽帧，并组织成包含多帧图像路径的字符串，最大帧数量max_frame_number
            frames_str = self._split_video_to_frames(video_info, max_frame_number=self.config.video_config.max_frame_num, decode_way=self.config.video_config.decode_way)
            return frames_str
        except Exception as e:
            print("**** get video error: {}, info: {} *****".format(str(e), str(video_info)))
            raise
        return ""
    
    def _get_audiotext(self, audiotext_info):
        try:
            ret = self._get_audio(audiotext_info)  # 重复取结果 cached result
            audiotext_info = ujson.loads(audiotext_info)
            if ('audiotext' not in audiotext_info) or (not audiotext_info['audiotext'].strip()):
                raise ValueError("can not find audiotext in audiotext_info")
            return ret, audiotext_info['audiotext'], audiotext_info
        except Exception as e:
            print("**** get audiotext error: {}, info: {} *****".format(str(e), str(audiotext_info)))
        return OmniProcessorOutput(), '', '', {}

    def _replace_image(self, image_text):

        image_info = re.sub(re.compile(self.image_start_tag + "|" + self.image_end_tag), '', image_text)
        ret = self._get_image(image_info)  # 重复取结果 cached result
        if ret.patch_nums is None:
            return ''
        return ret, self.image_start_tag + self.image_pad_tag * ret.patch_nums[0] + self.image_end_tag
    
    def _replace_video_frame(self, video_frame_text):
        if isinstance(video_frame_text, str):
            video_frame_info = re.sub(re.compile(self.image_start_tag + "|" + self.image_end_tag), '', video_frame_text) # 删除字符串中self.image_start_tag和self.image_end_tag包裹的字段
        elif isinstance(video_frame_text, list):
            video_frame_info = video_frame_text
        else:
            raise
        ret = self._get_video_frame(video_frame_info)  # 重复取结果 cached result
        # 获取时间戳list
        if "second" in video_frame_info:
            time_info = self._get_video_time(video_frame_info)
        else:
            time_info = []
        if ret.videos_patch_nums is None:
            return ret, ''
        '''
        split_video:
            if true: <img_start><video_place_tag>*ret.videos_grid[0][1] * ret.videos_grid[0][2]<img_end><img_start><video_place_tag>*ret.videos_grid[0][1] * ret.videos_grid[0][2]<img_end>...
            if false: <img_start><video_place_tag>*ret.videos_patch_nums[0]<img_end>
        '''
        if self.video_processor.split_video:
            if time_info is not None and len(time_info)>0:
                video_frame_str = "".join([self.image_start_tag + self.video_place_tag * (ret.videos_patch_nums[0] // ret.videos_grid[0][0]) + self.image_end_tag + time_info[idx * self.video_processor.temporal_patch_size] for idx in range(ret.videos_grid[0][0])])
            else:
                video_frame_str = "".join([self.image_start_tag + self.video_place_tag * (ret.videos_patch_nums[0] // ret.videos_grid[0][0]) + self.image_end_tag for _ in range(ret.videos_grid[0][0])])
        else:
            video_frame_str = self.image_start_tag + self.video_place_tag * ret.videos_patch_nums[0] + self.image_end_tag
        return ret, video_frame_str
        
    # 模态分配
    def split_multimodal_chunk(self, text_list, mm_label_list, trainable_list, mtype='audio'):
        # 抽取text中的json格式音频/图像信息，读取并转化为特征，同时估计encoder token数，填入对应数量的pad token
        if (self.audio_start_tag != None) and (mtype == 'audio'):
            match_regex = re.compile(self.audio_start_tag + '.*?' + self.audio_end_tag,re.S)
            drop_regex = re.compile(self.audio_start_tag + "|" + self.audio_end_tag,re.S)
        elif (self.image_start_tag != None) and (mtype == 'image'):
            match_regex = re.compile(self.image_start_tag + '.*?' + self.image_end_tag,re.S)
            drop_regex = re.compile(self.image_start_tag + "|" + self.image_end_tag,re.S)
        elif (self.audiotext_start_tag != None) and (mtype == 'audiotext'):
            match_regex = re.compile(self.audiotext_start_tag + '.*?' + self.audiotext_end_tag,re.S)
            drop_regex = re.compile(self.audiotext_start_tag + "|" + self.audiotext_end_tag,re.S)
        elif (self.audiogen_start_tag != None) and (mtype == 'audiogen'):
            match_regex = re.compile(self.audiogen_start_tag + '.*?' + self.audiogen_end_tag,re.S)
            drop_regex = re.compile(self.audiogen_start_tag + "|" + self.audiogen_end_tag,re.S)
        elif (self.video_start_tag != None) and (mtype == 'video'):
            match_regex = re.compile(self.video_start_tag + '.*?' + self.video_end_tag,re.S)
            drop_regex = re.compile(self.video_start_tag + "|" + self.video_end_tag,re.S)
        # elif (self.videoframe_start_tag != None) and (mtype == 'frame'):
        #     match_regex = re.compile(self.videoframe_start_tag + '.*?' + self.videoframe_end_tag,re.S)
        #     drop_regex = re.compile(self.videoframe_start_tag + "|" + self.videoframe_end_tag,re.S)
        else:
            raise ValueError(f"mtype not supportted!{mtype=}")
        new_text_list = []
        new_mm_label_list = []
        new_trainable_flag_list = []
        for text,mm_label,trainable in zip(text_list,mm_label_list,trainable_list):
            for t,m in zip(*split_text(text, match_regex)):
                new_trainable_flag_list.append(trainable)
                if m:
                    new_text_list.append(re.sub(drop_regex, '', t))
                    new_mm_label_list.append(mtype)
                else:
                    new_text_list.append(t)
                    new_mm_label_list.append(mm_label)
        return new_text_list, new_mm_label_list, new_trainable_flag_list
    
    def process_multimodal_chunk(self, text, mm_label, trainable): # 通过这个函数可以控制该模态是否参与训练
        ret = OmniProcessorOutput()
        if mm_label == 'audio':
            ret = self._get_audio(text)
            if ret.bridge_length is not None:    
                ret.input_ids = self.tokenizer.encode(self.audio_start_tag,add_special_tokens=False) + self.tokenizer.encode(self.audio_pad_tag,add_special_tokens=False) * sum(ret.bridge_length) + self.tokenizer.encode(self.audio_end_tag,add_special_tokens=False)
                ret.labels = [a if trainable else -100 for a in ret.input_ids]
                ret.audiotext_ids = self.tokenizer.encode(self.audiotext_pad_tag,add_special_tokens=False) * (sum(ret.bridge_length))
            else:
                raise ValueError(f"Get audio data Failed at Process audio chunk {text}")
        elif mm_label == 'audiogen':
            ret = self._get_audio(text)
            audio_info = ujson.loads(text)
            audiotext = audio_info['audiotext'] if 'audiotext' in audio_info.keys() else None
            
            if ret.bridge_length is not None:   
                if audiotext is not None:
                    audiotext_input_ids = self.tokenizer.encode(audiotext, add_special_tokens=False)
                    if len(audiotext_input_ids) > sum(ret.bridge_length) - 1 or len(audiotext_input_ids) > self.tokenizer.model_max_length - 1:  # 过滤长文本短音频或超长文本，这种属于不正常
                        raise ValueError(f"Audio text too long {len(audiotext_input_ids)} vs audio lengths {sum(ret.bridge_length)}, please check audio and text length！ 【{audiotext}】")
                    if 'delay' not in audio_info.keys():
                        if random.random() < 0.5:
                            delay_length = 0
                        else:
                            delay_length = random.randint(1, len(audiotext_input_ids))
                    else:
                        delay_length = min(len(audiotext_input_ids), audio_info['delay'])
                    ret.input_ids = self.tokenizer.encode(self.audiogen_start_tag,add_special_tokens=False) \
                        + self.tokenizer.encode(self.audiotext_pad_tag,add_special_tokens=False) * delay_length \
                        + self.tokenizer.encode(self.audiotext_start_tag,add_special_tokens=False) \
                        + self.tokenizer.encode(self.audio_pad_tag,add_special_tokens=False) * sum(ret.bridge_length) \
                        + self.tokenizer.encode(self.audiogen_end_tag,add_special_tokens=False)
                    ret.labels = [a if trainable else -100 for a in ret.input_ids]
                    ret.audiotext_ids = audiotext_input_ids + self.tokenizer.encode(self.audiotext_pad_tag,add_special_tokens=False) * (delay_length + 1 + sum(ret.bridge_length) - len(audiotext_input_ids))
                else:
                    ret.input_ids = self.tokenizer.encode(self.audiogen_start_tag,add_special_tokens=False) + self.tokenizer.encode(self.audio_pad_tag,add_special_tokens=False) * sum(ret.bridge_length) + self.tokenizer.encode(self.audiogen_end_tag,add_special_tokens=False)
                    ret.labels = [a if trainable else -100 for a in ret.input_ids]
                    ret.audiotext_ids = self.tokenizer.encode(self.audiotext_pad_tag,add_special_tokens=False) * (sum(ret.bridge_length))
            else:
                raise ValueError(f"Get audio data Failed at Process audio chunk {text}")
        elif mm_label == 'image':
            ret, input_str = self._replace_image(text)
            if input_str:
                ret.input_ids = self.tokenizer.encode(input_str, add_special_tokens=False)
                ret.labels = [a if trainable else -100 for a in ret.input_ids]
            else:
                raise ValueError("Get image data Failed at Process image chunk")
        # elif mm_label == 'video' or mm_label == 'frame': # video和frame都是视频模态，前者以独立的视频文件（例如mp4）存在，后者以抽帧后的图片形式存在
        elif mm_label == 'video': # frame也合并到video处理
            video_frames = self._get_video_frame_str(text)
            if isinstance(video_frames, str):
                frame_str = self.video_start_tag+video_frames+self.video_end_tag
            elif isinstance(video_frames, list):
                frame_str = video_frames
            ret, input_str = self._replace_video_frame(frame_str)
            # print("处理过后给模型encode的数据：", input_str) # <img_start><video_place>*N<img_end>
            if input_str:
                ret.input_ids = self.tokenizer.encode(input_str, add_special_tokens=False)
                ret.labels = [a if trainable else -100 for a in ret.input_ids]
            else:
                raise ValueError("Get video data Failed at Process video chunk")
        elif mm_label == 'audiotext':  # 这类数据音频和文本都训练
            raise ValueError("audiotext not supportted!")
            # ret, audiotext, full_info = self._get_audiotext(text)
            # if ret.bridge_length is not None:
            #     audiotext_input_ids = self.tokenizer.encode(audiotext, add_special_tokens=False)
            #     if len(audiotext_input_ids) > sum(ret.bridge_length) - 1 or len(audiotext_input_ids) > self.tokenizer.model_max_length - 1:  # 过滤长文本短音频或超长文本，这种属于不正常
            #         raise ValueError(f"Audio text too long or too short {len(audiotext_input_ids)}, please check audio and text length！ 【{audiotext}】")
            #     audiotext_pad_ids = self.tokenizer.encode(self.audiotext_pad_tag,add_special_tokens=False) * (sum(ret.bridge_length) - len(audiotext_input_ids))
            #     ret.audiotext_ids = audiotext_input_ids + audiotext_pad_ids
            #     ret.input_ids = self.tokenizer.encode(self.audiotext_start_tag, add_special_tokens=False) + self.tokenizer.encode(self.audio_pad_tag, add_special_tokens=False) * sum(ret.bridge_length) + self.tokenizer.encode(self.audiotext_end_tag, add_special_tokens=False)
            #     ret.labels = [a if trainable else -100 for a in ret.input_ids]
            # else:
            #     raise ValueError("Get audio file Failed at Process audiotext chunk")
        elif mm_label == 'text':
            ret.input_ids = self.tokenizer.encode(text, add_special_tokens=False)
            if len(ret.input_ids) > self.tokenizer.model_max_length-1:  # 过滤长文本
                raise ValueError(f"Text too long, please check text length！ 【{text[:5]+'...'*6+text[-5:]}】")
            ret.labels = [a if trainable else -100 for a in ret.input_ids]
        else:
            raise ValueError(f"mm_label not supportted! must in ['audio', 'image', 'audiotext', 'text'] but get {mm_label}")
        return ret
    
    def process_one(self, text, index=0, raw_only=False):
        ret = OmniProcessorOutput(index=index)
        all_text_list = []
        all_mm_label_list = []
        all_trainable_flag_list = []
        # 处理预训练中的trainable标记 
        text_list, match_flag = split_text(text, re.compile("<trainable_start>.*?<trainable_end>",re.S))
        if len(text_list) == 1:
            text = re.sub(re.compile("<trainable_start>|<trainable_end>",re.S), '', text_list[0])
            all_text_list.append(text)
            all_mm_label_list.append('text')
            all_trainable_flag_list.append(True)
        else:
            for text, match in zip(text_list, match_flag):
                text = re.sub(re.compile("<trainable_start>|<trainable_end>",re.S), '', text)
                if text.strip() == '':
                    continue  # 把多余的空格干掉
                all_text_list.append(text)
                all_mm_label_list.append('text')
                all_trainable_flag_list.append(match)
        # 处理多模态信息
        for mtype in self.config.multimodal:  # 循环获取音频 图像结果 
            all_text_list, all_mm_label_list, all_trainable_flag_list = self.split_multimodal_chunk(all_text_list, all_mm_label_list, all_trainable_flag_list, mtype)
        if len(all_text_list) == 0:
            print(f"Process {text} chunk error: No valid Text data!!!!!")
            return OmniProcessorOutput(index=index)
        
        for text, mm_label, trainable in zip(all_text_list, all_mm_label_list, all_trainable_flag_list):
            try:
                mret = self.process_multimodal_chunk(text, mm_label, trainable)
                ret = ret.concatenate(mret)
            except ValueError as e:
                tt = text[:24].replace('\n','<LF>')
                print(f"Process {tt if mm_label == 'text' else text} {mm_label} chunk error: {str(e)}")
                return OmniProcessorOutput(index=index)

        if raw_only:
            ret.raw_text = self.tokenizer.decode(ret.input_ids, skip_special_tokens=False)
            return ret
        return ret

    def _convert_text_and_media_to_example(self, text, images=None, videos=None, audios=None):
        """
        将text和媒体文件路径转换为example格式
        text: 包含媒体占位符的文本，如 "<C_Q><img_start><img_end>question<C_A>"
        images: 图像文件路径列表
        videos: 视频文件路径列表  
        audios: 音频文件路径列表
        返回: 完整的对话字符串，媒体占位符被替换为实际路径
        """
        if not isinstance(text, list):
            text = [text]
        
        result_texts = []
        image_idx = 0  # 用于追踪当前使用的图像索引
        video_idx = 0  # 用于追踪当前使用的视频索引  
        audio_idx = 0  # 用于追踪当前使用的音频索引

        for single_text in text:
            # 检查图像占位符数量
            img_placeholders = single_text.count('<img_start><img_end>')
            
            # 检查视频占位符数量
            video_placeholders = single_text.count('<video_start><video_end>')
            
            # 检查音频占位符数量
            audio_placeholders = single_text.count('<audio_start><audio_end>')
            
            # 对于batch处理，检查总的媒体数量是否足够
            if images is not None and image_idx + img_placeholders > len(images):
                raise ValueError(f"Not enough images: need {image_idx + img_placeholders}, got {len(images)}")
            
            if videos is not None and video_idx + video_placeholders > len(videos):
                raise ValueError(f"Not enough videos: need {video_idx + video_placeholders}, got {len(videos)}")
                
            if audios is not None and audio_idx + audio_placeholders > len(audios):
                raise ValueError(f"Not enough audios: need {audio_idx + audio_placeholders}, got {len(audios)}")
            
            # 替换图像占位符
            if images is not None and img_placeholders > 0:
                for i in range(img_placeholders):
                    image_path = images[image_idx + i]
                    single_text = single_text.replace('<img_start><img_end>', f'<img_start>{image_path}<img_end>', 1)
                image_idx += img_placeholders
            
            # 替换视频占位符
            if videos is not None and video_placeholders > 0:
                for i in range(video_placeholders):
                    video_path = videos[video_idx + i]
                    single_text = single_text.replace('<video_start><video_end>', f'<video_start>{video_path}<video_end>', 1)
                video_idx += video_placeholders
            
            # 替换音频占位符
            if audios is not None and audio_placeholders > 0:
                for i in range(audio_placeholders):
                    audio_path = audios[audio_idx + i]
                    single_text = single_text.replace('<audio_start><audio_end>', f'<audio_start>{audio_path}<audio_end>', 1)
                audio_idx += audio_placeholders
            
            result_texts.append(single_text)
        
        return result_texts
    
    def _convert_text_and_media_to_example_v1(self, text, images=None, videos=None, audios=None):
        """
        将text和媒体文件路径转换为example格式
        text: 包含媒体占位符的文本，如 "<C_Q><img_start><img_end>question<C_A>"
        images: 图像文件路径列表
        videos: 视频文件路径列表  
        audios: 音频文件路径列表
        返回: 完整的对话字符串，媒体占位符被替换为实际路径
        """
        if not isinstance(text, list):
            text = [text]
        
        result_texts = []
        
        for single_text in text:
            # 检查图像占位符数量
            img_placeholders = single_text.count('<img_start><img_end>')
            if images is not None and len(images) != img_placeholders:
                raise ValueError(f"Image placeholder count ({img_placeholders}) doesn't match images count ({len(images)})")
            
            # 检查视频占位符数量
            video_placeholders = single_text.count('<video_start><video_end>')
            if videos is not None and len(videos) != video_placeholders:
                raise ValueError(f"Video placeholder count ({video_placeholders}) doesn't match videos count ({len(videos)})")
            
            # 检查音频占位符数量
            audio_placeholders = single_text.count('<audio_start><audio_end>')
            if audios is not None and len(audios) != audio_placeholders:
                raise ValueError(f"Audio placeholder count ({audio_placeholders}) doesn't match audios count ({len(audios)})")
            
            # 替换图像占位符
            if images is not None:
                for image_path in images:
                    single_text = single_text.replace('<img_start><img_end>', f'<img_start>{image_path}<img_end>', 1)
            
            # 替换视频占位符
            if videos is not None:
                for video_path in videos:
                    single_text = single_text.replace('<video_start><video_end>', f'<video_start>{video_path}<video_end>', 1)
            
            # 替换音频占位符
            if audios is not None:
                for audio_path in audios:
                    single_text = single_text.replace('<audio_start><audio_end>', f'<audio_start>{audio_path}<audio_end>', 1)
            
            result_texts.append(single_text)
        
        return result_texts

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        """
        应用聊天模板，将messages转换为raw_prompt格式，支持多模态和多轮对话
        messages格式: [
            {'content': [{'type': 'image', 'image': 'path'}, {'type': 'text', 'text': 'question'}], 'role': 'user'},
            {'content': [{'type': 'text', 'text': 'answer'}, {'type': 'image', 'image': 'path'}], 'role': 'assistant'},
            {'content': [{'type': 'video', 'video': 'path'}, {'type': 'text', 'text': 'follow-up'}], 'role': 'user'}
        ]
        返回: 完整的对话字符串，但媒体路径被替换为占位符
        所有模态都按照content的顺序添加，不区分text还是media
        """
        if not isinstance(messages, list) or len(messages) == 0:
            raise ValueError("messages must be a non-empty list")
        
        conversation_parts = []
        
        for message in messages:
            role = message.get('role')
            content = message.get('content', [])
            
            if not isinstance(content, list):
                raise ValueError("content must be a list")
            
            # 处理所有消息（user和assistant都支持多模态）
            message_parts = []
            
            for item in content:
                item_type = item.get('type')
                if item_type == 'text':
                    message_parts.append(item.get('text', ''))
                elif item_type == 'image':
                    # 只保留占位符，不包含实际路径
                    message_parts.append("<img_start><img_end>")
                elif item_type == 'video':
                    message_parts.append("<video_start><video_end>")
                elif item_type == 'audio':
                    message_parts.append("<audio_start><audio_end>")
            
            # 组合消息内容
            combined_content = ''.join(message_parts)
            
            if role == 'user':
                conversation_parts.append(f"<C_Q>{combined_content}")
            elif role == 'assistant':
                if combined_content:
                    conversation_parts.append(f"<C_A>{combined_content}")
        
        # 组合所有对话部分
        raw_prompt = ''.join(conversation_parts)

        if add_generation_prompt:
            raw_prompt = raw_prompt + "<C_A>"

        if tokenize:
            if not hasattr(self, 'tokenizer'):
                raise ValueError("tokenizer is required when tokenize=True")
            return self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        
        return raw_prompt

    @torch.no_grad()
    def __call__(self, example=None, text=None, images=None, videos=None, audios=None, parallel=128, return_tensors=None):  # 支持预训练数据string，sft数据message， 以及batch推理数据listofstring 3种形式
        # 如果传入了text和媒体文件，先转换为example格式
        if text is not None:
            example = self._convert_text_and_media_to_example(text, images, videos, audios)
        if isinstance(example, Dict):
            pass 
        elif isinstance(example, str):
            return self.process_one(example)
        elif isinstance(example, List):  # batch推理 异步多线程处理
            with cf.ThreadPoolExecutor(min(parallel, len(example))) as executor:
                future_list = [executor.submit(self.process_one, di, idx) for idx, di in enumerate(example)]
                batch_data = [key.result() for key in cf.as_completed(future_list)]
            valid_num = sum([1 if x.input_ids is not None else 0 for x in batch_data])
            assert(valid_num == len(batch_data)), batch_data # 推理数据严格要求数量对齐
            batch_data = sorted(batch_data, key=lambda x: x.index)  # 保证顺序不变
            
            ret = OmniProcessorOutput()
            for i in range(len(batch_data)):
                ret = ret.concatenate(batch_data[i])
            self.tokenizer.padding_side = "left"
            max_len = min(max([len(x.input_ids) for x in batch_data]),self.tokenizer.model_max_length)
            padding_result = self.tokenizer.pad({"input_ids": [r.input_ids for r in batch_data]}, return_tensors='pt')
            ret.input_ids = padding_result["input_ids"]
            ret.attention_mask = padding_result["attention_mask"]  # batch推理不pack 不需要seqlens
            ret.labels = torch.LongTensor([(max(0,max_len - len(x.labels))*[-100] + x.labels)[:max_len] for x in batch_data])
            
            if ret.audios is not None:
                max_audios_len = max([x.shape[-1] for x in ret.audios])
                ret.audios = default_collate([np.pad(x, ((0,0),(0,max_audios_len - x.shape[-1])), 'constant', constant_values=0) for x in ret.audios])
            
                ret.encoder_length = default_collate(ret.encoder_length)
                ret.bridge_length = default_collate(ret.bridge_length)
            
            if ret.audiotext_ids is not None:
                ret.audiotext_ids = default_collate(ret.audiotext_ids)

            if ret.images is not None:
                ret.images = [torch.from_numpy(np.asarray(image, dtype=np.float32))  for image in ret.images]
                ret.patch_nums = default_collate(ret.patch_nums)
                
            if ret.videos is not None:
                ret.videos = [torch.from_numpy(np.asarray(image, dtype=np.float32))  for image in ret.videos]
                ret.videos_patch_nums = default_collate(ret.videos_patch_nums)

            return ret

        else:
            raise ValueError("example format supported yet")

    @torch.no_grad()
    def pack_batch_pretrain(self, raw_batch, max_sequence_length=None, parallel=8):
        if self.parallel is not None:
            parallel = self.parallel
        if max_sequence_length is None:
            max_sequence_length = self.tokenizer.model_max_length
        # 将N条数据pack为M条 max_sequence_length长度的数据, 每条数据包含所属的多模态输入
        assert isinstance(raw_batch, List)
        has_eosloss_list = [True] * len(raw_batch)  # 表示数据是否有eos/audioeos loss 默认为true
        start_ts = time.time()
        if parallel > 1:
            with cf.ThreadPoolExecutor(max_workers=parallel) as executor:
            # with cf.ProcessPoolExecutor(max_workers=parallel) as executor:
                future_list = []
                for idx, json_text in enumerate(raw_batch):
                    try:  # 读取json
                        json_obj = ujson.loads(json_text.strip())
                    except:
                        try: 
                            json_obj = ast.literal_eval(json_text.strip())
                        except:
                            print("parse json obj faild: {}....".format(json_text[:300]))
                            continue
                    if isinstance(json_obj, list):
                        content = json_obj[1]
                    elif "input" in json_obj.keys():
                        content = (json_obj["title"] if "title" in json_obj.keys() else "") + json_obj["input"]
                    else:
                        content = (json_obj["title"] if "title" in json_obj.keys() else "") + json_obj["content"]
                    if not isinstance(json_obj, list):  # 获取是否有eos loss
                        has_eosloss_list[idx] = json_obj.get('eosloss', True)
                    future_list.append(executor.submit(self.process_one, content, idx))
                # 获取结果 乱序
                batch_data = [key.result() for key in cf.as_completed(future_list)]
        else:  # debug only
            batch_data = [self.process_one(ujson.loads(json_text.strip())['content'], idx) for idx, json_text in enumerate(raw_batch)]

        if (time.time() - start_ts) / (len(batch_data) + 1e-3) > 1.0:
            print('[WARNING] processing each data cost more than 1.0s')

        # packing 文本部分的输入，不做任何截断
        current_length, packed_output, output = 0, OmniProcessorOutput(position_ids=[], seqlens=[]), []
        empty_data = OmniProcessorOutput(input_ids=[], labels=[], index=0)
        for idx, bd in enumerate(batch_data + [empty_data]):  # 加空数据方便append最后一个数据到output，防止遗漏
            if bd.input_ids is None and idx < len(batch_data):
                continue  # 数据没取到 并且不是最后一个
            if (len(bd.input_ids) <= 0 or len(bd.input_ids) + 1 > max_sequence_length) and idx < len(batch_data):
                continue  # 太长的直接不要 并且不是最后一个
            if current_length + len(bd.input_ids) + 1 > max_sequence_length or idx == len(batch_data):
                pad_nums = max_sequence_length - current_length  # right padding
                packed_output.input_ids += [self.tokenizer.pad_token_id] * pad_nums
                packed_output.labels += [-100] * pad_nums
                packed_output.attention_mask = [1] * current_length + [0] * pad_nums
                packed_output.position_ids += [0] * pad_nums
                packed_output.seqlens += [0] * (max_sequence_length - len(packed_output.seqlens))
                output.append(packed_output)
                packed_output = OmniProcessorOutput(position_ids=[], seqlens=[])  # reset empty
            packed_output = packed_output.concatenate(bd)
            packed_output.input_ids.append(self.tokenizer.eos_token_id)  # </s>需要单独加
            if (len(packed_output.input_ids) > 0 and has_eosloss_list[bd.index]):  # 部分数据的文本结束不加eos label
                packed_output.labels.append(self.tokenizer.eos_token_id)
            else:
                packed_output.labels.append(-100)
            
            packed_output.position_ids.extend(list(range(len(bd.input_ids) + 1)))
            packed_output.seqlens.append(len(bd.input_ids) + 1)

            current_length = len(packed_output.input_ids)
        return output
    
    @torch.no_grad()
    def collect_batch_pretrain(self, batch_data):
        ret = OmniProcessorOutput()
        for i in range(len(batch_data)):
            ret = ret.concatenate(batch_data[i])
        ret.input_ids = default_collate([np.asarray(x.input_ids, dtype=np.int64) for x in batch_data]).cuda(non_blocking=True)
        ret.labels = default_collate([np.asarray(x.labels, dtype=np.int64) for x in batch_data]).cuda(non_blocking=True)
        ret.attention_mask = default_collate([np.asarray(x.attention_mask, dtype=np.float32) for x in batch_data]).cuda(non_blocking=True)
        ret.position_ids = default_collate([np.asarray(x.position_ids, dtype=np.int64) for x in batch_data]).cuda(non_blocking=True)
        ret.seqlens = default_collate([np.asarray(x.seqlens, dtype=np.int64) for x in batch_data]).cuda(non_blocking=True)

        ret.raw_text = None
        if ret.audios is not None:
            max_audios_len = max([x.shape[-1] for x in ret.audios])
            ret.audios = default_collate([np.pad(x, ((0,0),(0,max_audios_len - x.shape[-1])), 'constant', constant_values=0) for x in ret.audios]).cuda(non_blocking=True)
            ret.encoder_length = default_collate(np.asarray(ret.encoder_length, dtype=np.int32)).cuda(non_blocking=True)
            ret.bridge_length = default_collate(np.asarray(ret.bridge_length, dtype=np.int32)).cuda(non_blocking=True)
        if ret.audiotext_ids is not None:
            ret.audiotext_ids = default_collate(ret.audiotext_ids).cuda(non_blocking=True)
        if ret.images is not None:
            ret.images = [torch.from_numpy(np.asarray(image, dtype=np.float32)).cuda(non_blocking=True) for image in ret.images]
            ret.patch_nums = default_collate(np.asarray(ret.patch_nums, dtype=np.int32)).cuda(non_blocking=True)
        if ret.videos is not None:
            ret.videos = [torch.from_numpy(np.asarray(video, dtype=np.float32)).cuda(non_blocking=True) for video in ret.videos]
            ret.videos_patch_nums = default_collate(np.asarray(ret.videos_patch_nums, dtype=np.int32)).cuda(non_blocking=True)
        
        return ret

    @torch.no_grad()
    def collect_batch_sft(self, batch_data):
        # list of dict to dataclass
        batch_data = [OmniProcessorOutput(**bd) for bd in batch_data]
        ret = OmniProcessorOutput()
        for i in range(len(batch_data)):
            ret = ret.concatenate(batch_data[i])
        ret.input_ids = default_collate([np.asarray(x.input_ids, dtype=np.int64) for x in batch_data])
        ret.labels = default_collate([np.asarray(x.labels, dtype=np.int64) for x in batch_data])
        ret.position_ids = default_collate([np.asarray(x.position_ids, dtype=np.int64) for x in batch_data])
        ret.seqlens = default_collate([np.asarray(x.seqlens, dtype=np.int64) for x in batch_data])

        ret.raw_text = None
        if ret.audios is not None:
            max_audios_len = max([x.shape[-1] for x in ret.audios])
            ret.audios = default_collate([np.pad(x, ((0,0),(0,max_audios_len - x.shape[-1])), 'constant', constant_values=0) for x in ret.audios])
            ret.encoder_length = default_collate(np.asarray(ret.encoder_length, dtype=np.int32))
            ret.bridge_length = default_collate(np.asarray(ret.bridge_length, dtype=np.int32))
        if ret.audiotext_ids is not None:
            ret.audiotext_ids = default_collate(np.asarray(ret.audiotext_ids, dtype=np.int64))
        if ret.images is not None:
            # 转换 每个image 为torch tensor
            ret.images = [torch.from_numpy(np.asarray(image, dtype=np.float32))  for image in ret.images]
            ret.patch_nums = default_collate(np.asarray(ret.patch_nums, dtype=np.int32))
        if ret.videos is not None:
            ret.videos = [torch.from_numpy(np.asarray(video, dtype=np.float32))  for video in ret.videos]
            ret.videos_patch_nums = default_collate(np.asarray(ret.videos_patch_nums, dtype=np.int32))
        
        ret = ret.__dict__
        del ret['images_size']
        del ret['videos_size']
        del ret['crop_size']
        del ret['videos_crop_size']
        del ret['raw_text']
        del ret['index']
        del ret['attention_mask']
        return ret
