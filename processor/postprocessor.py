# from processor.decoder.flash_omni_omnigen2.omnigen2_decoder import load_omnigen2_decoder, decode_and_save_img_omnigen2
from processor.preprocessor import PreProcessor
from processor.decoder.cosy24k_vocoder.cosy24k_vocoder import Cosy24kVocoder
from processor.decoder.audio_decode import decode_save_concat
# from processor.decoder.omni_gen2.modular_longcat_next_visual import LongcatNextVisualTokenizer
from processor.decoder.omni_gen2_new.modular_longcat_next_visual import VisionTransformerDecoder, decode_image
from processor.decoder.omni_gen2_new.refiner_modules import FlowMatchEulerDiscreteScheduler
from processor.decoder.omni_gen2_new.image_refiner import (
    ImageRefinerContainer,
    RefinerImageProcessor,
    RefinerPipeline,
    de_transform,
    tensor2pil,
)
import traceback
import os
from safetensors.torch import load_file
from transformers import AutoConfig
import torch


class PostProcessor:
    def __init__(self, configs, encoder=None):
        if not encoder:
            encoder = PreProcessor(configs, oe_delegate_fn="ok")
        self.encoder = encoder
        model_path = self.encoder.model_path
        # 音频解码
        vocoder_path = configs.get("vocoder-path", os.path.join(model_path, "cosy24k_vocoder/hift.pt"))
        self.audio_tokenizer = getattr(self.encoder, 'audio_tokenizer', None)
        self.vocoder = Cosy24kVocoder.from_pretrained(vocoder_path).cuda()
        # 视觉解码
        decoder_path = configs.get("decoder-path", os.path.join(model_path, "image_decoder/image_decoder.safetensors"))
        assert vocoder_path is not None, "vocoder_path is None"
        assert decoder_path is not None, "decoder_path is None"
        vd_config = self.encoder.config.visual_decoder_config
        self.image_decoder = VisionTransformerDecoder.from_pretrained(
            vd_config.image_decoder_config,
            decoder_path,
        ).to(device="cuda", dtype=torch.bfloat16)
        image_refiner = ImageRefinerContainer.from_pretrained(vd_config, decoder_path).to(device="cuda", dtype=torch.bfloat16)

        sc = vd_config.scheduler_config
        scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=sc.num_train_timesteps,
            dynamic_time_shift=sc.dynamic_time_shift)
        self.refiner_pipeline = RefinerPipeline(
            vae=image_refiner.vae,
            transformer=image_refiner.base_transformer,
            scheduler=scheduler,
            cond_proj=image_refiner.cond_proj,
        )
        self.refiner_pipeline.set_progress_bar_config(disable=False)
        
    @torch.no_grad()  
    def decode_multi(self, multi_data, file_name, gen_image, gen_audio, tokens_h=18, tokens_w=18):
        # 注：图像解码时multi_data为[324，8]的tensor，音频解码时为多段[n*8]tensor组成的list，每段结尾需要有一个8192
        try:
            if gen_image:
                refined_images = decode_image(multi_data, self.encoder.model.visual_model, self.image_decoder,
                                                self.refiner_pipeline,tokens_h, tokens_w)
                refined_images[0].save(f"{file_name}.png")
            if gen_audio:
                processed_sequences = [one_data.unsqueeze(0).to(self.audio_tokenizer.device) for one_data in multi_data]
                decode_save_concat(
                    response_list=processed_sequences,
                    vocoder=self.vocoder,
                    audio_tokenizer=self.audio_tokenizer,
                    codebook_sizes=self.encoder.config.audio_config.vq_config.codebook_sizes,
                    path=f"{file_name}.wav",
                    sampling_rate=24000,  # 使用默认采样率
                    wave_concat_overlap=1200  # 使用默认重叠长度
                )
                print(f"decode_save_concat success: {file_name}")
                
        except Exception as e:
            print(e)
            print(traceback.format_exc())
