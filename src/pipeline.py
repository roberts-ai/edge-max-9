import torch
from PIL.Image import Image
from diffusers import StableDiffusionXLPipeline, AutoencoderTiny
from pipelines.models import TextToImageRequest
from torch import Generator
from diffusers import AutoencoderKL

def load_pipeline() -> StableDiffusionXLPipeline:
    pipeline = StableDiffusionXLPipeline.from_pretrained("./models/edge-zk", local_files_only=True, torch_dtype=torch.float16)
    pipeline.vae = AutoencoderTiny.from_pretrained("./models/taesdxl",local_file_only=True, torch_dtype=torch.float16)
    pipeline = pipeline.to("cuda")
    pipeline(prompt="")

    return pipeline


def infer(request: TextToImageRequest, pipeline: StableDiffusionXLPipeline) -> Image:
    generator = Generator(pipeline.device).manual_seed(request.seed) if request.seed else None

    return pipeline(
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        width=request.width,
        height=request.height,
        generator=generator,
        num_inference_steps=25,
    ).images[0]
