"""
Copyright 2022 HuggingFace, ShivamShrirao

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import keyboard
import gradio as gr
import argparse
import random
import hashlib
import itertools
import json
import math
import os
import copy
from contextlib import nullcontext
from pathlib import Path
import shutil
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import numpy as np
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel,DiffusionPipeline, DPMSolverMultistepScheduler,EulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from torchvision.transforms import functional
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from typing import Dict, List, Generator, Tuple
from PIL import Image, ImageFile
from diffusers.utils.import_utils import is_xformers_available
from trainer_util import *
from dataloaders_util import *
from discriminator import Discriminator2D
from lion_pytorch import Lion
logger = get_logger(__name__)
def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--shuffle_per_epoch",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Will shffule the dataset per epoch",
    )
    parser.add_argument(
        "--attention",
        type=str,
        choices=["xformers", "flash_attention"],
        default="xformers",
        help="Type of attention to use."
    )
    parser.add_argument(
        "--model_variant",
        type=str,
        default='base',
        required=False,
        help="Train Base/Inpaint/Depth2Img",
    )
    parser.add_argument(
        "--aspect_mode",
        type=str,
        default='dynamic',
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--aspect_mode_action_preference",
        type=str,
        default='add',
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument('--use_lion',default=False,action=argparse.BooleanOptionalAction, help='Use the new LION optimizer')
    parser.add_argument('--use_ema',default=False,action=argparse.BooleanOptionalAction, help='Use EMA for finetuning')
    parser.add_argument('--clip_penultimate',default=False,action=argparse.BooleanOptionalAction, help='Use penultimate CLIP layer for text embedding')
    parser.add_argument("--conditional_dropout", type=float, default=None,required=False, help="Conditional dropout probability")
    parser.add_argument('--disable_cudnn_benchmark', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--use_text_files_as_captions', default=False, action=argparse.BooleanOptionalAction)
    
    parser.add_argument(
            "--sample_from_batch",
            type=int,
            default=0,
            help=("Number of prompts to sample from the batch for inference"),
        )
    parser.add_argument(
        "--flatten_sample_folder",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Will save samples in one folder instead of per-epoch",
    )
    parser.add_argument(
            "--stop_text_encoder_training",
            type=int,
            default=999999999999999,
            help=("The epoch at which the text_encoder is no longer trained"),
        )
    parser.add_argument(
        "--use_bucketing",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Will save and generate samples before training",
    )
    parser.add_argument(
        "--regenerate_latent_cache",
        default=None,
        action=argparse.BooleanOptionalAction,
        help="Will save and generate samples before training",
    )
    parser.add_argument(
        "--sample_on_training_start",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Will save and generate samples before training",
    )

    parser.add_argument(
        "--add_class_images_to_dataset",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="will generate and add class images to the dataset without using prior reservation in training",
    )
    parser.add_argument(
        "--auto_balance_concept_datasets",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="will balance the number of images in each concept dataset to match the minimum number of images in any concept dataset",
    )
    parser.add_argument(
        "--sample_aspect_ratios",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="sample different aspect ratios for each image",
    )
    parser.add_argument(
        "--dataset_repeats",
        type=int,
        default=1,
        help="repeat the dataset this many times",
    )
    parser.add_argument(
        "--save_every_n_epoch",
        type=int,
        default=1,
        help="save on epoch finished",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained vae or vae identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--save_sample_prompt",
        type=str,
        default=None,
        help="The prompt used to generate sample outputs to save.",
    )
    parser.add_argument(
        "--n_save_sample",
        type=int,
        default=4,
        help="The number of samples to save.",
    )
    parser.add_argument(
        "--sample_height",
        type=int,
        default=512,
        help="The number of samples to save.",
    )
    parser.add_argument(
        "--sample_width",
        type=int,
        default=512,
        help="The number of samples to save.",
    )
    parser.add_argument(
        "--save_guidance_scale",
        type=float,
        default=7.5,
        help="CFG for save sample.",
    )
    parser.add_argument(
        "--save_infer_steps",
        type=int,
        default=30,
        help="The number of inference steps for save sample.",
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument(
        "--with_offset_noise",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Flag to offset noise applied to latents.",
    )

    parser.add_argument("--offset_noise_weight", type=float, default=0.1, help="The weight of offset noise applied during training.")
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If not have enough images, additional images will be"
            " sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop", default=False, action=argparse.BooleanOptionalAction, help="Whether to center crop images before resizing to resolution"
    )
    parser.add_argument("--train_text_encoder", default=False, action=argparse.BooleanOptionalAction, help="Whether to train the text encoder")
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=float, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", default=False, action=argparse.BooleanOptionalAction, help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", default=False, action=argparse.BooleanOptionalAction, help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument("--log_interval", type=int, default=10, help="Log every N steps.")
    parser.add_argument("--sample_step_interval", type=int, default=100000000000000, help="Sample images every N steps.")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16","tf32"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--concepts_list",
        type=str,
        default=None,
        help="Path to json containing multiple concepts, will overwrite parameters like instance_prompt, class_prompt, etc.",
    )
    parser.add_argument("--save_sample_controlled_seed", type=int, action='append', help="Set a seed for an extra sample image to be constantly saved.")
    parser.add_argument("--detect_full_drive", default=True, action=argparse.BooleanOptionalAction, help="Delete checkpoints when the drive is full.")
    parser.add_argument("--send_telegram_updates", default=False, action=argparse.BooleanOptionalAction, help="Send Telegram updates.")
    parser.add_argument("--telegram_chat_id", type=str, default="0", help="Telegram chat ID.")
    parser.add_argument("--telegram_token", type=str, default="0", help="Telegram token.")
    parser.add_argument("--use_deepspeed_adam", default=False, action=argparse.BooleanOptionalAction, help="Use experimental DeepSpeed Adam 8.")
    parser.add_argument('--append_sample_controlled_seed_action', action='append')
    parser.add_argument('--add_sample_prompt', type=str, action='append')
    parser.add_argument('--use_image_names_as_captions', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--shuffle_captions', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--masked_training", default=False, required=False, action='store_true', help="Whether to mask parts of the image during training")
    parser.add_argument("--normalize_masked_area_loss", default=False, required=False, action='store_true', help="Normalize the loss, to make it independent of the size of the masked area")
    parser.add_argument("--unmasked_probability", type=float, default=1, required=False, help="Probability of training a step without a mask")
    parser.add_argument("--max_denoising_strength", type=float, default=1, required=False, help="Max denoising steps to train on")
    parser.add_argument('--add_mask_prompt', type=str, default=None, action="append", dest="mask_prompts", help="Prompt for automatic mask creation")
    parser.add_argument('--with_gan', default=False, action=argparse.BooleanOptionalAction, help="Use GAN (experimental)")
    parser.add_argument("--gan_weight", type=float, default=0.2, required=False, help="Strength of effect GAN has on training")
    parser.add_argument("--gan_warmup", type=float, default=0, required=False, help="Slowly increases GAN weight from zero over this many steps, useful when initializing a GAN discriminator from scratch")
    parser.add_argument('--discriminator_config', default="configs/discriminator_large.json", help="Location of config file to use when initializing a new GAN discriminator")
    parser.add_argument('--sample_from_ema', default=True, action=argparse.BooleanOptionalAction, help="Generate sample images using the EMA model")
    parser.add_argument('--run_name', type=str, default=None, help="Adds a custom identifier to the sample and checkpoint directories")
    parser.add_argument('--gan_ema', default=False, action=argparse.BooleanOptionalAction, help="Use GAN EMA (experimental)")
    parser.add_argument('--train_unet', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "--discriminator_learning_rate",
        type=float,
        default=None,
        help="Initial learning rate (after the potential warmup period) to use for the discriminator. Defaults to same as --learning-rate.",
    )
    parser.add_argument(
        "--with_perturbation_noise",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Flag to apply perturbation noise to latents.",
    )
    parser.add_argument("--perturbation_noise_weight", type=float, default=0.1, help="The weight of perturbation noise applied during training.")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args
    
def main():
    print(f" {bcolors.OKBLUE}Booting Up StableTuner{bcolors.ENDC}") 
    print(f" {bcolors.OKBLUE}Please wait a moment as we load up some stuff...{bcolors.ENDC}") 
    #torch.cuda.set_per_process_memory_fraction(0.5)
    args = parse_args()
    #temp arg
    args.batch_tokens = None
    if args.disable_cudnn_benchmark:
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
    if args.send_telegram_updates:
        send_telegram_message(f"Booting up StableTuner!\n", args.telegram_chat_id, args.telegram_token)
    logging_dir = Path(args.output_dir, "logs", args.logging_dir)
    if not args.pretrained_vae_name_or_path:
        args.pretrained_vae_name_or_path = os.path.join(args.pretrained_model_name_or_path, "vae")
    if args.run_name:
        main_sample_dir = os.path.join(args.output_dir, f"samples_{args.run_name}")
    else:
        main_sample_dir = os.path.join(args.output_dir, "samples")
    if os.path.exists(main_sample_dir):
            shutil.rmtree(main_sample_dir)
            os.makedirs(main_sample_dir)
    #create logging directory
    if not logging_dir.exists():
        logging_dir.mkdir(parents=True)
    #create output directory
    if not Path(args.output_dir).exists():
        Path(args.output_dir).mkdir(parents=True)
    

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision if args.mixed_precision != 'tf32' else 'no',
        log_with="tensorboard",
        logging_dir=logging_dir,
    )

    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
    if args.train_text_encoder and args.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )

    if args.seed is not None:
        set_seed(args.seed)

    if args.concepts_list is None:
        args.concepts_list = [
            {
                "instance_prompt": args.instance_prompt,
                "class_prompt": args.class_prompt,
                "instance_data_dir": args.instance_data_dir,
                "class_data_dir": args.class_data_dir
            }
        ]
    else:
        with open(args.concepts_list, "r") as f:
            args.concepts_list = json.load(f)

    if args.with_prior_preservation or args.add_class_images_to_dataset:
        pipeline = None
        for concept in args.concepts_list:
            class_images_dir = Path(concept["class_data_dir"])
            class_images_dir.mkdir(parents=True, exist_ok=True)
            cur_class_images = len(list(class_images_dir.iterdir()))

            if cur_class_images < args.num_class_images:
                torch_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
                if pipeline is None:

                    pipeline = DiffusionPipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        safety_checker=None,
                        vae=AutoencoderKL.from_pretrained(args.pretrained_vae_name_or_path, safe_serialization=True),
                        torch_dtype=torch_dtype,
                        requires_safety_checker=False,
                    )
                    pipeline.set_progress_bar_config(disable=True)
                    pipeline.to(accelerator.device)
                
                #if args.use_bucketing == False:
                num_new_images = args.num_class_images - cur_class_images
                logger.info(f"Number of class images to sample: {num_new_images}.")

                sample_dataset = PromptDataset(concept["class_prompt"], num_new_images)
                sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=args.sample_batch_size)
                sample_dataloader = accelerator.prepare(sample_dataloader)
                #else:
                    #create class images that match up to the concept target buckets
                #    instance_images_dir = Path(concept["instance_data_dir"])
                #    cur_instance_images = len(list(instance_images_dir.iterdir()))
                    #target_wh = min(self.aspects, key=lambda aspects:abs(aspects[0]/aspects[1] - image_aspect))
                #    num_new_images = cur_instance_images - cur_class_images
                
                

                with torch.autocast("cuda"):
                    for example in tqdm(
                        sample_dataloader, desc="Generating class images", disable=not accelerator.is_local_main_process
                    ):
                        with torch.autocast("cuda"):
                            images = pipeline(example["prompt"],height=args.resolution,width=args.resolution).images
                        for i, image in enumerate(images):
                            hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                            image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                            image.save(image_filename)

        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name )
    elif args.pretrained_model_name_or_path:
        #print(os.getcwd())
        tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer" )

    # Load models and create wrapper for stable diffusion
    #text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder" )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
        torch_dtype=torch.float32
    )
    
    if args.with_gan:
        if os.path.isdir(os.path.join(args.pretrained_model_name_or_path, "discriminator")):
            discriminator = Discriminator2D.from_pretrained(
                args.pretrained_model_name_or_path,
                subfolder="discriminator",
                revision=args.revision,
            )
        else:
            print(f" {bcolors.WARNING}Discriminator network (GAN) not found. Initializing a new network. It may take a very large number of steps to train.{bcolors.ENDC}")
            if args.train_unet and not args.gan_warmup:
                print(f" {bcolors.WARNING}Consider using --gan_warmup to stabilize the model while the discriminator is being trained.{bcolors.ENDC}")
            with open(args.discriminator_config, "r") as f:
                discriminator_config = json.load(f)
            discriminator = Discriminator2D.from_config(discriminator_config)
        ema_discriminator = copy.deepcopy(discriminator)
        ema_discriminator.config["step"] = 0
        
    
    if is_xformers_available() and args.attention=='xformers':
        try:
            vae.enable_xformers_memory_efficient_attention()
            unet.enable_xformers_memory_efficient_attention()
            if args.with_gan:
                discriminator.enable_xformers_memory_efficient_attention()
        except Exception as e:
            logger.warning(
                "Could not enable memory efficient attention. Make sure xformers is installed"
                f" correctly and a GPU is available: {e}"
            )
    elif args.attention=='flash_attention':
        replace_unet_cross_attn_to_flash_attention()

    if args.use_ema == True:
        if os.path.isdir(os.path.join(args.pretrained_model_name_or_path, "unet_ema")):
            ema_unet = UNet2DConditionModel.from_pretrained(
                args.pretrained_model_name_or_path,
                subfolder="unet_ema",
                revision=args.revision,
                torch_dtype=torch.float32
            )
        else:
            ema_unet = copy.deepcopy(unet)
            ema_unet.config["step"] = 0
        ema_unet.requires_grad_(False)

    if args.model_variant == "depth2img":
        d2i = Depth2Img(unet,text_encoder,args.mixed_precision,args.pretrained_model_name_or_path,accelerator)
    vae.requires_grad_(False)
    vae.enable_slicing()
    if not args.train_text_encoder:
        text_encoder.requires_grad_(False)
    if not args.train_unet:
        unet.requires_grad_(False)

    if args.gradient_checkpointing:
        if args.train_unet:
            unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()
        if args.with_gan:
            discriminator.enable_gradient_checkpointing()

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )
        if args.discriminator_learning_rate != None:
            args.discriminator_learning_rate = (
                args.discriminator_learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
            )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam and args.use_deepspeed_adam==False and args.use_lion==False:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )
        optimizer_class = bnb.optim.AdamW8bit
        print("Using 8-bit Adam")
    elif args.use_8bit_adam and args.use_deepspeed_adam==True:
        try:
            from deepspeed.ops.adam import DeepSpeedCPUAdam
        except ImportError:
            raise ImportError(
                "To use 8-bit DeepSpeed Adam, try updating your cuda and deepspeed integrations."
            )
        optimizer_class = DeepSpeedCPUAdam
    elif args.use_lion == True:
        print("Using LION optimizer")
        optimizer_class = Lion
    elif args.use_deepspeed_adam==False and args.use_lion==False and args.use_8bit_adam==False:
        optimizer_class = torch.optim.AdamW
    params_to_optimize = (
        itertools.chain(unet.parameters(), text_encoder.parameters()) if args.train_text_encoder else unet.parameters()
    )
    if args.use_lion == False:
        if args.train_unet:
            optimizer = optimizer_class(
                params_to_optimize,
                lr=args.learning_rate,
                betas=(args.adam_beta1, args.adam_beta2),
                weight_decay=args.adam_weight_decay,
                eps=args.adam_epsilon,
            )
        if args.with_gan:
            optimizer_discriminator = optimizer_class(
                discriminator.parameters(),
                lr=args.discriminator_learning_rate or args.learning_rate,
                betas=(args.adam_beta1, args.adam_beta2),
                weight_decay=args.adam_weight_decay,
                eps=args.adam_epsilon,
            )            
    else:
        if args.train_unet:
            optimizer = optimizer_class(
                params_to_optimize,
                lr=args.learning_rate,
                betas=(args.adam_beta1, args.adam_beta2),
                weight_decay=args.adam_weight_decay,
                #eps=args.adam_epsilon,
            )
        if args.with_gan:
            optimizer_discriminator = optimizer_class(
                discriminator.parameters(),
                lr=args.discriminator_learning_rate or args.learning_rate,
                betas=(args.adam_beta1, args.adam_beta2),
                weight_decay=args.adam_weight_decay,
                #eps=args.adam_epsilon,
            )
    noise_scheduler = DDPMScheduler.from_config(args.pretrained_model_name_or_path, subfolder="scheduler")

    if args.use_bucketing:
        train_dataset = AutoBucketing(
            concepts_list=args.concepts_list,
            use_image_names_as_captions=args.use_image_names_as_captions,
            shuffle_captions=args.shuffle_captions,
            batch_size=args.train_batch_size,
            tokenizer=tokenizer,
            add_class_images_to_dataset=args.add_class_images_to_dataset,
            balance_datasets=args.auto_balance_concept_datasets,
            resolution=args.resolution,
            with_prior_loss=False,#args.with_prior_preservation,
            repeats=args.dataset_repeats,
            use_text_files_as_captions=args.use_text_files_as_captions,
            aspect_mode=args.aspect_mode,
            action_preference=args.aspect_mode_action_preference,
            seed=args.seed,
            model_variant=args.model_variant,
            extra_module=None if args.model_variant != "depth2img" else d2i,
            mask_prompts=args.mask_prompts,
            load_mask=args.masked_training,
        )
    else:
        train_dataset = NormalDataset(
        concepts_list=args.concepts_list,
        tokenizer=tokenizer,
        with_prior_preservation=args.with_prior_preservation,
        size=args.resolution,
        center_crop=args.center_crop,
        num_class_images=args.num_class_images,
        use_image_names_as_captions=args.use_image_names_as_captions,
        shuffle_captions=args.shuffle_captions,
        repeats=args.dataset_repeats,
        use_text_files_as_captions=args.use_text_files_as_captions,
        seed = args.seed,
        model_variant=args.model_variant,
        extra_module=None if args.model_variant != "depth2img" else d2i,
        mask_prompts=args.mask_prompts,
        load_mask=args.masked_training,
    )
    def collate_fn(examples):
        #print(examples)
        #print('test')
        input_ids = [example["instance_prompt_ids"] for example in examples]
        tokens = input_ids
        pixel_values = [example["instance_images"] for example in examples]
        mask = None
        if "mask" in examples[0]:
            mask = [example["mask"] for example in examples]
        if args.model_variant == 'depth2img':
            depth = [example["instance_depth_images"] for example in examples]

        #print('test')
        # Concat class and instance examples for prior preservation.
        # We do this to avoid doing two forward passes.
        if args.with_prior_preservation:
            input_ids += [example["class_prompt_ids"] for example in examples]
            pixel_values += [example["class_images"] for example in examples]
            if "mask" in examples[0]:
                mask += [example["class_mask"] for example in examples]
            if args.model_variant == 'depth2img':
                depth = [example["class_depth_images"] for example in examples]
        mask_values = None
        if mask is not None:
            mask_values = torch.stack(mask)
            mask_values = mask_values.to(memory_format=torch.contiguous_format).float()
        if args.model_variant == 'depth2img':
            depth_values = torch.stack(depth)
            depth_values = depth_values.to(memory_format=torch.contiguous_format).float()
        ### no need to do it now when it's loaded by the multiAspectsDataset
        #if args.with_prior_preservation:
        #    input_ids += [example["class_prompt_ids"] for example in examples]
        #    pixel_values += [example["class_images"] for example in examples]
        
        #print(pixel_values)
        #unpack the pixel_values from tensor to list


        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = tokenizer.pad(
            {"input_ids": input_ids},
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",\
            ).input_ids

        extra_values = None
        if args.model_variant == 'depth2img':
            extra_values = depth_values

        return {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "extra_values": extra_values,
            "mask_values": mask_values,
            "tokens": tokens
        }

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=False, collate_fn=collate_fn, pin_memory=True
    )
    #get the length of the dataset
    train_dataset_length = len(train_dataset)
    #code to check if latent cache needs to be resaved
    if args.regenerate_latent_cache == None:
        #check if last_run.json file exists in logging_dir
        if os.path.exists(logging_dir / "last_run.json"):
            #if it exists, load it
            with open(logging_dir / "last_run.json", "r") as f:
                last_run = json.load(f)
                last_run_batch_size = last_run["batch_size"]
                last_run_dataset_length = last_run["dataset_length"]
                if last_run_batch_size != args.train_batch_size:
                    print(f" {bcolors.WARNING}The batch_size has changed since the last run. Regenerating Latent Cache.{bcolors.ENDC}") 

                    args.regenerate_latent_cache = True
                    #save the new batch_size and dataset_length to last_run.json
                if last_run_dataset_length != train_dataset_length:
                    print(f" {bcolors.WARNING}The dataset length has changed since the last run. Regenerating Latent Cache.{bcolors.ENDC}") 

                    args.regenerate_latent_cache = True
                    #save the new batch_size and dataset_length to last_run.json
            with open(logging_dir / "last_run.json", "w") as f:
                json.dump({"batch_size": args.train_batch_size, "dataset_length": train_dataset_length}, f)
                    
        else:
            #if it doesn't exist, create it
            last_run = {"batch_size": args.train_batch_size, "dataset_length": train_dataset_length}
            #create the file
            with open(logging_dir / "last_run.json", "w") as f:
                json.dump(last_run, f)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        print("Using fp16")
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        print("Using bf16")
        weight_dtype = torch.bfloat16
    elif args.mixed_precision == "tf32":
        torch.backends.cuda.matmul.allow_tf32 = True
        #torch.set_float32_matmul_precision("medium")

    # Move text_encode and vae to gpu.
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    vae.to(accelerator.device, dtype=weight_dtype)
    if args.use_ema == True:
        ema_unet.to(accelerator.device)
    if args.gan_ema:
        ema_discriminator.to(accelerator.device)
    if not args.train_text_encoder:
        text_encoder.to(accelerator.device, dtype=weight_dtype)
    if not args.train_unet:
        unet.to(accelerator.device)

    if args.use_bucketing:
        wh = set([tuple(x.target_wh) for x in train_dataset.image_train_items])
    else:
        wh = set([tuple([args.resolution, args.resolution]) for x in train_dataset.image_paths])
    full_mask_by_aspect = {shape: vae.encode(torch.zeros(1, 3, shape[1], shape[0]).to(accelerator.device, dtype=weight_dtype)).latent_dist.mean * 0.18215 for shape in wh}

    cached_dataset = CachedLatentsDataset(batch_size=args.train_batch_size,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    dtype=weight_dtype,
    model_variant=args.model_variant,
    shuffle_per_epoch=args.shuffle_per_epoch,
    args = args,)

    gen_cache = False
    data_len = len(train_dataloader)
    latent_cache_dir = Path(args.output_dir, "logs", "latent_cache")
    #check if latents_cache.pt exists in the output_dir
    if not os.path.exists(latent_cache_dir):
        os.makedirs(latent_cache_dir)
    for i in range(0,data_len-1):
        if not os.path.exists(os.path.join(latent_cache_dir, f"latents_cache_{i}.pt")):
            gen_cache = True
            break
    if args.regenerate_latent_cache == True:
            files = os.listdir(latent_cache_dir)
            gen_cache = True
            for file in files:
                os.remove(os.path.join(latent_cache_dir,file))
    if gen_cache == False :
        print(f" {bcolors.OKGREEN}Loading Latent Cache from {latent_cache_dir}{bcolors.ENDC}")
        del vae
        if not args.train_text_encoder:
            del text_encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        #load all the cached latents into a single dataset
        for i in range(0,data_len-1):
            cached_dataset.add_pt_cache(os.path.join(latent_cache_dir,f"latents_cache_{i}.pt"))
    if gen_cache == True:
        #delete all the cached latents if they exist to avoid problems
        print(f" {bcolors.WARNING}Generating latents cache...{bcolors.ENDC}")
        train_dataset = LatentsDataset([], [], [], [], [], [])
        counter = 0
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        with torch.no_grad():
            for batch in tqdm(train_dataloader, desc="Caching latents", bar_format='%s{l_bar}%s%s{bar}%s%s{r_bar}%s'%(bcolors.OKBLUE,bcolors.ENDC, bcolors.OKBLUE, bcolors.ENDC,bcolors.OKBLUE,bcolors.ENDC,)):
                cached_extra = None
                cached_mask = None
                batch["pixel_values"] = batch["pixel_values"].to(accelerator.device, non_blocking=True, dtype=weight_dtype)
                batch["input_ids"] = batch["input_ids"].to(accelerator.device, non_blocking=True)
                cached_latent = vae.encode(batch["pixel_values"]).latent_dist
                if batch["mask_values"] is not None:
                    cached_mask = functional.resize(batch["mask_values"], size=cached_latent.mean.shape[2:])
                if batch["mask_values"] is not None and args.model_variant == "inpainting":
                    batch["mask_values"] = batch["mask_values"].to(accelerator.device, non_blocking=True, dtype=weight_dtype)
                    cached_extra = vae.encode(batch["pixel_values"] * (1 - batch["mask_values"])).latent_dist
                if args.model_variant == "depth2img":
                    batch["extra_values"] = batch["extra_values"].to(accelerator.device, non_blocking=True, dtype=weight_dtype)
                    cached_extra = functional.resize(batch["extra_values"], size=cached_latent.mean.shape[2:])
                if args.train_text_encoder:
                    cached_text_enc = batch["input_ids"]
                else:
                    cached_text_enc = text_encoder(batch["input_ids"])[0]
                train_dataset.add_latent(cached_latent, cached_text_enc, cached_mask, cached_extra, batch["tokens"])
                del batch
                del cached_latent
                del cached_text_enc
                del cached_mask
                del cached_extra
                torch.save(train_dataset, os.path.join(latent_cache_dir,f"latents_cache_{counter}.pt"))
                cached_dataset.add_pt_cache(os.path.join(latent_cache_dir,f"latents_cache_{counter}.pt"))
                counter += 1
                train_dataset = LatentsDataset([], [], [], [], [], [])
                #if counter % 300 == 0:
                    #train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, collate_fn=lambda x: x, shuffle=False)
                #    gc.collect()
                #    torch.cuda.empty_cache()
                #    accelerator.free_memory()

        #clear vram after caching latents
        del vae
        if not args.train_text_encoder:
            del text_encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        #load all the cached latents into a single dataset
    train_dataloader = torch.utils.data.DataLoader(cached_dataset, batch_size=1, collate_fn=lambda x: x, shuffle=False)
    print(f" {bcolors.OKGREEN}Latents are ready.{bcolors.ENDC}")
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = len(train_dataloader)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
        
    if args.lr_warmup_steps < 1:
        args.lr_warmup_steps = math.floor(args.lr_warmup_steps * args.max_train_steps / args.gradient_accumulation_steps)

    if args.train_unet:
        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
            num_training_steps=args.max_train_steps,
        )
        unet, optimizer, lr_scheduler = accelerator.prepare(unet, optimizer, lr_scheduler)
    else:
        unet = accelerator.prepare(unet)

    train_dataloader = accelerator.prepare(train_dataloader)

    if args.train_text_encoder and not args.use_ema:
        text_encoder = accelerator.prepare(text_encoder)
    if args.use_ema:
        ema_unet = accelerator.prepare(ema_unet)

    if args.with_gan:
        lr_scheduler_discriminator = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer_discriminator,
            num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
            num_training_steps=args.max_train_steps,
        )
        discriminator, optimizer_discriminator, lr_scheduler_discriminator = accelerator.prepare(discriminator, optimizer_discriminator, lr_scheduler_discriminator)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = len(train_dataloader)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        #print(args.max_train_steps, num_update_steps_per_epoch)
    # Afterwards we recalculate our number of training epochs
    #print(args.max_train_steps, num_update_steps_per_epoch)
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(args.run_name or "dreambooth")
    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    def mid_train_playground(step):
        
        tqdm.write(f"{bcolors.WARNING} Booting up GUI{bcolors.ENDC}")
        epoch = step // num_update_steps_per_epoch
        if args.train_text_encoder and args.stop_text_encoder_training == True:
            text_enc_model = accelerator.unwrap_model(text_encoder,True)
        elif args.train_text_encoder and args.stop_text_encoder_training > epoch:
            text_enc_model = accelerator.unwrap_model(text_encoder,True)
        elif args.train_text_encoder == False:
            text_enc_model = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder" )
        elif args.train_text_encoder and args.stop_text_encoder_training <= epoch:
            if 'frozen_directory' in locals():
                text_enc_model = CLIPTextModel.from_pretrained(frozen_directory, subfolder="text_encoder")
            else:
                text_enc_model = accelerator.unwrap_model(text_encoder,True)
        scheduler = DPMSolverMultistepScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        unwrapped_unet = accelerator.unwrap_model(ema_unet if args.use_ema else unet,True)
            
        pipeline = DiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=unwrapped_unet,
            text_encoder=text_enc_model,
            vae=AutoencoderKL.from_pretrained(args.pretrained_vae_name_or_path, safe_serialization=True),
            safety_checker=None,
            torch_dtype=weight_dtype,
            local_files_only=False,
            requires_safety_checker=False,
        )
        pipeline.scheduler = scheduler
        if is_xformers_available() and args.attention=='xformers':
            try:
                vae.enable_xformers_memory_efficient_attention()
                unet.enable_xformers_memory_efficient_attention()
            except Exception as e:
                logger.warning(
                    "Could not enable memory efficient attention. Make sure xformers is installed"
                    f" correctly and a GPU is available: {e}"
                )
        elif args.attention=='flash_attention':
            replace_unet_cross_attn_to_flash_attention()
        pipeline = pipeline.to(accelerator.device)
        def inference(prompt, negative_prompt, num_samples, height=512, width=512, num_inference_steps=50,seed=-1,guidance_scale=7.5):
            with torch.autocast("cuda"), torch.inference_mode():
                if seed != -1:
                    if g_cuda is None:
                        g_cuda = torch.Generator(device='cuda')
                    else:
                        g_cuda.manual_seed(int(seed))
                else:
                    seed = random.randint(0, 100000)
                    g_cuda = torch.Generator(device='cuda')
                    g_cuda.manual_seed(seed)
                    return pipeline(
                            prompt, height=int(height), width=int(width),
                            negative_prompt=negative_prompt,
                            num_images_per_prompt=int(num_samples),
                            num_inference_steps=int(num_inference_steps), guidance_scale=guidance_scale,
                            generator=g_cuda).images, seed
        
        with gr.Blocks() as demo:
            with gr.Row():
                with gr.Column():
                    prompt = gr.Textbox(label="Prompt", value="photo of zwx dog in a bucket")
                    negative_prompt = gr.Textbox(label="Negative Prompt", value="")
                    run = gr.Button(value="Generate")
                    with gr.Row():
                        num_samples = gr.Number(label="Number of Samples", value=4)
                        guidance_scale = gr.Number(label="Guidance Scale", value=7.5)
                    with gr.Row():
                        height = gr.Number(label="Height", value=512)
                        width = gr.Number(label="Width", value=512)
                    with gr.Row():
                        num_inference_steps = gr.Slider(label="Steps", value=25)
                        seed = gr.Number(label="Seed", value=-1)
                with gr.Column():
                    gallery = gr.Gallery()
                    seedDisplay = gr.Number(label="Used Seed:", value=0)

            run.click(inference, inputs=[prompt, negative_prompt, num_samples, height, width, num_inference_steps,seed, guidance_scale], outputs=[gallery,seedDisplay])
        
        demo.launch(share=True,prevent_thread_lock=True)
        tqdm.write(f"{bcolors.WARNING}Gradio Session is active, Press 'F12' to resume training{bcolors.ENDC}")
        keyboard.wait('f12')
        demo.close()
        del demo
        del text_enc_model
        del unwrapped_unet
        del pipeline
        return
    
    def save_and_sample_weights(step,context='checkpoint',save_model=True):
        try:
            #check how many folders are in the output dir
            #if there are more than 5, delete the oldest one
            #save the model
            #save the optimizer
            #save the lr_scheduler
            #save the args
            height = args.sample_height
            width = args.sample_width
            batch_prompts = []
            if args.sample_from_batch > 0:
                num_samples = args.sample_from_batch if args.sample_from_batch < args.train_batch_size else args.train_batch_size
                batch_prompts = []
                tokens = args.batch_tokens
                if tokens != None:
                    allPrompts = list(set([tokenizer.decode(p).replace('<|endoftext|>','').replace('<|startoftext|>', '') for p in tokens]))
                    if len(allPrompts) < num_samples:
                        num_samples = len(allPrompts)
                    batch_prompts = random.sample(allPrompts, num_samples)
                        

            if args.sample_aspect_ratios:
                #choose random aspect ratio from ASPECTS    
                aspect_ratio = random.choice(ASPECTS)
                height = aspect_ratio[0]
                width = aspect_ratio[1]
            if os.path.exists(args.output_dir):
                if args.detect_full_drive==True:
                    folders = os.listdir(args.output_dir)
                    #check how much space is left on the drive
                    total, used, free = shutil.disk_usage("/")
                    if (free // (2**30)) < 4:
                        #folders.remove("0")
                        #get the folder with the lowest number
                        #oldest_folder = min(folder for folder in folders if folder.isdigit())
                        tqdm.write(f"{bcolors.FAIL}Drive is almost full, Please make some space to continue training.{bcolors.ENDC}")
                        if args.send_telegram_updates:
                            try:
                                send_telegram_message(f"Drive is almost full, Please make some space to continue training.", args.telegram_chat_id, args.telegram_token)
                            except:
                                pass
                        #count time
                        import time
                        start_time = time.time()
                        import platform
                        while input("Press Enter to continue... if you're on linux we'll wait 5 minutes for you to make space and continue"):
                            #check if five minutes have passed
                            #check if os is linux
                            if 'Linux' in platform.platform():
                                if time.time() - start_time > 300:
                                    break

                        
                        #oldest_folder_path = os.path.join(args.output_dir, oldest_folder)
                        #shutil.rmtree(oldest_folder_path)
            # Create the pipeline using using the trained modules and save it.
            if accelerator.is_main_process:
                if 'step' in context:
                    #what is the current epoch
                    epoch = step // num_update_steps_per_epoch
                else:
                    epoch = step
                if args.train_text_encoder and args.stop_text_encoder_training == True:
                    text_enc_model = accelerator.unwrap_model(text_encoder,True)
                elif args.train_text_encoder and args.stop_text_encoder_training > epoch:
                    text_enc_model = accelerator.unwrap_model(text_encoder,True)
                elif args.train_text_encoder == False:
                    text_enc_model = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder" )
                elif args.train_text_encoder and args.stop_text_encoder_training <= epoch:
                    if 'frozen_directory' in locals():
                        text_enc_model = CLIPTextModel.from_pretrained(frozen_directory, subfolder="text_encoder")
                    else:
                        text_enc_model = accelerator.unwrap_model(text_encoder,True)
                    
                if args.run_name:
                    save_dir = os.path.join(args.output_dir, f"{context}_{step}_{args.run_name}")
                else:
                    save_dir = os.path.join(args.output_dir, f"{context}_{step}")
                if args.flatten_sample_folder:
                    sample_dir = main_sample_dir
                else:
                    sample_dir = os.path.join(main_sample_dir, f"{context}_{step}")
                if args.stop_text_encoder_training == True:
                    save_dir = frozen_directory
                
                #scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
                #scheduler = EulerDiscreteScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler", prediction_type="v_prediction")
                scheduler = DPMSolverMultistepScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

                unwrapped_unet = accelerator.unwrap_model(unet,True)
                pipeline = DiffusionPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    unet=unwrapped_unet,
                    text_encoder=text_enc_model,
                    vae=AutoencoderKL.from_pretrained(args.pretrained_vae_name_or_path),
                    safety_checker=None,
                    torch_dtype=weight_dtype,
                    local_files_only=False,
                    requires_safety_checker=False,
                )
                pipeline.scheduler = scheduler
                if is_xformers_available() and args.attention=='xformers':
                    try:
                        unet.enable_xformers_memory_efficient_attention()
                    except Exception as e:
                        logger.warning(
                            "Could not enable memory efficient attention. Make sure xformers is installed"
                            f" correctly and a GPU is available: {e}"
                        )
                elif args.attention=='flash_attention':
                    replace_unet_cross_attn_to_flash_attention()
                    
                if save_model:
                    os.makedirs(save_dir, exist_ok=True)
                    with open(os.path.join(save_dir, "args.json"), "w") as f:
                        json.dump(args.__dict__, f, indent=2)
                    if args.train_unet:
                        pipeline.save_pretrained(save_dir,safe_serialization=True)
                    if args.with_gan:
                        discriminator.save_pretrained(os.path.join(save_dir, "discriminator"), safe_serialization=True)
                    if args.use_ema:
                        ema_unet.save_pretrained(os.path.join(save_dir, "unet_ema"), safe_serialization=True)
                    tqdm.write(f"{bcolors.OKGREEN}Weights saved to {save_dir}{bcolors.ENDC}")
                        
                if args.stop_text_encoder_training == True:
                    #delete every folder in frozen_directory but the text encoder
                    for folder in os.listdir(save_dir):
                        if folder != "text_encoder" and os.path.isdir(os.path.join(save_dir, folder)):
                            shutil.rmtree(os.path.join(save_dir, folder))
                            
                imgs = []
                if args.use_ema and args.sample_from_ema:
                    pipeline.unet = ema_unet
                    
                unet.requires_grad_(False)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    
                if args.add_sample_prompt is not None or batch_prompts != [] and args.stop_text_encoder_training != True:
                    prompts = []
                    if args.add_sample_prompt is not None:
                        for prompt in args.add_sample_prompt:
                            prompts.append(prompt)
                    if batch_prompts != []:
                        for prompt in batch_prompts:
                            prompts.append(prompt)

                    pipeline = pipeline.to(accelerator.device)
                    pipeline.set_progress_bar_config(disable=True)
                    #sample_dir = os.path.join(save_dir, "samples")
                    #if sample_dir exists, delete it
                    if os.path.exists(sample_dir):
                        if not args.flatten_sample_folder:
                            shutil.rmtree(sample_dir)
                    os.makedirs(sample_dir, exist_ok=True)
                    with torch.autocast("cuda"), torch.inference_mode():
                        if args.send_telegram_updates:
                            try:
                                send_telegram_message(f"Generating samples for <b>{step}</b> {context}", args.telegram_chat_id, args.telegram_token)
                            except:
                                pass
                        n_sample = args.n_save_sample
                        if args.save_sample_controlled_seed:
                            n_sample += len(args.save_sample_controlled_seed)
                        progress_bar_sample = tqdm(total=len(prompts)*n_sample,desc="Generating samples")
                        for samplePrompt in prompts:
                            sampleIndex = prompts.index(samplePrompt)
                            #convert sampleIndex to number in words
                            # Data to be written
                            sampleProperties = {
                                "samplePrompt" : samplePrompt
                            }
                            
                            # Serializing json
                            json_object = json.dumps(sampleProperties, indent=4)
                            
                            if args.flatten_sample_folder:
                                sampleName = f"{context}_{step}_prompt_{sampleIndex+1}"
                            else:
                                sampleName = f"prompt_{sampleIndex+1}"
                            
                            if not args.flatten_sample_folder:
                                os.makedirs(os.path.join(sample_dir,sampleName), exist_ok=True)
                            
                            if args.model_variant == 'inpainting':
                                conditioning_image = torch.zeros(1, 3, height, width)
                                mask = torch.ones(1, 1, height, width)
                            if args.model_variant == 'depth2img':
                                #pil new white image
                                test_image = Image.new('RGB', (width, height), (255, 255, 255))
                                depth_image = Image.new('RGB', (width, height), (255, 255, 255))
                                depth = np.array(depth_image.convert("L"))
                                depth = depth.astype(np.float32) / 255.0
                                depth = depth[None, None]
                                depth = torch.from_numpy(depth)
                            for i in range(n_sample):
                                #check if the sample is controlled by a seed
                                if i < args.n_save_sample:
                                    if args.model_variant == 'inpainting':
                                        images = pipeline(samplePrompt, conditioning_image, mask, height=height,width=width, guidance_scale=args.save_guidance_scale, num_inference_steps=args.save_infer_steps).images
                                    if args.model_variant == 'depth2img':
                                        images = pipeline(samplePrompt,image=test_image, guidance_scale=args.save_guidance_scale, num_inference_steps=args.save_infer_steps,strength=1.0).images
                                    elif args.model_variant == 'base':
                                        images = pipeline(samplePrompt,height=height,width=width, guidance_scale=args.save_guidance_scale, num_inference_steps=args.save_infer_steps).images
                                    
                                    if not args.flatten_sample_folder:
                                        images[0].save(os.path.join(sample_dir,sampleName, f"{sampleName}_{i}.png"))
                                    else:
                                        images[0].save(os.path.join(sample_dir, f"{sampleName}_{i}.png"))
                                                                       
                                else:
                                    seed = args.save_sample_controlled_seed[i - args.n_save_sample]
                                    generator = torch.Generator("cuda").manual_seed(seed)
                                    if args.model_variant == 'inpainting':
                                        images = pipeline(samplePrompt,conditioning_image, mask,height=height,width=width, guidance_scale=args.save_guidance_scale, num_inference_steps=args.save_infer_steps, generator=generator).images
                                    if args.model_variant == 'depth2img':
                                        images = pipeline(samplePrompt,image=test_image, guidance_scale=args.save_guidance_scale, num_inference_steps=args.save_infer_steps,generator=generator,strength=1.0).images
                                    elif args.model_variant == 'base':
                                        images = pipeline(samplePrompt,height=height,width=width, guidance_scale=args.save_guidance_scale, num_inference_steps=args.save_infer_steps, generator=generator).images
                                    
                                    if not args.flatten_sample_folder:
                                        images[0].save(os.path.join(sample_dir,sampleName, f"{sampleName}_controlled_seed_{str(seed)}.png"))
                                    else:
                                        images[0].save(os.path.join(sample_dir, f"{sampleName}_controlled_seed_{str(seed)}.png"))
                                progress_bar_sample.update(1)
                            
                            if args.send_telegram_updates:
                                imgs = []
                                #get all the images from the sample folder
                                if not args.flatten_sample_folder:
                                    dir = os.listdir(os.path.join(sample_dir,sampleName))
                                else:
                                    dir = sample_dir

                                for file in dir:
                                    if file.endswith(".png"):
                                        #open the image with pil
                                        img = Image.open(os.path.join(sample_dir,sampleName,file))
                                        imgs.append(img)
                                try:
                                    send_media_group(args.telegram_chat_id,args.telegram_token,imgs, caption=f"Samples for the <b>{step}</b> {context} using the prompt:\n\n<b>{samplePrompt}</b>")
                                except:
                                    pass
                    del pipeline
                    del unwrapped_unet
                    if args.train_unet:
                        unet.requires_grad_(True)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                    
        except Exception as e:
            tqdm.write(e)
            tqdm.write(f"{bcolors.FAIL} Error occured during sampling, skipping.{bcolors.ENDC}")
            pass

    @torch.no_grad()
    def update_ema(ema_model, model, max_decay = 0.9999):
        ema_step = ema_model.config["step"]
        decay = min((ema_step + 1) / (ema_step + 10), max_decay)
        ema_model.config["step"] += 1
        for (s_param, param) in zip(ema_model.parameters(), model.parameters()):
            if param.requires_grad:
                s_param.add_((1 - decay) * (param - s_param))
            else:
                s_param.copy_(param)
    

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps),bar_format='%s{l_bar}%s%s{bar}%s%s{r_bar}%s'%(bcolors.OKBLUE,bcolors.ENDC, bcolors.OKBLUE, bcolors.ENDC,bcolors.OKBLUE,bcolors.ENDC,), disable=not accelerator.is_local_main_process)
    progress_bar_inter_epoch = tqdm(range(num_update_steps_per_epoch),bar_format='%s{l_bar}%s%s{bar}%s%s{r_bar}%s'%(bcolors.OKBLUE,bcolors.ENDC, bcolors.OKGREEN, bcolors.ENDC,bcolors.OKBLUE,bcolors.ENDC,), disable=not accelerator.is_local_main_process)
    progress_bar_e = tqdm(range(args.num_train_epochs),bar_format='%s{l_bar}%s%s{bar}%s%s{r_bar}%s'%(bcolors.OKBLUE,bcolors.ENDC, bcolors.OKGREEN, bcolors.ENDC,bcolors.OKBLUE,bcolors.ENDC,), disable=not accelerator.is_local_main_process)

    progress_bar.set_description("Overall Steps")
    progress_bar_inter_epoch.set_description("Steps To Epoch")
    progress_bar_e.set_description("Overall Epochs")
    global_step = 0
    loss_avg = AverageMeter("loss_avg", max_eta=0.999)
    gan_loss_avg = AverageMeter("gan_loss_avg", max_eta=0.999)
    discriminator_loss_avg = AverageMeter("discriminator_loss_avg", max_eta=0.999)
    text_enc_context = nullcontext() if args.train_text_encoder else torch.no_grad()
    if args.send_telegram_updates:
        try:
            send_telegram_message(f"Starting training with the following settings:\n\n{format_dict(args.__dict__)}", args.telegram_chat_id, args.telegram_token)
        except:
            pass
    try:
        tqdm.write(f"{bcolors.OKBLUE}Starting Training!{bcolors.ENDC}")
        try:
            def toggle_gui(event=None):
                nonlocal mid_generation
                if mid_generation == True:
                    mid_generation = False
                    tqdm.write(f"{bcolors.WARNING}Cancelled GUI.{bcolors.ENDC}")
                else:
                    mid_generation = True
                tqdm.write(f"{bcolors.WARNING}GUI will boot as soon as the current step is done.{bcolors.ENDC}")

            def toggle_checkpoint(event=None):
                nonlocal mid_checkpoint
                if mid_checkpoint == True:
                    mid_checkpoint = False
                    tqdm.write(f"{bcolors.WARNING}Cancelled Checkpointing.{bcolors.ENDC}")
                else:
                    mid_checkpoint = True
                    tqdm.write(f"{bcolors.WARNING}Saving the model as soon as this epoch is done.{bcolors.ENDC}")

            def toggle_sample(event=None):
                nonlocal mid_sample
                if mid_sample == True:
                    mid_sample = False
                    tqdm.write(f"{bcolors.WARNING}Cancelled Sampling.{bcolors.ENDC}")
                else:
                    mid_sample = True
                    tqdm.write(f"{bcolors.WARNING}Sampling will begin as soon as this epoch is done.{bcolors.ENDC}")
            def toggle_checkpoint_step(event=None):
                nonlocal mid_checkpoint_step
                if mid_checkpoint_step == True:
                    mid_checkpoint_step = False
                    tqdm.write(f"{bcolors.WARNING}Cancelled Checkpointing.{bcolors.ENDC}")
                else:
                    mid_checkpoint_step = True
                    tqdm.write(f"{bcolors.WARNING}Saving the model as soon as this step is done.{bcolors.ENDC}")

            def toggle_sample_step(event=None):
                nonlocal mid_sample_step
                if mid_sample_step == True:
                    mid_sample_step = False
                    tqdm.write(f"{bcolors.WARNING}Cancelled Sampling.{bcolors.ENDC}")
                else:
                    mid_sample_step = True
                    tqdm.write(f"{bcolors.WARNING}Sampling will begin as soon as this step is done.{bcolors.ENDC}")
            def toggle_quit_and_save_epoch(event=None):
                nonlocal mid_quit
                if mid_quit == True:
                    mid_quit = False
                    tqdm.write(f"{bcolors.WARNING}Cancelled Quitting.{bcolors.ENDC}")
                else:
                    mid_quit = True
                    tqdm.write(f"{bcolors.WARNING}Quitting and saving the model as soon as this epoch is done.{bcolors.ENDC}")
            def toggle_quit_and_save_step(event=None):
                nonlocal mid_quit_step
                if mid_quit_step == True:
                    mid_quit_step = False
                    tqdm.write(f"{bcolors.WARNING}Cancelled Quitting.{bcolors.ENDC}")
                else:
                    mid_quit_step = True
                    tqdm.write(f"{bcolors.WARNING}Quitting and saving the model as soon as this step is done.{bcolors.ENDC}")
            def help(event=None):
                print_instructions()
            keyboard.add_hotkey("ctrl+shift+g", toggle_gui)
            keyboard.add_hotkey("ctrl+shift+s", toggle_checkpoint)
            keyboard.add_hotkey("ctrl+shift+p", toggle_sample)
            keyboard.add_hotkey("ctrl+alt+shift+s", toggle_checkpoint_step)
            keyboard.add_hotkey("ctrl+alt+shift+p", toggle_sample_step)
            keyboard.add_hotkey("ctrl+shift+q", toggle_quit_and_save_epoch)
            keyboard.add_hotkey("ctrl+alt+shift+q", toggle_quit_and_save_step)
            keyboard.add_hotkey("ctrl+h", help)
            print_instructions()
        except Exception as e:
            pass

        mid_generation = False
        mid_checkpoint = False
        mid_sample = False
        mid_checkpoint_step = False
        mid_sample_step = False
        mid_quit = False
        mid_quit_step = False
        #lambda set mid_generation to true
        if args.run_name:
            frozen_directory = os.path.join(args.output_dir, f"frozen_text_encoder_{args.run_name}")
        else:
            frozen_directory = os.path.join(args.output_dir, "frozen_text_encoder")
        
        unet_stats = {}
        discriminator_stats = {}

        os.makedirs(main_sample_dir, exist_ok=True)
        with open(os.path.join(main_sample_dir, "args.json"), "w") as f:
            json.dump(args.__dict__, f, indent=2)
        if args.with_gan:
            with open(os.path.join(main_sample_dir, "discriminator_config.json"), "w") as f:
                json.dump(discriminator.config, f, indent=2)
        
        for epoch in range(args.num_train_epochs):
            #every 10 epochs print instructions
            unet.train()
            if args.train_text_encoder:
                text_encoder.train()
            
            #save initial weights
            if args.sample_on_training_start==True and epoch==0:
                save_and_sample_weights(epoch,'start',save_model=False)
            
            if args.train_text_encoder and args.stop_text_encoder_training == epoch:
                args.stop_text_encoder_training = True
                if accelerator.is_main_process:
                    tqdm.write(f"{bcolors.WARNING} Stopping text encoder training{bcolors.ENDC}")   
                    current_percentage = (epoch/args.num_train_epochs)*100
                    #round to the nearest whole number
                    current_percentage = round(current_percentage,0)
                    try:
                        send_telegram_message(f"Text encoder training stopped at epoch {epoch} which is {current_percentage}% of training. Freezing weights and saving.", args.telegram_chat_id, args.telegram_token)   
                    except:
                        pass        
                    if os.path.exists(frozen_directory):
                        #delete the folder if it already exists
                        shutil.rmtree(frozen_directory)
                    os.mkdir(frozen_directory)
                    save_and_sample_weights(epoch,'epoch')
                    args.stop_text_encoder_training = epoch
            progress_bar_inter_epoch.reset(total=num_update_steps_per_epoch)
            for step, batch in enumerate(train_dataloader):
                with accelerator.accumulate(unet):
                    # Convert images to latent space
                    with torch.no_grad():

                        latent_dist = batch[0][0]
                        latents = latent_dist.sample() * 0.18215
                        
                        if args.model_variant == 'inpainting':
                            mask = batch[0][2]
                            mask_mean = batch[0][3]
                            conditioning_latent_dist = batch[0][4]
                            conditioning_latents = conditioning_latent_dist.sample() * 0.18215
                        if args.model_variant == 'depth2img':
                            depth = batch[0][4]
                    if args.sample_from_batch > 0:
                        args.batch_tokens = batch[0][5]
                    # Sample noise that we'll add to the latents
                    # and some extra bits to make it so that the model learns to change the zero-frequency of the component freely
                    # https://www.crosslabs.org/blog/diffusion-with-offset-noise
                    if (args.with_offset_noise == True):
                        noise = torch.randn_like(latents) + (args.offset_noise_weight * torch.randn(latents.shape[0], latents.shape[1], 1, 1).to(accelerator.device))
                    else:
                        noise = torch.randn_like(latents)
                    

                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(0, int(noise_scheduler.config.num_train_timesteps * args.max_denoising_strength), (bsz,), device=latents.device)
                    timesteps = timesteps.long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    if args.with_perturbation_noise:
                        # https://arxiv.org/pdf/2301.11706.pdf
                        noisy_latents = noise_scheduler.add_noise(latents, noise + args.perturbation_noise_weight * torch.randn_like(latents), timesteps)
                    else:
                        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    # Get the text embedding for conditioning
                    with text_enc_context:
                        if args.train_text_encoder:
                            if args.clip_penultimate == True:
                                encoder_hidden_states = text_encoder(batch[0][1],output_hidden_states=True)
                                encoder_hidden_states = text_encoder.text_model.final_layer_norm(encoder_hidden_states['hidden_states'][-2])
                            else:
                                encoder_hidden_states = text_encoder(batch[0][1])[0]
                        else:
                            encoder_hidden_states = batch[0][1]

                    
                    # Predict the noise residual
                    mask=None
                    if args.model_variant == 'inpainting':
                        if mask is not None and random.uniform(0, 1) < args.unmasked_probability:
                            # for some steps, predict the unmasked image
                            conditioning_latents = torch.stack([full_mask_by_aspect[tuple([latents.shape[3]*8, latents.shape[2]*8])].squeeze()] * bsz)
                            mask = torch.ones(bsz, 1, latents.shape[2], latents.shape[3]).to(accelerator.device, dtype=weight_dtype)
                        noisy_inpaint_latents = torch.concat([noisy_latents, mask, conditioning_latents], 1)
                        model_pred = unet(noisy_inpaint_latents, timesteps, encoder_hidden_states).sample
                    elif args.model_variant == 'depth2img':
                        noisy_depth_latents = torch.cat([noisy_latents, depth], dim=1)
                        model_pred = unet(noisy_depth_latents, timesteps, encoder_hidden_states, depth).sample
                    elif args.model_variant == "base":
                        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                    

                    # Get the target for loss depending on the prediction type
                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                    
                    # GAN stuff
                    # Input: noisy_latents
                    # True output: target
                    # Fake output: model_pred

                    if args.with_gan:                    
                        # Turn on learning for the discriminator, and do an optimization step
                        discriminator.requires_grad_(True)

                        discriminator_input = get_discriminator_input(
                            discriminator=discriminator,
                            noise_scheduler=noise_scheduler,
                            noisy_latents=noisy_latents,
                            model_pred=torch.cat((target, model_pred), 0),
                            timesteps=timesteps,
                            noise=noise,
                        )
                        discriminator_input.detach_()
                        discriminator_pred = discriminator(discriminator_input, timesteps.repeat(2), encoder_hidden_states.repeat(2, 1, 1))
                        discriminator_target = torch.cat((torch.ones(bsz, 1, device=accelerator.device), torch.zeros(bsz, 1, device=accelerator.device)), 0)
                        discriminator_loss = 2 * F.mse_loss(discriminator_pred, discriminator_target, reduction="mean")
                        if discriminator_loss.isnan():
                            tqdm.write(f"{bcolors.WARNING}Discriminator loss is NAN, skipping GAN update.{bcolors.ENDC}")
                        else:
                            accelerator.backward(discriminator_loss)
                            if accelerator.sync_gradients:
                                accelerator.clip_grad_norm_(discriminator.parameters(), args.max_grad_norm)
                            optimizer_discriminator.step()
                            lr_scheduler_discriminator.step()
                            # Hack to fix NaNs caused by GAN training
                            for name, p in discriminator.named_parameters():
                                if p.isnan().any():
                                    fix_nans_(p, name, discriminator_stats[name])
                                else:
                                    (std, mean) = torch.std_mean(p)
                                    discriminator_stats[name] = (std.item(), mean.item())
                                    del std, mean
                            optimizer_discriminator.zero_grad()
                        del discriminator_input, discriminator_pred, discriminator_target

                        if args.gan_ema == True:
                            update_ema(ema_discriminator, discriminator, 0.9)
                        discriminator_loss.detach_()
                        discriminator_loss_avg.update(discriminator_loss)
                        
                        # Turn off learning for the discriminator for the generator optimization step
                        discriminator.requires_grad_(False)
                            

                    if args.train_unet:
                        if args.with_prior_preservation:
                            # Chunk the noise and noise_pred into two parts and compute the loss on each part separately.
                            """
                            noise_pred, noise_pred_prior = torch.chunk(noise_pred, 2, dim=0)
                            noise, noise_prior = torch.chunk(noise, 2, dim=0)

                            # Compute instance loss
                            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="none").mean([1, 2, 3]).mean()

                            # Compute prior loss
                            prior_loss = F.mse_loss(noise_pred_prior.float(), noise_prior.float(), reduction="mean")

                            # Add the prior loss to the instance loss.
                            loss = loss + args.prior_loss_weight * prior_loss
                            """
                            # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                            model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                            target, target_prior = torch.chunk(target, 2, dim=0)
                            if mask is not None and args.model_variant != "inpainting":
                                loss = masked_mse_loss(model_pred.float(), target.float(), mask, reduction="none").mean([1, 2, 3]).mean()
                                prior_loss = masked_mse_loss(model_pred_prior.float(), target_prior.float(), mask, reduction="mean")
                            else:
                                loss = F.mse_loss(model_pred.float(), target.float(), reduction="none").mean([1, 2, 3]).mean()
                                prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

                            # Add the prior loss to the instance loss.
                            loss = loss + args.prior_loss_weight * prior_loss

                            if mask is not None and args.normalize_masked_area_loss:
                                loss = loss / mask_mean

                        else:
                            if mask is not None and args.model_variant != "inpainting":
                                loss = masked_mse_loss(model_pred.float(), target.float(), mask, reduction="none").mean([1, 2, 3])
                                loss = loss.mean()
                            else:
                                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                            if mask is not None and args.normalize_masked_area_loss:
                                loss = loss / mask_mean
                                
                        base_loss = loss
                                
                        if args.with_gan:
                            # Add loss from the GAN
                            discriminator_input = get_discriminator_input(
                                discriminator=discriminator,
                                noise_scheduler=noise_scheduler,
                                noisy_latents=noisy_latents,
                                model_pred=model_pred,
                                timesteps=timesteps,
                                noise=noise,
                            )
                            if args.gan_ema:
                                discriminator_pred = ema_discriminator(discriminator_input, timesteps, encoder_hidden_states)
                            else:
                                discriminator_pred = discriminator(discriminator_input, timesteps, encoder_hidden_states)
                            gan_loss = F.mse_loss(discriminator_pred, torch.ones_like(discriminator_pred), reduction="mean")
                            if gan_loss.isnan():
                                tqdm.write(f"{bcolors.WARNING}GAN loss is NAN, skipping GAN loss.{bcolors.ENDC}")
                            else:
                                gan_weight = args.gan_weight
                                if args.gan_warmup and global_step < args.gan_warmup:
                                    gan_weight *= global_step / args.gan_warmup
                                loss += gan_weight * gan_loss
                            del discriminator_input, discriminator_pred

                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            params_to_clip = (
                                itertools.chain(unet.parameters(), text_encoder.parameters())
                                if args.train_text_encoder
                                else unet.parameters()
                            )
                            accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                        optimizer.step()
                        lr_scheduler.step()
                        # Hack to fix NaNs caused by GAN training
                        for name, p in unet.named_parameters():
                            if p.isnan().any():
                                fix_nans_(p, name, unet_stats[name])
                            else:
                                (std, mean) = torch.std_mean(p)
                                unet_stats[name] = (std.item(), mean.item())
                                del std, mean
                        optimizer.zero_grad()
                        base_loss.detach_()
                        loss_avg.update(base_loss)
                        if args.with_gan:
                            gan_loss.detach_()
                            gan_loss_avg.update(gan_loss)
                        if args.use_ema == True:
                            update_ema(ema_unet, unet)
 
                        del loss, model_pred
                        if args.with_prior_preservation:
                            del model_pred_prior

                logs = {}
                raw_logs = {}
                if args.train_unet:
                    logs["loss"] = loss_avg.avg.item()
                    raw_logs["loss"] = base_loss.mean().item()
                    logs["lr"] = raw_logs["lr"] = lr_scheduler.get_last_lr()[0]
                elif args.with_gan:
                    logs["lr"] = raw_logs["lr"] = lr_scheduler_discriminator.get_last_lr()[0]
                if args.with_gan:
                    logs["d_loss"] = discriminator_loss_avg.avg.item()
                    raw_logs["d_loss"] = discriminator_loss.mean().item()
                if args.train_unet and args.with_gan:
                    logs["gan_loss"] = gan_loss_avg.avg.item()
                    raw_logs["gan_loss"] = gan_loss.mean().item()
                progress_bar.set_postfix(**logs)
                try:
                    accelerator.log(raw_logs, step=global_step)
                except:
                    pass

                if global_step > 0 and args.sample_step_interval and not global_step % args.sample_step_interval:
                    save_and_sample_weights(global_step,'step',save_model=False)

                progress_bar.update(1)
                progress_bar_inter_epoch.update(1)
                progress_bar_e.refresh()
                global_step += 1

                if mid_quit_step==True:
                    accelerator.wait_for_everyone()
                    save_and_sample_weights(global_step,'quit_step')
                    quit()
                if global_step >= args.max_train_steps and step < num_update_steps_per_epoch - 1:
                    accelerator.wait_for_everyone()
                    save_and_sample_weights(global_step,'final_step')
                    quit()
                if mid_generation==True:
                    mid_train_playground(global_step)
                    mid_generation=False
                if mid_checkpoint_step == True:
                    save_and_sample_weights(global_step,'step',save_model=True)
                    mid_checkpoint_step=False
                    mid_sample_step=False
                elif mid_sample_step == True:
                    save_and_sample_weights(global_step,'step',save_model=False)
                    mid_sample_step=False
            progress_bar_e.update(1)
            if mid_quit==True:
                accelerator.wait_for_everyone()
                save_and_sample_weights(epoch,'quit_epoch')
                quit()
            if epoch == args.num_train_epochs - 1:
                accelerator.wait_for_everyone()
                save_and_sample_weights(epoch,'final_epoch')
                quit()
            elif args.save_every_n_epoch and (epoch + 1) % args.save_every_n_epoch == 0:
                save_and_sample_weights(epoch,'epoch',True)
            elif mid_checkpoint==True:
                save_and_sample_weights(epoch,'epoch',True)
                mid_checkpoint=False
                mid_sample=False
            elif mid_sample==True:
                save_and_sample_weights(epoch,'epoch',False)
                mid_sample=False
            accelerator.wait_for_everyone()
    except Exception:
        try:
            send_telegram_message("Something went wrong while training! :(", args.telegram_chat_id, args.telegram_token)
            #save_and_sample_weights(global_step,'checkpoint')
            send_telegram_message(f"Saved checkpoint {global_step} on exit", args.telegram_chat_id, args.telegram_token)
        except Exception:
            pass
        raise
    except KeyboardInterrupt:
        send_telegram_message("Training stopped", args.telegram_chat_id, args.telegram_token)
    try:
        send_telegram_message("Training finished!", args.telegram_chat_id, args.telegram_token)
    except:
        pass

    accelerator.end_training()
    


if __name__ == "__main__":
    main()
