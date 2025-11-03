# import pdb; pdb.set_trace()
import logging, math, os, shutil
from pathlib import Path
from tqdm.auto import tqdm
import copy
import json
import numpy as np
from PIL import Image
# from datetime import datetime
# from typing import List, Optional, Tuple, Union, Dict, Any


import torch
import torch.nn.functional as nn_func
import gc
from accelerate.logging import get_logger

from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params, free_memory
from diffusers.utils.import_utils import is_wandb_available
from peft import LoraConfig, get_peft_model, PeftModel


from worker.base import prepare_config, prepare_everything, get_optimizer
from dataset.hy_video_audio import build_video_loader
from wan.modules.t5 import T5EncoderModel
from wan.modules.vae2_2 import Wan2_2_VAE
from wan.modules.model import WanModel
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from wan.utils.utils import masks_like
from wan.utils.utils import crop_img, pad_img, expand_timestep, save_video
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from PIL import Image



logger = get_logger(__name__)


def log_validation(
    config,
    
    video_model,
    video_vae,
    video_text_encoder,

    infer_dtype,
    accelerator,
    global_step
    ):

    with open(config.prompt_index_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        data = list(data.items())
    idx = accelerator.process_index if accelerator.process_index < len(data) else 0
    step_range = accelerator.num_processes if accelerator.num_processes <= len(data) else len(data)
    sync_times = len(data) // step_range
    data = data[idx::step_range]
    output_dir = os.path.join(accelerator.project_dir, args.logging_subdir, str(global_step))
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving path :{output_dir}")

    
    generator = torch.Generator(device=accelerator.device).manual_seed(config.seed) if config.seed is not None else None
    neg_prompt = config.neg_prompt
    patch_size = config.patch_size
    vae_stride = config.vae_stride
    frame_num = config.frame_num
    guide_scale = config.guide_scale
    num_inference_steps = config.num_inference_steps
    size = config.size
    fps = config.fps
    # max_area = MAX_AREA_CONFIGS[size]


    


    video_model.eval()
    with ( torch.amp.autocast('cuda', dtype=infer_dtype), torch.no_grad() ):
        for path, info in data:
            prompt = info['video_caption']
            img = Image.open(info['img_path']).convert("RGB")
            ow, oh = int(size.split('*')[0]), int(size.split('*')[1])
            img, ow, oh = pad_img(img, ow, oh, accelerator.device)
            # img, ow, oh = crop_img(img, patch_size, vae_stride, max_area, frame_num, accelerator.device)

            seq_len = ((frame_num - 1) // vae_stride[0] + 1) * (oh // vae_stride[1]) * (ow //vae_stride[2]) // (patch_size[1] * patch_size[2])
            noise = torch.randn(
                video_vae.model.z_dim, 
                (frame_num-1) // vae_stride[0] + 1,
                oh // vae_stride[1], 
                ow // vae_stride[2],
                dtype=infer_dtype, 
                generator=generator, 
                device=accelerator.device
            )
            _, mask = masks_like([noise], zero=True)
            z       = video_vae.encode([img])
            latent  = (1. - mask[0]) * z[0] + mask[0] * noise

            context = video_text_encoder([prompt], accelerator.device)
            context_null = video_text_encoder([neg_prompt], accelerator.device)

            step_scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=config.num_train_timesteps, shift=1, use_dynamic_shifting=False)
            step_scheduler.set_timesteps(num_inference_steps, device=accelerator.device, shift=config.sample_shift) 
            timesteps = step_scheduler.timesteps
            for t in tqdm(timesteps):
                latent_model_input = latent.to(accelerator.device)
                timestep = expand_timestep(t, mask, patch_size, seq_len, accelerator.device)

                noise_pred = video_model([latent_model_input], t=timestep, context=context, seq_len=seq_len)[0]
                noise_pred_uncond = video_model([latent_model_input], t=timestep, context=context_null, seq_len=seq_len)[0]
                noise_pred = noise_pred_uncond + guide_scale * (noise_pred - noise_pred_uncond)

                temp_x0 = step_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latent.unsqueeze(0),
                    return_dict=False,
                    generator=generator)[0]
                latent = temp_x0.squeeze(0)
                latent = (1. - mask[0]) * z[0] + mask[0] * latent


            out = video_vae.decode([latent])
            save_video(
                tensor=out[0][None],
                save_file=f"{output_dir}/{path.split('/')[-1].replace('.mp4', '')}.mp4",
                fps=fps, nrow=1, normalize=True, value_range=(-1, 1)
            )
            print(f"Saving Video to {output_dir}/{path.split('/')[-1].replace('.mp4', '')}.mp4")


            if sync_times != 0: 
                sync_times -= 1
                accelerator.wait_for_everyone()

                    
        # Clear cache after validation
        accelerator.wait_for_everyone()
        torch.cuda.empty_cache()
        gc.collect()
        free_memory()
        video_model.train()
    
    

def training_step(batch, 
                  video_model,
                  video_vae,
                  video_text_encoder,
                  frame_num,
                  fps,
                  size,
                  patch_size,
                  vae_stride,
                  accelerator, ):
    prompt       = batch['prompt']      # list
    # img_paths    = batch['img_path']    # list
    videos_pixel = batch['video_pixel'] # tensor
    imgs         = batch['img']         # tensor
    seq_len      = int(batch['seq_len'][0])
    batch_size  = len(prompt)
    assert batch_size == 1 # 暂时不考虑其他情况，尽快实现一版先

    
    # with autocast(dtype=load_dtype):
    with torch.no_grad():
        context     = video_text_encoder(prompt, accelerator.device)
        ref_latents = video_vae.encode([img.to(video_vae.dtype) for img in imgs])
        latents     = video_vae.encode([video_pixel.to(video_vae.dtype) for video_pixel in videos_pixel])
        noise       = [torch.randn_like(latents[0])]
        _, mask     = masks_like(noise, zero=True)
        latents     = [(1. - mask[0]) * ref_latents[0] + mask[0] * latents[0]]
        noise       = [(1. - mask[0]) * ref_latents[0] + mask[0] * noise[0]]
        
        timesteps = torch.randint( 0, 1000, (batch_size,), dtype=torch.int64, device=video_model.device).detach()
        t = (timesteps / 1000).to(latents[0].dtype).view(-1, *([1] * (latents[0].ndim - 1)))
        latent_model_input = (1  - t) * latents[0] + t * noise[0]
        target = noise[0] - latents[0]
        timesteps = expand_timestep(t.squeeze(), mask, patch_size, seq_len, accelerator.device)


        # # # TODO : Check latent
        # # Decode latents to final outputs
        # out_video = video_vae.decode(latents)
        # for i in range(len(out_video)):
        #     save_video(
        #         tensor=out_video[i][None],
        #         save_file=f"test{i}.mp4",
        #         fps=fps, nrow=1, normalize=True, value_range=(-1, 1)
        #     )
        # out_video = video_vae.decode([latent_model_input])
        # for i in range(len(out_video)):
        #     save_video(
        #         tensor=out_video[i][None],
        #         save_file=f"test_latent{i}.mp4",
        #         fps=fps, nrow=1, normalize=True, value_range=(-1, 1)
        #     )
        # out_images = video_vae.decode(ref_latents)
        # for i in range(len(out_video)):
        #     img_tensor = out_images[i].squeeze(1)
        #     img_np = img_tensor.cpu().numpy()
        #     img_np = (img_np * 0.5 + 0.5)  # 恢复到 [0, 1]
        #     img_np = np.transpose(img_np, (1, 2, 0))  # (832, 832, 3)
        #     img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
        #     img_pil = Image.fromarray(img_np)
        #     img_pil.save(f"test_ref{i}.jpg")  # 或 .png
        

    
    v_noise_pred = video_model([latent_model_input], t=timesteps, context=context, seq_len=seq_len)[0]
    loss_v = nn_func.mse_loss(mask[0] * v_noise_pred.float(),  target.float(), reduction="mean")
    return loss_v


def checkpointing_step(video_model, accelerator, logger, args, ckpt_idx = 0):
    # ckpt_dir = os.path.join(args.output_dir, args.ckpt_subdir)
    ckpt_dir = os.path.join(accelerator.project_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
    if args.checkpoints_total_limit is not None:
        checkpoints = os.listdir(ckpt_dir)
        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
        checkpoints = sorted(checkpoints, key=lambda x: float(x.split("-step")[1]) if "-step" in x else 0)

        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
        if len(checkpoints) >= args.checkpoints_total_limit:
            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]

            logger.info( f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints" )
            logger.info( f"Removing checkpoints: {', '.join(removing_checkpoints)}" )

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(ckpt_dir, removing_checkpoint)
                shutil.rmtree(removing_checkpoint)

    # Save whole model lora
    accelerator.save_state()
    # Save lora
    lora_save_path = f"{ckpt_dir}/checkpoint_{ckpt_idx}"
    video_model.module.save_pretrained(lora_save_path) if hasattr(video_model, "module") else model.save_pretrained(lora_save_path)
    logger.info(f"Saved state to {ckpt_dir}")


def main(args, accelerator):

    """ ****************************  Model setting.  **************************** """
    logger.info("=> Preparing models and scheduler...", main_process_only=True)
    load_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float32 
    infer_dtype = torch.bfloat16

    # Wan
    video_model = WanModel.from_pretrained(args.diffusion_ckpt_idx).to(load_dtype).eval().requires_grad_(False)
    # Set Training Parameters
    if args.gradient_checkpointing:
        video_model.enable_gradient_checkpointing()



    """ ****************************  Weights dtype setting.  **************************** """
    # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = load_dtype
    if accelerator.state.deepspeed_plugin:
        # DeepSpeed is handling precision, use what's in the DeepSpeed config
        if (
            "fp16" in accelerator.state.deepspeed_plugin.deepspeed_config
            and accelerator.state.deepspeed_plugin.deepspeed_config["fp16"]["enabled"]
        ):
            weight_dtype = torch.float16
        if (
            "bf16" in accelerator.state.deepspeed_plugin.deepspeed_config
            and accelerator.state.deepspeed_plugin.deepspeed_config["bf16"]["enabled"]
        ):
            weight_dtype = torch.float16
    else:
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
    # Due to pytorch#99272, MPS does not yet support bfloat16.
    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        raise ValueError("Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead.")
    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        cast_training_params([video_model], dtype=torch.float32)


    """ ****************************  Data setting.  **************************** """
    # TODO: Add dataloader
    logger.info("=> Preparing training data...", main_process_only=True)
    dataset_cfg = args.hy_dataloader
    train_dataloader = build_video_loader(args=dataset_cfg)
    if accelerator.state.deepspeed_plugin is not None:
        accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = args.train_batch_size_local


    """ ****************************  Optimization setting.  **************************** """
    video_model.requires_grad_(False)
    # TODO: Add load lora
    if args.lora_config.use_lora == True:
        lora_config = LoraConfig(
            r=args.lora_config.rank,
            lora_alpha=args.lora_config.lora_alpha,
            target_modules=list(args.lora_config.lora_modules),
            lora_dropout=0.1,
            bias="none",)
        video_model = get_peft_model(video_model, lora_config, adapter_name="learner")
        video_model.print_trainable_parameters()

    if args.optimize_params is not None:
        for name, param in video_model.named_parameters():
            if any(opt_param in name for opt_param in args.optimize_params):
                param.requires_grad = True


    # Filter parameters based on optimize_params
    transformer_training_parameters = list(filter(lambda p: p.requires_grad, video_model.parameters()))
    if args.scale_lr:
        args.learning_rate = ( args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size_local * accelerator.num_processes )
    print(len(transformer_training_parameters), args.learning_rate)

    # Optimization parameters
    transformer_parameters_with_lr = {"params": transformer_training_parameters, "lr": args.learning_rate}
    params_to_optimize = [transformer_parameters_with_lr]
    use_deepspeed_optimizer = ( accelerator.state.deepspeed_plugin is not None ) and ( "optimizer" in accelerator.state.deepspeed_plugin.deepspeed_config )
    use_deepspeed_scheduler = ( accelerator.state.deepspeed_plugin is not None ) and ( "scheduler" in accelerator.state.deepspeed_plugin.deepspeed_config )
    optimizer = get_optimizer(args, params_to_optimize, use_deepspeed=use_deepspeed_optimizer)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    if use_deepspeed_scheduler:
        from accelerate.utils import DummyScheduler
        lr_scheduler = DummyScheduler(
            name=args.lr_scheduler,
            optimizer=optimizer,
            total_num_steps=args.max_train_steps * accelerator.num_processes,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        )
    else:
        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
            num_cycles=args.lr_num_cycles,
            power=args.lr_power,
        )

    """ ****************************  Accelerator setting.  **************************** """
    logger.info("=> Prepare everything with accelerator ...", main_process_only=True)
    video_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare( 
        video_model, optimizer, train_dataloader, lr_scheduler 
    )
    video_vae = Wan2_2_VAE(vae_pth=args.vae_ckpt, device=accelerator.device, dtype=load_dtype)
    video_text_encoder = T5EncoderModel(
        text_len=512,
        dtype=load_dtype,
        device=accelerator.device,
        checkpoint_path=args.t5_ckpt,
        tokenizer_path=args.t5_tokenizer
    )
    video_text_encoder.model.eval().requires_grad_(False)


    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)


    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        if args.report_to == "wandb":
            if not is_wandb_available():
                raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
            else:
                accelerator.init_trackers(
                    project_name=args.wandb_init_args.project,
                    config=dict(args),
                    init_kwargs={
                        "wandb": {
                            "name": args.wandb_init_args.name,
                            "tags": args.wandb_init_args.tags.split(",") if isinstance(args.wandb_init_args.tags, str) else args.wandb_init_args.tags,
                            "dir": str(Path(args.output_dir)), 
                            "mode": args.wandb_init_args.mode,
                        }
                    }
                )
        else:
            tracker_name = args.tracker_name or "some-training"
            accelerator.init_trackers(tracker_name, config=vars(args))



    """ ****************************  Training info and resume.  **************************** """
    total_batch_size = args.train_batch_size_local * accelerator.num_processes * args.gradient_accumulation_steps
    num_trainable_parameters = sum(param.numel() for model in params_to_optimize for param in model["params"])

    logger.info("***** Running training *****")
    logger.info(f"  Num trainable parameters = {num_trainable_parameters}")
    # logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size_local}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    

    # Potentially load in the weights and states from a previous save
    if not args.resume_from_checkpoint:
        initial_global_step = 0
    else:
        if args.resume_from_checkpoint != "latest": 
            dir_name = os.path.basename(args.resume_from_checkpoint)
            cur_full_path = args.resume_from_checkpoint
        else:
            # Get the mos recent checkpoint
            # dirs = os.listdir(os.path.join(args.output_dir, args.ckpt_subdir))
            dirs = os.listdir(os.path.join(accelerator.project_dir, 'checkpoints'))
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: float(x.split("-step")[1]) if "-step" in x else 0)
            dir_name = dirs[-1] if len(dirs) > 0 else None
            cur_full_path = os.path.join(accelerator.project_dir, 'checkpoints', dir_name)
        if dir_name is None:
            accelerator.print( f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run." )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print( f"Resuming from checkpoint {cur_full_path}" )
            accelerator.load_state(cur_full_path)
            global_step = (int(dir_name.split('_')[-1]) + 1) * args.checkpointing_steps
            # global_step = 0 ####
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
            iteration = int(global_step // args.checkpointing_steps)
            accelerator.project_configuration.iteration = iteration
            



    """ ****************************  Training.  **************************** """
    # Only show the progress bar once on each machine.
    progress_bar = tqdm( range(0, args.max_train_steps), initial=initial_global_step, desc="Steps", disable=not accelerator.is_local_main_process, )
    for epoch in range(first_epoch, args.num_train_epochs):
        video_model.train()
        for step, batch in enumerate(train_dataloader):
            
            if global_step % args.validation.eval_steps == 0:
                log_validation(
                    config = args.validation,
                    video_model = video_model,
                    video_vae = video_vae,
                    video_text_encoder = video_text_encoder,
                    infer_dtype = infer_dtype,
                    accelerator = accelerator,
                    global_step = global_step
                )
            

            # TODO: Add training step
            # TRAIN
            models_to_accumulate = [video_model]
            with accelerator.accumulate(models_to_accumulate):
                loss = training_step(batch = batch, 
                                       video_model = video_model,
                                       video_vae = video_vae,
                                       video_text_encoder = video_text_encoder,
                                       frame_num = args.frame_num,
                                       fps = args.fps,
                                       size = args.size,
                                       patch_size = args.patch_size,
                                       vae_stride = args.vae_stride,
                                       accelerator = accelerator)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = video_model.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                if accelerator.state.deepspeed_plugin is None:
                    optimizer.step()
                    optimizer.zero_grad()
                lr_scheduler.step()


            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if global_step % args.checkpointing_steps == 0:
                    checkpointing_step(video_model = video_model, 
                                        accelerator = accelerator, 
                                        logger = logger, 
                                        args = args, 
                                        ckpt_idx = int(global_step // args.checkpointing_steps - 1))


            logs = {"loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            if global_step >= args.max_train_steps:
                break


    # Save the lora layers
    logger.info("=> Saving the trained model ...")
    checkpointing_step(video_model = video_model, 
                        accelerator = accelerator, 
                        logger = logger, 
                        args = args, 
                        ckpt_idx = int(global_step // args.checkpointing_steps - 1))

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    args = prepare_config()
    args, accelerator = prepare_everything(args)
    main(args, accelerator)
