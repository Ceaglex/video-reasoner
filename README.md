## 1. Env
- flash_attn 2.8.3
- torch 2.6.0+cu126
- 其他的看着安就行

## 2. Training
- git lfs clone https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B

- data format
    - 把数据集整理成json的形式，每个 {video_path: video_info}, video_path是视频mp4的路径，video_info也是一个dict，{"video_caption": video_prompt, "img_path":img_path}, video_prompt是输入的文本，img_path是首帧图片的路径
    - 参考 data/meta/test_maze_5_easy.json

- 修改config/ttv_wan.yaml 
    - 根据Wan2.2-TI2V-5B的下载位置，修改diffusion_ckpt_idx, t5_ckpt, t5_tokenizer, vae_ckpt
    - 根据data_config，修改 hy_dataloader.video_index_file 以及 validation.prompt_index_file
    - 其他的num_train_epochs checkpointing_steps, learning_rate, lora_config, validation.neg_prompt 之类的超参数可以自己调