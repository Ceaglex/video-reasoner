
import sys
sys.path.append('/home/chengxin/chengxin/video-reasoner')
import json
import torch
from torch.utils.data import Dataset, DataLoader
import imageio
from PIL import Image
import torchvision.transforms as transforms
import random

from wan.utils.utils import crop_img, pad_img, expand_timestep, save_video
from wan.configs import MAX_AREA_CONFIGS
from wan.utils.utils import best_output_size

class VideoAudioDataset(Dataset):
    def __init__(self, 
                 json_path, 
                 load_mode = 'video_pixel', 
                 uncond_prob = 0.2,
                 size="480*832", 
                 patch_size=[1,2,2],
                 vae_stride=[4,16,16],
                 fps=24, 
                 frame_num=121,
                 value_range=[-1,1]):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        # loading info
        self.samples = list(self.data.items())
        print(f"Dataset total items {len(self.samples)}")
        self.load_mode = load_mode
        self.uncond_prob = uncond_prob
        self.patch_size = patch_size
        self.vae_stride = vae_stride

        # video processing params
        self.size = size
        self.fps = fps
        self.frame_num = frame_num
        self.value_range = value_range
        # self.resize_transform = transforms.Resize((int(size.split('*')[0]), int(size.split('*')[1])), antialias=True)



    # TODO: VAE 有些色差，有空check一下
    def read_video(self, video_path, ow, oh, start_duration = 0.0):
        reader = imageio.get_reader(video_path, fps=self.fps)
        frames = [torch.tensor(frame).to(torch.float32) for frame in reader]
        reader.close()
        video = torch.stack(frames)  # Shape: [T, H, W, C]
        video = video / 255.0
        video = video * (self.value_range[1] - self.value_range[0]) + self.value_range[0]
        
    
        ih, iw = video.shape[1], video.shape[2]
        scale = min(ow / iw, oh / ih)
        new_h = round(ih * scale)
        new_w = round(iw * scale)

        pad_left   = (ow - new_w) // 2
        pad_right  = ow - new_w - pad_left
        pad_top    = (oh - new_h) // 2
        pad_bottom = oh - new_h - pad_top

        resize_and_pad = transforms.Compose([
            transforms.Resize((new_h, new_w), antialias=True),
            transforms.Pad((pad_left, pad_top, pad_right, pad_bottom), padding_mode='constant', fill=self.value_range[0]),
        ])

        video = video.permute(0, 3, 1, 2)
        video = torch.stack([resize_and_pad(frame) for frame in video])
        video = video.permute(0, 2, 3, 1)
            
        # Ensure number of frames is 4n+1 by trimming excess frames
        start_frames = int(start_duration * self.fps)
        video_frames = video.shape[0] - start_frames
        target_frames = video_frames - (video_frames - 1) % 4  # Closest 4n+1
        target_frames = min(target_frames, self.frame_num)

        video = video[start_frames : start_frames + target_frames]    
        if target_frames < self.frame_num:
            print(f"Pad Frame from {video.shape[0]} to {self.frame_num}")
            video = torch.cat([video, video[-1:].repeat(self.frame_num - target_frames, 1, 1, 1)], dim=0)

        # Permute to [C, T, H, W]
        video = video.permute(3, 0, 1, 2)
        return video, start_duration


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        try:
            video_path, info = self.samples[idx]
            if self.load_mode == 'video_pixel':
                return self.load_video_pixel(video_path, info)
            else:
                raise NotImplementedError(f"Load mode {self.load_mode} not implemented.")
            
        except BaseException as e:  
            # print(f"Fail to load file {self.samples[idx]}")
            index = random.randint(0, len(self.samples) - 1)
            return self.__getitem__(index)
        


    def load_video_pixel(self, video_path, info):
        prompt = info['video_caption']
        img_path = info['img_path']
        img = Image.open(img_path).convert("RGB")
        ow, oh = int(self.size.split('*')[0]), int(self.size.split('*')[1])
        img, ow, oh = pad_img(img, ow, oh, 'cpu')
        # img, ow, oh = crop_img(img, self.patch_size, self.vae_stride, MAX_AREA_CONFIGS[self.size], self.frame_num, 'cpu')
        video_pixel, start_duration = self.read_video(video_path, ow, oh, 0)  # [T, C, H, W]
        seq_len     = ((self.frame_num - 1) // self.vae_stride[0] + 1) * (oh // self.vae_stride[1]) * (ow //self.vae_stride[2]) // (self.patch_size[1] * self.patch_size[2])

        if random.random() < self.uncond_prob:
            prompt = ""
            
        return {
            "video_path" : video_path,
            "video_pixel": video_pixel,
            "img"        : img,
            "img_path"   : img_path,
            "prompt"     : prompt,
            "seq_len"    : seq_len
        }



def build_video_loader(args):
    dataset = VideoAudioDataset(
        json_path=args.video_index_file,
        load_mode=args.load_mode,
        uncond_prob=args.uncond_prob if 'uncond_prob' in args else 0.1,
        size=args.size if 'size' in args else '480*832',
        patch_size=args.patch_size if 'patch_size' in args else [1,2,2],
        vae_stride=args.vae_stride if 'vae_stride' in args else [4,16,16],
        fps=args.fps if 'fps' in args else 24,
        frame_num=args.frame_num if 'frame_num' in args else 121,

    )
    dataloader = DataLoader(dataset, 
                            batch_size=args.batch_size, 
                            num_workers=args.num_workers, 
                            prefetch_factor=args.prefetch_factor,
                            shuffle=args.shuffle,
                            pin_memory=True,
                            )
    return dataloader



if __name__ == "__main__":

    from omegaconf import OmegaConf
    # dataset = VideoAudioDataset(
    #     json_path="/home/chengxin/chengxin/video-reasoner/data/debug.json"  ,
    #     load_mode="video_pixel",
    #     uncond_prob=0.1,
    #     size='480*832',
    #     patch_size=[1,2,2],
    #     vae_stride=[4,16,16],
    #     fps=24,
    #     frame_num=121,

    # )
    # print(dataset[0])
    config = OmegaConf.load("/home/chengxin/chengxin/video-reasoner/config/ttv_wan.yaml")
    dataloader = build_video_loader(config.hy_dataloader)
    for i, batch in enumerate(dataloader):
        print(i)
        if i > 10:
            break