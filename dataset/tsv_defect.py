import os
import json
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from PIL import Image


class DefectDataset(Dataset):
    def __init__(self, root_dir, tsv_path, prob_use_caption=1, image_size=256, random_flip=False):
        """
        tsv_path : TSV 파일 경로
        transform: 이미지에 적용할 transform (e.g. Resize, ToTensor 등)
        mask_transform: 마스크에 적용할 transform (e.g. Resize, ToTensor 등)
        """
        super().__init__()
        self.tsv_path = tsv_path
        # 이미지/마스크가 들어 있는 루트 디렉토리
        self.root_dir = root_dir

        # TSV 파일 파싱하여 샘플 목록 생성
        self.samples = self.load_tsv(tsv_path)
        self.prob_use_caption = prob_use_caption
        self.image_size = image_size
        self.random_flip = random_flip
        self.to_tensor = T.ToTensor()
    
    def total_images(self):
        return len(self)

    def load_tsv(self, tsv_file):
        print(tsv_file)
        samples = []
        with open(tsv_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                columns = line.split('\t')
                image_name = columns[0]         # 이미지 파일명
                caption = columns[1]           # 텍스트 캡션
                anno_dict = json.loads(columns[2])  # JSON: {"mask_path": "...", "defect_label": "..."}

                mask_path = anno_dict.get("mask_path", None)
                defect_label = anno_dict.get("defect_label", None)

                samples.append({
                    "image": image_name,
                    "caption": caption,
                    "mask_path": mask_path,
                    "defect_label": defect_label
                })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # TSV에서 읽은 정보들
        image_name   = sample["image"]
        caption      = sample["caption"]
        mask_relpath = sample["mask_path"]
        defect_label = sample["defect_label"]

        # (1) 원본 이미지 로드
        image_path = os.path.join(self.root_dir, "Source_Images", image_name)
        img = Image.open(image_path).convert("RGB")

        # (2) 마스크 로드 (있다면)
        if mask_relpath is not None:
            mask_path = os.path.join(self.root_dir, mask_relpath)
            mask = Image.open(mask_path).convert("L")
        else:
            H, W = img.size  # (width, height)
            mask = Image.new('L', (W, H), 0)
        
        assert img.size ==  mask.size

        # - - - - - center_crop, resize and random_flip - - - - - - #  

        crop_size = min(img.size)
        img = TF.center_crop(img, crop_size)
        img = img.resize( (self.image_size, self.image_size) )
        # 만약 위에서 mask_tensor가 필요하다면, 그 시점에 img.size와 동일 크기로 만드는 로직이 필요
        img = self.to_tensor(img)  

        mask = TF.center_crop(mask, crop_size)
        mask = mask.resize( (self.image_size, self.image_size), Image.NEAREST )
        

        # (4) 반환
        return {
            "image":        img,          # (3,H,W) 텐서
            "caption":      caption,
            "mask":         torch.tensor(np.array(mask), dtype=torch.long),  # (1,H,W) 텐서
            "defect_label": defect_label
        }

