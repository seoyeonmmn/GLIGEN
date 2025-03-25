import torch
import torch.nn as nn
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf

config = OmegaConf.load("/home/seoyeon/fake_defect_generator/GLIGEN/configs/mvtec.yaml")

class GroundingNetInput:
    def __init__(self):
        self.set = False
        self.text_encoder = instantiate_from_config(config.text_encoder)
        self.device = torch.device("cuda")
        self.text_encoder = self.text_encoder.to(self.device)
    
    def prepare(self, batch):
        self.set = True

        mask = batch['mask']
        mask = batch['mask'].float()
        label = batch['defect_label']
        label = self.text_encoder.encode(label)

        self.batch_size, self.seq_length, self.embed_dim = label.shape
        self.mask_shape = mask.shape       

        return {'mask':mask, "label":label}
    
    def get_null_input(self, batch=None, device=None, dtype=None):
        assert self.set, "not set yet, cannot call this funcion"
        
        device = device if device is not None else self.device
        dtype = dtype if dtype is not None else torch.float32

        null_label = torch.zeros((self.batch_size, self.seq_length, self.embed_dim),
                                 dtype=dtype, device=device)
        null_mask = torch.zeros(self.mask_shape, dtype=dtype, device=device)

        return {'mask': null_mask, 'label': null_label}




# class GroundingNetInput:
#     def __init__(self):
#         self.text_encoder = instantiate_from_config(config.text_encoder)
#         self.device = torch.device("cuda")
#         self.text_encoder = self.text_encoder.to(self.device)
#         self.mask_expansion_layer = nn.Conv2d(1, 768, kernel_size=3, stride=1, padding=1)
#         self.set = True
#         # 원하는 경우 미리 H, W 사이즈 정의(Null Input용)
#         self.H = 256
#         self.W = 256

#     def prepare(self, batch):
#         """
#         batch는 { 'image', 'caption', 'mask', 'defect_label', ... }를 포함한다고 가정.
#         이 중 'mask': [B, 1, H_resized, W_resized], 'defect_label' (리스트 of str) 등이 필요.
#         """
#         # 1. 마스크
#         mask = batch['mask'].to(self.device)  # [B, 1, H, W]
#         # 2. 텍스트 임베딩
#         defect_label = batch['defect_label']  # 예: ["scratch", "anomaly", ...] 등
#         label_emb = self.text_encoder.encode(defect_label)  # [B, seq_len, 768] 형태라고 가정
#         label_emb = label_emb.mean(dim=1)                   # [B, 768] - 문장 단위로 평균 pooling

#         # 3. 마스크를 플래튼 후 합산 → [B], 이어서 [B, 1]로 reshape
#         mask_flat = mask.view(mask.shape[0], -1)    # [B, H*W]
#         mask_feat = mask_flat.sum(dim=1, keepdim=True)  # [B, 1]

#         # 4. 텍스트 임베딩 [B,768] + 마스크 스칼라 [B,1] → [B,769]
#         grounding_tokens = torch.cat([label_emb, mask_feat], dim=-1)

#         # 5. 최종 반환
#         # "grounding_tokens"만 담은 dict 반환
#         return {"grounding_tokens": grounding_tokens}

#     def get_null_input(self, batch=None, device=None, dtype=None):
#         """
#         prepare()와 같은 로직으로 'null' (모두 0)인 마스크/텍스트를 만들어
#         동일한 [B, 769] 형태의 grounding_tokens를 반환
#         """
#         # device/dtype 설정
#         device = device or self.device
#         dtype = dtype or torch.float32

#         if batch is None:
#             # 1) batch가 없으면 가상의 null 입력 만들기
#             batch_size = 8  # 예시로 8
#             H, W = self.H, self.W

#             # (A) null text embeddings
#             text_embeddings = torch.zeros(batch_size, 768, dtype=dtype, device=device)  # [B,768]
            
#             # (B) null mask
#             #     일관성을 위해 prepare()와 동일하게 마스크를 flatten 후 합산해야 함.
#             mask = torch.zeros(batch_size, 1, H, W, dtype=dtype, device=device)  # [B,1,H,W]
#             mask_feat = mask.view(batch_size, -1).sum(dim=1, keepdim=True)      # [B,1]

#             # (C) [B,769] 만들기
#             grounding_tokens = torch.cat([text_embeddings, mask_feat], dim=-1)  # [B,769]

#             return {"grounding_tokens": grounding_tokens}
#         else:
#             # 2) 기존 batch가 있으면, 그 batch 크기만큼 null 입력 생성
#             batch_size = batch['grounding_tokens'].size(0)
#             H, W = self.H, self.W

#             text_embeddings = torch.zeros(batch_size, 768, dtype=dtype, device=device)  # [B,768]
#             mask = torch.zeros(batch_size, 1, H, W, dtype=dtype, device=device)         # [B,1,H,W]
#             mask_feat = mask.view(batch_size, -1).sum(dim=1, keepdim=True)             # [B,1]

#             grounding_tokens = torch.cat([text_embeddings, mask_feat], dim=-1)         # [B,769]

#             return {"grounding_tokens": grounding_tokens}
