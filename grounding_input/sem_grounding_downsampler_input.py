import torch
from ..ldm.util import instantiate_from_config
from omegaconf import OmegaConf

config = OmegaConf.load("/home/seoyeon/fake_defect_generator/GLIGEN/configs/mvtec.yaml")

class GroundingDSInput:
    def __init__(self):
        self.text_encoder = instantiate_from_config(config.text_encoder)

    def prepare(self, batch):
        """
        batch should contain 'sem' (mask), 'text_embeddings' (defect label embeddings),
        and other data like image if needed.
        """
        # 1. 마스크와 텍스트 임베딩을 결합
        mask = batch['mask']  # [B, 1, H_resized, W_resized]
        label = batch['defect_label']  # [B, 768]
        label = self.text_encoder.encode(label)

        # 2. 마스크 feature (하나의 토큰으로 풀링된 값)
        # 예시로, 마스크가 1인 영역을 pooling하여 feature를 얻음.
        mask_flat = mask.view(mask.shape[0], -1)  # Flatten mask
        mask_feat = torch.sum(mask_flat, dim=-1)  # Pooling to get features per mask

        # 3. 결합된 텍스트 임베딩과 마스크 feature
        # 텍스트 임베딩과 마스크 feature를 결합하거나, 함께 처리하여 grounding tokens 생성
        grounding_tokens = torch.cat([label, mask_feat.unsqueeze(-1)], dim=-1)  # Concatenation or any custom logic

        batch['grounding_tokens'] = grounding_tokens  # Final grounding token output
        return batch
