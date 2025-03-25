import torch
import torch.nn as nn
# from ldm.modules.attention import BasicTransformerBlock
# from ldm.modules.diffusionmodules.util import checkpoint, FourierEmbedder
import torch.nn.functional as F
# from ..attention import SelfAttention, FeedForward
# from .convnext import convnext_tiny


# class PositionNet(nn.Module):
#     def __init__(self, resize_input=448, in_dim=152, out_dim=768):
#         super().__init__()
        
#         self.resize_input = resize_input
#         self.down_factor = 32 # determined by the convnext backbone 
#         self.out_dim = out_dim
#         assert self.resize_input % self.down_factor == 0
        
#         self.in_conv = nn.Conv2d(in_dim,3,3,1,1) # from num_sem to 3 channels
#         self.convnext_tiny_backbone = convnext_tiny(pretrained=True)
        
#         self.num_tokens = (self.resize_input // self.down_factor) ** 2
        
#         convnext_feature_dim = 768
#         self.pos_embedding = nn.Parameter(torch.empty(1, self.num_tokens, convnext_feature_dim).normal_(std=0.02))  # from BERT
      
#         self.linears = nn.Sequential(
#             nn.Linear( convnext_feature_dim, 512),
#             nn.SiLU(),
#             nn.Linear( 512, 512),
#             nn.SiLU(),
#             nn.Linear(512, out_dim),
#         )

#         self.null_feature = torch.nn.Parameter(torch.zeros([convnext_feature_dim]))


#     def forward(self, sem, mask):
#         B = sem.shape[0] 

#         # token from edge map 
#         sem = torch.nn.functional.interpolate(sem, self.resize_input, mode="nearest")
#         sem = self.in_conv(sem)
#         sem_feature = self.convnext_tiny_backbone(sem)
#         objs = sem_feature.reshape(B, -1, self.num_tokens)
#         objs = objs.permute(0, 2, 1) # N*Num_tokens*dim

#         # expand null token
#         null_objs = self.null_feature.view(1,1,-1)
#         null_objs = null_objs.repeat(B,self.num_tokens,1)
        
#         # mask replacing 
#         mask = mask.view(-1,1,1)
#         objs = objs*mask + null_objs*(1-mask)
        
#         # add pos 
#         objs = objs + self.pos_embedding

#         # fuse them 
#         objs = self.linears(objs)

#         assert objs.shape == torch.Size([B,self.num_tokens,self.out_dim])        
#         return objs


class PositionNetRegionPooling(nn.Module):
    def __init__(self, in_dim=1, label_embed_dim=768, out_dim=768, num_tokens=77):
        super().__init__()
        self.num_tokens = num_tokens
        # 작은 CNN으로 mask에서 feature 추출
        self.mask_cnn = nn.Sequential(
            nn.Conv2d(in_dim, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, out_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        # mask의 spatial dimension을 num_tokens(77) 길이의 시퀀스로 만들기 위해 adaptive pooling 사용
        # 예: 입력 mask [B, 1, 256, 256] → CNN → [B, output_dim, 16, 16]
        # 이를 [B, output_dim, 1, 77]로 바꾼 후 시퀀스 형태로 변환
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, num_tokens))
        # label embedding을 projection
        self.label_proj = nn.Linear(label_embed_dim, out_dim)
    
    def forward(self, mask, label):
        # mask: [B, 256, 256] 이므로 채널 차원 추가
        mask = mask.unsqueeze(1)  # → [B, 1, 256, 256]
        mask_feat = self.mask_cnn(mask)  # → [B, output_dim, H, W], 예를 들어 [B, 768, 16, 16]
        mask_feat = self.adaptive_pool(mask_feat)  # → [B, output_dim, 1, num_tokens]
        mask_feat = mask_feat.squeeze(2)  # → [B, output_dim, num_tokens]
        mask_feat = mask_feat.transpose(1, 2)  # → [B, num_tokens, output_dim]
        
        # label embedding: [B, 77, 768]를 projection
        label_feat = self.label_proj(label)  # → [B, 77, output_dim]
        
        # 두 feature를 결합 (여기서는 element-wise sum 사용)
        grounding_input = label_feat + mask_feat  # [B, 77, output_dim]
        # 또는, concatenate하는 방식: grounding_input = torch.cat([label_feat, mask_feat], dim=-1)
        
        return grounding_input