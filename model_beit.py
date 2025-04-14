import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
from functools import partial

from unilm.beit3.modeling_utils import _get_base_config, BEiT3Wrapper
from AAF import AAF, AAF_RandWeight

class BEiT3Segmentation(BEiT3Wrapper):
    def __init__(
        self,
        arg=None,
        num_classes=20,
        reduction=4,
        pool="avg",
        AdaptiveAttentionFusion=None,
        feat_reduction=4,
        **kwargs
    ):
        super(BEiT3Segmentation, self).__init__(args=arg)

        self.embed_dim = arg.encoder_embed_dim
        self.num_classes = num_classes
        self.patch_size = arg.patch_size
        self.num_heads = arg.encoder_attention_heads
        self.depth = arg.encoder_layers
        num_patches = self.beit3.vision_embed.num_patches
        self.head = nn.Conv2d(self.embed_dim, self.num_classes, kernel_size=3, stride=1, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head.apply(self._init_weights)

        self.cls_token = nn.Parameter(torch.zeros(1, self.num_classes, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_classes, self.embed_dim))  # Placeholder 1000 patches
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)

        feats_channel = self.embed_dim // self.num_heads
        aaf_params = dict(
            channel=self.depth * self.num_heads,
            reduction=reduction,
            feats_channel=feats_channel,
            feat_reduction=feat_reduction,
            pool=pool
        )
        self.adaptive_attention_fusion = AdaptiveAttentionFusion(**aaf_params) if AdaptiveAttentionFusion else None
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
                
    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - self.num_classes
        N = self.pos_embed.shape[1] - self.num_classes
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0:self.num_classes]
        patch_pos_embed = self.pos_embed[:, self.num_classes:]
        dim = x.shape[-1]

        w0 = w // self.beit3.vision_embed.patch_size[0]
        h0 = h // self.beit3.vision_embed.patch_size[0]
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)
        
    def extract_features(self, x):
        B, C, H, W = x.shape

        # Use the BEiT3 vision embedding to convert the image to patch tokens
        encoder_out = self.beit3.vision_embed(x)
        patch_tokens = encoder_out[:, 1:, :]                    # [B, N, D]
        cls_tokens = self.cls_token.expand(B, -1, -1)           # [B, num_classes, D]
        combined_tokens = torch.cat((cls_tokens, patch_tokens), dim=1)  # [B, 20 + 196, D]
        
        # Add positional embeddings
        pos_embed = self.interpolate_pos_encoding(combined_tokens, H, W)
        x = combined_tokens + pos_embed
        
        # This will store attention weights and encoder states for each layer
        attn_weights_list = []
        encoder_states = []
        
        # Now manually process through each encoder layer
        # Access the encoder layers from self.beit3.encoder.layers
        for layer in self.beit3.encoder.layers:
            # The EncoderLayer's forward method returns x, attn_weights, l_aux
            x, attn_weights, _ = layer(
                x,
                encoder_padding_mask=None,  # No padding mask needed for fully visible tokens
                attn_mask=None,             # No attention mask needed
                rel_pos=None                # No relative position bias
            )
            encoder_states.append(x)
            attn_weights_list.append(attn_weights)
        
        return x[:, :self.num_classes], x[:, self.num_classes:], attn_weights_list, encoder_states


    def forward(self, x, return_att=False, attention_type='fused'):
        B, C, W, H = x.shape
        x_cls, x_patch, attn_weights, attn_feats = self.extract_features(x)
        #print("x_cls", x_cls.shape, "x_patch", x_patch.shape, "attn_weights", len(attn_weights))
        n, p, c = x_patch.shape
        if W != H:
            w0 = W // self.patch_size
            h0 = H // self.patch_size
            x_patch = torch.reshape(x_patch, [n, w0, h0, c])
        else:
            x_patch = torch.reshape(x_patch, [n, int(p ** 0.5), int(p ** 0.5), c])
        x_patch = x_patch.permute([0, 3, 1, 2])
        x_patch = x_patch.contiguous()
        #print("x_patch", x_patch.shape)  # B, Dim, H, W 4,768,14,14  
        x_patch = self.head(x_patch)
        #print("x_patch", x_patch.shape)     # B, C, H, W 4,20,14,14
        coarse_cam_pred = self.avgpool(x_patch).squeeze(3).squeeze(2)
    
        attn_weights = torch.stack(attn_weights)  # [L, H, B, N, N]
        attn_feats = torch.stack(attn_feats)      # [L, B, N, embed_dim]
        #print("attn_weights", attn_weights.shape) #([12, 12, 4, 216, 216])
        #print("attn_feats", attn_feats.shape)      #([12, 4, 216, 768])
        attn_weights_detach = attn_weights.detach().clone()
        k, h, b, n, m = attn_weights_detach.shape
        attn_weights_detach = attn_weights_detach.permute([2, 1, 0, 3, 4]).contiguous()
        attn_weights_detach = attn_weights_detach.view(b, h * k, n, m)
        #attn_weights = attn_weights.permute(2, 1, 0, 3, 4).reshape(B, H * L, N, N)
        #print("attn_weights", attn_weights_detach.shape) #([4, 144, 216, 216])
        
        attn_feats_detach = attn_feats.detach().clone()
        #print("attn_feats_detach", attn_feats_detach.shape) #([12, 4, 216, 768])
        k, b, n, c = attn_feats_detach.shape
        attn_feats_detach = attn_feats_detach.view(k, b, n, -1, h)
        attn_feats_detach = attn_feats_detach.permute([1, 4, 0, 2, 3]).contiguous()
        attn_feats_detach = attn_feats_detach.view(b, h * k, n, -1)
        #L2, B3, N2, C2 = attn_feats.shape
        #head_dim = C2 // self.num_heads

        #attn_feats = attn_feats.view(L2, B3, N2, self.num_heads, head_dim).permute(1, 3, 0, 2, 4)
        #attn_feats = attn_feats.reshape(B3, self.num_heads * L2, N2, head_dim)
        #print("attn_feats_detach", attn_feats_detach.shape) #([4, 144, 216, 64])
        cross_attn_map, patch_attn_map = self.adaptive_attention_fusion(attn_feats_detach, attn_weights_detach)
        #print("cross_attn_map", cross_attn_map.shape, "patch_attn_map", patch_attn_map.shape)
        coarse_cam = F.relu(x_patch.detach().clone())
        
        n, c, h, w = coarse_cam.shape 
        
        cross_attn = cross_attn_map.mean(1)[:, 0:self.num_classes, self.num_classes:].reshape([n, c, h, w])
        
        if attention_type == 'fused':
            cams = cross_attn * coarse_cam  # B * C * 14 * 14
        elif attention_type == 'patchcam':
            cams = coarse_cam
        else:
            cams = cross_attn      #[2,20,196]    # [B, C, N]
            
        patch_attn = patch_attn_map.mean(1)[:, self.num_classes:, self.num_classes:]   #[2,196,196]   # [B, N, N]
        fine_cam = torch.matmul(patch_attn.unsqueeze(1), cams.view(cams.shape[0],
                                                                        cams.shape[1], -1, 1)). \
            reshape(cams.shape[0], cams.shape[1], h, w)

        fine_cam_pred = self.avgpool(fine_cam).squeeze(3).squeeze(2)
        patch_attn = patch_attn.unsqueeze(0)
        cls_token_pred = x_cls.mean(-1)  # [B, C]

        # Return
        if return_att:
            return cls_token_pred, cams, patch_attn
        else:
            return cls_token_pred, coarse_cam_pred, fine_cam_pred



@register_model
def beit3_base_wsss_patch16_224(pretrained=False, **kwargs):
    args_cfg = _get_base_config(img_size=224, patch_size=16)
    model = BEiT3Segmentation(
        arg=args_cfg,
        AdaptiveAttentionFusion=AAF,
        **kwargs
    )
    return model

@register_model
def beit3_base_wsss_aaf_randweight_patch16_224(pretrained=False, **kwargs):
    args_cfg = _get_base_config(img_size=224, patch_size=16)
    model = BEiT3Segmentation(
        config=args_cfg,
        AdaptiveAttentionFusion=AAF_RandWeight,
        **kwargs
    )
    return model

def create_model(model_name, pretrained=False, **kwargs):
    if model_name == "beit3_base_wsss_patch16_224":
        return beit3_base_wsss_patch16_224(pretrained=pretrained, **kwargs)
    elif model_name == "beit3_base_wsss_aaf_randweight_patch16_224":
        return beit3_base_wsss_aaf_randweight_patch16_224(pretrained=pretrained, **kwargs)
    else:
        raise NotImplementedError(f"Model {model_name} not implemented")