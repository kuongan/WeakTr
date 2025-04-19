import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
from functools import partial

from unilm.beit3.modeling_utils import _get_base_config, BEiT3Wrapper
from AAF import AAF, AAF_RandWeight

from transformers import XLMRobertaTokenizer


categories = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
              'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
              'tvmonitor']
categories_coco = ['background',
                   'person',
                   'bicycle',
                   'car',
                   'motorcycle',
                   'airplane',
                   'bus',
                   'train',
                   'truck',
                   'boat',
                   'traffic light',
                   'fire hydrant',
                   'street sign',
                   'stop sign',
                   'parking meter',
                   'bench',
                   'bird',
                   'cat',
                   'dog',
                   'horse',
                   'sheep',
                   'cow',
                   'elephant',
                   'bear',
                   'zebra',
                   'giraffe',
                   'hat',
                   'backpack',
                   'umbrella',
                   'shoe',
                   'eye glasses',
                   'handbag',
                   'tie',
                   'suitcase',
                   'frisbee',
                   'skis',
                   'snowboard',
                   'sports ball',
                   'kite',
                   'baseball bat',
                   'baseball glove',
                   'skateboard',
                   'surfboard',
                   'tennis racket',
                   'bottle',
                   'plate',
                   'wine glass',
                   'cup',
                   'fork',
                   'knife',
                   'spoon',
                   'bowl',
                   'banana',
                   'apple',
                   'sandwich',
                   'orange',
                   'broccoli',
                   'carrot',
                   'hot dog',
                   'pizza',
                   'donut',
                   'cake',
                   'chair',
                   'couch',
                   'potted plant',
                   'bed',
                   'mirror',
                   'dining table',
                   'window',
                   'desk',
                   'toilet',
                   'door',
                   'tv',
                   'laptop',
                   'mouse',
                   'remote',
                   'keyboard',
                   'cell phone',
                   'microwave',
                   'oven',
                   'toaster',
                   'sink',
                   'refrigerator',
                   'blender',
                   'book',
                   'clock',
                   'vase',
                   'scissors',
                   'teddy bear',
                   'hair drier',
                   'toothbrush']


class BEiT3Segmentation(BEiT3Wrapper):
    def __init__(
        self,
        arg=None,
        num_classes=20,
        reduction=4,
        pool_type="avg",
        AdaptiveAttentionFusion=AAF,
        feat_reduction=4,
        class_names=None,
        dataset='voc',
        **kwargs
    ):
        assert arg is not None, "Missing backbone config `arg`"
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
        if dataset == 'voc':
            self.class_names = categories[:num_classes]
        elif dataset == 'coco':
            self.class_names = categories_coco[:num_classes]
        self.tokenizer = XLMRobertaTokenizer("beit3.spm")
        
        # Projecteur textuel pour adapter les embeddings textuels à l'espace visuel si nécessaire
        self.text_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self._init_weights(self.text_proj)
        
        # Initialize cls_token as a temporary placeholder (will be set in forward)
        self.cls_token = nn.Parameter(torch.zeros(1, self.num_classes, self.embed_dim))
        trunc_normal_(self.cls_token, std=.02)
        
        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_classes, self.embed_dim))
        trunc_normal_(self.pos_embed, std=.02)

        feats_channel = self.embed_dim // self.num_heads
        aaf_params = dict(
            channel=self.depth * self.num_heads,
            reduction=reduction,
            feats_channel=feats_channel,
            feat_reduction=feat_reduction,
            pool=pool_type
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
    
    def initialize_semantic_class_tokens(self, targets=None):
        """Initialize class tokens using text embeddings from BEiT3's text encoder"""
        device = next(self.beit3.parameters()).device
        
        # Always create prompts for all classes
        text_prompts = [f"a photo of a {name}" for name in self.class_names]
        
        # Tokenize text prompts
        with torch.no_grad():
            tokenized_inputs = self.tokenizer(
                text_prompts,
                padding="max_length",
                truncation=True,
                max_length=32,
                return_tensors="pt"
            ).to(device)
            
            # Get text embeddings from BEiT3 text encoder
            text_embeddings = self.beit3.text_embed(tokenized_inputs["input_ids"])
            
            if len(text_embeddings.shape) > 2:
                text_embeddings = text_embeddings.mean(dim=1)
            
            projected_embeddings = self.text_proj(text_embeddings)
            print(f"projected_embeddings shape: {projected_embeddings.shape}")
            assert projected_embeddings.shape[0] == self.num_classes, \
                f"Expected {self.num_classes} embeddings but got {projected_embeddings.shape[0]}"
            
            # Reshape to [1, num_classes, embed_dim] for cls_token format
            cls_tokens = projected_embeddings.unsqueeze(0)
            print(f"cls_tokens shape: {cls_tokens.shape}")
            return cls_tokens
    def extract_features(self, x, cls_tokens=None):
        B, C, H, W = x.shape

        # Use the BEiT3 vision embedding to convert the image to patch tokens
        encoder_out = self.beit3.vision_embed(x)
        patch_tokens = encoder_out[:, 1:, :]  # [B, N, D]            
        combined_tokens = torch.cat((cls_tokens, patch_tokens), dim=1)  # [B, num_classes + N, D]
        print(f"combined_tokens shape: {combined_tokens.shape}")
        # Add positional embeddings
        pos_embed = self.interpolate_pos_encoding(combined_tokens, H, W)
        x = combined_tokens + pos_embed
        
        # Store attention weights and encoder states
        attn_weights_list = []
        encoder_states = []
        
        # Process through each encoder layer
        for i, layer in enumerate(self.beit3.encoder.layers):
            # The EncoderLayer's forward method returns x, attn_weights, l_aux
            x, attn_weights, _ = layer(
                x,
                encoder_padding_mask=None,
                attn_mask=None,
                rel_pos=None
            )
            encoder_states.append(x)
            attn_weights_list.append(attn_weights)
        
        return x[:, :self.num_classes], x[:, self.num_classes:], attn_weights_list, encoder_states        

    def forward(self, x, targets=None, return_att=False, attention_type='fused'):
        B, C, H, W = x.shape
        
        # Using class tokens
        if targets is not None:
            # Generate semantic class tokens
            with torch.no_grad():  # We don't want to backprop through text encoder
                semantic_tokens = self.initialize_semantic_class_tokens(targets)
                semantic_tokens = semantic_tokens.to(x.device)
                print('semantic_tokens shape:', semantic_tokens.shape)
                cls_tokens = semantic_tokens.expand(B, -1, -1)  # [B, num_classes, D]
        else:
            # Use the default cls_token
            cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, num_classes, D]
        
        # Extract features using the provided class tokens
        x_cls, x_patch, attn_weights, attn_feats = self.extract_features(x, cls_tokens)
        
        w0, h0 = H // self.patch_size, W // self.patch_size
        if x_patch.shape[1] > (w0 * h0):
            x_patch = x_patch[:, : (w0 * h0), :]

        x_patch = x_patch.view(B, w0, h0, self.embed_dim).permute(0, 3, 1, 2)
        x_patch = self.head(x_patch)

        coarse_cam_pred = self.avgpool(x_patch).squeeze(-1).squeeze(-1)

        attn_weights = torch.stack(attn_weights)  # [L, H, B, N, N]
        attn_feats = torch.stack(attn_feats)      # [L, B, N, embed_dim]

        L, H, B, N, _ = attn_weights.shape
        attn_weights = attn_weights.permute(2, 1, 0, 3, 4).reshape(B, H * L, N, N)
        
        L2, B3, N2, C2 = attn_feats.shape
        head_dim = C2 // self.num_heads

        attn_feats = attn_feats.view(L2, B3, N2, self.num_heads, head_dim).permute(1, 3, 0, 2, 4)
        attn_feats = attn_feats.reshape(B3, self.num_heads * L2, N2, head_dim)
        
        cross_attn_map, patch_attn_map = self.adaptive_attention_fusion(attn_feats, attn_weights)
        
        coarse_cam = F.relu(x_patch)
        
        n, c, h, w = coarse_cam.shape 
        
        cross_attn = cross_attn_map.mean(1)[:, 0:self.num_classes, self.num_classes:].reshape([n, c, h, w])
        
        if attention_type == 'fused':
            cams = cross_attn * coarse_cam  # B * C * 14 * 14
        elif attention_type == 'patchcam':
            cams = coarse_cam
        else:
            cams = cross_attn      # [B, C, N]
            
        patch_attn = patch_attn_map.mean(1)[:, self.num_classes:, self.num_classes:]   # [B, N, N]
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
    # Tách các tham số backbone ra khỏi kwargs
    img_size = kwargs.pop("img_size", 224)
    patch_size = kwargs.pop("patch_size", 16)
    drop_path_rate = kwargs.pop("drop_path_rate", 0.1)
    mlp_ratio = kwargs.pop("mlp_ratio", 4.0)
    vocab_size = kwargs.pop("vocab_size", 64010)
    checkpoint_activations = kwargs.pop("checkpoint_activations", False)

    # Tạo config cho backbone
    args_cfg = _get_base_config(
        img_size=img_size,
        patch_size=patch_size,
        drop_path_rate=drop_path_rate,
        mlp_ratio=mlp_ratio,
        vocab_size=vocab_size,
        checkpoint_activations=checkpoint_activations,
    )

    # Tạo model segmentation
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
        arg=args_cfg,  # Fix: Changed from config to arg
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