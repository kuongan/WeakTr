from torch import nn
import torch


# 1. using attention feature to generate dynamic weight
class AAF(nn.Module):
    def __init__(self, channel, reduction=16, feats_channel=64, feat_reduction=8, pool="avg"):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if pool == "max":
            self.avg_pool = nn.AdaptiveMaxPool2d(1)
        self.attn_head_ffn = nn.Sequential(
            nn.Linear(channel, int(channel / reduction), bias=False),
            nn.ReLU(inplace=True),  # inplace=True sometimes slightly decrease the memory usage
            # nn.Sigmoid(),
            nn.Linear(int(channel / reduction), channel, bias=False),
            nn.Sigmoid()
        )
        self.attn_feat_ffn = nn.Sequential(
                                    nn.Linear(feats_channel, int(feats_channel / feat_reduction)),
                                    nn.Linear(int(feats_channel / feat_reduction), 1),
                                )
    
    def forward_weight(self, x):
        b, c, n, m = x.size() # batchsize, attn heads num=72, class tokens + patch tokens, embedding_dim=64

        # 1. pooling for tokens
        x = x.permute(0, 1, 3, 2).contiguous().view(b, c*m, n, 1) 
        attn_feat_pool = self.avg_pool(x)

        # 2. FFN for channels, generate dynamic weight
        attn_feat_pool = attn_feat_pool.view(b*c, m)
        attn_weight = self.attn_feat_ffn(attn_feat_pool)

        # 3. FFN for attn heads generate last weight
        attn_weight = attn_weight.view(b, c)
        attn_weight = self.attn_head_ffn(attn_weight).view(b, c, -1, 1)

        return attn_weight

    def forward(self, attn_feat, x):
        weight = self.forward_weight(attn_feat)
        return x * weight.expand_as(x), x * weight.expand_as(x)

# 2. using randomly initialized weight to generate dynamic weight
class AAF_RandWeight(AAF):
    def __init__(self, channel, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.query = torch.randn(1, channel, requires_grad=False).cuda()
    
    def forward_weight(self, x):
        b, c, n, m = x.size() # batchsize, attn heads num=72, class tokens + patch tokens, embedding_dim=64

        attn_weight = self.attn_head_ffn(self.query.expand(b, -1)).unsqueeze(2).unsqueeze(3)

        return attn_weight
    
    
    # 3. New CAAF module: Class-Contrastive Adaptive Attention Fusion
class CAAF(nn.Module):
    def __init__(self, channel, num_classes=20, reduction=16, feats_channel=64, feat_reduction=8, pool="avg"):
        super().__init__()
        self.num_classes = num_classes
        # Spatial pooling (original AAF component)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if pool == "max":
            self.avg_pool = nn.AdaptiveMaxPool2d(1)
        
        # Feature transformation for attention features
        self.attn_feat_ffn = nn.Sequential(
            nn.Linear(feats_channel, int(feats_channel / feat_reduction)),
            nn.ReLU(inplace=True),
            nn.Linear(int(feats_channel / feat_reduction), 1)
        )
        print("channel: ", channel)
        # Class-specific head weight generator
        self.class_specific_head_weight = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel * num_classes),
            nn.Tanh()  # Allow both suppression and enhancement
        )
        
        # Head reliability estimator
        self.head_reliability = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, 1),
            nn.Sigmoid()
        )
        
        # Class feature extractor (to determine class relevance)
        self.class_feature_extractor = nn.Sequential(
            nn.Linear(feats_channel, feats_channel // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feats_channel // 2, num_classes)
        )
        
        # Class contrast module
        self.class_contrast = nn.Sequential(
            nn.Linear(2 * channel, 1),
            nn.Sigmoid()
        )

        
        # Store last class relevance for contrastive loss
        self.register_buffer('last_class_relevance', torch.zeros(1, channel, num_classes))
    
    def forward(self, attn_feat, x):
        b, c, n, m = attn_feat.size()  # [batch, num_heads, num_tokens, feat_dim]
        
        # 1. Base attention weight computation (similar to original AAF)
        x_permute = attn_feat.permute(0, 1, 3, 2).contiguous().view(b, c*m, n, 1)
        attn_feat_pool = self.avg_pool(x_permute)
        attn_feat_pool = attn_feat_pool.view(b*c, m)
        attn_base = self.attn_feat_ffn(attn_feat_pool)
        attn_base = attn_base.view(b, c)
        
        # 2. Extract class relevance for each attention head
        # First, compute head representations
        head_repr = attn_feat.mean(dim=2)  # [b, c, m]
        
        # For each head, compute its relevance to each class
        class_relevance = []
        for i in range(c):
            head_i_feat = head_repr[:, i, :]  # [b, m]
            rel_i = self.class_feature_extractor(head_i_feat)  # [b, num_classes]
            class_relevance.append(rel_i)
        
        class_relevance = torch.stack(class_relevance, dim=1)  # [b, c, num_classes]
        
        # Store for contrastive loss computation
        if self.training:
            self.last_class_relevance = class_relevance.detach()
        
        # 3. Generate class-specific weights
        # Compute global head representation
        global_head_repr = attn_feat.mean(dim=2).mean(dim=2)  # [b, c]
        
        # Generate contrastive weights
        class_weights = self.class_specific_head_weight(global_head_repr)  # [b, c*num_classes]
        class_weights = class_weights.view(b, self.num_classes, c)  # [b, num_classes, c]
        
        # 4. Apply contrastive mechanism
        # For each head, compute positive and negative class contribution
        positive_relevance = torch.softmax(class_relevance, dim=2)  # [b, c, num_classes]
        negative_relevance = 1 - positive_relevance  # [b, c, num_classes]
        
        # Compute contrastive score by comparing positive and negative relevance
        contrastive_input = torch.cat([
            positive_relevance.permute(0, 2, 1),  # [b, num_classes, c]
            negative_relevance.permute(0, 2, 1)   # [b, num_classes, c]
        ], dim=2)  # [b, num_classes, 2*c]
        
        contrastive_input = contrastive_input.reshape(b * self.num_classes, 2 * c)
        contrastive_score = self.class_contrast(contrastive_input)  # [b*num_classes, 1]
        contrastive_score = contrastive_score.reshape(b, self.num_classes, 1)  # [b, num_classes, 1]
        
        # 5. Estimate head reliability
        head_rel = self.head_reliability(global_head_repr)  # [b, 1]
        print(f"class_weights shape: {class_weights.shape}")
        print(f"head_rel shape: {head_rel.shape}")
        print(f"contrastive_score shape: {contrastive_score.shape}")
        head_rel = head_rel.unsqueeze(2).expand(-1, 20, 1) 
        
        # 6. Generate final class-specific attention weights
        # Apply reliability-based scaling and contrastive enhancement
        refined_weights = class_weights * head_rel * contrastive_score  # [b, num_classes, c]
        
        # 7. Apply weights to generate class-specific attention maps
        weighted_maps = []
        for i in range(self.num_classes):
            w = refined_weights[:, i, :].view(b, c, 1, 1)  # [b, c, 1, 1]
            weighted_map = x * w  # [b, c, n, n]
            weighted_maps.append(weighted_map)
        
        weighted_x_all = torch.stack(weighted_maps, dim=1)  # [b, num_classes, c, n, n]
        
        # For compatibility with original interface, return first class and all classes
        return weighted_maps[0], weighted_x_all
    
    def compute_contrastive_loss(self, class_relevance, targets):
        """
        Compute additional contrastive loss to enhance class-specific attention
        
        Args:
            class_relevance: Tensor of shape [b, c, num_classes] - relevance of each head to each class
            targets: Tensor of shape [b, num_classes] - binary targets indicating presence of classes
            
        Returns:
            contrastive_loss: Scalar loss value
        """
        b, c, num_classes = class_relevance.shape
        
        # For each class, compute positive and negative samples
        pos_mask = targets.unsqueeze(1).expand(-1, c, -1)  # [b, c, num_classes]
        neg_mask = 1 - pos_mask
        
        # For positive classes, increase relevance
        pos_relevance = (class_relevance * pos_mask).sum(dim=2) / (pos_mask.sum(dim=2) + 1e-6)
        
        # For negative classes, decrease relevance
        neg_relevance = (class_relevance * neg_mask).sum(dim=2) / (neg_mask.sum(dim=2) + 1e-6)
        
        # Contrastive loss: maximize positive relevance, minimize negative relevance
        contrastive_loss = -torch.log(pos_relevance + 1e-6).mean() - torch.log(1 - neg_relevance + 1e-6).mean()
        
        return contrastive_loss