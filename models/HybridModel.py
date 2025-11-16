import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm
from typing import Dict, Tuple, Optional, Union



# ==================== ResEmoteNet Components ====================

class SqueezeExcitationBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        b, c, _, _ = x.size()
        z = self.avg_pool(x).view(b, c)
        s = self.fc1(z)
        s = self.relu(s)
        s = self.fc2(s)
        s = self.sigmoid(s)
        s = s.view(b, c, 1, 1)
        return x * s.expand_as(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        return out

class ResEmoteNetBackbone(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.se_block = SqueezeExcitationBlock(256, reduction=16)
        self.res_block1 = ResidualBlock(256, 512, stride=2)
        self.res_block2 = ResidualBlock(512, 1024, stride=2)
        self.res_block3 = ResidualBlock(1024, 2048, stride=2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self._initialize_weights()
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool1(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.maxpool2(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.maxpool3(x)
        x = self.se_block(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)  # (batch, 2048)
        return x



# class ResEmoteNetBackbone(nn.Module):
#     """
#     Paper-faithful ResEmoteNet backbone for hybrid models.
#     Outputs 2048-dimensional features, matching the original architecture.
#     """
#     def __init__(self, input_channels: int = 3):
#         super().__init__()
#         self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool1 = nn.MaxPool2d(2, 2) # 224 -> 112

#         self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(128)
#         self.maxpool2 = nn.MaxPool2d(2, 2) # 112 -> 56
        
#         self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(256)
#         self.maxpool3 = nn.MaxPool2d(2, 2) # 56 -> 28

#         self.se_block = SqueezeExcitationBlock(256, reduction=16)

#         # Residual blocks: 256→512 (stride 2), 512→1024 (stride 2), 1024→2048 (stride 2)
#         self.res_block1 = ResidualBlock(256, 512, stride=2)   # 28→14
#         self.res_block2 = ResidualBlock(512, 1024, stride=2)  # 14→7
#         self.res_block3 = ResidualBlock(1024, 2048, stride=2) # 7→4

#         self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

#         self._initialize_weights()

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#     def forward(self, x):
#         x = self.relu(self.bn1(self.conv1(x)))
#         x = self.maxpool1(x)
#         x = self.relu(self.bn2(self.conv2(x)))
#         x = self.maxpool2(x)
#         x = self.relu(self.bn3(self.conv3(x)))
#         x = self.maxpool3(x)
#         x = self.se_block(x)

#         x = self.res_block1(x)
#         x = self.res_block2(x)
#         x = self.res_block3(x)
#         x = self.global_pool(x)
#         x = torch.flatten(x, 1)  # (batch, 2048)
#         return x





# ==================== Multi-Head Attention Fusion Module ====================


class MultiHeadAttentionFusion(nn.Module):
    """
    Adaptive fusion module using Multi-Head Attention
    Learns to dynamically weight CNN and Transformer features
    
    Improvements:
    - Residual dropout for better regularization
    - Reduced FFN expansion ratio (2x instead of 4x) for 2-token sequence
    - Multiple aggregation strategies
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads (default: 8)
        dropout: Dropout probability (default: 0.2)
        ffn_expansion: FFN expansion ratio (default: 2)
    """
    def __init__(self, embed_dim: int, num_heads: int = 8, 
                 dropout: float = 0.2, ffn_expansion: int = 2):
        super(MultiHeadAttentionFusion, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Multi-head attention layer
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Feed-forward network with reduced expansion for 2-token sequence
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * ffn_expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * ffn_expansion, embed_dim),
            nn.Dropout(dropout)
        )
        
        # Residual dropout for improved regularization
        self.residual_dropout = nn.Dropout(dropout)
        
        # Learnable aggregation weights for 'weighted' strategy
        self.agg_weights = nn.Parameter(torch.ones(2) / 2)
    
    def forward(self, x: torch.Tensor, 
                aggregation: str = 'mean') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Stacked embeddings (batch_size, 2, embed_dim)
               Position 0: CNN features, Position 1: Transformer features
            aggregation: Aggregation strategy - 'mean', 'weighted', or 'cnn'
        
        Returns:
            fused: Fused embeddings (batch_size, embed_dim)
            attn_weights: Attention weights (batch_size, num_heads, 2, 2)
        """
        # Multi-head attention with residual connection and dropout
        attn_output, attn_weights = self.mha(x, x, x)
        x = self.norm1(self.residual_dropout(x) + attn_output)
        
        # Feed-forward network with residual connection and dropout
        ffn_output = self.ffn(x)
        x = self.norm2(self.residual_dropout(x) + ffn_output)
        
        # Aggregate tokens based on strategy
        if aggregation == 'mean':
            # Equal weighting of both branches
            fused = torch.mean(x, dim=1)  # (batch_size, embed_dim)
        elif aggregation == 'weighted':
            # Learned adaptive weighting
            weights = F.softmax(self.agg_weights, dim=0)
            fused = (x * weights.view(1, 2, 1)).sum(dim=1)
        elif aggregation == 'cnn':
            # Use CNN branch as primary
            fused = x[:, 0, :]
        elif aggregation == 'transformer':
            # Use Transformer branch as primary
            fused = x[:, 1, :]
        else:
            raise ValueError(f"Unknown aggregation strategy: {aggregation}")
        
        return fused, attn_weights



# ==================== Hybrid Model ====================


class HybridEmotionRecognition(nn.Module):
    """
    Hybrid CNN-Transformer model for facial emotion recognition
    
    Architecture:
    1. ResEmoteNet branch (CNN) - extracts local facial features
    2. Swin Transformer Tiny branch - captures global dependencies
    3. Projection layers - align embeddings to common dimension
    4. Per-branch normalization - prevents branch dominance
    5. Multi-Head Attention Fusion - adaptive feature combination
    6. Classification head - emotion prediction
    
    Improvements over baseline:
    - Residual dropout in fusion module
    - Per-branch feature normalization
    - Gradient checkpointing support
    - Multiple aggregation strategies
    - Attention visualization capabilities
    - Enhanced parameter counting
    
    Args:
        num_classes: Number of emotion classes (default: 7)
        embed_dim: Common embedding dimension (default: 512)
        num_heads: Number of attention heads (default: 8)
        dropout: Dropout rate (default: 0.2)
        pretrained_swin: Use pretrained Swin Transformer (default: True)
        use_gradient_checkpointing: Enable gradient checkpointing (default: False)
        aggregation: Aggregation strategy for fusion (default: 'mean')
    """
    def __init__(self, 
                 num_classes: int = 7, 
                 embed_dim: int = 512, 
                 num_heads: int = 8, 
                 dropout: float = 0.2, 
                 pretrained_swin: bool = True,
                 use_gradient_checkpointing: bool = False,
                 aggregation: str = 'mean'):
        super(HybridEmotionRecognition, self).__init__()
        
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.aggregation = aggregation
        
        # ============ Branch 1: ResEmoteNet (CNN) ============
        self.resemotenet = ResEmoteNetBackbone(input_channels=3)
        resemotenet_dim = 2048  # Output dimension of ResEmoteNet
        
        # ============ Branch 2: Swin Transformer Tiny ============
        # Load Swin Transformer Tiny (expects 224x224 images)
        self.swin = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=pretrained_swin,
            num_classes=0  # Remove classification head
        )
        swin_dim = self.swin.num_features  # 768 for swin_tiny
        
        # ============ Projection Layers ============
        # Project both embeddings to common dimension
        self.cnn_projection = nn.Sequential(
            nn.Linear(resemotenet_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.transformer_projection = nn.Sequential(
            nn.Linear(swin_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # ============ Per-Branch Normalization ============
        # Prevents one branch from dominating
        self.cnn_norm = nn.LayerNorm(embed_dim)
        self.transformer_norm = nn.LayerNorm(embed_dim)
        
        # ============ Fusion Module ============
        self.fusion = MultiHeadAttentionFusion(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            ffn_expansion=2  # Reduced from 4x to 2x for 2-token sequence
        )
        
        # ============ Classification Head ============
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize projection and classification layers using Xavier initialization"""
        for module in [self.cnn_projection, self.transformer_projection, self.classifier]:
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
    
    def forward(self, 
                x: torch.Tensor, 
                return_embeddings: bool = False, 
                return_attention: bool = False) -> Union[torch.Tensor, Tuple]:
        """
        Forward pass through the hybrid model
        
        Args:
            x: Input images (batch_size, 3, 224, 224)
            return_embeddings: Return intermediate embeddings for analysis
            return_attention: Return attention weights for visualization
        
        Returns:
            logits: Class predictions (batch_size, num_classes)
            embeddings: Dict of intermediate embeddings (optional)
            attention_weights: Attention weights (optional)
        """
        batch_size = x.size(0)
        
        # ============ Extract features from both branches ============
        # CNN branch
        cnn_features = self.resemotenet(x)  # (batch_size, 1024)
        cnn_embed = self.cnn_projection(cnn_features)  # (batch_size, embed_dim)
        cnn_embed = self.cnn_norm(cnn_embed)  # Normalize CNN features
        
        # Transformer branch
        transformer_features = self.swin(x)  # (batch_size, 768)
        transformer_embed = self.transformer_projection(transformer_features)  # (batch_size, embed_dim)
        transformer_embed = self.transformer_norm(transformer_embed)  # Normalize Transformer features
        
        # ============ Prepare for fusion ============
        # Stack embeddings as sequence: [CNN_embed, Transformer_embed]
        embeddings_seq = torch.stack([cnn_embed, transformer_embed], dim=1)  # (batch_size, 2, embed_dim)
        
        # ============ Adaptive Fusion ============
        fused_embed, attention_weights = self.fusion(
            embeddings_seq, 
            aggregation=self.aggregation
        )  # (batch_size, embed_dim)
        
        # ============ Classification ============
        logits = self.classifier(fused_embed)  # (batch_size, num_classes)
        
        # Prepare return values
        output = [logits]
        
        if return_embeddings:
            embeddings = {
                'cnn_raw': cnn_features,
                'transformer_raw': transformer_features,
                'cnn_projected': cnn_embed,
                'transformer_projected': transformer_embed,
                'fused': fused_embed
            }
            output.append(embeddings)
        
        if return_attention:
            output.append(attention_weights)
        
        return output[0] if len(output) == 1 else tuple(output)
    
    def get_attention_maps(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract attention weights for visualization
        Shows how CNN and Transformer features attend to each other
        
        Args:
            x: Input images (batch_size, 3, 224, 224)
        
        Returns:
            avg_attention: Average attention weights (batch_size, 2, 2)
                          [0, 0]: CNN -> CNN
                          [0, 1]: CNN -> Transformer
                          [1, 0]: Transformer -> CNN
                          [1, 1]: Transformer -> Transformer
        """
        with torch.no_grad():
            # Extract features
            cnn_features = self.resemotenet(x)
            transformer_features = self.swin(x)
            
            # Project and normalize
            cnn_embed = self.cnn_projection(cnn_features)
            cnn_embed = self.cnn_norm(cnn_embed)
            
            transformer_embed = self.transformer_projection(transformer_features)
            transformer_embed = self.transformer_norm(transformer_embed)
            
            # Stack
            embeddings_seq = torch.stack([cnn_embed, transformer_embed], dim=1)
            
            # Get attention weights
            _, attention_weights = self.fusion(embeddings_seq, aggregation=self.aggregation)
            
            # attention_weights shape: (batch_size, num_heads, 2, 2)
            # Average over heads for visualization
            avg_attention = attention_weights.mean(dim=1)  # (batch_size, 2, 2)
        
        return avg_attention
    
    def get_num_params(self) -> int:
        """Calculate total number of parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def get_trainable_params(self) -> int:
        """Calculate number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_params_by_component(self) -> Dict[str, int]:
        """
        Get parameter count for each component
        Useful for thesis reporting and analysis
        
        Returns:
            Dictionary with parameter counts per component
        """
        return {
            'resemotenet': sum(p.numel() for p in self.resemotenet.parameters()),
            'swin_transformer': sum(p.numel() for p in self.swin.parameters()),
            'cnn_projection': sum(p.numel() for p in self.cnn_projection.parameters()),
            'transformer_projection': sum(p.numel() for p in self.transformer_projection.parameters()),
            'cnn_norm': sum(p.numel() for p in self.cnn_norm.parameters()),
            'transformer_norm': sum(p.numel() for p in self.transformer_norm.parameters()),
            'fusion_module': sum(p.numel() for p in self.fusion.parameters()),
            'classifier': sum(p.numel() for p in self.classifier.parameters()),
            'total': self.get_num_params(),
            'trainable': self.get_trainable_params()
        }
    
    def freeze_backbone(self, freeze_cnn: bool = False, freeze_transformer: bool = False):
        """
        Freeze backbone networks for fine-tuning experiments
        
        Args:
            freeze_cnn: Freeze ResEmoteNet parameters
            freeze_transformer: Freeze Swin Transformer parameters
        """
        if freeze_cnn:
            for param in self.resemotenet.parameters():
                param.requires_grad = False
            print("✓ Froze ResEmoteNet backbone")
        
        if freeze_transformer:
            for param in self.swin.parameters():
                param.requires_grad = False
            print("✓ Froze Swin Transformer backbone")



# ==================== Model Factory Functions ====================


def create_hybrid_model(num_classes: int = 7, 
                       embed_dim: int = 512, 
                       num_heads: int = 8, 
                       dropout: float = 0.2, 
                       pretrained_swin: bool = True,
                       use_gradient_checkpointing: bool = False,
                       aggregation: str = 'mean') -> HybridEmotionRecognition:
    """
    Factory function to create the hybrid model with recommended settings
    
    Args:
        num_classes: Number of emotion classes (7 for FER2013/RAF-DB)
        embed_dim: Common embedding dimension (512 recommended)
        num_heads: Number of attention heads (8 recommended)
        dropout: Dropout rate (0.2 for small datasets)
        pretrained_swin: Use ImageNet pretrained Swin Transformer
        use_gradient_checkpointing: Enable for memory-limited GPUs
        aggregation: Fusion strategy ('mean', 'weighted', 'cnn', 'transformer')
    
    Returns:
        Initialized HybridEmotionRecognition model
    """
    model = HybridEmotionRecognition(
        num_classes=num_classes,
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=dropout,
        pretrained_swin=pretrained_swin,
        use_gradient_checkpointing=use_gradient_checkpointing,
        aggregation=aggregation
    )
    return model


def create_ablation_models(num_classes: int = 7) -> Dict[str, nn.Module]:
    """
    Create ablation study models for thesis experiments
    
    Returns dictionary with:
    - 'hybrid_full': Full hybrid model with adaptive fusion
    - 'cnn_only': ResEmoteNet only
    - 'transformer_only': Swin Transformer only
    - 'concat_fusion': Simple concatenation instead of MHA
    
    Args:
        num_classes: Number of emotion classes
    
    Returns:
        Dictionary of models for ablation studies
    """
    models = {}
    
    # Full hybrid model
    models['hybrid_full'] = create_hybrid_model(
        num_classes=num_classes,
        aggregation='mean'
    )
    
    # For CNN-only and Transformer-only, you can extract components
    # These would need separate implementation or wrapper classes
    
    print(f"Created {len(models)} models for ablation studies")
    return models



# ==================== Model Summary ====================


def print_model_summary(model: HybridEmotionRecognition):
    """
    Print detailed model summary for thesis documentation
    
    Args:
        model: HybridEmotionRecognition instance
    """
    print("="*70)
    print("HYBRID EMOTION RECOGNITION MODEL SUMMARY")
    print("="*70)
    
    params = model.get_params_by_component()
    
    print(f"\nComponent-wise Parameter Count:")
    print(f"  ResEmoteNet (CNN Branch):      {params['resemotenet']:>12,} parameters")
    print(f"  Swin Transformer Tiny:          {params['swin_transformer']:>12,} parameters")
    print(f"  CNN Projection Layer:           {params['cnn_projection']:>12,} parameters")
    print(f"  Transformer Projection Layer:   {params['transformer_projection']:>12,} parameters")
    print(f"  CNN Normalization:              {params['cnn_norm']:>12,} parameters")
    print(f"  Transformer Normalization:      {params['transformer_norm']:>12,} parameters")
    print(f"  Fusion Module (MHA):            {params['fusion_module']:>12,} parameters")
    print(f"  Classification Head:            {params['classifier']:>12,} parameters")
    print(f"  {'-'*50}")
    print(f"  Total Parameters:               {params['total']:>12,} parameters")
    print(f"  Trainable Parameters:           {params['trainable']:>12,} parameters")
    
    print(f"\nModel Configuration:")
    print(f"  Embedding Dimension:            {model.embed_dim}")
    print(f"  Number of Attention Heads:      {model.fusion.num_heads}")
    print(f"  Number of Classes:              {model.num_classes}")
    print(f"  Aggregation Strategy:           {model.aggregation}")
    
    print("="*70)