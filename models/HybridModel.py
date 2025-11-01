import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm


# ==================== ResEmoteNet Components ====================

class SqueezeExcitationBlock(nn.Module):
    """Squeeze and Excitation Block for channel attention"""
    def __init__(self, channels, reduction=16):
        super(SqueezeExcitationBlock, self).__init__()
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
    """Residual Block with skip connection"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


class ResEmoteNetBackbone(nn.Module):
    """
    ResEmoteNet without classification head - extracts embeddings
    Modified to output feature embeddings instead of class predictions
    """
    def __init__(self, input_channels=3):
        super(ResEmoteNetBackbone, self).__init__()
        
        # Initial Convolutional Network
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Squeeze and Excitation Block
        self.se_block = SqueezeExcitationBlock(256, reduction=16)
        
        # Residual Blocks
        self.res_block1 = ResidualBlock(256, 256)
        self.res_block2 = ResidualBlock(256, 512, stride=2)
        self.res_block3 = ResidualBlock(512, 1024, stride=2)
        
        # Adaptive Average Pooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', 
                                       nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Convolutional layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.maxpool3(x)
        
        # SE block
        x = self.se_block(x)
        
        # Residual blocks
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        
        # Global pooling
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        
        return x  # Returns 1024-dim embedding


# ==================== Multi-Head Attention Fusion Module ====================

class MultiHeadAttentionFusion(nn.Module):
    """
    Adaptive fusion module using Multi-Head Attention
    Learns to weight CNN and Transformer features dynamically
    """
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
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
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, num_tokens, embed_dim)
               where num_tokens = 2 (CNN + Transformer embeddings)
        Returns:
            fused: Tensor of shape (batch_size, embed_dim)
        """
        # Multi-head attention with residual connection
        attn_output, attn_weights = self.mha(x, x, x)
        x = self.norm1(x + attn_output)
        
        # Feed-forward network with residual connection
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        
        # Aggregate tokens (mean pooling)
        fused = torch.mean(x, dim=1)  # (batch_size, embed_dim)
        
        return fused, attn_weights


# ==================== Hybrid Model ====================

class HybridEmotionRecognition(nn.Module):
    """
    Hybrid CNN-Transformer model for facial emotion recognition
    
    Architecture:
    1. ResEmoteNet branch (CNN) - extracts local features
    2. Swin Transformer Tiny branch - captures global dependencies
    3. Projection layers - align both embeddings to common dimension
    4. Multi-Head Attention Fusion - adaptive feature combination
    5. Classification head - predicts emotions
    
    Args:
        num_classes: Number of emotion classes (default: 7)
        embed_dim: Common embedding dimension (default: 512)
        num_heads: Number of attention heads (default: 8)
        dropout: Dropout rate (default: 0.1)
        pretrained_swin: Use pretrained Swin Transformer (default: True)
    """
    def __init__(self, num_classes=7, embed_dim=512, num_heads=8, 
                 dropout=0.1, pretrained_swin=True):
        super(HybridEmotionRecognition, self).__init__()
        
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        # ============ Branch 1: ResEmoteNet (CNN) ============
        self.resemotenet = ResEmoteNetBackbone(input_channels=3)
        resemotenet_dim = 1024  # Output dimension of ResEmoteNet
        
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
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        self.transformer_projection = nn.Sequential(
            nn.Linear(swin_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # ============ Fusion Module ============
        self.fusion = MultiHeadAttentionFusion(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # ============ Classification Head ============
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize projection and classification layers"""
        for m in [self.cnn_projection, self.transformer_projection, self.classifier]:
            for layer in m.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
    
    def forward(self, x, return_embeddings=False, return_attention=False):
        """
        Args:
            x: Input images (batch_size, 3, 224, 224)
            return_embeddings: If True, return intermediate embeddings
            return_attention: If True, return attention weights
        
        Returns:
            logits: Class predictions (batch_size, num_classes)
            Optional: embeddings dict, attention weights
        """
        batch_size = x.size(0)
        
        # ============ Extract features from both branches ============
        # CNN branch
        cnn_features = self.resemotenet(x)  # (batch_size, 1024)
        cnn_embed = self.cnn_projection(cnn_features)  # (batch_size, embed_dim)
        
        # Transformer branch
        transformer_features = self.swin(x)  # (batch_size, 768)
        transformer_embed = self.transformer_projection(transformer_features)  # (batch_size, embed_dim)
        
        # ============ Prepare for fusion ============
        # Stack embeddings as sequence: [CNN_embed, Transformer_embed]
        embeddings_seq = torch.stack([cnn_embed, transformer_embed], dim=1)  # (batch_size, 2, embed_dim)
        
        # ============ Adaptive Fusion ============
        fused_embed, attention_weights = self.fusion(embeddings_seq)  # (batch_size, embed_dim)
        
        # ============ Classification ============
        logits = self.classifier(fused_embed)  # (batch_size, num_classes)
        
        # Prepare return values
        output = [logits]
        
        if return_embeddings:
            embeddings = {
                'cnn_features': cnn_features,
                'transformer_features': transformer_features,
                'cnn_projected': cnn_embed,
                'transformer_projected': transformer_embed,
                'fused': fused_embed
            }
            output.append(embeddings)
        
        if return_attention:
            output.append(attention_weights)
        
        return output[0] if len(output) == 1 else tuple(output)
    
    def get_num_params(self):
        """Calculate total number of parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def get_trainable_params(self):
        """Calculate number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ==================== Model Variants ====================

def create_hybrid_model(num_classes=7, embed_dim=512, num_heads=8, 
                       dropout=0.1, pretrained_swin=True):
    """
    Factory function to create the hybrid model
    
    Args:
        num_classes: Number of emotion classes
        embed_dim: Common embedding dimension
        num_heads: Number of attention heads in fusion module
        dropout: Dropout rate
        pretrained_swin: Use pretrained Swin Transformer
    """
    model = HybridEmotionRecognition(
        num_classes=num_classes,
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=dropout,
        pretrained_swin=pretrained_swin
    )
    return model


# ==================== Testing and Examples ====================

# if __name__ == '__main__':
#     print("=" * 60)
#     print("Testing Hybrid CNN-Transformer Model")
#     print("=" * 60)
    
#     # Create model
#     model = create_hybrid_model(
#         num_classes=7,
#         embed_dim=512,
#         num_heads=8,
#         dropout=0.1,
#         pretrained_swin=False  # Set to True if you have internet
#     )
    
#     # Test input (batch of 4 images, 224x224 RGB)
#     x = torch.randn(4, 3, 224, 224)
    
#     print(f"\nInput shape: {x.shape}")
#     print(f"Total parameters: {model.get_num_params():,}")
#     print(f"Trainable parameters: {model.get_trainable_params():,}")
    
#     # Test forward pass
#     print("\n" + "=" * 60)
#     print("1. Basic forward pass")
#     print("=" * 60)
#     logits = model(x)
#     print(f"Output logits shape: {logits.shape}")
#     print(f"Predictions: {torch.argmax(logits, dim=1)}")
    
#     # Test with embeddings
#     print("\n" + "=" * 60)
#     print("2. Forward pass with embeddings")
#     print("=" * 60)
#     logits, embeddings = model(x, return_embeddings=True)
#     print(f"CNN features shape: {embeddings['cnn_features'].shape}")
#     print(f"Transformer features shape: {embeddings['transformer_features'].shape}")
#     print(f"CNN projected shape: {embeddings['cnn_projected'].shape}")
#     print(f"Transformer projected shape: {embeddings['transformer_projected'].shape}")
#     print(f"Fused embedding shape: {embeddings['fused'].shape}")
    
#     # Test with attention weights
#     print("\n" + "=" * 60)
#     print("3. Forward pass with attention weights")
#     print("=" * 60)
#     logits, embeddings, attention = model(x, return_embeddings=True, return_attention=True)
#     print(f"Attention weights shape: {attention.shape}")
#     print(f"Attention weights (first sample):\n{attention[0]}")
    
#     print("\n" + "=" * 60)
#     print("Model architecture summary:")
#     print("=" * 60)
#     print(model)
    
#     print("\n✅ All tests passed!")