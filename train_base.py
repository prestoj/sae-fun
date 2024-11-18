import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import math

# Small ViT for MNIST
class TinyViT(nn.Module):
    def __init__(self, patch_size=4, embed_dim=64, depth=3, num_heads=4, mlp_ratio=4.):
        super().__init__()
        self.patch_size = patch_size
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x, return_intermediates=False):
        # Patch embedding
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)  # [B, N, embed_dim]
        
        # Process through transformer blocks
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)
        
        if return_intermediates:
            return x, [x]  # Return final layer only
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, embed_dim)
        )
        
    def forward(self, x):
        # Self attention
        x = x + self._attention_block(self.norm1(x))
        # MLP
        x = x + self.mlp(self.norm2(x))
        return x
        
    def _attention_block(self, x):
        x = x.transpose(0, 1)  # [N, B, E]
        x, _ = self.attn(x, x, x)
        x = x.transpose(0, 1)  # [B, N, E]
        return x

def get_positional_embedding(target_mask, embed_dim, device):
    """Creates positional embeddings for target block prediction."""
    patches_per_side = int(math.sqrt(target_mask.shape[0]))
    
    target_indices = target_mask.nonzero().squeeze()
    rows = target_indices // patches_per_side
    cols = target_indices % patches_per_side
    
    rows = (2 * rows.float() / (patches_per_side - 1)) - 1
    cols = (2 * cols.float() / (patches_per_side - 1)) - 1
    
    pos_dim = embed_dim // 2
    inv_freq = 1.0 / (10000 ** (torch.arange(0, pos_dim, 2).float() / pos_dim)).to(device)
    
    row_pos = rows.unsqueeze(1) @ inv_freq.unsqueeze(0)
    row_embed = torch.cat([torch.sin(row_pos), torch.cos(row_pos)], dim=1)
    
    col_pos = cols.unsqueeze(1) @ inv_freq.unsqueeze(0)
    col_embed = torch.cat([torch.sin(col_pos), torch.cos(col_pos)], dim=1)
    
    pos_embed = torch.cat([row_embed, col_embed], dim=1)
    return pos_embed.unsqueeze(0)

class Predictor(nn.Module):
    def __init__(self, in_dim=64, depth=8, num_heads=4):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(in_dim, num_heads, mlp_ratio=2.)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(in_dim)
        
    def forward(self, x, pos_embed):
        B = x.size(0)
        pos_embed = pos_embed.expand(B, -1, -1)
        x = torch.cat([x, pos_embed], dim=1)
        
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        
        return x[:, -pos_embed.size(1):]

class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=10):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        # x shape: [B, N, D]
        x = x.transpose(1, 2)  # [B, D, N]
        x = self.avgpool(x).squeeze(-1)  # [B, D]
        return self.fc(x)

def evaluate_classifier(target_encoder, classifier, test_loader, device):
    target_encoder.eval()
    classifier.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            features = target_encoder(images)
            outputs = classifier(features)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return 100. * correct / total

def train_classifier(target_encoder, classifier, train_loader, device, num_epochs=1):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    
    target_encoder.eval()
    classifier.train()
    
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                features = target_encoder(images)
            outputs = classifier(features)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

# Training functions
def get_block_dims(min_scale, max_scale, patches_per_side, min_aspect=0.75, max_aspect=1.5):
    """
    Get dimensions for a block following I-JEPA paper specifications.
    Args:
        min_scale: Minimum scale of block relative to image
        max_scale: Maximum scale of block relative to image
        patches_per_side: Number of patches per side of the image
        min_aspect: Minimum aspect ratio (default: 0.75)
        max_aspect: Maximum aspect ratio (default: 1.5)
    Returns:
        height, width of block in number of patches
    """
    area = torch.rand(1) * (max_scale - min_scale) + min_scale
    aspect = torch.rand(1) * (max_aspect - min_aspect) + min_aspect
    area = area * (patches_per_side ** 2)
    w = torch.sqrt(area * aspect)
    h = area / w
    return max(1, int(h.item())), max(1, int(w.item()))

def create_masks(batch_size, image_size, patch_size, device):
    """Creates masks following I-JEPA paper strategy"""
    n_patches = (image_size // patch_size) ** 2
    patches_per_side = image_size // patch_size
    
    target_blocks = []  # Will be a list of lists of masks
    context_masks = []
    
    for _ in range(batch_size):
        # Sample 4 target blocks
        curr_target_blocks = []
        combined_target_mask = torch.zeros(n_patches, dtype=torch.bool)
        
        for _ in range(4):
            h, w = get_block_dims(
                min_scale=0.15, 
                max_scale=0.2,
                patches_per_side=patches_per_side
            )
            max_start_h = patches_per_side - h
            max_start_w = patches_per_side - w
            
            if max_start_h > 0 and max_start_w > 0:
                start_h = torch.randint(0, max_start_h + 1, (1,)).item()
                start_w = torch.randint(0, max_start_w + 1, (1,)).item()
                
                # Create mask for this target block
                block_mask = torch.zeros(n_patches, dtype=torch.bool)
                for i in range(h):
                    for j in range(w):
                        idx = (start_h + i) * patches_per_side + (start_w + j)
                        block_mask[idx] = True
                        combined_target_mask[idx] = True
                
                curr_target_blocks.append(block_mask)
        
        target_blocks.append(curr_target_blocks)
        
        # Create one large context block
        h, w = get_block_dims(
            min_scale=0.85, 
            max_scale=1.0,
            patches_per_side=patches_per_side,
            min_aspect=1.0, 
            max_aspect=1.0
        )
        max_start_h = patches_per_side - h
        max_start_w = patches_per_side - w
        
        if max_start_h > 0 and max_start_w > 0:
            start_h = torch.randint(0, max_start_h + 1, (1,)).item()
            start_w = torch.randint(0, max_start_w + 1, (1,)).item()
            
            # Create context mask
            context_mask = torch.zeros(n_patches, dtype=torch.bool)
            for i in range(h):
                for j in range(w):
                    idx = (start_h + i) * patches_per_side + (start_w + j)
                    context_mask[idx] = True
            
            # Remove overlapping regions with target
            context_mask = context_mask & ~combined_target_mask
        else:
            context_mask = torch.zeros(n_patches, dtype=torch.bool)
        
        context_masks.append(context_mask)
    
    context_masks = torch.stack(context_masks).to(device)
    return context_masks, target_blocks

def print_mask_stats(context_masks, target_masks):
    total_patches = target_masks.shape[1]
    
    # Per batch stats
    target_ratios = target_masks.sum(1).float() / total_patches
    context_ratios = context_masks.sum(1).float() / total_patches
    
    print(f"\nMasking Statistics:")
    print(f"Target masking ratio: {target_ratios.mean():.1%} ± {target_ratios.std():.1%}")
    print(f"Context masking ratio: {(1 - context_ratios.mean()):.1%} ± {context_ratios.std():.1%}")
    print(f"Total visible ratio: {context_ratios.mean():.1%}")
    print(f"Average patches per mask:")
    print(f"- Target: {(target_ratios * total_patches).mean():.1f} patches")
    print(f"- Context: {(context_ratios * total_patches).mean():.1f} patches")

def train_epoch(context_encoder, target_encoder, predictor, train_loader, optimizer, device):
    context_encoder.train()
    predictor.train()
    target_encoder.eval()
    
    total_loss = 0
    for batch_idx, (images, _) in enumerate(train_loader):
        images = images.to(device)
        batch_size = images.shape[0]
        
        # Create masks
        context_masks, target_blocks = create_masks(
            batch_size, image_size=28, patch_size=4, device=device
        )
        
        # Get representations for final layer only
        with torch.no_grad():
            _, target_intermediates = target_encoder(images, return_intermediates=True)
        _, context_intermediates = context_encoder(images, return_intermediates=True)
        
        # Process final layer
        context_repr = context_intermediates[0]  # Get final layer representation
        target_repr = target_intermediates[0]    # Get final layer representation
        
        B, N, D = context_repr.shape
        context_masks_reshaped = context_masks.view(B, N)
        
        # Get context representations for each image
        context_reprs = []
        target_reprs = []
        pos_embeds = []
        
        for i in range(batch_size):
            # Get context representation
            curr_context_mask = context_masks_reshaped[i]
            curr_context_repr = context_repr[i, curr_context_mask]
            
            # Process all 4 target blocks for this image
            for t in range(4):
                curr_target_mask = target_blocks[i][t]
                curr_target_reprs = target_repr[i, curr_target_mask.to(device)]
                curr_pos_embeds = get_positional_embedding(curr_target_mask, D, device)
                
                context_reprs.append(curr_context_repr)
                target_reprs.append(curr_target_reprs)
                pos_embeds.append(curr_pos_embeds.squeeze(0))
        
        # Pad sequences within the batch
        max_context_len = max(r.size(0) for r in context_reprs)
        max_target_len = max(r.size(0) for r in target_reprs)
        
        # Create padded tensors
        padded_context = torch.zeros(len(context_reprs), max_context_len, D, device=device)
        padded_targets = torch.zeros(len(target_reprs), max_target_len, D, device=device)
        padded_pos_embeds = torch.zeros(len(pos_embeds), max_target_len, D, device=device)
        
        # Get lengths for masking
        context_lengths = torch.tensor([r.size(0) for r in context_reprs], device=device)
        target_lengths = torch.tensor([r.size(0) for r in target_reprs], device=device)
        
        # Fill padded tensors
        for i in range(len(context_reprs)):
            padded_context[i, :context_lengths[i]] = context_reprs[i]
            padded_targets[i, :target_lengths[i]] = target_reprs[i]
            padded_pos_embeds[i, :target_lengths[i]] = pos_embeds[i]
        
        # Create attention mask for valid elements
        target_mask = torch.arange(max_target_len, device=device)[None, :] < target_lengths[:, None]
        
        # Forward pass through predictor
        pred_reprs = predictor(padded_context, padded_pos_embeds)
        
        # Compute masked loss
        mse = (pred_reprs - padded_targets) ** 2
        masked_mse = mse * target_mask.unsqueeze(-1)
        batch_loss = masked_mse.sum() / (target_mask.sum() * D)
        
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        
        # Update target encoder with momentum
        with torch.no_grad():
            m = 0.996
            for param_q, param_k in zip(context_encoder.parameters(), 
                                    target_encoder.parameters()):
                param_k.data = param_k.data * m + param_q.data * (1. - m)
        
        total_loss += batch_loss.item()
    
    return total_loss / len(train_loader)

def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
    
    # Add test loader for evaluation
    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)
    
    # Initialize models
    context_encoder = TinyViT(patch_size=4, embed_dim=64).to(device)
    target_encoder = TinyViT(patch_size=4, embed_dim=64).to(device)
    predictor = Predictor(in_dim=64, depth=2).to(device)  # Single predictor for final layer
    
    # Copy initial weights from context to target encoder
    target_encoder.load_state_dict(context_encoder.state_dict())
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        list(context_encoder.parameters()) + list(predictor.parameters()),
        lr=1e-3,
        weight_decay=0.05
    )
    
    # Training loop
    num_epochs = 1000
    eval_frequency = 5  # Evaluate every 5 epochs
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_loss = train_epoch(
            context_encoder, target_encoder, predictor, 
            trainloader, optimizer, device
        )
        print(f"Epoch {epoch+1} average loss: {train_loss:.4f}")
        
        # Periodically evaluate linear classifier
        if (epoch + 1) % eval_frequency == 0:
            print("\nTraining linear classifier...")
            classifier = LinearClassifier(input_dim=64).to(device)
            train_classifier(target_encoder, classifier, trainloader, device)
            acc = evaluate_classifier(target_encoder, classifier, testloader, device)
            print(f"Linear classifier accuracy: {acc:.2f}%\n")

            # Save the trained models
            torch.save({
                'context_encoder': context_encoder.state_dict(),
                'target_encoder': target_encoder.state_dict(),
                'predictor': predictor.state_dict(),
                'classifier': classifier.state_dict()
            }, f'mnist_jepa.pth')
    
    # Final evaluation
    print("\nFinal evaluation:")
    classifier = LinearClassifier(input_dim=64).to(device)
    train_classifier(target_encoder, classifier, trainloader, device)
    acc = evaluate_classifier(target_encoder, classifier, testloader, device)
    print(f"Final linear classifier accuracy: {acc:.2f}%")
    
    # Save the trained models
    torch.save({
        'context_encoder': context_encoder.state_dict(),
        'target_encoder': target_encoder.state_dict(),
        'predictor': predictor.state_dict(),
        'classifier': classifier.state_dict()
    }, 'mnist_jepa.pth')

if __name__ == '__main__':
    main()