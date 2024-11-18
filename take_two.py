import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import math
from tqdm import tqdm

class TinyViT(nn.Module):
    def __init__(self, image_size=28, patch_size=4, in_channels=1, 
                 embed_dim=64, depth=4, num_heads=4, mlp_ratio=2.):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, 
                                   kernel_size=patch_size, stride=patch_size)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x, mask=None):
        # Patch embedding [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        
        # Apply mask if provided
        if mask is not None:
            x = x * mask.unsqueeze(-1)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )
        
    def forward(self, x):
        # Self attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        return x

class Predictor(nn.Module):
    def __init__(self, embed_dim=64, predictor_dim=32, depth=2, num_heads=4, num_patches=49):
        super().__init__()
        self.embedding = nn.Linear(embed_dim, predictor_dim)
        # Learnable position embeddings for each possible patch position
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.normal_(self.pos_embed, std=0.02)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(predictor_dim, num_heads, mlp_ratio=2.)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(predictor_dim)
        self.head = nn.Linear(predictor_dim, embed_dim)
        
    def forward(self, x, target_indices):
        # Project context features to predictor dimension
        x = self.embedding(x)
        
        # Get position embeddings for target locations
        mask_tokens = self.pos_embed.expand(x.shape[0], -1, -1)[:, target_indices]
        mask_tokens = self.embedding(mask_tokens)
        
        # Concatenate along sequence dimension
        x = torch.cat([x, mask_tokens], dim=1)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)
        x = self.head(x)
        
        # Return only the predictions for masked tokens
        return x[:, -mask_tokens.size(1):]

class LinearClassifier(nn.Module):
    def __init__(self, embed_dim=64, num_classes=10):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        # Average pool across patches [B, num_patches, dim] -> [B, dim]
        x = self.avgpool(x.transpose(1, 2)).squeeze(-1)
        # Classify
        return self.fc(x)

class MNISTJEPA:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.num_patches = 49  # 7x7 patches for 28x28 image with patch_size=4
        
        # Initialize models
        self.context_encoder = TinyViT().to(device)
        self.target_encoder = TinyViT().to(device)
        self.predictor = Predictor(num_patches=self.num_patches).to(device)
        
        # Copy weights from context to target encoder
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        
        # Initialize optimizers
        self.optimizer = torch.optim.AdamW([
            {'params': self.context_encoder.parameters()},
            {'params': self.predictor.parameters()}
        ], lr=1e-3, weight_decay=0.05)
        
        # EMA update momentum
        self.momentum = 0.996

    @torch.no_grad()
    def get_representations(self, images):
        """Get representations from target encoder"""
        images = images.to(self.device)
        return self.target_encoder(images)
        
    @torch.no_grad()
    def update_target_encoder(self):
        """Update target encoder using exponential moving average"""
        for param_q, param_k in zip(self.context_encoder.parameters(),
                                  self.target_encoder.parameters()):
            param_k.data = param_k.data * self.momentum + \
                          param_q.data * (1. - self.momentum)
    
    def generate_masks(self, batch_size):
        """Generate context and target masks"""
        num_patches = 49  # 7x7 patches for 28x28 image with patch_size=4
        
        # Generate 4 target blocks
        target_masks = []
        for _ in range(4):
            # Random block size between 15-20% of patches
            block_size = torch.randint(2, 4, (1,)).item()
            start_idx = torch.randint(0, num_patches - block_size, (1,)).item()
            mask = torch.zeros(batch_size, num_patches)
            mask[:, start_idx:start_idx + block_size] = 1
            target_masks.append(mask)
        
        # Combine target masks
        target_mask = torch.max(torch.stack(target_masks), dim=0)[0]
        
        # Generate context mask (85-100% of non-target patches)
        context_mask = 1 - target_mask
        context_mask = context_mask * torch.bernoulli(
            torch.ones_like(context_mask) * 0.85
        )
        
        return context_mask.to(self.device), target_mask.to(self.device)
    
    def train_step(self, images):
        images = images.to(self.device)
        batch_size = images.size(0)
        
        # Generate masks and get target indices
        context_mask, target_mask = self.generate_masks(batch_size)
        target_indices = torch.where(target_mask[0] == 1)[0]  # Get indices where mask is 1
        
        # Get target representations
        with torch.no_grad():
            target_repr = self.target_encoder(images)
            target_repr = target_repr[target_mask.bool()].view(
                batch_size, -1, target_repr.size(-1)
            )
        
        # Get context representations
        context_repr = self.context_encoder(images, context_mask)
        
        # Predict target representations using positional information
        pred_repr = self.predictor(context_repr, target_indices)
        
        # Compute L2 loss
        loss = nn.MSELoss()(pred_repr, target_repr)
        
        # Update models
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target encoder
        self.update_target_encoder()
        
        return loss.item()

def train_linear_classifier(encoder, trainloader, testloader, device, epochs=1):
    """Train and evaluate a linear classifier on frozen representations"""
    classifier = LinearClassifier().to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # Train
    classifier.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            
            # Get frozen representations
            with torch.no_grad():
                representations = encoder(images)
            
            # Forward pass
            outputs = classifier(representations)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    # Evaluate
    classifier.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            representations = encoder(images)
            outputs = classifier(representations)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
    
    train_acc = 100. * correct / total
    test_acc = 100. * test_correct / test_total
    
    return train_acc, test_acc

def train_mnist_jepa(num_epochs=100, batch_size=256):
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    trainset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    # Initialize model
    model = MNISTJEPA()
    
    # Lists to store metrics
    train_losses = []
    linear_eval_accuracies = []
    
    # Training loop
    for epoch in range(num_epochs):
        # Train JEPA
        model.context_encoder.train()
        model.target_encoder.eval()
        total_loss = 0
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("Training JEPA...")
        for batch_idx, (images, _) in enumerate(tqdm(trainloader)):
            loss = model.train_step(images)
            total_loss += loss
            
        avg_loss = total_loss / len(trainloader)
        train_losses.append(avg_loss)
        print(f"Average JEPA Loss: {avg_loss:.4f}")
        
        # Linear evaluation
        print("Evaluating representations...")
        model.target_encoder.eval()
        train_acc, test_acc = train_linear_classifier(
            model.target_encoder, 
            trainloader, 
            testloader, 
            model.device,
            epochs=1  # Reduced epochs for faster evaluation
        )
        linear_eval_accuracies.append(test_acc)
        
        print(f"Linear Eval - Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")
        
    return model, train_losses, linear_eval_accuracies

if __name__ == "__main__":
    model, losses, accuracies = train_mnist_jepa()
    
    # Print final results
    print("\nTraining completed!")
    print(f"Final test accuracy: {accuracies[-1]:.2f}%")
    
    # Optionally, plot learning curves
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(losses)
        plt.title('JEPA Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        plt.subplot(1, 2, 2)
        plt.plot(accuracies)
        plt.title('Linear Evaluation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Test Accuracy (%)')
        
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("matplotlib not available for plotting")