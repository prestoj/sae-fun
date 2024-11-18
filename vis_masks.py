import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms
from train_layerwise_jepa import create_masks
import numpy as np

def visualize_masks(num_samples=8):
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load some MNIST images
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = torchvision.datasets.MNIST(root='./data', train=True,
                                       download=True, transform=transform)
    
    # Get random indices
    random_indices = torch.randperm(len(dataset))[:num_samples]
    
    # Get a batch of random images
    batch_size = num_samples
    images = torch.stack([dataset[i.item()][0] for i in random_indices])
    images = images.to(device)
    
    # Create masks
    context_masks, target_blocks = create_masks(
        batch_size=batch_size,
        image_size=28,
        patch_size=4,
        device=device
    )

    # Setup the plot with adjusted size
    fig, axes = plt.subplots(num_samples, 6, figsize=(15, 2*num_samples))
    fig.suptitle('Mask Visualization', fontsize=16)
    
    for idx in range(num_samples):
        # Original image
        axes[idx, 0].imshow(images[idx].cpu().squeeze(), cmap='gray')
        axes[idx, 0].set_title('Original')
        axes[idx, 0].axis('off')
        
        # Context mask visualization
        img_with_context_mask = images[idx].cpu().squeeze().numpy()
        mask_vis = context_masks[idx].reshape(7, 7).cpu().numpy()
        mask_vis = np.repeat(np.repeat(mask_vis, 4, axis=0), 4, axis=1)
        
        # Create RGB image with grey tint for non-context regions
        masked_img = np.stack([img_with_context_mask]*3, axis=-1)
        # Scale down values and add grey to non-context regions
        masked_img[~mask_vis] = np.clip(masked_img[~mask_vis] * 0.1 + 0.5, 0, 1)
        
        axes[idx, 1].imshow(masked_img)
        axes[idx, 1].set_title('Context Mask')
        axes[idx, 1].axis('off')
        
        # Target blocks visualization (4 blocks)
        for block_idx, target_mask in enumerate(target_blocks[idx]):
            img_with_target_mask = images[idx].cpu().squeeze().numpy()
            mask_vis = target_mask.reshape(7, 7).cpu().numpy()
            mask_vis = np.repeat(np.repeat(mask_vis, 4, axis=0), 4, axis=1)
            
            # Create RGB image with grey tint for non-target regions
            masked_img = np.stack([img_with_target_mask]*3, axis=-1)
            # Scale down values and add grey to non-target regions
            masked_img[~mask_vis] = np.clip(masked_img[~mask_vis] * 0.1 + 0.5, 0, 1)
            
            axes[idx, block_idx + 2].imshow(masked_img)
            axes[idx, block_idx + 2].set_title(f'Target Block {block_idx+1}')
            axes[idx, block_idx + 2].axis('off')
    
    plt.tight_layout()
    plt.show()

def print_mask_stats(context_masks, target_blocks):
    """Print statistics about the masks"""
    total_patches = context_masks.shape[1]
    
    # Context mask stats
    context_visible = context_masks.sum(1).float().mean()
    print(f"\nMask Statistics:")
    print(f"Context visible patches: {context_visible:.1f} ({context_visible/total_patches:.1%})")
    
    # Target block stats
    for i in range(len(target_blocks[0])):
        block_masks = torch.stack([blocks[i] for blocks in target_blocks])
        block_size = block_masks.sum(1).float().mean()
        print(f"Target block {i+1} size: {block_size:.1f} patches ({block_size/total_patches:.1%})")
        
        # Check overlap with context
        overlap = (context_masks & block_masks).sum(1).float().mean()
        print(f"Overlap with context: {overlap:.1f} patches")

if __name__ == '__main__':
    visualize_masks()
