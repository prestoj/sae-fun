import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torchvision import transforms

# Define the training transforms
train_transform = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor()
])

def visualize_transforms(num_examples=20):
    # Original transform (just convert to tensor)
    original_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # Get a few sample images
    original_dataset = MNIST(root='./data', train=True, download=True, transform=original_transform)
    augmented_dataset = MNIST(root='./data', train=True, download=True, transform=train_transform)
    
    # Create a figure with a better layout for more examples
    num_rows = 4  # We'll show 4 rows
    num_cols = 5  # And 5 columns (4x5=20 examples)
    fig, axes = plt.subplots(num_rows, num_cols*2, figsize=(15, 12))  # Wider figure
    fig.suptitle('Original vs Augmented Images', fontsize=16, y=1.02)
    
    # Reshape axes for easier indexing
    axes = axes.reshape(num_rows, num_cols*2)
    
    for i in range(num_examples):
        row = i // num_cols
        col = (i % num_cols) * 2  # Multiply by 2 because each example needs 2 columns
        
        # Get the same image from both datasets
        img_orig, _ = original_dataset[i]
        img_aug, _ = augmented_dataset[i]
        
        # Plot original
        axes[row, col].imshow(img_orig.squeeze(), cmap='gray')
        axes[row, col].axis('off')
        if i < num_cols:  # Only show titles for the first row
            axes[row, col].set_title('Original')
        
        # Plot augmented
        axes[row, col+1].imshow(img_aug.squeeze(), cmap='gray')
        axes[row, col+1].axis('off')
        if i < num_cols:  # Only show titles for the first row
            axes[row, col+1].set_title('Augmented')
    
    plt.tight_layout()
    plt.show()

# You can call this function from your main() or directly:
if __name__ == '__main__':
    visualize_transforms()