import torch
import torch.nn as nn
from train_sae import SparseAutoencoder, TinyViT
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def generate_neuron_maximizing_image(base_model, layer_name, sae, neuron_idx, 
                                   input_shape=(1, 28, 28), learning_rate=1e-2, 
                                   n_steps=1000, device=None, mode='center'):
    """
    Generate an image that maximizes activation of a specific neuron in the SAE.
    """
    if device is None:
        device = next(base_model.parameters()).device
        
    base_model.eval()
    sae.eval()
    
    # Setup hook to capture intermediate activations
    activations = {}
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            act = output[0]
        else:
            act = output
            
        # For all layers, reshape to [B*N, D]
        if len(act.shape) == 4:  # [B, C, H, W] from patch_embed
            B, C, H, W = act.shape
            act = act.permute(0, 2, 3, 1)  # [B, H, W, C]
            act = act.reshape(-1, C)        # [B*H*W, C]
        else:  # [B, N, D] from transformer blocks
            B, N, D = act.shape
            act = act.reshape(-1, D)        # [B*N, D]
            
        activations['features'] = act
    
    # Register hook
    for name, module in base_model.named_modules():
        if name == layer_name:
            handle = module.register_forward_hook(hook_fn)
            break
    
    # Initialize image with random noise and ensure non-zero activation
    max_attempts = 1000
    found_activation = False
    for attempt in range(max_attempts):
        image = torch.rand(1, *input_shape, requires_grad=True, device=device)
        
        # Forward pass through base model and SAE to check activation
        base_model(image)
        features = activations['features']
        
        if mode == 'center':
            center_patch = features[24:25]
            _, encoded = sae(center_patch)
            activation = encoded[0, neuron_idx]
        elif mode == 'max':
            _, encoded = sae(features)
            activation = encoded[:, neuron_idx].max()
        elif mode == 'mean':
            _, encoded = sae(features)
            activation = encoded[:, neuron_idx].mean()
            
        if activation > 0:
            found_activation = True
            break
            
        if attempt == max_attempts - 1:
            print(f"Warning: Could not find initialization with non-zero activation for neuron {neuron_idx} after {max_attempts} attempts")
            return None  # Return None for dead neurons
    
    optimizer = torch.optim.Adam([image], lr=learning_rate)
    
    # Optimize image
    pbar = tqdm(range(n_steps), desc=f"Optimizing for neuron {neuron_idx}")
    for step in pbar:
        optimizer.zero_grad()
        
        # Forward pass through base model to get activations
        base_model(image)
        features = activations['features']  # [B, N*D]
        
        # Process features
        if mode == 'center':
            # Use the center patch (patch 24 for 7x7 grid)
            center_patch = features[24:25]  # Take just the center patch
            _, encoded = sae(center_patch)  # [1, hidden_dim]
            target_activation = encoded[0, neuron_idx]
            
            # Penalize other neurons being active
            other_neurons = torch.cat([encoded[0, :neuron_idx], encoded[0, neuron_idx+1:]])
            interference_penalty = (other_neurons ** 2).mean()
            
        elif mode == 'max':
            # Process each patch separately and take the maximum activation
            _, encoded = sae(features)  # [49, hidden_dim]
            target_activation = encoded[:, neuron_idx].max()
            
            # Penalize other neurons being active across all patches
            other_neurons = torch.cat([encoded[:, :neuron_idx], encoded[:, neuron_idx+1:]], dim=1)
            interference_penalty = (other_neurons ** 2).mean()
            
        elif mode == 'mean':
            # Process each patch separately and take the mean activation
            _, encoded = sae(features)  # [49, hidden_dim]
            target_activation = encoded[:, neuron_idx].mean()
            
            # Penalize other neurons being active across all patches
            other_neurons = torch.cat([encoded[:, :neuron_idx], encoded[:, neuron_idx+1:]], dim=1)
            interference_penalty = (other_neurons ** 2).mean()
        
        # Loss combines negative target activation and interference penalty
        loss = -target_activation + 1 * interference_penalty
        
        loss.backward()
        optimizer.step()
        
        # Clip values to valid image range
        with torch.no_grad():
            image.clamp_(0.0, 1.0)
            
        if step % 50 == 0:
            pbar.set_postfix({
                'activation': f'{target_activation.item():.4f}',
                'interference': f'{interference_penalty.item():.4f}'
            })
    # Remove hook
    handle.remove()
    return image.detach()

def visualize_layer_neurons(base_model, sae, layer_name, n_neurons=24, nrows=4, mode='center'):
    """
    Visualize multiple neurons from a layer
    
    Args:
        mode: How to handle multiple patches ('max', 'mean', or 'center')
    """
    ncols = n_neurons // nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(2*ncols, 2*nrows))
    
    for idx in range(n_neurons):
        row, col = idx // ncols, idx % ncols
        image = generate_neuron_maximizing_image(base_model, layer_name, sae, idx, mode=mode)
        
        if image is not None:  # Only plot if we found a valid activation
            axes[row, col].imshow(image[0, 0].cpu().numpy(), cmap='gray')
            axes[row, col].set_title(f'N{idx}')
        else:
            axes[row, col].axis('off')  # Hide axis for dead neurons
            axes[row, col].set_title(f'N{idx}\n(dead)', color='red')
        
        axes[row, col].axis('off')
    
    plt.suptitle(f'Layer: {layer_name} (mode: {mode})')
    plt.tight_layout()
    plt.show()

def visualize_neuron_examples(base_model, sae, layer_name, dataloader, n_neurons=24, 
                            n_examples=3, nrows=4, device=None):
    """
    Visualize neurons by finding real examples from the dataset that maximally activate them.
    
    Args:
        base_model: The base transformer model
        sae: The sparse autoencoder
        layer_name: Name of the layer to analyze
        dataloader: DataLoader containing the dataset
        n_neurons: Number of neurons to visualize
        n_examples: Number of top examples to show per neuron
        nrows: Number of rows in the visualization grid
    """
    if device is None:
        device = next(base_model.parameters()).device
        
    base_model.eval()
    sae.eval()
    
    # Setup hook to capture intermediate activations
    activations = {}
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            act = output[0]
        else:
            act = output
            
        # For all layers, reshape to [B*N, D]
        if len(act.shape) == 4:  # [B, C, H, W] from patch_embed
            B, C, H, W = act.shape
            act = act.permute(0, 2, 3, 1)  # [B, H, W, C]
            act = act.reshape(-1, C)        # [B*H*W, C]
        else:  # [B, N, D] from transformer blocks
            B, N, D = act.shape
            act = act.reshape(-1, D)        # [B*N, D]
            
        activations['features'] = act
    
    # Register hook
    for name, module in base_model.named_modules():
        if name == layer_name:
            handle = module.register_forward_hook(hook_fn)
            break
    
    # Storage for top activations
    top_activations = {i: {} for i in range(n_neurons)}  # neuron_idx -> {image_idx -> (activation, image, patch_idx)}
    
    # Process dataset
    print("Finding top activating examples...")
    with torch.no_grad():
        for batch, _ in tqdm(dataloader):
            batch = batch.to(device)
            batch_size = batch.shape[0]
            
            # Get activations
            base_model(batch)
            features = activations['features']
            
            # Get SAE activations
            _, encoded = sae(features)  # [B*N, hidden_dim]
            
            # For each neuron we care about
            for neuron_idx in range(n_neurons):
                neuron_acts = encoded[:, neuron_idx]
                
                # Find top activations in this batch
                batch_top_vals, batch_top_indices = neuron_acts.topk(3)
                
                # Calculate which image and which patch
                image_indices = batch_top_indices // 49  # 7x7 patches = 49 patches per image
                patch_indices = batch_top_indices % 49
                
                # Store results, keying by image_idx to ensure different images
                for val, img_idx, patch_idx in zip(batch_top_vals, image_indices, patch_indices):
                    curr_img_idx = img_idx.item()
                    if curr_img_idx not in top_activations[neuron_idx] or val > top_activations[neuron_idx][curr_img_idx][0]:
                        top_activations[neuron_idx][curr_img_idx] = (
                            val.item(),
                            batch[img_idx].cpu(),
                            patch_idx.item()
                        )
                
                # Keep only top 3 different images
                sorted_activations = sorted(top_activations[neuron_idx].items(), key=lambda x: x[1][0], reverse=True)[:3]
                top_activations[neuron_idx] = dict(sorted_activations)
    
    # Remove hook
    handle.remove()
    
    # Visualization
    ncols = n_neurons // nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(2*ncols, 2*nrows))

    neuron_idx = 0
    plot_idx = 0
    while plot_idx < n_neurons and neuron_idx < hidden_dim:
        row, col = plot_idx // ncols, plot_idx % ncols
        
        # Skip dead neurons (activation < 0.01)
        while neuron_idx < hidden_dim:
            if neuron_idx not in top_activations:
                neuron_idx += 1
                continue
            
            examples = list(top_activations[neuron_idx].values())
            max_activation = max(act for act, _, _ in examples) if examples else 0
            if max_activation >= 0.01:  # Found a live neuron
                break
            neuron_idx += 1
        
        if neuron_idx >= hidden_dim:  # No more neurons to check
            break
            
        # Create a 1x3 subplot for the top 3 examples
        gs = axes[row, col].get_gridspec()
        subgs = gs[row, col].subgridspec(1, 3, wspace=0.1)
        for i, (act, img, patch_idx) in enumerate(examples):
            subax = fig.add_subplot(subgs[0, i])
            subax.imshow(img[0].numpy(), cmap='gray')
            
            # Highlight the patch
            patch_row = patch_idx // 7
            patch_col = patch_idx % 7
            patch_size = 4
            rect = plt.Rectangle(
                (patch_col * patch_size, patch_row * patch_size),
                patch_size, patch_size,
                fill=False, color='red', linewidth=1
            )
            subax.add_patch(rect)
            subax.axis('off')
            
            if i == 0:  # Only add title to first image
                subax.set_title(f'N{neuron_idx}\nAct: {act:.3f}')
        
        plot_idx += 1
        neuron_idx += 1

    # Hide any unused subplots
    for idx in range(plot_idx, nrows * ncols):
        row, col = idx // ncols, idx % ncols
        axes[row, col].axis('off')

    plt.suptitle(f'Layer: {layer_name} - Top activating patches')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Setup MNIST dataset and dataloader
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
    
    # Load the trained JEPA model
    checkpoint = torch.load('mnist_layerwise_jepa.pth', map_location=device)
    base_model = TinyViT(patch_size=4, embed_dim=64).to(device)
    base_model.load_state_dict(checkpoint['target_encoder'])
    base_model.eval()
    
    # Define layers we trained autoencoders for
    layer_names = [
        'patch_embed',
        'blocks.0',
        'blocks.1',
        'blocks.2'
    ]
    
    # Try different visualization modes
    modes = ['center']  # Start with just center mode for debugging
    for layer_name in layer_names:
        print(f"\nVisualizing {layer_name}")
        
        # Load the saved autoencoder
        checkpoint = torch.load(f'sae_{layer_name.replace(".", "_")}.pth', map_location=device)
        
        # Create and load the autoencoder
        input_dim = checkpoint['state_dict']['encoder.weight'].shape[1]  # Should be 64
        hidden_dim = checkpoint['hidden_dim']
        print(f"Creating SAE with input_dim={input_dim}, hidden_dim={hidden_dim}")
        
        sae = SparseAutoencoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim
        ).to(device)
        sae.load_state_dict(checkpoint['state_dict'])
        
        # Generate and display visualizations for each mode
        for mode in modes:
            print(f"\nMode: {mode}")
            # visualize_layer_neurons(base_model, sae, layer_name, mode=mode)
            visualize_neuron_examples(base_model, sae, layer_name, dataloader, n_neurons=24, n_examples=3, nrows=4, device=device)
