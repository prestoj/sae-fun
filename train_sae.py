import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms

from train_base import TinyViT

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # Pre-encoder bias (will be set to negative of decoder bias)
        self.pre_encoder_bias = nn.Parameter(torch.zeros(input_dim))
        
        # Encoder
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)
        self.relu = nn.ReLU()
        
        # Decoder with normalized columns
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=True)
        
        # Initialize with Kaiming Uniform
        nn.init.kaiming_uniform_(self.encoder.weight)
        nn.init.kaiming_uniform_(self.decoder.weight)
        
        # Normalize decoder columns
        with torch.no_grad():
            self.decoder.weight.div_(self.decoder.weight.norm(dim=0, keepdim=True))
    
    def forward(self, x):
        # Handle both 2D and 1D inputs
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension
            
        # Apply pre-encoder bias
        x = x - self.pre_encoder_bias
        
        # Encode
        encoded = self.relu(self.encoder(x))
        
        # Decode
        decoded = self.decoder(encoded) + self.pre_encoder_bias
        
        return decoded, encoded
    
    def normalize_decoder(self):
        """Normalize decoder columns to unit norm"""
        with torch.no_grad():
            self.decoder.weight.div_(self.decoder.weight.norm(dim=0, keepdim=True))

class ActivationDataset(Dataset):
    def __init__(self, activations):
        self.activations = activations
        
    def __len__(self):
        return len(self.activations)
    
    def __getitem__(self, idx):
        return self.activations[idx]

def collect_activations(model, dataloader, layer_name, device):
    """Collect activations from a specific layer"""
    activations = []
    
    def hook_fn(module, input, output):
        # Handle different output types
        if isinstance(output, tuple):
            act = output[0].detach()
        else:
            act = output.detach()
            
        # Reshape to [B*N, D] where D is the feature dimension (64)
        if len(act.shape) == 4:  # From patch_embed: [B, C, H, W]
            B, C, H, W = act.shape
            act = act.permute(0, 2, 3, 1)  # [B, H, W, C]
            act = act.reshape(-1, C)        # [B*H*W, C]
        else:  # From transformer blocks: [B, N, D]
            B, N, D = act.shape
            act = act.reshape(-1, D)        # [B*N, D]
            
        activations.append(act.cpu())
    
    # Register hook
    for name, module in model.named_modules():
        if name == layer_name:
            handle = module.register_forward_hook(hook_fn)
            break
    
    # Collect activations
    model.eval()
    with torch.no_grad():
        for batch, _ in tqdm(dataloader, desc=f"Collecting {layer_name} activations"):
            batch = batch.to(device)
            model(batch)
    
    # Remove hook
    handle.remove()
    
    # Concatenate all activations
    return torch.cat(activations, dim=0)  # Will be [total_patches, 64]

def resample_dead_neurons(sae, optimizer, loader, dead_threshold=1e-4):
    """Resample dead neurons with stable, diverse initialization"""
    device = next(sae.parameters()).device
    
    # Find dead neurons
    print("Checking neuron activity...")
    all_activations = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            _, activations = sae(batch)
            all_activations.append(activations)
    all_activations = torch.cat(all_activations, dim=0)
    
    # Compute neuron activity
    neuron_activity = (all_activations > 0).float().mean(dim=0)
    dead_neurons = (neuron_activity < dead_threshold).nonzero().squeeze()
    
    if len(dead_neurons) == 0:
        print("No dead neurons found")
        return 0
    
    print(f"Found {len(dead_neurons)} dead neurons")
    print(f"Activity range before resampling: {neuron_activity[neuron_activity > 0].min():.6f} - {neuron_activity.max():.6f}")
    
    # Only resample a portion of dead neurons at a time
    # max_resample = min(50, len(dead_neurons))  # Limit number of neurons resampled at once
    max_resample = len(dead_neurons)
    dead_neurons = dead_neurons[:max_resample]
    print(f"Resampling {len(dead_neurons)} neurons this iteration")
    
    # Get diverse set of high-loss examples
    batch_size = min(16384, len(loader.dataset))
    sample_idx = torch.randint(0, len(loader.dataset), (batch_size,))
    samples = loader.dataset[sample_idx].to(device)
    
    recon, _ = sae(samples)
    losses = ((recon - samples) ** 2).mean(dim=1)
    
    # Get more candidates than needed
    n_candidates = min(batch_size, len(dead_neurons) * 20)
    candidate_idx = losses.argsort(descending=True)[:n_candidates]
    candidate_samples = samples[candidate_idx]
    
    # Calculate reference norms from active neurons
    with torch.no_grad():
        encoder_norms = sae.encoder.weight.norm(dim=1)
        alive_mask = neuron_activity > dead_threshold
        if alive_mask.any():
            target_norm = encoder_norms[alive_mask].median()  # Use median instead of mean
        else:
            target_norm = encoder_norms.median()
        if target_norm == 0:
            target_norm = 1.0
            
        print(f"Target encoder norm: {target_norm:.4f}")
    
    # Initialize arrays to store selected vectors
    selected_encoder = []
    selected_decoder = []
    
    # For each dead neuron, find a distinct direction
    with torch.no_grad():
        remaining_candidates = candidate_samples.clone()
        
        for i in range(len(dead_neurons)):
            if len(remaining_candidates) == 0:
                break
                
            # Compute angles with previously selected vectors
            angles = torch.zeros(len(remaining_candidates))
            if selected_decoder:
                prev_directions = torch.stack(selected_decoder)
                similarities = remaining_candidates @ prev_directions.T
                angles = similarities.max(dim=1)[0]
            
            # Find candidate with smallest maximum angle to existing vectors
            best_idx = angles.argmin()
            new_vector = remaining_candidates[best_idx]
            
            # Remove similar vectors from candidate pool
            similarities = (remaining_candidates @ new_vector) / (remaining_candidates.norm(dim=1) * new_vector.norm())
            mask = similarities.abs() < 0.3  # Keep only vectors sufficiently different
            remaining_candidates = remaining_candidates[mask]
            
            # Normalize and store
            decoder_vector = new_vector / new_vector.norm()
            selected_decoder.append(decoder_vector)
            
            # Create corresponding encoder vector with noise
            noise = torch.randn_like(new_vector) * 0.1
            encoder_vector = (new_vector + noise) * target_norm / new_vector.norm()
            selected_encoder.append(encoder_vector)
    
    # Apply updates
    with torch.no_grad():
        for i, dead_idx in enumerate(dead_neurons):
            if i < len(selected_decoder):
                # Update decoder
                sae.decoder.weight[:, dead_idx] = selected_decoder[i]
                
                # Update encoder with slightly random initialization
                sae.encoder.weight[dead_idx, :] = selected_encoder[i]
                sae.encoder.bias[dead_idx] = torch.randn(1).item() * 0.01  # Random small bias
    
    successful_reinits = len(selected_decoder)
    print(f"Successfully reinitialized {successful_reinits}/{len(dead_neurons)} neurons")
    
    # Verify neuron activity after resampling
    with torch.no_grad():
        _, new_activations = sae(samples)
        new_activity = (new_activations > 0).float().mean(dim=0)
        resampled_activity = new_activity[dead_neurons]
        if len(resampled_activity) > 0:
            print(f"Resampled neurons activity range: {resampled_activity.min():.6f} - {resampled_activity.max():.6f}")
    
    # Reset optimizer state for resampled neurons
    if optimizer is not None:
        state = optimizer.state.get(sae.encoder.weight, {})
        if 'exp_avg' in state:
            state['exp_avg'][dead_neurons] = 0
        if 'exp_avg_sq' in state:
            state['exp_avg_sq'][dead_neurons] = 0
        
        state = optimizer.state.get(sae.decoder.weight, {})
        if 'exp_avg' in state:
            state['exp_avg'][:, dead_neurons] = 0
        if 'exp_avg_sq' in state:
            state['exp_avg_sq'][:, dead_neurons] = 0
    
    return successful_reinits

def train_sparse_autoencoder(model, dataloader, layer_name, hidden_dim, 
                           l1_coeff=1e-3, device='cuda', num_epochs=10,
                           batch_size=512, learning_rate=1e-3):
    """Train a sparse autoencoder on activations from a specific layer"""
    
    # Collect activations
    print(f"Collecting activations from {layer_name}...")
    activations = collect_activations(model, dataloader, layer_name, device)
    input_dim = activations.shape[1]
    
    # Create dataset and loader for activations
    dataset = ActivationDataset(activations)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize autoencoder
    sae = SparseAutoencoder(input_dim, hidden_dim).to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(sae.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        recon_loss = 0
        sparse_loss = 0
        
        for batch_idx, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            batch = batch.to(device)
            
            # Forward pass
            recon, encoded = sae(batch)
            
            # Compute losses
            batch_recon_loss = torch.nn.functional.mse_loss(recon, batch)
            batch_sparse_loss = l1_coeff * encoded.abs().mean()
            loss = batch_recon_loss + batch_sparse_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Remove gradients on decoder weights in direction of current weights
            with torch.no_grad():
                norm_grad = (sae.decoder.weight.grad * sae.decoder.weight).sum(dim=0, keepdim=True)
                sae.decoder.weight.grad -= norm_grad * sae.decoder.weight
            
            optimizer.step()
            
            # Normalize decoder weights
            sae.normalize_decoder()
            
            # Update statistics
            total_loss += loss.item()
            recon_loss += batch_recon_loss.item()
            sparse_loss += batch_sparse_loss.item()
        
        # # Resample dead neurons every 25 steps
        # if (epoch) % 2 == 0:
        #     n_resampled = resample_dead_neurons(sae, optimizer, loader)
        #     if n_resampled > 0:
        #         print(f"Resampled {n_resampled} dead neurons")
        
        # Print epoch statistics
        avg_loss = total_loss / len(loader)
        avg_recon = recon_loss / len(loader)
        avg_sparse = sparse_loss / len(loader)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f} "
              f"(Recon = {avg_recon:.4f}, Sparse = {avg_sparse:.4f})")
        
        # Compute feature statistics
        with torch.no_grad():
            # Process activations in batches
            batch_size = 1024
            n_batches = (len(activations) + batch_size - 1) // batch_size
            all_active = []
            
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(activations))
                batch = activations[start_idx:end_idx].to(device)
                
                _, batch_encoded = sae(batch)
                batch_active = (batch_encoded > 0).float().mean(dim=0)
                all_active.append(batch_active)
            
            active = torch.stack(all_active).mean(dim=0)
            print(f"Active neurons: {(active > 0).sum()}/{hidden_dim}")
            print(f"Mean activation frequency: {active.mean():.4f}")
    return sae

def train_all_layers(model, dataloader, layer_names, hidden_dims, device='cuda'):
    """Train sparse autoencoders for multiple layers"""
    autoencoders = {}
    
    for layer_name, hidden_dim in zip(layer_names, hidden_dims):
        print(f"\nTraining autoencoder for {layer_name}")
        sae = train_sparse_autoencoder(
            model=model,
            dataloader=dataloader,
            layer_name=layer_name,
            hidden_dim=hidden_dim,
            device=device
        )
        autoencoders[layer_name] = sae
        
        # Save checkpoint
        torch.save({
            'state_dict': sae.state_dict(),
            'layer_name': layer_name,
            'hidden_dim': hidden_dim
        }, f'sae_{layer_name.replace(".", "_")}.pth')
    
    return autoencoders

if __name__ == "__main__":
    # Define layers to analyze
    layer_names = [
        'patch_embed',
        'blocks.0',
        'blocks.1',
        'blocks.2'
    ]

    # Define hidden dimensions for each layer
    hidden_dims = [512, 512, 512, 512]

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load the trained Layerwise JEPA model
    checkpoint = torch.load('mnist_layerwise_jepa.pth')
    model = TinyViT(patch_size=4, embed_dim=64).to(device)
    model.load_state_dict(checkpoint['target_encoder'])
    model.to(device)
    model.eval()

    # Create dataset and dataloader
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Train autoencoders
    autoencoders = train_all_layers(model, trainloader, layer_names, hidden_dims, device)