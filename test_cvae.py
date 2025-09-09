#!/usr/bin/env python
"""
Test conditional VAE with counts_Size_Factor and time as conditions.
Conditions are used only during decoding, not encoding.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from embedding.vae import VAE

# Set up scanpy settings
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=100, facecolor='white', frameon=False)


def load_and_prepare_data_with_conditions(data_path, use_log1p=True):
    """Load data and prepare conditions for training."""
    print("Loading scRNA-seq data...")
    adata = sc.read_h5ad(data_path)
    
    print(f"Data shape: {adata.shape}")
    print(f"Available layers: {list(adata.layers.keys())}")
    print(f"Available annotations: {list(adata.obs.columns)}")
    
    # Use log1p layer
    if use_log1p and "log1p" in adata.layers:
        X = adata.layers["log1p"]
        print("Using log1p layer")
    else:
        X = adata.X
        print("Using X (default layer)")
    
    # Convert to dense if sparse
    if hasattr(X, "toarray"):
        X = X.toarray()
    
    # Extract conditions
    counts_size_factor = adata.obs['counts_Size_Factor'].values.reshape(-1, 1)
    time = adata.obs['time'].values.reshape(-1, 1)
    
    # Normalize conditions
    scaler_size = StandardScaler()
    scaler_time = StandardScaler()
    
    counts_size_factor_norm = scaler_size.fit_transform(counts_size_factor)
    time_norm = scaler_time.fit_transform(time)
    
    # Combine conditions
    conditions = np.concatenate([counts_size_factor_norm, time_norm], axis=1)
    
    print(f"\nCondition information:")
    print(f"  counts_Size_Factor range: [{counts_size_factor.min():.2f}, {counts_size_factor.max():.2f}]")
    print(f"  time points: {np.unique(time.flatten())}")
    print(f"  Combined condition dim: {conditions.shape[1]}")
    
    return adata, X, conditions, (scaler_size, scaler_time)


def prepare_tensors_with_conditions(X_train, X_val, cond_train, cond_val):
    """Convert to tensors."""
    X_train_t = torch.FloatTensor(X_train)
    X_val_t = torch.FloatTensor(X_val)
    cond_train_t = torch.FloatTensor(cond_train)
    cond_val_t = torch.FloatTensor(cond_val)
    
    return X_train_t, X_val_t, cond_train_t, cond_val_t


def train_cvae(model, train_loader, val_loader, n_epochs=30, lr=5e-4, device='cuda'):
    """Train the conditional VAE."""
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    best_model_state = None
    train_losses = []
    val_losses = []
    
    # Print conditioning configuration
    print(f"\nConditioning configuration:")
    print(f"  Encoder uses conditions: {model.condition_on_encoder}")
    print(f"  Decoder uses conditions: {model.condition_on_decoder}")
    
    # Start training timer
    train_start_time = time.time()
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0
        train_recon = 0
        train_kl = 0
        n_train_batches = 0
        
        for batch in train_loader:
            x_batch = batch[0].to(device)
            cond_batch = batch[1].to(device)
            
            # Forward pass with conditions
            outputs = model(x_batch, cond_batch)
            losses = model.loss(x_batch, outputs)
            
            # Skip if loss is NaN
            if torch.isnan(losses['total']):
                print(f"Warning: NaN loss detected at epoch {epoch+1}, skipping batch")
                continue
            
            # Backward pass
            optimizer.zero_grad()
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += losses['total'].item()
            train_recon += losses['reconstruction'].item()
            train_kl += losses['kl'].item()
            n_train_batches += 1
        
        # Validation
        model.eval()
        val_loss = 0
        val_recon = 0
        val_kl = 0
        n_val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                x_batch = batch[0].to(device)
                cond_batch = batch[1].to(device)
                
                outputs = model(x_batch, cond_batch)
                losses = model.loss(x_batch, outputs)
                
                val_loss += losses['total'].item()
                val_recon += losses['reconstruction'].item()
                val_kl += losses['kl'].item()
                n_val_batches += 1
        
        # Average losses
        train_loss = train_loss / n_train_batches
        train_recon = train_recon / n_train_batches
        train_kl = train_kl / n_train_batches
        val_loss = val_loss / n_val_batches
        val_recon = val_recon / n_val_batches
        val_kl = val_kl / n_val_batches
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            best_epoch = epoch + 1
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {train_loss:.6f} (Recon: {train_recon:.6f}, KL: {train_kl:.6f}), "
                  f"Val Loss: {val_loss:.6f} (Recon: {val_recon:.6f}, KL: {val_kl:.6f})")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model from epoch {best_epoch} with val loss {best_val_loss:.6f}")
    
    # Calculate training time
    train_time = time.time() - train_start_time
    print(f"Total training time: {train_time:.2f} seconds ({train_time/60:.2f} minutes)")
    
    return model, train_losses, val_losses, train_time


def test_conditional_generation(model, adata, X, conditions, scalers, device='cuda'):
    """Test conditional generation with different size factors and time points."""
    model.eval()
    scaler_size, scaler_time = scalers
    
    # Select a few representative cells
    np.random.seed(42)
    test_indices = np.random.choice(len(X), 100, replace=False)
    X_test = torch.FloatTensor(X[test_indices]).to(device)
    conditions_test = torch.FloatTensor(conditions[test_indices]).to(device)
    
    with torch.no_grad():
        # Encode without conditions (since condition_on_encoder=False)
        encoder_output = model.encode(X_test, conditions_test)
        mu = encoder_output['mu']
        
        # Test 1: Decode with original conditions
        recon_original = model.decode(mu, conditions_test)
        
        # Test 2: Decode with swapped time (keep size factor)
        conditions_swapped = conditions_test.clone()
        # Shuffle time component (index 1)
        conditions_swapped[:, 1] = conditions_swapped[torch.randperm(len(conditions_swapped)), 1]
        recon_swapped_time = model.decode(mu, conditions_swapped)
        
        # Test 3: Decode with swapped size factor (keep time)
        conditions_swapped2 = conditions_test.clone()
        # Shuffle size factor component (index 0)
        conditions_swapped2[:, 0] = conditions_swapped2[torch.randperm(len(conditions_swapped2)), 0]
        recon_swapped_size = model.decode(mu, conditions_swapped2)
    
    # Calculate differences
    diff_time = torch.mean((recon_original - recon_swapped_time).pow(2)).item()
    diff_size = torch.mean((recon_original - recon_swapped_size).pow(2)).item()
    
    print(f"\nConditional generation test:")
    print(f"  MSE when swapping time: {diff_time:.6f}")
    print(f"  MSE when swapping size factor: {diff_size:.6f}")


def main():
    # Configuration
    data_path = os.path.expanduser("~/oak/analysis-proj/dynode-training/mm-heart-flow-v2/adata_merged.h5ad")
    latent_dim = 32
    n_epochs = 50  # Shorter for testing
    batch_size = 256
    learning_rate = 5e-4
    test_size = 0.1
    n_cells_per_time = 5000
    beta = 0.1
    
    # Load data with conditions
    adata, X, conditions, scalers = load_and_prepare_data_with_conditions(data_path, use_log1p=True)
    
    print(f"\nDataset info:")
    print(f"  Total cells: {adata.shape[0]:,}")
    print(f"  Total genes: {adata.shape[1]:,}")
    print(f"  Condition dimensions: {conditions.shape[1]}")
    
    # Train/validation split (stratify by time to ensure balance)
    time_labels = adata.obs['time'].values
    X_train, X_val, cond_train, cond_val, time_train, time_val = train_test_split(
        X, conditions, time_labels, test_size=test_size, random_state=42, stratify=time_labels
    )
    
    print(f"\nTrain/Val split:")
    print(f"  Training: {X_train.shape[0]:,} cells")
    print(f"  Validation: {X_val.shape[0]:,} cells")
    
    # Convert to tensors
    X_train_t, X_val_t, cond_train_t, cond_val_t = prepare_tensors_with_conditions(
        X_train, X_val, cond_train, cond_val
    )
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_t, cond_train_t)
    val_dataset = TensorDataset(X_val_t, cond_val_t)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Create conditional VAE model
    model = VAE(
        input_dim=X.shape[1],
        latent_dim=latent_dim,
        encoder_hidden_dims=[512, 256],
        decoder_hidden_dims=[256, 512],
        activation='silu',
        dropout=0.0,
        bias=True,
        beta=beta,
        output_activation='softplus',
        condition_dim=conditions.shape[1],  # 2D: size factor + time
        condition_on_encoder=False,  # Don't use conditions in encoder
        condition_on_decoder=True,   # Use conditions only in decoder
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel architecture:")
    print(f"  Input dim: {X.shape[1]}")
    print(f"  Latent dim: {latent_dim}")
    print(f"  Condition dim: {conditions.shape[1]}")
    print(f"  Beta: {beta}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Condition on encoder: {model.condition_on_encoder}")
    print(f"  Condition on decoder: {model.condition_on_decoder}")
    
    # Train model
    print(f"\nTraining conditional VAE for {n_epochs} epochs...")
    model, train_losses, val_losses, train_time = train_cvae(
        model, train_loader, val_loader, 
        n_epochs=n_epochs, lr=learning_rate, device=device
    )
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train', linewidth=2)
    plt.plot(val_losses, label='Validation', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.title('Training Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    # Plot last 20% of epochs
    start_idx = int(0.8 * len(train_losses))
    plt.plot(range(start_idx, len(train_losses)), train_losses[start_idx:], 
             'o-', label='Train', markersize=4)
    plt.plot(range(start_idx, len(val_losses)), val_losses[start_idx:], 
             'o-', label='Validation', markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.title('Final Epochs Detail')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.suptitle('Conditional VAE Training Progress')
    plt.tight_layout()
    plt.savefig('cvae_training_curves.png', dpi=150, bbox_inches='tight')
    print("Training curves saved to cvae_training_curves.png")
    
    # Test conditional generation
    test_conditional_generation(model, adata, X, conditions, scalers, device)
    
    # Encode and visualize latent space
    print("\nEncoding dataset and visualizing latent space...")
    model.eval()
    
    # Sample cells for visualization
    from test_ae import downsample_by_timepoint
    sampled_indices = downsample_by_timepoint(adata, n_cells_per_time=n_cells_per_time)
    
    X_vis = X[sampled_indices]
    conditions_vis = conditions[sampled_indices]
    
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_vis).to(device)
        cond_tensor = torch.FloatTensor(conditions_vis).to(device)
        
        # Encode (conditions not used since condition_on_encoder=False)
        encoder_output = model.encode(X_tensor, cond_tensor)
        latent_vis = encoder_output['mu'].cpu().numpy()
    
    # Create visualization
    adata_vis = adata[sampled_indices].copy()
    adata_vis.obsm['X_latent'] = latent_vis
    
    # UMAP on latent space
    sc.pp.neighbors(adata_vis, use_rep='X_latent', n_neighbors=30)
    sc.tl.umap(adata_vis)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Color by cell type
    sc.pl.umap(adata_vis, color='Level1_update', ax=axes[0], show=False,
               title='Cell Types', legend_loc='right margin', s=20)
    
    # Color by time
    sc.pl.umap(adata_vis, color='time', ax=axes[1], show=False,
               title='Time Points', s=20)
    
    # Color by size factor
    sc.pl.umap(adata_vis, color='counts_Size_Factor', ax=axes[2], show=False,
               title='Size Factor', s=20)
    
    plt.suptitle('Conditional VAE Latent Space (Decoder-only Conditioning)', fontsize=16)
    plt.tight_layout()
    plt.savefig('cvae_latent_visualization.png', dpi=150, bbox_inches='tight')
    print("\nLatent visualization saved to cvae_latent_visualization.png")
    
    # Save model
    model_path = 'cvae_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'input_dim': X.shape[1],
            'latent_dim': latent_dim,
            'encoder_hidden_dims': [512, 256],
            'decoder_hidden_dims': [256, 512],
            'activation': 'silu',
            'beta': beta,
            'condition_dim': conditions.shape[1],
            'condition_on_encoder': False,
            'condition_on_decoder': True,
        },
        'scalers': scalers,
        'train_time': train_time,
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1],
    }, model_path)
    print(f"\nModel saved to {model_path}")
    
    print("\n" + "="*50)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("="*50)
    print("Generated files:")
    print("  - cvae_latent_visualization.png")
    print("  - cvae_training_curves.png")
    print("  - cvae_model.pt")
    
    return model, adata_vis


if __name__ == "__main__":
    model, adata_vis = main()