#!/usr/bin/env python
"""
Test autoencoder on scRNA-seq data.
Train on full data, visualize with downsampled cells from each time point.
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
from embedding.autoencoder import Autoencoder

# Set up scanpy settings
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=100, facecolor='white', frameon=False)


def load_and_prepare_data(data_path, use_log1p=True):
    """Load data and prepare for training."""
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
    
    return adata, X


def prepare_tensors(X_train, X_val):
    """Convert to tensors without additional normalization (log1p is already normalized)."""
    X_train_t = torch.FloatTensor(X_train)
    X_val_t = torch.FloatTensor(X_val)
    
    return X_train_t, X_val_t


def train_autoencoder(model, train_loader, val_loader, n_epochs=30, lr=5e-4, device='cuda'):
    """Train the autoencoder."""
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    best_model_state = None
    train_losses = []
    val_losses = []
    
    # Start training timer
    train_start_time = time.time()
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0
        n_train_batches = 0
        for batch in train_loader:
            x_batch = batch[0].to(device)
            
            # Forward pass
            outputs = model(x_batch)
            losses = model.loss(x_batch, outputs)
            
            # Skip if loss is NaN
            if torch.isnan(losses['total']):
                print(f"Warning: NaN loss detected at epoch {epoch+1}, skipping batch")
                continue
            
            # Backward pass
            optimizer.zero_grad()
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Stricter gradient clipping
            optimizer.step()
            
            train_loss += losses['total'].item()
            n_train_batches += 1
        
        # Validation
        model.eval()
        val_loss = 0
        n_val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                x_batch = batch[0].to(device)
                outputs = model(x_batch)
                losses = model.loss(x_batch, outputs)
                val_loss += losses['total'].item()
                n_val_batches += 1
        
        # Average losses
        train_loss = train_loss / n_train_batches
        val_loss = val_loss / n_val_batches
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            best_epoch = epoch + 1
        
        if (epoch + 1) % 10 == 0:  # Print every 10 epochs for 100 epoch training
            print(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model from epoch {best_epoch} with val loss {best_val_loss:.6f}")
    
    # Calculate training time
    train_time = time.time() - train_start_time
    print(f"Total training time: {train_time:.2f} seconds ({train_time/60:.2f} minutes)")
    
    return model, train_losses, val_losses, train_time


def downsample_by_timepoint(adata, n_cells_per_time=5000, seed=42):
    """Downsample cells from each time point."""
    np.random.seed(seed)
    
    time_points = adata.obs['time'].unique()
    print(f"\nTime points found: {sorted(time_points)}")
    
    sampled_indices = []
    
    for tp in sorted(time_points):
        tp_indices = np.where(adata.obs['time'] == tp)[0]
        n_cells_tp = len(tp_indices)
        
        # Sample up to n_cells_per_time cells
        n_sample = min(n_cells_per_time, n_cells_tp)
        sampled_idx = np.random.choice(tp_indices, n_sample, replace=False)
        sampled_indices.extend(sampled_idx)
        
        print(f"  Time {tp}: {n_cells_tp} cells -> sampled {n_sample}")
    
    sampled_indices = np.array(sampled_indices)
    return sampled_indices


def encode_and_visualize(model, adata, X_full, sampled_indices, device='cuda'):
    """Encode data and create visualizations using scanpy style."""
    model.eval()
    
    # Start visualization timer
    vis_start_time = time.time()
    
    # Encode full dataset
    print("\nEncoding full dataset...")
    encode_start = time.time()
    X_tensor = torch.FloatTensor(X_full)
    
    with torch.no_grad():
        latent_list = []
        batch_size = 1000
        
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i+batch_size].to(device)
            latent = model.encode(batch)
            latent_list.append(latent.cpu())
        
        latent_full = torch.cat(latent_list, dim=0).numpy()
    
    print(f"Latent representation shape: {latent_full.shape}")
    
    # Check for NaN or Inf values
    if np.any(np.isnan(latent_full)) or np.any(np.isinf(latent_full)):
        print("Warning: NaN or Inf detected in latent representation!")
        # Replace NaN/Inf with zeros for visualization
        latent_full = np.nan_to_num(latent_full, nan=0.0, posinf=0.0, neginf=0.0)
        print("Replaced NaN/Inf values with zeros for visualization")
    
    # Create AnnData object with latent representation for visualization
    adata_vis = adata[sampled_indices].copy()
    
    # Store latent representation
    adata_vis.obsm['X_latent'] = latent_full[sampled_indices]
    
    # Compute UMAP on latent space
    print("\nComputing UMAP on latent space...")
    sc.pp.neighbors(adata_vis, use_rep='X_latent', n_neighbors=30)
    sc.tl.umap(adata_vis)
    
    # Compute diffusion map on latent space
    print("Computing diffusion map on latent space...")
    sc.tl.diffmap(adata_vis, n_comps=15)
    
    # Create visualizations using scanpy style
    with plt.rc_context():  # Use scanpy's default style
        # Create figure with better layout for legends
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # UMAP visualizations
        # Cell types with legend
        sc.pl.umap(adata_vis, color='Level1_update', ax=axes[0, 0], show=False, 
                   title='UMAP: Cell Types', legend_loc='right margin', s=20,
                   legend_fontsize=8, legend_fontoutline=2)
        
        # Time points
        sc.pl.umap(adata_vis, color='time', ax=axes[0, 1], show=False,
                   title='UMAP: Time Points', s=20, legend_loc='right margin')
        
        # UMAP with both annotations (smaller points for density)
        sc.pl.umap(adata_vis, color='Level1_update', ax=axes[0, 2], show=False,
                   title='UMAP: Cell Types (Overview)', s=5, legend_loc='none')
        
        # Diffusion map visualizations (skip 0-th component)
        adata_vis.obsm['X_diffmap_vis'] = adata_vis.obsm['X_diffmap'][:, 1:3]  # Components 1-2
        
        # Diffusion map with cell types and legend
        sc.pl.embedding(adata_vis, basis='X_diffmap_vis', color='Level1_update', 
                       ax=axes[1, 0], show=False, title='Diffusion Map (1-2): Cell Types', 
                       legend_loc='right margin', s=20, legend_fontsize=8,
                       legend_fontoutline=2)
        
        # Diffusion map with time
        sc.pl.embedding(adata_vis, basis='X_diffmap_vis', color='time',
                       ax=axes[1, 1], show=False, title='Diffusion Map (1-2): Time Points', 
                       s=20, legend_loc='right margin')
        
        # Diffusion components 2-3
        adata_vis.obsm['X_diffmap_23'] = adata_vis.obsm['X_diffmap'][:, 2:4]  # Components 2-3
        sc.pl.embedding(adata_vis, basis='X_diffmap_23', color='time',
                       ax=axes[1, 2], show=False, title='Diffusion Map (2-3): Time Points', 
                       s=20, legend_loc='right margin')
        
        plt.suptitle('Autoencoder Latent Space Analysis (100 epochs)', fontsize=16, y=0.98)
        plt.tight_layout()
        plt.savefig('ae_latent_visualization.png', dpi=150, bbox_inches='tight')
        print("\nVisualization saved to ae_latent_visualization.png")
    
    # Calculate timing
    encode_time = encode_start - vis_start_time  # This will be recalculated
    vis_total_time = time.time() - vis_start_time
    print(f"\nVisualization timing:")
    print(f"  Encoding time: {time.time() - encode_start:.2f} seconds")
    print(f"  Total visualization time: {vis_total_time:.2f} seconds ({vis_total_time/60:.2f} minutes)")
    
    return adata_vis


def main():
    # Configuration
    data_path = os.path.expanduser("~/oak/analysis-proj/dynode-training/mm-heart-flow-v2/adata_merged.h5ad")
    latent_dim = 32
    n_epochs = 100  # Train for 100 epochs
    batch_size = 256
    learning_rate = 5e-4  # Using the better learning rate
    test_size = 0.1
    n_cells_per_time = 5000
    
    # Load data
    adata, X = load_and_prepare_data(data_path, use_log1p=True)
    
    print(f"\nDataset info:")
    print(f"  Total cells: {adata.shape[0]:,}")
    print(f"  Total genes: {adata.shape[1]:,}")
    print(f"  Unique cell types: {adata.obs['Level1_update'].nunique()}")
    print(f"  Unique time points: {adata.obs['time'].nunique()}")
    
    # Train/validation split
    X_train, X_val = train_test_split(X, test_size=test_size, random_state=42)
    print(f"\nTrain/Val split:")
    print(f"  Training: {X_train.shape[0]:,} cells")
    print(f"  Validation: {X_val.shape[0]:,} cells")
    
    # Convert to tensors (no additional normalization needed for log1p)
    X_train_t, X_val_t = prepare_tensors(X_train, X_val)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_t)
    val_dataset = TensorDataset(X_val_t)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Create model
    model = Autoencoder(
        input_dim=X.shape[1],
        latent_dim=latent_dim,
        encoder_hidden_dims=[512, 256],
        decoder_hidden_dims=[256, 512],
        activation='swiglu',
        dropout=0.1,
        bias=True,
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel architecture:")
    print(f"  Input dim: {X.shape[1]}")
    print(f"  Latent dim: {latent_dim}")
    print(f"  Total parameters: {total_params:,}")
    
    # Train model
    print(f"\nTraining for {n_epochs} epochs...")
    model, train_losses, val_losses, train_time = train_autoencoder(
        model, train_loader, val_loader, 
        n_epochs=n_epochs, lr=5e-4, device=device  # Reduced learning rate
    )
    
    # Plot training curves with scanpy style and log scale
    with plt.rc_context({'figure.figsize': (12, 5)}):  # Use scanpy-like style
        sc.settings.set_figure_params(dpi=100, facecolor='white', frameon=True)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot 1: Log scale training curves
        axes[0].semilogy(train_losses, label='Train', linewidth=2, color='#1f77b4')
        axes[0].semilogy(val_losses, label='Validation', linewidth=2, color='#ff7f0e')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss (log scale)', fontsize=12)
        axes[0].set_title('Training Curves - Log Scale', fontsize=14)
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3, which='both')
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)
        
        # Plot 2: Linear scale (validation only since train might be too large)
        axes[1].plot(val_losses, linewidth=2, color='#ff7f0e')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Validation Loss', fontsize=12)
        axes[1].set_title('Validation Loss - Linear Scale', fontsize=14)
        axes[1].grid(True, alpha=0.3)
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)
        
        # Plot 3: Last 10 epochs in detail
        start_epoch = max(0, len(val_losses) - 10)
        epochs = range(start_epoch + 1, len(val_losses) + 1)
        axes[2].plot(epochs, val_losses[start_epoch:], 'o-', linewidth=2, 
                    markersize=6, color='#ff7f0e', label='Validation')
        # Only plot train if values are reasonable
        if max(train_losses[start_epoch:]) < 1e6:  # Avoid plotting exploded values
            axes[2].plot(epochs, train_losses[start_epoch:], 's-', linewidth=2,
                        markersize=5, color='#1f77b4', label='Train')
            axes[2].legend(fontsize=11)
        axes[2].set_xlabel('Epoch', fontsize=12)
        axes[2].set_ylabel('Loss', fontsize=12)
        axes[2].set_title('Last 10 Epochs', fontsize=14)
        axes[2].grid(True, alpha=0.3)
        axes[2].spines['top'].set_visible(False)
        axes[2].spines['right'].set_visible(False)
        
        plt.suptitle('Autoencoder Training Progress', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig('ae_training_curves.png', dpi=150, bbox_inches='tight')
        print("Training curves saved to ae_training_curves.png")
    
    # Save the trained model FIRST
    print("\n" + "="*50)
    print("SAVING TRAINED MODEL")
    print("="*50)
    model_path = 'ae_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'input_dim': X.shape[1],
            'latent_dim': latent_dim,
            'encoder_hidden_dims': [512, 256],
            'decoder_hidden_dims': [256, 512],
            'activation': 'swiglu',
        },
        'train_time': train_time,
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1],
    }, model_path)
    print(f"Model saved to {model_path}")
    
    # Clear the model from memory
    del model
    torch.cuda.empty_cache()
    
    # Load the model for inference
    print("\n" + "="*50)
    print("LOADING MODEL FOR INFERENCE")
    print("="*50)
    
    inference_start = time.time()
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    # Recreate model from config
    model_loaded = Autoencoder(
        input_dim=config['input_dim'],
        latent_dim=config['latent_dim'],
        encoder_hidden_dims=config['encoder_hidden_dims'],
        decoder_hidden_dims=config['decoder_hidden_dims'],
        activation=config['activation'],
        dropout=0.1,
        bias=True,
    )
    
    # Load weights
    model_loaded.load_state_dict(checkpoint['model_state_dict'])
    model_loaded = model_loaded.to(device)
    model_loaded.eval()
    
    print(f"Model loaded successfully from {model_path}")
    print(f"Training time was: {checkpoint['train_time']:.2f} seconds ({checkpoint['train_time']/60:.2f} minutes)")
    print(f"Final training loss: {checkpoint['final_train_loss']:.6f}")
    print(f"Final validation loss: {checkpoint['final_val_loss']:.6f}")
    
    # Downsample for visualization
    sampled_indices = downsample_by_timepoint(adata, n_cells_per_time=n_cells_per_time)
    print(f"\nTotal cells for visualization: {len(sampled_indices):,}")
    
    # Encode and visualize using LOADED model
    adata_vis = encode_and_visualize(
        model_loaded, adata, X, sampled_indices, device
    )
    
    inference_time = time.time() - inference_start
    print(f"\nTotal inference time (loading + visualization): {inference_time:.2f} seconds ({inference_time/60:.2f} minutes)")
    
    print("\n" + "="*50)
    print("TIMING SUMMARY")
    print("="*50)
    print(f"Training time: {train_time:.2f} seconds ({train_time/60:.2f} minutes)")
    print(f"Inference time: {inference_time:.2f} seconds ({inference_time/60:.2f} minutes)")
    print(f"Speed ratio: {train_time/inference_time:.1f}x (training takes {train_time/inference_time:.1f}x longer than inference)")
    
    print("\n" + "="*50)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("="*50)
    print("Generated files:")
    print("  - ae_latent_visualization.png")
    print("  - ae_training_curves.png")
    print("  - ae_model.pt")
    
    return model_loaded, adata_vis


if __name__ == "__main__":
    model, adata_vis = main()