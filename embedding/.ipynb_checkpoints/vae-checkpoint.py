import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.ffn import FFN


class VAE(nn.Module):
    """
    Conditional Variational Autoencoder using FFN modules for encoding and decoding.
    
    Architecture:
        Input (+ condition) -> FFN Encoder blocks -> (μ, log σ²) -> z ~ N(μ, σ²) -> z (+ condition) -> FFN Decoder blocks -> Output
    
    Args:
        input_dim (int): Input dimension.
        latent_dim (int): Latent space dimension.
        hidden_dim (int): Hidden dimension for encoder/decoder FFN blocks.
        condition_dim (int | None): Dimension of conditioning vector. If None, operates as standard VAE.
        condition_on_encoder (bool): Whether to use conditioning in encoder. Default True.
        condition_on_decoder (bool): Whether to use conditioning in decoder. Default True.
        encoder_kind (str): Activation type for encoder {"swiglu", "geglu", "gelu", "silu"}.
        decoder_kind (str): Activation type for decoder {"swiglu", "geglu", "gelu", "silu"}.
        dropout (float): Dropout probability for FFN blocks.
        bias (bool): Whether to use bias in linear layers.
        spectral_norm (bool): Whether to apply spectral normalization.
        align_to (int | None): Alignment for auto-inferred hidden dims.
    """
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int = 1024,
        condition_dim: int | None = None,
        condition_on_encoder: bool = True,
        condition_on_decoder: bool = True,
        encoder_kind: str = "silu",
        decoder_kind: str = "silu",
        dropout: float = 0.0,
        bias: bool = True,
        spectral_norm: bool = False,
        align_to: int | None = None,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.condition_dim = condition_dim
        self.condition_on_encoder = condition_on_encoder
        self.condition_on_decoder = condition_on_decoder
        
        # Determine input dimensions for encoder and decoder based on conditioning flags
        encoder_input_dim = input_dim
        decoder_input_dim = latent_dim
        
        if condition_dim is not None:
            if condition_on_encoder:
                encoder_input_dim += condition_dim
            if condition_on_decoder:
                decoder_input_dim += condition_dim
        
        # Encoder: 3 FFN blocks
        self.encoder_block1 = FFN(
            d_model=encoder_input_dim,
            hidden_dim=None,  # Auto-infer based on kind
            out_dim=hidden_dim,
            kind=encoder_kind,
            dropout=dropout,
            bias=bias,
            spectral_norm=spectral_norm,
            align_to=align_to,
        )
        self.encoder_norm1 = nn.LayerNorm(hidden_dim)
        
        self.encoder_block2 = FFN(
            d_model=hidden_dim,
            hidden_dim=None,
            out_dim=hidden_dim,
            kind=encoder_kind,
            dropout=dropout,
            bias=bias,
            spectral_norm=spectral_norm,
            align_to=align_to,
        )
        self.encoder_norm2 = nn.LayerNorm(hidden_dim)
        
        self.encoder_block3 = FFN(
            d_model=hidden_dim,
            hidden_dim=None,
            out_dim=hidden_dim,
            kind=encoder_kind,
            dropout=dropout,
            bias=bias,
            spectral_norm=spectral_norm,
            align_to=align_to,
        )
        self.encoder_norm3 = nn.LayerNorm(hidden_dim)
        
        # Mean and log variance projections
        self.fc_mu = nn.Linear(hidden_dim, latent_dim, bias=bias)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim, bias=bias)
        
        # Decoder: 2 FFN blocks + final projection
        self.decoder_block1 = FFN(
            d_model=decoder_input_dim,
            hidden_dim=None,
            out_dim=hidden_dim,
            kind=decoder_kind,
            dropout=dropout,
            bias=bias,
            spectral_norm=spectral_norm,
            align_to=align_to,
        )
        self.decoder_norm1 = nn.LayerNorm(hidden_dim)
        
        self.decoder_block2 = FFN(
            d_model=hidden_dim,
            hidden_dim=None,
            out_dim=hidden_dim,
            kind=decoder_kind,
            dropout=dropout,
            bias=bias,
            spectral_norm=spectral_norm,
            align_to=align_to,
        )
        self.decoder_norm2 = nn.LayerNorm(hidden_dim)
        
        # Final projection with softplus activation
        self.decoder_proj = nn.Linear(hidden_dim, input_dim, bias=bias)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Kaiming normal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def encode(self, x: torch.Tensor, condition: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent distribution parameters.
        
        Args:
            x: Input tensor of shape (..., input_dim).
            condition: Optional conditioning tensor of shape (..., condition_dim).
        
        Returns:
            Tuple of (mu, logvar):
                - mu: Mean of shape (..., latent_dim)
                - logvar: Log variance of shape (..., latent_dim)
        """
        # Concatenate condition if provided and encoder conditioning is enabled
        if condition is not None and self.condition_on_encoder:
            assert condition.shape[:-1] == x.shape[:-1], "Condition and input must have same batch dimensions"
            h = torch.cat([x, condition], dim=-1)
        else:
            h = x
            
        # Encoder blocks with layer norm
        h = self.encoder_norm1(self.encoder_block1(h))
        h = self.encoder_norm2(self.encoder_block2(h))
        h = self.encoder_norm3(self.encoder_block3(h))
        
        # Get distribution parameters
        mu = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), max=10)  # Clamp for numerical stability
        
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from N(0, 1).
        
        Args:
            mu: Mean of shape (..., latent_dim).
            logvar: Log variance of shape (..., latent_dim).
        
        Returns:
            Sampled latent vector of shape (..., latent_dim).
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        """
        Decode latent representation to output.
        
        Args:
            z: Latent tensor of shape (..., latent_dim).
            condition: Optional conditioning tensor of shape (..., condition_dim).
        
        Returns:
            Output tensor of shape (..., input_dim).
        """
        # Concatenate condition if provided and decoder conditioning is enabled
        if condition is not None and self.condition_on_decoder:
            assert condition.shape[:-1] == z.shape[:-1], "Condition and latent must have same batch dimensions"
            h = torch.cat([z, condition], dim=-1)
        else:
            h = z
            
        # Decoder blocks with layer norm
        h = self.decoder_norm1(self.decoder_block1(h))
        h = self.decoder_norm2(self.decoder_block2(h))
        
        # Final projection with softplus to ensure non-negative outputs
        return F.softplus(self.decoder_proj(h))
    
    def forward(self, x: torch.Tensor, condition: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the VAE.
        
        Args:
            x: Input tensor of shape (..., input_dim).
            condition: Optional conditioning tensor of shape (..., condition_dim).
        
        Returns:
            Tuple of (x_recon, mu, logvar):
                - x_recon: Reconstructed tensor of shape (..., input_dim)
                - mu: Mean of latent distribution of shape (..., latent_dim)
                - logvar: Log variance of latent distribution of shape (..., latent_dim)
        """
        mu, logvar = self.encode(x, condition)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, condition)
        return x_recon, mu, logvar
    
    def sample(self, num_samples: int, condition: torch.Tensor | None = None, device: torch.device | None = None) -> torch.Tensor:
        """
        Sample from the latent space and decode to generate new samples.
        
        Args:
            num_samples: Number of samples to generate.
            condition: Optional conditioning tensor of shape (num_samples, condition_dim) or 
                      (1, condition_dim) to be broadcasted.
            device: Device to generate samples on. If None, uses model's device.
        
        Returns:
            Generated samples of shape (num_samples, input_dim).
        """
        if device is None:
            device = next(self.parameters()).device
        
        z = torch.randn(num_samples, self.latent_dim, device=device)
        
        # Handle conditioning
        if condition is not None:
            if condition.shape[0] == 1 and num_samples > 1:
                # Broadcast condition to all samples
                condition = condition.expand(num_samples, -1)
            elif condition.shape[0] != num_samples:
                raise ValueError(f"Condition batch size {condition.shape[0]} must be 1 or {num_samples}")
        
        return self.decode(z, condition)


def vae_loss(x_recon, x, mu, logvar, beta=1.0):
    """
    Compute VAE loss = Reconstruction loss + β * KL divergence.
    
    Args:
        x_recon: Reconstructed data of shape (batch_size, ...)
        x: Original data of shape (batch_size, ...)
        mu: Latent mean of shape (batch_size, latent_dim)
        logvar: Latent log variance of shape (batch_size, latent_dim)
        beta: Weight for KL divergence term (β-VAE)
    
    Returns:
        Dictionary with total loss and individual components
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(x_recon, x, reduction='mean')
    
    # KL divergence loss
    # KL(N(μ, σ²) || N(0, 1)) = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss
    total_loss = recon_loss + beta * kl_loss
    
    return {
        'loss': total_loss,
        'reconstruction_loss': recon_loss,
        'kl_loss': kl_loss
    }