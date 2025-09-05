import torch
import torch.nn as nn
from utils.ffn import FFN


class Autoencoder(nn.Module):
    """
    Simple autoencoder using FFN modules for encoding and decoding.
    
    Architecture:
        Input -> FFN Encoder -> Latent -> FFN Decoder -> Output
    
    Args:
        input_dim (int): Input dimension.
        latent_dim (int): Latent space dimension.
        encoder_hidden_dim (int | None): Hidden dimension for encoder FFN.
        decoder_hidden_dim (int | None): Hidden dimension for decoder FFN.
        encoder_kind (str): Activation type for encoder {"swiglu", "geglu", "gelu", "silu"}.
        decoder_kind (str): Activation type for decoder {"swiglu", "geglu", "gelu", "silu"}.
        dropout (float): Dropout probability for both encoder and decoder.
        bias (bool): Whether to use bias in linear layers.
        spectral_norm (bool): Whether to apply spectral normalization.
        align_to (int | None): Alignment for auto-inferred hidden dims.
    """
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        encoder_hidden_dim: int | None = None,
        decoder_hidden_dim: int | None = None,
        encoder_kind: str = "swiglu",
        decoder_kind: str = "swiglu",
        dropout: float = 0.0,
        bias: bool = False,
        spectral_norm: bool = False,
        align_to: int | None = None,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder: input_dim -> latent_dim
        self.encoder = FFN(
            d_model=input_dim,
            hidden_dim=encoder_hidden_dim,
            out_dim=latent_dim,
            kind=encoder_kind,
            dropout=dropout,
            bias=bias,
            spectral_norm=spectral_norm,
            align_to=align_to,
        )
        
        # Decoder: latent_dim -> input_dim
        self.decoder = FFN(
            d_model=latent_dim,
            hidden_dim=decoder_hidden_dim,
            out_dim=input_dim,
            kind=decoder_kind,
            dropout=dropout,
            bias=bias,
            spectral_norm=spectral_norm,
            align_to=align_to,
        )
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent representation.
        
        Args:
            x: Input tensor of shape (..., input_dim).
        
        Returns:
            Latent tensor of shape (..., latent_dim).
        """
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to output.
        
        Args:
            z: Latent tensor of shape (..., latent_dim).
        
        Returns:
            Output tensor of shape (..., input_dim).
        """
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the autoencoder.
        
        Args:
            x: Input tensor of shape (..., input_dim).
        
        Returns:
            Tuple of (reconstructed, latent):
                - reconstructed: Output tensor of shape (..., input_dim)
                - latent: Latent representation of shape (..., latent_dim)
        """
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed, latent