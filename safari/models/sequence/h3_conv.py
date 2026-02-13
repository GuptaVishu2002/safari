import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from einops import rearrange

try:
    from safari.ops.fftconv import fftconv_func
    HAS_FFTCONV = True
except ImportError:
    fftconv_func = None
    HAS_FFTCONV = False


@torch.jit.script
def mul_sum(q, y):
    """Efficient multiplication and sum for multi-head gating."""
    return (q * y).sum(dim=1)


class H3Conv(nn.Module):
    def __init__(
            self,
            d_model,
            l_max=None,
            head_dim=1,
            use_fast_fftconv=False,
            dropout=0.0,
            layer_idx=None,
            device=None, 
            dtype=None,
            # Long convolution specific arguments
            lam=0.001,  # λ for squash operator (paper: 0.001-0.005)
            kernel_dropout=0.0,  # Dropout on kernel weights (paper: 0.1-0.5)
            kernel_init='geometric',  # 'geometric' or 'random'
            smooth_kernel=False,  # Apply smoothing operator
            smooth_width=3,  # Width for smoothing (2p+1)
            **kwargs,  # Absorb any extra kwargs for compatibility
        ):
        """
        Args:
            d_model: Model dimension
            l_max: Maximum sequence length
            head_dim: Dimension per head (d_model must be divisible by head_dim)
            use_fast_fftconv: Use optimized fftconv implementation
            dropout: Standard dropout (not used on kernels)
            layer_idx: Layer index
            
        Long convolution parameters:
            lam: Squash regularization parameter λ (default 0.001)
            kernel_dropout: Dropout rate for kernel weights (default 0.0)
            kernel_init: 'geometric' or 'random' initialization
            smooth_kernel: Whether to apply smoothing operator
            smooth_width: Width p for smoothing operator (uses 2p+1 average)
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        self.d_model = d_model
        self.head_dim = head_dim
        assert d_model % head_dim == 0, f"d_model ({d_model}) must be divisible by head_dim ({head_dim})"
        self.H = d_model // head_dim  # Number of heads
        self.L = l_max
        self.layer_idx = layer_idx
        self.use_fast_fftconv = use_fast_fftconv and HAS_FFTCONV
        
        # Regularization parameters
        self.lam = lam
        self.kernel_dropout = kernel_dropout
        self.kernel_init = kernel_init
        self.smooth_kernel = smooth_kernel
        self.smooth_width = smooth_width
        
        if self.use_fast_fftconv and not HAS_FFTCONV:
            print("Warning: fftconv not available, falling back to PyTorch FFT")
            self.use_fast_fftconv = False

        # Q, K, V projections (same as H3)
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=True, **factory_kwargs)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=True, **factory_kwargs)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=True, **factory_kwargs)
        
        # Direct convolution kernel parameterization (NOT SSM-based)
        # K shift kernel: one kernel per head
        self.k_kernel = nn.Parameter(torch.randn(self.H, self.L, **factory_kwargs))
        self.k_D = nn.Parameter(torch.randn(self.H, **factory_kwargs))
        
        # Main convolution kernel: one kernel per head
        self.kernel = nn.Parameter(torch.randn(self.H, self.L, **factory_kwargs))
        self.D = nn.Parameter(torch.randn(self.H, **factory_kwargs))

        # Initialize kernels
        self._initialize_kernels()

        # Pointwise output transformation
        self.output_linear = nn.Linear(self.d_model, self.d_model, **factory_kwargs)

    def _initialize_kernels(self):
        with torch.no_grad():
            if self.kernel_init == 'geometric':
                for h in range(self.H):
                    for k in range(self.L):
                        # Geometric decay: decays across sequence and across heads
                        decay = np.exp(-k / self.L * (self.H / 2) * h / self.H)
                        self.kernel.data[h, k] = torch.randn(1, device=self.kernel.device).item() * decay
                        self.k_kernel.data[h, k] = torch.randn(1, device=self.k_kernel.device).item() * decay
            else:
                # Random initialization
                nn.init.normal_(self.kernel, mean=0, std=0.02)
                nn.init.normal_(self.k_kernel, mean=0, std=0.02)
            
            # Initialize D parameters (skip connections)
            nn.init.normal_(self.D, mean=0, std=0.02)
            nn.init.normal_(self.k_D, mean=0, std=0.02)

    def _apply_squash(self, kernel):
        return torch.sign(kernel) * torch.clamp(torch.abs(kernel) - self.lam, min=0)
    
    def _apply_smooth(self, kernel, width=None):
        if width is None:
            width = self.smooth_width
        
        # Use 1D average pooling
        # kernel shape: (H, L)
        kernel = kernel.unsqueeze(1)  # (H, 1, L)
        smoothed = F.avg_pool1d(
            kernel, 
            kernel_size=2*width+1, 
            stride=1, 
            padding=width,
            count_include_pad=False  # Don't count padding in average
        )
        return smoothed.squeeze(1)  # (H, L)

    def _regularize_kernel(self, kernel):
        # 1. Dropout on kernel weights
        if self.training and self.kernel_dropout > 0:
            kernel = F.dropout(kernel, p=self.kernel_dropout, training=True)
        
        # 2. Smoothing (optional, paper shows squash alone works better)
        if self.smooth_kernel:
            kernel = self._apply_smooth(kernel)
        
        # 3. Squash (critical!)
        kernel = self._apply_squash(kernel)
        
        return kernel

    def forward(self, u, inference_params=None):
        L_og = u.size(-2)
        
        # Pad sequence length if using fast fftconv and length is odd
        if self.use_fast_fftconv and L_og % 2 != 0:
            u = F.pad(u, (0, 0, 0, 1))
        L = u.size(-2)

        # Regularize kernels (Algorithm 1)
        ssm_kernel = self._regularize_kernel(self.kernel)  # (H, L)
        k_kernel = self._regularize_kernel(self.k_kernel)  # (H, L)

        # Project to Q, K, V
        u_flat = rearrange(u, 'b l h -> (b l) h')
        dtype = (self.q_proj.weight.dtype if not torch.is_autocast_enabled()
                 else torch.get_autocast_gpu_dtype())
        
        q = self.q_proj.weight @ u_flat.T + self.q_proj.bias.to(dtype).unsqueeze(-1)
        k = self.k_proj.weight @ u_flat.T + self.k_proj.bias.to(dtype).unsqueeze(-1)
        v = self.v_proj.weight @ u_flat.T + self.v_proj.bias.to(dtype).unsqueeze(-1)
        q, k, v = [rearrange(x, 'h (b l) -> b h l', l=L) for x in [q, k, v]]

        # Apply shift convolution to K
        if not self.use_fast_fftconv:
            # Standard FFT convolution
            fft_size = 2 * L
            k_kernel_f = torch.fft.rfft(k_kernel, n=fft_size)  # (H, L+1)
            k_f = torch.fft.rfft(k.to(ssm_kernel.dtype), n=fft_size)  # (B, H, L+1)
            shift_k_out = torch.fft.irfft(k_kernel_f * k_f, n=fft_size)[..., :L]
            k = shift_k_out + rearrange(self.k_D, 'h -> 1 h 1') * k
        else:
            # Fast fftconv
            dropout_mask = None
            k = fftconv_func(k, k_kernel, self.k_D, dropout_mask, False, False, True)
            # Fix stride for batch_size=1 case
            k = rearrange(rearrange(k, 'b h l -> h b l'), 'h b l -> b h l')

        # Main H3 computation: y = q * (ssm_kernel * (k * v))
        if not self.use_fast_fftconv:
            fft_size = 2 * L
            
            # Compute element-wise k * v, reshape for multi-head
            kv = (rearrange(k, 'b (h d1) l -> b d1 1 h l', d1=self.head_dim)
                    * rearrange(v, 'b (h d2) l -> b 1 d2 h l', d2=self.head_dim))  # (B, d1, d2, H, L)
            
            # Apply main convolution to kv
            kv_f = torch.fft.rfft(kv.to(dtype=ssm_kernel.dtype), n=fft_size) / fft_size
            ssm_kernel_f = torch.fft.rfft(ssm_kernel, n=fft_size)  # (H, L+1)
            y = torch.fft.irfft(kv_f * ssm_kernel_f, n=fft_size, norm='forward')[..., :L]  # (B, d1, d2, H, L)
            
            # Add skip connection with D
            y = y + kv * self.D.unsqueeze(-1)  # (B, d1, d2, H, L)
            
            # Multiply by query (gating mechanism)
            q = rearrange(q, 'b (h d1) l -> b d1 1 h l', d1=self.head_dim)
            if self.head_dim > 1:
                y = mul_sum(y, q)
                y = rearrange(y, 'b d h l -> b (d h) l')
            else:
                y = rearrange(y * q, 'b 1 1 h l -> b h l')
        else:
            # Fast fftconv path
            dropout_mask = None
            y = fftconv_func(k, ssm_kernel, self.D,
                             dropout_mask, False, torch.is_autocast_enabled(), True,
                             v, self.head_dim, q)

        y = rearrange(y, 'b h l -> b l h')

        # Output projection
        if not torch.is_autocast_enabled():
            y = y.to(dtype=self.output_linear.weight.dtype)
        y = self.output_linear(y)
        
        # Remove padding if added
        if L_og < L:
            y = y[:, :L_og, :]

        return y

    def extra_repr(self):
        """String representation for debugging."""
        return (f'd_model={self.d_model}, heads={self.H}, head_dim={self.head_dim}, '
                f'l_max={self.L}, lam={self.lam}, kernel_dropout={self.kernel_dropout}, '
                f'init={self.kernel_init}, smooth={self.smooth_kernel}')