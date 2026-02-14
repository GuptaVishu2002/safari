import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from long_conv_kernel import LongConvKernel

try:
    from safari.ops.fftconv import fftconv_func
    HAS_FFTCONV = True
except ImportError:
    fftconv_func = None
    HAS_FFTCONV = False


@torch.jit.script
def mul_sum(q, y):
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
            # Long convolution parameters
            lam=0.001,
            kernel_dropout=0.0,
            kernel_init='geometric',  # 'geometric' or 'random'
            smooth_kernel=False,
            smooth_width=7,
            smooth_freq=False,
            learning_rate=None,
            **kwargs,
        ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        self.d_model = d_model
        self.head_dim = head_dim
        assert d_model % head_dim == 0
        self.H = d_model // head_dim
        self.L = l_max
        self.layer_idx = layer_idx
        self.use_fast_fftconv = use_fast_fftconv and HAS_FFTCONV
        
        if self.use_fast_fftconv and not HAS_FFTCONV:
            print("Warning: fftconv not available, falling back to PyTorch FFT")
            self.use_fast_fftconv = False

        # Q, K, V projections
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=True, **factory_kwargs)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=True, **factory_kwargs)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=True, **factory_kwargs)
        
        # Map kernel_init names
        weight_init_map = {
            'geometric': 'double_exp',
            'random': 'random',
        }
        weight_init = weight_init_map.get(kernel_init, 'random')
        
        # Use LongConvKernel for K shift
        self.k_kernel = LongConvKernel(
            H=self.H,
            L=self.L,
            channels=1,
            learning_rate=learning_rate,
            lam=lam,
            causal=True,
            kernel_dropout=kernel_dropout,
            weight_init=weight_init,
            use_ma_smoothing=smooth_kernel,
            ma_window_len=smooth_width,
            smooth_freq=smooth_freq,
        )
        self.k_D = nn.Parameter(torch.randn(self.H, **factory_kwargs))
        
        # Use LongConvKernel for main convolution
        self.kernel = LongConvKernel(
            H=self.H,
            L=self.L,
            channels=1,
            learning_rate=learning_rate,
            lam=lam,
            causal=True,
            kernel_dropout=kernel_dropout,
            weight_init=weight_init,
            use_ma_smoothing=smooth_kernel,
            ma_window_len=smooth_width,
            smooth_freq=smooth_freq,
        )
        self.D = nn.Parameter(torch.randn(self.H, **factory_kwargs))

        # Initialize D parameters
        with torch.no_grad():
            nn.init.normal_(self.D, mean=0, std=0.02)
            nn.init.normal_(self.k_D, mean=0, std=0.02)

        # Output projection
        self.output_linear = nn.Linear(self.d_model, self.d_model, **factory_kwargs)

    def forward(self, u, inference_params=None):
        L_og = u.size(-2)
        
        if self.use_fast_fftconv and L_og % 2 != 0:
            u = F.pad(u, (0, 0, 0, 1))
        L = u.size(-2)

        ssm_kernel, _ = self.kernel()  # (1, H, L)
        k_kernel, _ = self.k_kernel()  # (1, H, L)
        
        # Remove channel dimension (we only use 1 channel)
        ssm_kernel = ssm_kernel.squeeze(0)  # (H, L)
        k_kernel = k_kernel.squeeze(0)  # (H, L)

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
            fft_size = 2 * L
            k_kernel_f = torch.fft.rfft(k_kernel, n=fft_size)
            k_f = torch.fft.rfft(k.to(ssm_kernel.dtype), n=fft_size)
            shift_k_out = torch.fft.irfft(k_kernel_f * k_f, n=fft_size)[..., :L]
            k = shift_k_out + rearrange(self.k_D, 'h -> 1 h 1') * k
        else:
            dropout_mask = None
            k = fftconv_func(k, k_kernel, self.k_D, dropout_mask, False, False, True)
            k = rearrange(rearrange(k, 'b h l -> h b l'), 'h b l -> b h l')

        # Main H3 computation
        if not self.use_fast_fftconv:
            fft_size = 2 * L
            
            kv = (rearrange(k, 'b (h d1) l -> b d1 1 h l', d1=self.head_dim)
                    * rearrange(v, 'b (h d2) l -> b 1 d2 h l', d2=self.head_dim))
            
            kv_f = torch.fft.rfft(kv.to(dtype=ssm_kernel.dtype), n=fft_size) / fft_size
            ssm_kernel_f = torch.fft.rfft(ssm_kernel, n=fft_size)
            y = torch.fft.irfft(kv_f * ssm_kernel_f, n=fft_size, norm='forward')[..., :L]
            
            y = y + kv * self.D.unsqueeze(-1)
            
            q = rearrange(q, 'b (h d1) l -> b d1 1 h l', d1=self.head_dim)
            if self.head_dim > 1:
                y = mul_sum(y, q)
                y = rearrange(y, 'b d h l -> b (d h) l')
            else:
                y = rearrange(y * q, 'b 1 1 h l -> b h l')
        else:
            dropout_mask = None
            y = fftconv_func(k, ssm_kernel, self.D,
                             dropout_mask, False, torch.is_autocast_enabled(), True,
                             v, self.head_dim, q)

        y = rearrange(y, 'b h l -> b l h')

        if not torch.is_autocast_enabled():
            y = y.to(dtype=self.output_linear.weight.dtype)
        y = self.output_linear(y)
        
        if L_og < L:
            y = y[:, :L_og, :]

        return y