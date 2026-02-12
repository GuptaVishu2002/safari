"""
Hyena layers and components - properly aligned with original.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def fftconv_ref(u, k, D, dropout_mask, gelu=True, k_rev=None):
    seqlen = u.shape[-1]
    fft_size = 2 * seqlen
    k_f = torch.fft.rfft(k, n=fft_size) / fft_size
    if k_rev is not None:
        k_rev_f = torch.fft.rfft(k_rev, n=fft_size) / fft_size
        k_f = k_f + k_rev_f.conj()
    u_f = torch.fft.rfft(u.to(dtype=k.dtype), n=fft_size)
    
    if len(u.shape) > 3:
        k_f = k_f.unsqueeze(1)

    y = torch.fft.irfft(u_f * k_f, n=fft_size, norm='forward')[..., :seqlen]
    out = y + u * D.unsqueeze(-1)
    
    if gelu:
        out = F.gelu(out)
    if dropout_mask is not None:
        return (out * rearrange(dropout_mask, 'b H -> b H 1')).to(dtype=u.dtype)
    else:
        return out.to(dtype=u.dtype)


@torch.jit.script
def mul_sum(q, y):
    return (q * y).sum(dim=1)


class Sin(nn.Module):
    def __init__(self, dim, w=10, train_freq=True):
        super().__init__()
        self.freq = nn.Parameter(w * torch.ones(1, dim)) if train_freq else w * torch.ones(1, dim)

    def forward(self, x):
        return torch.sin(self.freq * x)


class PositionalEmbedding(nn.Module):
    def __init__(self, emb_dim: int, seq_len: int, **kwargs):
        super().__init__()
        self.seq_len = seq_len
        t = torch.linspace(0, 1, self.seq_len)[None, :, None]
        
        if emb_dim > 1:
            bands = (emb_dim - 1) // 2
        
        t_rescaled = torch.linspace(0, seq_len - 1, seq_len)[None, :, None]
        w = 2 * math.pi * t_rescaled / seq_len
        
        f = torch.linspace(1e-4, bands - 1, bands)[None, None]
        z = torch.exp(-1j * f * w)
        z = torch.cat([t, z.real, z.imag], dim=-1)
        
        self.register_buffer('z', z)
        self.register_buffer('t', t)
        
    def forward(self, L):
        return self.z[:, :L], self.t[:, :L]


class ExponentialModulation(nn.Module):
    def __init__(
        self,
        d_model,
        fast_decay_pct=0.3,
        slow_decay_pct=1.5,
        target=1e-2,
        modulate: bool = True,
        shift: float = 0.0,
        **kwargs
    ):
        super().__init__()
        self.modulate = modulate
        self.shift = shift
        max_decay = math.log(target) / fast_decay_pct
        min_decay = math.log(target) / slow_decay_pct
        deltas = torch.linspace(min_decay, max_decay, d_model)[None, None]
        self.register_buffer('deltas', deltas)
        
    def forward(self, t, x):
        if self.modulate:
            decay = torch.exp(-t * self.deltas.abs())
            x = x * (decay + self.shift)
        return x


class HyenaFilter(nn.Module):
    def __init__(
        self, 
        d_model,
        emb_dim=3,
        order=16,
        seq_len=1024,
        dropout=0.0,
        w=1,
        bias=True,
        num_inner_mlps=2,
        normalized=False,
        **kwargs
    ):
        super().__init__()
        self.d_model = d_model
        self.use_bias = bias
        self.bias = nn.Parameter(torch.randn(self.d_model))
        self.dropout = nn.Dropout(dropout)
        
        act = Sin(dim=order, w=w)
        self.emb_dim = emb_dim
        assert emb_dim % 2 != 0 and emb_dim >= 3
        self.seq_len = seq_len
  
        self.pos_emb = PositionalEmbedding(emb_dim, seq_len)

        layers = [nn.Linear(emb_dim, order), act]
        for i in range(num_inner_mlps):
            layers.append(nn.Linear(order, order))
            layers.append(act)
        layers.append(nn.Linear(order, d_model, bias=False))
        
        self.implicit_filter = nn.Sequential(*layers)
        self.modulation = ExponentialModulation(d_model, **kwargs)
        self.normalized = normalized

    def filter(self, L):
        z, t = self.pos_emb(L)
        h = self.implicit_filter(z)
        h = self.modulation(t, h)
        if self.normalized:
            h = h / torch.norm(h, dim=-1, p=1, keepdim=True)
        return h

    def forward(self, x, L, k=None, bias=None):
        if k is None:
            k = self.filter(L)
        
        k = k[0] if type(k) is tuple else k
        if bias is None:
            bias = self.bias
        bias = bias if self.use_bias else 0 * bias

        y = fftconv_ref(x, k, bias, dropout_mask=None, gelu=False)
        return y


class HyenaOperator(nn.Module):
    def __init__(
        self,
        d_model,
        l_max,
        order=2,
        filter_order=64,
        num_heads=1,
        inner_factor=1,
        num_blocks=1,
        dropout=0.0,
        filter_dropout=0.0,
        short_filter_order=3,
        **filter_args,
    ):
        super().__init__()
        
        assert d_model % num_heads == 0
        assert l_max % num_blocks == 0
        
        self.d_model = d_model
        self.order = order
        self.l_max = l_max
        self.num_heads = num_heads
        self.inner_factor = inner_factor
        self.num_blocks = num_blocks
        self.block_dim = l_max // num_blocks
        self.head_dim = d_model // num_heads
        
        self.dropout = nn.Dropout(dropout)
        
        self.out_proj = nn.Linear(self.d_model * inner_factor, self.d_model)
        self.in_proj = nn.Linear(self.d_model, (self.order + 1) * self.d_model)
        
        total_width = self.d_model * self.inner_factor * (self.order + 1)
        self.short_filter = nn.Conv1d(
            in_channels=total_width,
            out_channels=total_width,
            kernel_size=short_filter_order,
            groups=total_width,
            padding=short_filter_order - 1
        )
        
        self.filter_fn = HyenaFilter(
            self.head_dim * self.inner_factor * (self.order - 1),
            order=filter_order,
            seq_len=self.l_max,
            dropout=filter_dropout,
            **filter_args
        )

    def forward(self, u):
        l = u.size(-2)
        l_filter = min(l, self.l_max)
        
        u = self.in_proj(u)
        u = rearrange(u, 'b l d -> b d l')
        
        uc = self.short_filter(u)[..., :l_filter]
        
        uc = rearrange(
            uc, 'b (ho v) (z l) -> b ho v z l',
            z=self.num_blocks,
            ho=self.num_heads,
            v=self.head_dim * (self.order + 1)
        )

        *x, v = uc.split(self.head_dim, dim=2)
        
        k = self.filter_fn.filter(l_filter)
        
        k = rearrange(k, 'c l (v o) -> c o v l', v=self.head_dim, o=self.order - 1)[0]
        bias = rearrange(self.filter_fn.bias, '(v o) -> o v', v=self.head_dim, o=self.order - 1)

        for o, x_i in enumerate(reversed(x[1:])):
            v = self.dropout(v * x_i)
            v = self.filter_fn(v, l_filter, k=k[o], bias=bias[o, None, :, None])

        y = rearrange(
            v * x[0], 'b h v z l -> b (z l) (h v)',
            z=self.num_blocks,
            h=self.num_heads
        )
        y = self.out_proj(y)
        return y

    @property
    def d_output(self):
        return self.d_model


class HyenaModel(nn.Module):
    def __init__(
        self,
        vocabulary,
        n_layers=4,
        d_model=256,
        order=2,
        filter_order=64,
        num_heads=1,
        dropout=0.25,
        max_len=250,
        **hyena_args
    ):
        super(HyenaModel, self).__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vocabulary = vocabulary
        self.vocabulary_size = len(vocabulary)
        self.padding_idx = vocabulary.dictionary["<PAD>"]
        self.d_model = d_model
        self.n_layers = n_layers
        self.dropout = dropout
        self.max_len = max_len
        
        self.embedding = nn.Embedding(
            self.vocabulary_size, d_model, padding_idx=self.padding_idx
        )
        
        self.hyena_layers = nn.ModuleList([
            HyenaOperator(
                d_model=d_model,
                l_max=max_len,
                order=order,
                filter_order=filter_order,
                num_heads=num_heads,
                dropout=dropout,
                **hyena_args
            )
            for _ in range(n_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])
        
        self.dropout_layer = nn.Dropout(dropout)
        self.output_embedding = nn.Linear(d_model, self.vocabulary_size)
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=self.padding_idx, reduction="none"
        )
        
        if torch.cuda.is_available():
            self.cuda()
    
    def forward(self, x):
        x = self.embedding(x)
        
        for hyena_layer, layer_norm in zip(self.hyena_layers, self.layer_norms):
            residual = x
            x = layer_norm(x)
            x = hyena_layer(x)
            x = self.dropout_layer(x)
            x = x + residual
        
        return self.output_embedding(x)
    
    def loss(self, batch):
        if len(batch) == 3:
            padded, lengths, _ = batch
        else:
            padded, lengths = batch
        
        padded = padded.to(self.device)
        if padded.dim() == 2:
            padded = padded.transpose(0, 1)
        
        logits = self(padded)
        targets = padded[:, 1:]
        logits = logits[:, :-1, :]
        
        loss = 0.0
        actual_len = min(logits.shape[1], targets.shape[1])
        for char_idx in range(actual_len):
            loss += self.loss_fn(logits[:, char_idx, :], targets[:, char_idx])
        
        return loss.mean()
    
    def sample(
        self,
        *,
        n_sequences,
        max_len=None,
        return_smiles=True,
        return_losses=False,
        descriptors=None,
    ):
        if max_len is None:
            max_len = self.max_len
        
        self.eval()
        
        start_token = self.vocabulary.dictionary["SOS"]
        stop_token = self.vocabulary.dictionary["EOS"]
        pad_token = self.vocabulary.dictionary["<PAD>"]
        
        inputs = torch.empty(n_sequences).fill_(start_token).long().to(self.device)
        loss_fn = nn.NLLLoss(reduction="none", ignore_index=pad_token)
        
        finished = torch.zeros(n_sequences).byte().to(self.device)
        log_probs = torch.zeros(n_sequences).to(self.device)
        sequences = []
        
        with torch.no_grad():
            for step in range(max_len):
                if step == 0:
                    current_seq = inputs.unsqueeze(1)
                else:
                    seq_list = [inputs.unsqueeze(1)] + sequences
                    current_seq = torch.cat(seq_list, dim=1)
                
                logits = self(current_seq)
                logits = logits[:, -1, :]
                
                logits = torch.clamp(logits, min=-1e4, max=1e4)
                prob = F.softmax(logits, dim=-1)
                
                if torch.isnan(prob).any() or torch.isinf(prob).any():
                    break
                
                outputs = torch.multinomial(prob, num_samples=1).squeeze(1)
                sequences.append(outputs.view(-1, 1))
                
                log_prob = F.log_softmax(logits, dim=-1)
                losses = loss_fn(log_prob, outputs)
                losses[finished.bool()] = 0
                log_probs += losses
                
                finished = torch.ge(finished + (outputs == stop_token), 1)
                if torch.prod(finished) == 1:
                    break
        
        seqs = torch.cat(sequences, 1)
        if return_smiles:
            outputs = [self.vocabulary.decode(seq.cpu().numpy()) for seq in seqs]
        else:
            outputs = sequences
        
        if return_losses:
            return outputs, log_probs.detach().cpu().numpy()
        else:
            return outputs