import torch.nn as nn
import torch


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config['emb_dim'] % config['n_heads'] == 0, "d_out needs to divisible by heads"

        self.d_out = config['emb_dim']
        self.num_heads = config['n_heads']
        self.head_dim = self.d_out // self.num_heads

        self.W_query = nn.Linear(self.d_out, self.d_out, bias=False, dtype = config['dtype'])
        self.W_key = nn.Linear(self.d_out, self.d_out, bias=False, dtype = config['dtype'])
        self.W_value = nn.Linear(self.d_out, self.d_out, bias=False, dtype = config['dtype'])
        self.out_proj = nn.Linear(self.d_out, self.d_out, bias=False, dtype = config['dtype'])

        self.register_buffer('mask', torch.triu(torch.ones(config['context_length'], 
                                                            config['context_length']), diagonal = 1))

        cos, sin = self.precompute_rope_params(head_dim = self.head_dim, context_length = config["context_length"])

        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def precompute_rope_params(head_dim, theta_base=10_000, context_length=4096):
        assert head_dim % 2 == 0, "Embedding dimension must be even"

        inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim))

        positions = torch.arange(context_length)

        angles = positions[:, None] * inv_freq[None, :] 

        angles = torch.cat([angles, angles], dim=1) 

        cos = torch.cos(angles)
        sin = torch.sin(angles)

        return cos, sin

    def compute_rope(x, cos, sin):
        # x: (batch_size, num_heads, seq_len, head_dim)
        batch_size, num_heads, seq_len, head_dim = x.shape
        assert head_dim % 2 == 0, "Head dimension must be even"

        # Split x into first half and second half
        x1 = x[..., : head_dim // 2]  # First half
        x2 = x[..., head_dim // 2 :]  # Second half

        # Adjust sin and cos shapes
        cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, head_dim)
        sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

        # Apply the rotary transformation
        rotated = torch.cat((-x2, x1), dim=-1)
        x_rotated = (x * cos) + (rotated * sin)

        return x_rotated.to(dtype=x.dtype)
    
    def forward(self, x):
        batch_size, num_tokens, d_in = x.shape

        keys = self.W_k(x)
        queries = self.W_q(x)
        values = self.W_v(x)

        keys = keys.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        values = values.view(batch_size, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # RoPE Embedding
        keys = self.compute_rope(keys, self.cos, self.sin)
        queries = self.compute_rope(queries, self.cos, self.sin)

        #QK^T
        attn_scores = queries @ keys.transpose(2, 3)

        # Masking
        mask_bool  = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim = -1)
        
        # Value 
        context_vec = (attn_weights @ values).transpose(1, 2)
    
        context_vec = context_vec.reshape(batch_size, num_tokens, self.d_out)

        # Out Projection
        context_vec = self.out_proj(context_vec)

        return context_vec


class SiLU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x * torch.sigmoid(x)

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ffn_1 = nn.Linear(config['emb_dim'], config['hidden_dim'], dtpye = config["dtype"], bias = False)
        self.ffn_2 = nn.Linear(config['emb_dim'], config['hidden_dim'], dtpye = config["dtype"], bias = False)
        self.ffn_3 = nn.Linear(config["hidden_dim"], config["emb_dim"], dtype = config["dtype"], bias = False)
        self.silu = SiLU()

    def forward(self, x):
        x = self.silu(self.ffn_1(x)) * self.ffn_2(x)
        x = self.ffn_3(x)
        return x


class RMSNorm(nn.Module):
    def __init__(self, config, eps = 1e-5):
        super().__init__()
        self.emb_dim = config['emb_dim']
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(self.emb_dim)).float()
    
    def forward(self, x):

        means = x.pow(2).mean(dim = -1, keepdim = True)
        x_normed = x * torch.rsqrt(means + self.eps)
        return (x_normed * self.weight).to(dtype = x.dtype)
    

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = MultiHeadAttention(config)
        self.ff = FeedForward(config)
        self.norm1 = RMSNorm(config['emb_dim'])
        self.norm2 = RMSNorm(config['emb_dim'])

    def forward(self, x):
        x = self.attn(self.norm1(x)) + x
        x = self.ff(self.norm2(x)) + x
        return x

class Llama2Model(nn.Module):
    
    def __init__(self, config):
        super().__init__()

        self.token_emb = nn.Embedding(config['vocab_size'], config['emb_dim'])
        
        self.transfomer_block = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config['n_layers'])]
        )

        self.final_norm = RMSNorm(config['emb_dim'])
        self.out_head = nn.Linear(config['emb_dim'], config['vocab_size'], bias = False, dtype= config['dtype'])

    
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        
        tok_emb = self.token_emb(in_idx)

        x = tok_emb

        x = self.transfomer_block(x)

        x = self.final_norm(x)

        x = self.out_head(x)

        return x

