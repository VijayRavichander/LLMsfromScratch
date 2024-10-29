import torch.nn as nn
import torch



class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config['emb_dim'] % config['n_heads'] == 0, "d_out needs to divisible by heads"

        self.d_out = config['emb_dim']
        self.num_heads = config['n_heads']
        self.head_dim = self.d_out // self.num_heads

        self.W_q = nn.Linear(self.d_out, self.d_out)
        self.W_k = nn.Linear(self.d_out, self.d_out)
        self.W_v = nn.Linear(self.d_out, self.d_out)

        self.out_proj = nn.Linear(self.d_out, self.d_out)
        self.register_buffer('mask', torch.triu(torch.ones(config['context_length'], 
                                                            config['context_length']), diagonal = 1))


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

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool  = self.mask.bool()[:num_tokens, :num_tokens]

        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim = -1)
        
        context_vec = (attn_weights @ values).transpose(1, 2)
    
        context_vec = context_vec.reshape(batch_size, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ffn_1 = nn.Linear(config['emb_dim'], 4 * config['emb_dim'])
        self.ffn_2 = nn.Linear(4 *  config['emb_dim'], config['emb_dim'])
        self.gelu = GELU()

    def forward(self, x):
        return self.ffn_2(self.gelu(self.ffn_1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = MultiHeadAttention(config)
        self.ff = FeedForward(config)
        self.norm1 = nn.LayerNorm(config['emb_dim'])
        self.norm2 = nn.LayerNorm(config['emb_dim'])

    def forward(self, x):

        x = self.attn(self.norm1(x)) + x
        x = self.ff(self.norm2(x)) + x
        return x


class GPTModel(nn.Module):
    
    def __init__(self, config):
        super().__init__()

        self.token_emb = nn.Embedding(config['vocab_size'], config['emb_dim'])
        self.pos_emb = nn.Embedding(config['context_length'], config['emb_dim'])
        
        self.transfomer_block = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config['n_layers'])]
        )

        self.final_norm = nn.LayerNorm(config['emb_dim'])
        self.out_head = nn.Linear(config['emb_dim'], config['vocab_size'])

    
    def forward(self, in_idx):

        batch_size, seq_len = in_idx.shape
        tok_emb = self.token_emb(in_idx)
        pos_emb = self.pos_emb(torch.arange(seq_len, device = in_idx.device))

        x = tok_emb + pos_emb
        x = self.transfomer_block(x)
        x = self.final_norm(x)
        x = self.out_head(x)

        return x

