import torch
import torch.nn as nn

from attention import SelfAttention

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super().__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, values, keys, queries, mask):
        attention = self.attention(values, keys, queries, mask)
        x = self.norm1(attention + queries)  # skip connection and normalization
        x = self.dropout(x)
        fed_forward = self.feed_forward(x)
        out = self.norm2(fed_forward + x)  # skip connection and normalization
        return self.dropout(out)


class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        heads,
        num_layers,
        forward_expansion,
        dropout,
        max_length,
        device,
    ):
        super().__init__()
        self.device = device
        self.embed_size = embed_size
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size=embed_size,
                    heads=heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, inp, mask):
        N, seq_length = inp.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        out = self.word_embedding(inp) + self.position_embedding(positions)
        out = self.dropout(out)

        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out