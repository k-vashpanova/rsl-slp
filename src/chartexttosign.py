class PositionalEncoding(nn.Module):

    def __init__(self, dim: int, max_len: int = 5000):
        super().__init__()
        '''
        1d sinusoidal positional encoding
        '''

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))

        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add a batch dimension: (1, max_len, dim)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:, :x.size(1)]
        return x


class CharCNN(nn.Module):

    def __init__(self, out_channels, vocab_size=92, padding_index=0):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=out_channels, padding_idx=padding_index)
        self.conv = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=3)
        self.maxpool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        B, L, D = x.shape
        x = x.reshape(B*L, D).contiguous()
        x = self.embed(x)
        x = x.transpose(1, 2).contiguous()
        x = self.conv(x)
        x = self.maxpool(x)
        x = x.view(B, L, -1)
        return x


class LengthRegulator(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()

        self.linear1 = nn.Linear(in_features, in_features//2)
        self.linear2 = nn.Linear(in_features//2, out_features)
        self.dropout = nn.Dropout(0.5)
        self.norm = nn.LayerNorm(out_features)
        self.gelu = nn.GELU()
        if in_features == out_features:
            self.res = nn.Identity()
        else:
            self.res = nn.Linear(in_features, out_features)

    def forward(self, input):

        x = self.linear1(input)
        x = self.gelu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        x += self.res(input)
        x = self.norm(x)

        return x


class CharText2Sign(nn.Module):

    def __init__(self,
                 mae,
                 bert,
                 char_embedding_dim = 16,
                 char_vocab_size = 92,
                 max_seq = 32,
                 num_encoder_layers = 4,
                 num_decoder_layers = 4,
                 num_heads = 8,
                 freeze_bert = True,
                 freeze_mae = True):
        super().__init__()

        self.max_seq = max_seq

        # --------------------------------------------------------------------------
        # BERT specifications
        self.bert = bert
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        self.text_embedding_dim = self.bert.config.dim
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE specifications
        self.mae = mae.decoder
        if freeze_mae:
            for param in self.mae.parameters():
                param.requires_grad = False
        self.pose_embedding_dim = self.mae.embedding_dim
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # Text and character embeddings
        if char_embedding_dim:
            self.char_embedding_dim = char_embedding_dim
            self.char_vocab_size = char_vocab_size
            self.char_embed = CharCNN(self.char_embedding_dim, vocab_size=self.char_vocab_size)
        else:
            self.char_embedding_dim = False
            self.char_embed = self.return_empty

        self.linear = nn.Linear(self.text_embedding_dim + self.char_embedding_dim, self.pose_embedding_dim)
        self.dropout = nn.Dropout(0.5) #
        self.pos_embed = PositionalEncoding(self.pose_embedding_dim) #
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # Encoder-decoder specifications
        self.nar = nn.Transformer(d_model=self.pose_embedding_dim,
                                  nhead=num_heads,
                                  num_encoder_layers=num_encoder_layers,
                                  num_decoder_layers=num_decoder_layers,
                                  dim_feedforward=self.pose_embedding_dim*4,
                                  dropout=0.1,
                                  activation='gelu',
                                  norm_first=True, #
                                  batch_first=True,)
        self.queries = nn.Parameter(torch.zeros(1, self.max_seq, self.pose_embedding_dim))
        self.mask = LengthRegulator(self.text_embedding_dim, self.max_seq)
        self.norm = nn.LayerNorm(self.pose_embedding_dim) #
        # --------------------------------------------------------------------------

    def return_empty(self, *args, **kwargs):
        return torch.empty(0)

    def forward(self, input_ids, attention_mask=None, chars=None):

        x = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        length_mask = self.mask(x[:, 0, :])

        char_embeds = self.char_embed(chars[:, 1:, :]).to(chars.device)
        x = torch.cat([x[:, 1:, :], char_embeds], dim=-1)
        x = self.linear(x)

        queries = self.queries.expand(x.shape[0], -1, -1)
        #x = self.nar(tgt=queries, src=x)
        x = self.pos_embed(x)
        x = self.dropout(x)
        x = self.nar(tgt=queries, src=x, src_key_padding_mask=~attention_mask[:, 1:].bool())
        x = self.norm(x)

        B, L, D = x.shape

        '''
        segs = []
        for i in range(B):
            seg = x[i].view(L, 1, -1)
            seg = self.mae(seg)
            _, T, V, C = seg.shape
            segs.append(seg.view(L*T, V, C))
        x = torch.stack(segs)
        '''

        x = x.view(B*L, 1, D)
        x = self.mae(x)
        _, T, V, C = x.shape
        x = x.view(B, L*T, V, C)

        return {'frames':x, 'length_mask':length_mask}
