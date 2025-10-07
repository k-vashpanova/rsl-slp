class CTRGC(nn.Module):
    def __init__(self, in_channels, out_channels, num_points, hidden_channels=8):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        self.conv1 = nn.Conv2d(self.in_channels, self.hidden_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(self.in_channels, self.hidden_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.hidden_channels, self.out_channels, kernel_size=1)

        self.A = nn.Parameter(torch.rand(1, 1, num_points, num_points)) # N,C,V,V
        self.alpha = nn.Parameter(torch.randn(1))

        self.tanh = nn.Tanh()

    def forward(self, x):

        x1 = self.conv1(x).mean(-2)
        x2 = self.conv2(x).mean(-2)
        spat = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))

        x4 = self.conv4(spat)
        spat = x4 * self.alpha + self.A

        x3 = self.conv3(x)
        spat = torch.einsum('ncuv,nctv->nctu', spat, x3)

        return spat

class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super().__init__()

        pad = (kernel_size + (kernel_size-1) * (dilation-1) - 1) // 2
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=(kernel_size, 1),
                              padding=(pad, 0),
                              stride=(stride, 1),
                              dilation=(dilation, 1))

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class MSTCN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilations=[1,2,3,4],
                 residual=True,
                 residual_kernel_size=1):

        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches (= len(dilations) + 2)'

        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches

        # Temporal Convolution branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                TemporalConv(branch_channels, branch_channels, kernel_size=kernel_size, stride=stride, dilation=dilation),
            )
            for dilation in dilations
        ])

        # Additional Max branch
        self.branches.append(
            nn.Sequential(
                nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(3,1), stride=(stride,1), padding=(1,0)),
                nn.BatchNorm2d(branch_channels)
                )
            )

        # 1x1 branch
        self.branches.append(
            nn.Sequential(
                nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride,1)),
                nn.BatchNorm2d(branch_channels)
                )
            )

        # Residual connection
        if not residual:
            self.residual = torch.zeros_like
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = nn.Identity()
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)

    def forward(self, x):
        # Input dim: (N,C,T,V)

        res = self.residual(x)

        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        out += res

        return out

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


class SignMAEEncoder(nn.Module):

    def __init__(self,
                 n_frames,
                 n_joints,
                 n_channels,
                 nhead=8,
                 num_layers=4,
                 embedding_dim=128,
                 spatio_temporal_features=96,
                 mask_ratio=0.25
                 ):

        super().__init__()
        self.n_frames = n_frames
        self.n_joints = n_joints
        self.n_channels = n_channels
        self.embedding_dim = embedding_dim
        self.spatio_temporal_features = spatio_temporal_features
        self.mask_ratio = mask_ratio

        self.spatial = CTRGC(self.n_channels, self.spatio_temporal_features, self.n_joints)
        if self.spatio_temporal_features % 6 == 0:
            self.temporal = MSTCN(self.spatio_temporal_features,
                                  self.spatio_temporal_features, dilations=[1, 2, 3, 4])
        else:
            self.temporal = MSTCN(self.spatio_temporal_features,
                                  self.spatio_temporal_features, dilations=[1, 2])
        self.linear = nn.Linear(self.spatio_temporal_features * self.n_joints, self.embedding_dim)

        self.pos_embed = PositionalEncoding(self.embedding_dim)

        self.glor = nn.Parameter(torch.randn(1, 1, self.embedding_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model = self.embedding_dim,
            nhead = nhead,
            dim_feedforward = self.embedding_dim * 4,
            activation = 'gelu',
            norm_first = True,
            batch_first = True)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer,
                                             num_layers=num_layers)
        self.norm = nn.LayerNorm(self.embedding_dim)

    def embeddings(self, x):
        """
        Extract spatio-temporal features
        Flatten and project to the transformer input dimension
        x: [B, T, V, C], sequence
        """
        B, T, V, _ = x.shape
        x = x.permute(0, 3, 1, 2) #(B, C, T, V)

        ## Extract spatial and temporal features
        x = self.spatial(x) # (B, C*, T, V)
        x = self.temporal(x) # (B, C**, T, V)

        x = x.permute(0, 2, 1, 3).reshape(B, T, V*self.spatio_temporal_features)
        x = self.linear(x)

        return x

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [B, T, D], sequence
        """
        B, T, D = x.shape  # batch, length, dim
        len_keep = int(T * (1 - mask_ratio))

        noise = torch.rand(B, T, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        return x_masked

    def forward(self, x):
        """
        SignMAE encoder
        Return glor token
        x: [B, T, V, C], sequence
        """

        # spatio-temporal embedding
        x = self.embeddings(x)

        # add positional embedding w/o glor token
        x = self.pos_embed(x)

        # masking: length -> length * mask_ratio
        x = self.random_masking(x, self.mask_ratio)

        # append glor token
        glor_tokens = self.glor.expand(x.shape[0], -1, -1)
        x = torch.cat((glor_tokens, x), dim=1)

        # apply Transformer
        x = self.encoder(x)
        x = self.norm(x)

        x = x[:, :1, :]

        return x


class SignMAEDecoder(nn.Module):

    def __init__(self,
                 n_frames,
                 n_joints,
                 n_channels,
                 nhead=8,
                 num_layers=4,
                 embedding_dim=128,
                 ):

        super().__init__()
        self.n_frames = n_frames
        self.n_joints = n_joints
        self.n_channels = n_channels
        self.embedding_dim = embedding_dim

        self.queries = nn.Parameter(torch.randn(1, self.n_frames, self.embedding_dim))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model = self.embedding_dim,
            nhead = nhead,
            dim_feedforward = self.embedding_dim * 4,
            activation = 'gelu',
            norm_first = True,
            batch_first = True)
        self.decoder = nn.TransformerDecoder(decoder_layer=decoder_layer,
                                             num_layers=num_layers)
        self.norm = nn.LayerNorm(self.embedding_dim)
        self.linear = nn.Linear(self.embedding_dim, self.n_channels * self.n_joints)

    def forward(self, x):

        queries = self.queries.expand(x.shape[0], -1, -1)

        # apply Transformer blocks
        x = self.decoder(queries, x)
        x = self.norm(x)

        # predictor projection
        x = self.linear(x)
        x = x.view(x.shape[0], x.shape[1], self.n_joints, self.n_channels)

        return x


class SignMAE(nn.Module):
    def __init__(self,

                 n_frames,
                 n_joints,
                 in_channels,
                 out_channels,

                 nhead=8,
                 n_encoder_layers=4,
                 n_decoder_layers=4,

                 spatio_temporal_features=96,
                 embedding_dim=128,
                 mask_ratio=0.25,
                 ):

        super().__init__()
        self.n_frames = n_frames
        self.n_joints = n_joints
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embedding_dim = embedding_dim

        self.encoder = SignMAEEncoder(self.n_frames,
                                      self.n_joints,
                                      self.in_channels,
                                      embedding_dim=self.embedding_dim,
                                      nhead=nhead,
                                      num_layers=n_encoder_layers,
                                      spatio_temporal_features=spatio_temporal_features,
                                      mask_ratio=mask_ratio)

        self.decoder = SignMAEDecoder(self.n_frames,
                                      self.n_joints,
                                      self.out_channels,
                                      embedding_dim=self.embedding_dim,
                                      nhead=nhead,
                                      num_layers=n_decoder_layers)


    def forward(self, x):

        x = self.encoder(x)
        x = self.decoder(x)

        return x