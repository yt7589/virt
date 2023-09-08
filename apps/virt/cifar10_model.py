#
import torch
from torch import nn
from retnet.configuration_retnet import RetNetConfig
from retnet.modeling_retnet import RetNetModel, RetNetModelWithLMHead
from apps.virt.patch_embedding import PatchEmbedding

class Cifar10Model(nn.Module):
    def __init__(self):
        super(Cifar10Model, self).__init__()
        hidden_size = 64
        config = RetNetConfig(num_layers=8,
                            hidden_size=hidden_size,
                            num_heads=4,
                            qk_dim=64,
                            v_dim=128,
                            ffn_proj_size=12,
                            use_default_gamma=False)
        self.model = RetNetModel(config)
        self.flatten = nn.Flatten()
        self.neck = nn.Linear(64, 16)
        self.head = nn.Linear(16, 10)
        self.pos_embed = PatchEmbedding(in_channels=3, patch_size=8, emb_size=64, img_size=32)
    
    def forward(self, x):
        x = self.pos_embed(x)
        outputs = self.model(inputs_embeds=x, forward_impl='parallel', use_cache=True)
        ys = outputs.last_hidden_state[:, -1, :]
        a1 = self.flatten(ys)
        a2 = self.neck(a1)
        return self.head(a2)