#
import argparse
import torch

from retnet.configuration_retnet import RetNetConfig
from retnet.modeling_retnet import RetNetModel, RetNetModelWithLMHead
from apps.virt.cifar10_app import Cifar10App

class VirtApp(object):
    def __init__(self):
        self.name = 'apps.virt.virt_app.VirtApp'

    def startup(self, args:argparse.Namespace = {}) -> None:
        print(f'Visual Retnet v0.0.2')
        app = Cifar10App()
        app.startup(args=args)

    def test_sample_app(self):
        torch.manual_seed(0)
        hidden_size = 64
        config = RetNetConfig(num_layers=8,
                            hidden_size=hidden_size,
                            num_heads=4,
                            qk_dim=64,
                            v_dim=128,
                            ffn_proj_size=12,
                            use_default_gamma=False)

        model = RetNetModel(config)
        model.eval()

        device = 'cuda:0'  # cuda, cpu, mps for m1 mac
        model = model.to(device)

        input_ids = torch.LongTensor([[1,2,3,4,1,2,3,4,5,6]]).to(device)
        seq_len = 16
        inputs_embeds = torch.randn(1, seq_len, hidden_size).to(device)

        parallel_outputs = model(inputs_embeds=inputs_embeds, forward_impl='parallel', use_cache=True)
        parallel_state = parallel_outputs.last_hidden_state
        parallel_cache = parallel_outputs.past_key_values
        print(f'parallel_outputs: {type(parallel_outputs)}; parallel_state: {parallel_state.shape}; parallel_cache: {type(parallel_cache)};')
        print(f'最后输出：{parallel_state[:, -1, :]}')

        print(f'############# ^_^   The End!   ^_^ ################################################')