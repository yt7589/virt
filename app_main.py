#
import argparse
from apps.virt.virt_app import VirtApp

def t001():
    import torch
    from apps.virt.patch_embedding import PatchEmbedding
    x = torch.randn(1, 3, 32, 32)
    pe = PatchEmbedding(in_channels=3, patch_size=8, emb_size=64, img_size=32)
    y = pe(x)
    print(f'y: {y.shape};')

def main(args:argparse.Namespace = {}) -> None:
    i_debug = 10
    if 1 == i_debug:
        t001()
        exit(0)
    app = VirtApp()
    app.startup()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_mode', action='store', type=int, default=1, dest='runMode', help='run mode')
    parser.parse_args()

if '__main__' == __name__:
    args = parse_args()
    main(args=args)