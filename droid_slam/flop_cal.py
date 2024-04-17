from modules.extractor import BasicEncoder, DualEncoder
from ptflops import get_model_complexity_info
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--fusion_method', type=str, default='concat', choices=['concat', 'self_att', 'deform_att', 'none'])
argparser.add_argument('--net', type=str, default='cnet', choices=['fnet', 'cnet'])
argparser.add_argument('--stereo', action='store_true')

args = argparser.parse_args()

if args.stereo:
    assert args.fusion_method == 'none', 'Stereo mode only supports none fusion method'

if args.fusion_method == 'none':
    if args.net == 'fnet':
        encoder = BasicEncoder(output_dim=128, norm_fn='instance')
    else:
        encoder = BasicEncoder(output_dim=256, norm_fn='none')
else:
    if args.net == 'fnet':
        encoder = DualEncoder(output_dim=128, norm_fn='instance', args=args, ori_shape=(640, 320))
    else:
        encoder = DualEncoder(output_dim=256, norm_fn='none', args=args, ori_shape=(640, 320))
        
flops, params = get_model_complexity_info(encoder, (2 if args.stereo else 1, 3, 640, 320), as_strings=True, print_per_layer_stat=True)
print(flops)