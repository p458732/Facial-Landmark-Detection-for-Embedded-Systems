

import argparse
from typing import Optional
import argparse
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, Any
from torch import Tensor, nn
import torch
import sys
sys.path.insert(0, './ml-cvnets')
from cvnets.models.classification.base_image_encoder import BaseImageEncoder 
from options.utils import load_config_file
from cvnets import arguments_model, arguments_nn_layers
from cvnets.layers import ConvLayer2d, GlobalPool, Identity, LinearLayer
from cvnets.models.classification.base_image_encoder import BaseImageEncoder
from cvnets.models.classification.config.mobilevit_v2 import get_configuration
from cvnets.modules import InvertedResidual
from cvnets.modules import MobileViTBlockv2 as Block


def get_training_arguments(parse_args=True, config_path=None):
    parser = argparse.ArgumentParser()
    parser = arguments_nn_layers(parser=parser)
    parser = arguments_model(parser=parser)
    parser.add_argument('--common.config-file', type=str, default='./checkpoints/mobilevitv2-0.5.yaml')
    parser.add_argument('--dataset.category', type=str, default='classification')
    if parse_args:
        if config_path:
            opts = parser.parse_args(['--common.config-file', config_path])
        else:
            opts = parser.parse_args()
        opts = load_config_file(opts)
        return opts
    else:
        return parser


edge_info = (
                (True, (0, 1, 2, 3, 4)),  # RightEyebrow
                (True, (5, 6, 7, 8, 9)),  # LeftEyebrow
                (False, (10, 11, 12, 13)),  # NoseLine
                (False, (14, 15, 16, 17, 18)),  # Nose
                (True, (19, 20, 21, 22, 23, 24)),  # RightEye
                (True, (25, 26, 27, 28, 29, 30)),  # LeftEye
                (True, (31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42)),  # OuterLip
                (True, (43, 44, 45, 46, 47, 48, 49, 50)),  # InnerLip
            )

class E2HTransform(nn.Module):
    def __init__(self, edge_info, num_points, num_edges):
        super().__init__()

        e2h_matrix = np.zeros([num_points, num_edges])
        for edge_id, isclosed_indices in enumerate(edge_info):
            is_closed, indices = isclosed_indices
            for point_id in indices:
                e2h_matrix[point_id, edge_id] = 1
        e2h_matrix = torch.from_numpy(e2h_matrix).float()

        # pn x en x 1 x 1.
        self.register_buffer('weight', e2h_matrix.view(
            e2h_matrix.size(0), e2h_matrix.size(1), 1, 1))

        # some keypoints are not coverred by any edges,
        # in these cases, we must add a constant bias to their heatmap weights.
        bias = ((e2h_matrix @ torch.ones(e2h_matrix.size(1)).to(
            e2h_matrix)) < 0.5).to(e2h_matrix)
        # pn x 1.
        self.register_buffer('bias', bias)

    def forward(self, edgemaps):
        # input: batch_size x en x hw x hh.
        # output: batch_size x pn x hw x hh.
        return F.conv2d(edgemaps, weight=self.weight, bias=self.bias)
    
class ConvBlock(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, groups=1):
        super(ConvBlock, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size,
                              stride, padding=(kernel_size - 1) // 2, groups=groups, bias=True)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU()
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
    
class Activation(nn.Module):
    def __init__(self, kind: str = 'relu', channel=None):
        super().__init__()
        self.kind = kind

        if '+' in kind:
            norm_str, act_str = kind.split('+')
        else:
            norm_str, act_str = 'none', kind

        self.norm_fn = {
            'in': F.instance_norm,
            'bn': nn.BatchNorm2d(channel),
            'bn_noaffine': nn.BatchNorm2d(channel, affine=False, track_running_stats=True),
            'none': None
        }[norm_str]

        self.act_fn = {
            'relu': F.relu,
            'softplus': nn.Softplus(),
            'exp': torch.exp,
            'sigmoid': torch.sigmoid,
            'tanh': torch.tanh,
            'none': None
        }[act_str]

        self.channel = channel

    def forward(self, x):
        if self.norm_fn is not None:
            x = self.norm_fn(x)
        if self.act_fn is not None:
            x = self.act_fn(x)
        return x

    def extra_repr(self):
        return f'kind={self.kind}, channel={self.channel}'
    
class decoder_default:
    def __init__(self, weight=1, use_weight_map=False):
        self.weight = weight
        self.use_weight_map = use_weight_map

    def _make_grid(self, h, w):
        yy, xx = torch.meshgrid(
            torch.arange(h).float() / (h - 1) * 2 - 1,
            torch.arange(w).float() / (w - 1) * 2 - 1)
        return yy, xx

    def my_get_coords_from_heatmap(self, heatmap):
        batch, npoints, h, w = heatmap.shape
        yy, xx = self._make_grid(h, w)
        yy = yy.view(1, 1, h, w).to(heatmap)
        xx = xx.view(1, 1, h, w).to(heatmap)
        
        yy_coord = (yy * heatmap).sum([2, 3]) 
        xx_coord = (xx * heatmap).sum([2, 3]) 
        coords = torch.stack([xx_coord, yy_coord], dim=-1)

        return coords
    def get_coords_from_heatmap(self, heatmap):
        """
            inputs:
            - heatmap: batch x npoints x h x w

            outputs:
            - coords: batch x npoints x 2 (x,y), [-1, +1]
            - radius_sq: batch x npoints
        """
        batch, npoints, h, w = heatmap.shape
        if self.use_weight_map:
            heatmap = heatmap * self.weight

        yy, xx = self._make_grid(h, w)
        yy = yy.view(1, 1, h, w).to(heatmap)
        xx = xx.view(1, 1, h, w).to(heatmap)

        heatmap_sum = torch.clamp(heatmap.sum([2, 3]), min=1e-6)

        yy_coord = (yy * heatmap).sum([2, 3]) / heatmap_sum  # batch x npoints
        xx_coord = (xx * heatmap).sum([2, 3]) / heatmap_sum  # batch x npoints
        coords = torch.stack([xx_coord, yy_coord], dim=-1)

        return coords


class MobileViTv2(BaseImageEncoder):
    """
    This class defines the `MobileViTv2 <https://arxiv.org/abs/2206.02680>`_ architecture
    """

    def __init__(self, opts, *args, **kwargs) -> None:
        num_classes = getattr(opts, "model.classification.n_classes", 1000)
        pool_type = getattr(opts, "model.layer.global_pool", "mean")

        mobilevit_config = get_configuration(opts=opts)
        image_channels = mobilevit_config["layer0"]["img_channels"]
        out_channels = mobilevit_config["layer0"]["out_channels"]

        super().__init__(opts, *args, **kwargs)

        # store model configuration in a dictionary
        self.model_conf_dict = dict()
        self.conv_1 = ConvLayer2d(
            opts=opts,
            in_channels=image_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            use_norm=True,
            use_act=True,
        )

        self.model_conf_dict["conv1"] = {"in": image_channels, "out": out_channels}

        in_channels = out_channels
        self.layer_1, out_channels = self._make_layer(
            opts=opts, input_channel=in_channels, cfg=mobilevit_config["layer1"]
        )
        self.model_conf_dict["layer1"] = {"in": in_channels, "out": out_channels}

        in_channels = out_channels
        self.layer_2, out_channels = self._make_layer(
            opts=opts, input_channel=in_channels, cfg=mobilevit_config["layer2"]
        )
        self.model_conf_dict["layer2"] = {"in": in_channels, "out": out_channels}

        in_channels = out_channels
        self.layer_3, out_channels = self._make_layer(
            opts=opts, input_channel=in_channels, cfg=mobilevit_config["layer3"]
        )
        self.model_conf_dict["layer3"] = {"in": in_channels, "out": out_channels}

        in_channels = out_channels
        self.layer_4, out_channels = self._make_layer(
            opts=opts,
            input_channel=in_channels,
            cfg=mobilevit_config["layer4"],
            dilate=self.dilate_l4,
        )
        self.model_conf_dict["layer4"] = {"in": in_channels, "out": out_channels}

        in_channels = out_channels
        self.layer_5, out_channels = self._make_layer(
            opts=opts,
            input_channel=in_channels,
            cfg=mobilevit_config["layer5"],
            dilate=self.dilate_l5,
        )
        self.model_conf_dict["layer5"] = {"in": in_channels, "out": out_channels}

        self.conv_1x1_exp = Identity()
        self.model_conf_dict["exp_before_cls"] = {
            "in": out_channels,
            "out": out_channels,
        }

        self.final_upsample =nn.Upsample(scale_factor=8, mode='nearest')
        self.final_upsample1 = nn.Sequential(nn.ConvTranspose2d(out_channels, out_channels, 6, stride=4, padding=1),nn.BatchNorm2d(out_channels), nn.ReLU())
        self.final_upsample2 = nn.Sequential(nn.ConvTranspose2d(out_channels, out_channels, 4, stride=2, padding=1),nn.BatchNorm2d(out_channels), nn.ReLU())
        
        self.out_pointmaps = ConvBlock(out_channels, 51, 1, relu=False, bn=False)
        self.out_edgemaps = ConvBlock(out_channels, 8, 1, relu=False, bn=False)
        self.out_heatmaps = ConvBlock(out_channels, 51, 1, relu=False, bn=False)
        self.pointmap_act = Activation("sigmoid", 51)
        self.edgemap_act = Activation("sigmoid", 8)
        self.heatmap_act = Activation("in+relu", 51)
        self.decoder = decoder_default()
        self.e2h_transform = E2HTransform(edge_info, 51, 8)
        # check model
        self.check_model()

        # weight initialization
        self.reset_parameters(opts=opts)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(title=cls.__name__)
        group.add_argument(
            "--model.classification.mitv2.attn-dropout",
            type=float,
            default=0.0,
            help="Dropout in attention layer. Defaults to 0.0",
        )
        group.add_argument(
            "--model.classification.mitv2.ffn-dropout",
            type=float,
            default=0.0,
            help="Dropout between FFN layers. Defaults to 0.0",
        )
        group.add_argument(
            "--model.classification.mitv2.dropout",
            type=float,
            default=0.0,
            help="Dropout in attention layer. Defaults to 0.0",
        )
        group.add_argument(
            "--model.classification.mitv2.width-multiplier",
            type=float,
            default=1.0,
            help="Width multiplier. Defaults to 1.0",
        )
        group.add_argument(
            "--model.classification.mitv2.attn-norm-layer",
            type=str,
            default="layer_norm_2d",
            help="Norm layer in attention block. Defaults to LayerNorm",
        )
        return parser

    def _make_layer(
        self, opts, input_channel, cfg: Dict, dilate: Optional[bool] = False
    ) -> Tuple[nn.Sequential, int]:
        block_type = cfg.get("block_type", "mobilevit")
        if block_type.lower() == "mobilevit":
            return self._make_mit_layer(
                opts=opts, input_channel=input_channel, cfg=cfg, dilate=dilate
            )
        else:
            return self._make_mobilenet_layer(
                opts=opts, input_channel=input_channel, cfg=cfg
            )

    @staticmethod
    def _make_mobilenet_layer(
        opts, input_channel: int, cfg: Dict
    ) -> Tuple[nn.Sequential, int]:
        output_channels = cfg.get("out_channels")
        num_blocks = cfg.get("num_blocks", 2)
        expand_ratio = cfg.get("expand_ratio", 4)
        block = []

        for i in range(num_blocks):
            stride = cfg.get("stride", 1) if i == 0 else 1

            layer = InvertedResidual(
                opts=opts,
                in_channels=input_channel,
                out_channels=output_channels,
                stride=stride,
                expand_ratio=expand_ratio,
            )
            block.append(layer)
            input_channel = output_channels
        return nn.Sequential(*block), input_channel

    def _make_mit_layer(
        self, opts, input_channel, cfg: Dict, dilate: Optional[bool] = False
    ) -> Tuple[nn.Sequential, int]:
        prev_dilation = self.dilation
        block = []
        stride = cfg.get("stride", 1)

        if stride == 2:
            if dilate:
                self.dilation *= 2
                stride = 1

            layer = InvertedResidual(
                opts=opts,
                in_channels=input_channel,
                out_channels=cfg.get("out_channels"),
                stride=stride,
                expand_ratio=cfg.get("mv_expand_ratio", 4),
                dilation=prev_dilation,
            )

            block.append(layer)
            input_channel = cfg.get("out_channels")

        attn_unit_dim = cfg["attn_unit_dim"]
        ffn_multiplier = cfg.get("ffn_multiplier")

        dropout = getattr(opts, "model.classification.mitv2.dropout", 0.0)

        block.append(
            Block(
                opts=opts,
                in_channels=input_channel,
                attn_unit_dim=attn_unit_dim,
                ffn_multiplier=ffn_multiplier,
                n_attn_blocks=cfg.get("attn_blocks", 1),
                patch_h=cfg.get("patch_h", 2),
                patch_w=cfg.get("patch_w", 2),
                dropout=dropout,
                ffn_dropout=getattr(
                    opts, "model.classification.mitv2.ffn_dropout", 0.0
                ),
                attn_dropout=getattr(
                    opts, "model.classification.mitv2.attn_dropout", 0.0
                ),
                conv_ksize=3,
                attn_norm_layer=getattr(
                    opts, "model.classification.mitv2.attn_norm_layer", "layer_norm_2d"
                ),
                dilation=self.dilation,
            )
        )

        return nn.Sequential(*block), input_channel

    def forward_classifier(self, x: Tensor, *args, **kwargs) -> Tensor:
        """A helper function to extract features and running a classifier."""
        # We add another classifier function so that the classifiers
        # that do not adhere to the structure of BaseEncoder can still
        # use neural augmentor
        y, fusionmaps = [], []
        x = self.extract_features(x)
        x = self.final_upsample(x)
        #x = self.final_upsample1(x)
        #x = self.final_upsample2(x)
        #x = self.upsample(x)
        # x : 16 384 8 8
        #heatmaps0 = self.final_upsample(x)
        
        heatmaps = self.heatmap_act(self.out_heatmaps(x))
         
        pointmaps0 = self.out_pointmaps(x)
        pointmaps = self.pointmap_act(pointmaps0)
        edgemaps0 = self.out_edgemaps(x)
        edgemaps = self.edgemap_act(edgemaps0)
        mask = self.e2h_transform(edgemaps) * pointmaps
        fusion_heatmaps = mask * heatmaps
        
        
        landmarks = self.decoder.get_coords_from_heatmap(fusion_heatmaps)
        y.append(landmarks)
            
        y.append(pointmaps)
        y.append(edgemaps)

        fusionmaps.append(fusion_heatmaps)
        return y, fusionmaps, landmarks
    
    def forward(self, x: Any, *args, **kwargs) -> Any:
        """A forward function of the model, optionally training the model with
        neural augmentation."""
        return self.forward_classifier(x)
        
        
def mobile_vit_v2():
    opts = get_training_arguments(config_path='./checkpoints/mobilevitv2-0.5.yaml')
    state_dict = torch.load('./checkpoints/mobilevitv2-0.5.pt')

    model = MobileViTv2(opts)
    del state_dict['classifier.1.weight']
    del state_dict['classifier.1.bias']
    model.load_state_dict(state_dict, strict=False)
    #stat(model, (3,256,256))
    return model
    

#model = mobile_vit_v2()

#print("Params: ", params)
#x = model(torch.rand(1,3,256,256))
# stat(model, (3,256,256))
# device = torch.device("cuda")
# model.to(device)
# dummy_input = torch.randn(16, 3,256,256, dtype=torch.float).to(device)

# # INIT LOGGERS
# starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
# repetitions = 300
# timings=np.zeros((repetitions,1))
# #GPU-WARM-UP
# for _ in range(10):
#     _ = model(dummy_input)
# # MEASURE PERFORMANCE
# with torch.no_grad():
#     for rep in range(repetitions):
#         starter.record()
#         _ = model(dummy_input)
#         ender.record()
#         # WAIT FOR GPU SYNC
#         torch.cuda.synchronize()
#         curr_time = starter.elapsed_time(ender)
#         timings[rep] = curr_time

# mean_syn = np.sum(timings) / repetitions
# std_syn = np.std(timings)
# print(mean_syn)

# x = torch.rand((1,3,256,256))

# y, fusionmaps, landmarks = mobile_vit_v2()(x)
# print(fusionmaps[0].shape)
# stat(mobile_vit_v2(), (3,256,256))



