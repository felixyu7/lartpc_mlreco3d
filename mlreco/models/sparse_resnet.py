import torch
import torch.nn as nn
import time
import sparseconvnet as scn
from collections import OrderedDict

def coords_padding(coords, val):
    coords[:, :3] += val
    return coords

class SparseResNet(nn.Module):
    
    def __init__(self, cfg, name='sparse_resnet'):
        super(SparseResNet,self).__init__()

        self.model_config = cfg[name]
        
        self.dim_in = self.model_config.get('dim_in', 64)
        self.dim_out = self.model_config.get('dim_out', 1024)
        self.dimension = self.model_config.get('data_dim', 3)
        self.pad = self.model_config.get('pad', 11)
        self.input_features = self.model_config.get('features', 1)
        self.num_output = self.model_config.get('num_output', 5)
        self.spatial_size = self.model_config.get('spatial_size', 1024)
        self.pool_mode = self.model_config.get('pool_mode', 'max')
        
        self.final_tensor_shape = int(((self.spatial_size + self.pad) / (2**4)) + 0.5)
        
        self.res1 = scn.Sequential(OrderedDict([('conv1', scn.Convolution(self.dimension, self.input_features, self.dim_in, 7, 2, False)),
            ('bn1', SparseAffineChannel3d(self.dim_in)),
            ('relu', scn.ReLU()),
            ('maxpool', scn.MaxPooling(self.dimension, 3, 2))
            ]))
        self.res2 = scn.Sequential(sparse_bottleneck(self.dimension, self.dim_in, 256, self.dim_in, downsample=basic_bn_shortcut(self.dimension, self.dim_in, 256, 1)),
            sparse_bottleneck(self.dimension, 256, 256, self.dim_in),
            sparse_bottleneck(self.dimension, 256, 256, self.dim_in))
        self.res3 = scn.Sequential(sparse_bottleneck(self.dimension, 256, 512, 128, stride=2, downsample=basic_bn_shortcut(self.dimension, 256, 512, 2)),
            sparse_bottleneck(self.dimension, 512, 512, 128),
            sparse_bottleneck(self.dimension, 512, 512, 128),
            sparse_bottleneck(self.dimension, 512, 512, 128))
        self.res4 = scn.Sequential(sparse_bottleneck(self.dimension, 512, self.dim_out, 256, stride=2, downsample=basic_bn_shortcut(self.dimension, 512, self.dim_out, 2)),
            sparse_bottleneck(self.dimension, self.dim_out, self.dim_out, 256),
            sparse_bottleneck(self.dimension, self.dim_out, self.dim_out, 256),
            sparse_bottleneck(self.dimension, self.dim_out, self.dim_out, 256),
            sparse_bottleneck(self.dimension, self.dim_out, self.dim_out, 256),
            sparse_bottleneck(self.dimension, self.dim_out, self.dim_out, 256))
        
        self.desparsify = scn.SparseToDense(self.dimension, self.dim_out)
        self.sparsifier = scn.InputLayer(self.dimension, self.dim_out + self.pad, mode=3)
        
        if self.pool_mode == 'max':
            self.pool = scn.MaxPooling(self.dimension, self.final_tensor_shape, self.final_tensor_shape)
        else:
            self.pool = scn.AveragePooling(self.dimension, self.final_tensor_shape, self.final_tensor_shape)
            
        self.linear = nn.Linear(self.dim_out, self.num_output)

        self.apply(lambda m: freeze_params(m) if isinstance(m, SparseAffineChannel3d) else None)
    
    def forward(self, x):
        coords = x[:, :self.dimension+1].float()
        features = x[:, self.dimension+1:].float()
        features = features[:, -1].view(-1, 1)
        batch_size = coords[:, 3].unique().shape[0]
        
        coords = coords_padding(coords, self.pad)
        
        x = self.sparsifier((coords, features))

        out = self.res1(x)
        out = self.res2(out)
        out = self.res3(out)
        out = self.res4(out)

#         out = self.pool(out)
        out = self.desparsify(out)
#         out = out.view(batch_size, -1)
#         out = self.linear(out)
        
        return out

class sparse_bottleneck(nn.Module):
    """ Sparse Bottleneck Residual Block """

    def __init__(self, dimension, inplanes, outplanes, innerplanes, stride=1, dilation=1, group=1,
                 downsample=None):
        super(sparse_bottleneck,self).__init__()
        # In original resnet, stride=2 is on 1x1.
        # In fb.torch resnet, stride=2 is on 3x3.
        #(str1x1, str3x3) = (stride, 1) if cfg.RESNETS.STRIDE_1X1 else (1, stride)
        #self.stride = stride

        self.conv1 = scn.SubmanifoldConvolution(dimension,
            inplanes, innerplanes, 1, False)
        if (stride !=1):
            self.conv1 = scn.Convolution(dimension, inplanes, innerplanes, 1, stride, False)

        self.bn1 = SparseAffineChannel3d(innerplanes)

        self.conv2 = scn.SubmanifoldConvolution(dimension,
            innerplanes, innerplanes, 3, False)

        self.bn2 = SparseAffineChannel3d(innerplanes)

        self.conv3 = scn.SubmanifoldConvolution(dimension,
            innerplanes, outplanes, 1, False)

        self.bn3 = SparseAffineChannel3d(outplanes)

        self.downsample = downsample
        self.relu = scn.ReLU()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out.features += residual.features
        out = self.relu(out)

        return out

def basic_bn_shortcut(dimension, inplanes, outplanes, stride):
    if stride == 1:
        return nn.Sequential(
            scn.SubmanifoldConvolution(dimension, inplanes,
                    outplanes,
                    1,
                    False),
            SparseAffineChannel3d(outplanes),
        )
    else:
        return nn.Sequential(
            scn.SubmanifoldConvolution(dimension, inplanes,
                    outplanes,
                    1,
                    False),
            SparseAffineChannel3d(outplanes),
            scn.MaxPooling(dimension, 1, 2),
        )

class SparseAffineChannel3d(nn.Module):
    """ A simple channel-wise affine transformation operation """
    def __init__(self, num_features):
        super(SparseAffineChannel3d,self).__init__()
        self.num_features = num_features
        self.weight = nn.Parameter(torch.Tensor(num_features))
        self.bias = nn.Parameter(torch.Tensor(num_features))
        self.weight.data.uniform_()
        self.bias.data.zero_()

    def forward(self, x): #use torch.multiply()
    
        x.features = x.features * self.weight.view(1, self.num_features) + self.bias.view(1, self.num_features)
        return x
    
class ResNet_roi_conv5_head(nn.Module):
    def __init__(self, dim_in, roi_xform_func, spatial_scale, validation=False):
        super(ResNet_roi_conv5_head,self).__init__()
        self.dimension = 3
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale

#         dim_bottleneck = cfg.RESNETS.NUM_GROUPS * cfg.RESNETS.WIDTH_PER_GROUP
#         stride_init = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION // 7
        
        self.dim_out = 2048
        
        self.sparsify = scn.DenseToSparse(self.dimension)
        
        self.res5 = scn.Sequential(sparse_bottleneck(self.dimension, 1024, 2048, dim_in, downsample=basic_bn_shortcut(self.dimension, 1024, 2048, 1)),
            sparse_bottleneck(self.dimension, 2048, 2048, dim_in),
            sparse_bottleneck(self.dimension, 2048, 2048, dim_in))
        
        self.desparsify = scn.SparseToDense(self.dimension, 2048)
        
        self.avgpool = nn.AvgPool3d(7)
        self.validation = validation

        self._init_modules()

    def _init_modules(self):
        # Freeze all bn (affine) layers !!!
        self.apply(lambda m: freeze_params(m) if isinstance(m, SparseAffineChannel3d) else None)

    def forward(self, x, rpn_ret):

        x = self.roi_xform(
            x, 
            rpn_ret,
            blob_rois='rois',
            method='RoIAlign',
            resolution=14,
            spatial_scale=self.spatial_scale,
            sampling_ratio=0
        )
        # print('x in RESNET.PY', x.shape)
        
#         import pdb; pdb.set_trace()

        print(x.shape)
        
        sparse_x = self.sparsify(x.contiguous())
        
        res5_feat = self.res5(sparse_x)
        
        res5_feat = self.desparsify(res5_feat)
        
        # print('res5_feat in RESNET.PY', res5_feat.shape)
        
        x = self.avgpool(res5_feat)
        if ( self.training or self.validation ):
            return x, res5_feat
        else:
            return x

def freeze_params(m):
    """Freeze all the weights by setting requires_grad to False
    """
    for p in m.parameters():
        p.requires_grad = False