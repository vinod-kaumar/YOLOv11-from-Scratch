import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# model: [depth(d), width(w), max_channels(mc)]
model_scales = {
  "yolo11n": [0.50, 0.25, 1024],
  "yolo11s": [0.50, 0.50, 1024],
  "yolo11m": [0.50, 1.00, 512],
  "yolo11l": [1.00, 1.00, 512],
  "yolo11x": [1.00, 1.50, 512]}

# Functions for backbone module

def autopad(k, p = None, d = 1):
    ''' Pad to same shape outputs'''
    if d > 1:
        d = d*(k-1) + 1 if isinstance(k, int) else [d*(x-1) + 1 for x in k] # actual kernel size

    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k] # auto - pad
    return p

class Conv(nn.Module):
    # Standard Convolution 
    default_act = nn.SiLU() # default activation
    
    def __init__(self, c1, c2, k = 1, s = 1, p = None, g = 1, d = 1 , act = True):
        # initialize Conv layer with given parameters
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups = g, dilation = d, bias = False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = (self.default_act if act is True 
                   else act if isinstance(act, nn.Module) 
                   else nn.Identity()
        )    
    def forward(self, x):
        # forward propagation : convolution --> normalization --> activation to input tensor
        return self.act(self.bn(self.conv(x)))
    
    def forward_fuse(self, x):
        # forward propagation without batch normalization (for inference optimization)
        return self.act(self.conv(x))
    
class Bottleneck(nn.Module):
    '''Standard bottleneck'''
    
    def __init__(self, c1, c2, shortcut = True, g = 1, k = (3, 3), e = 0.5):
        '''Initialize a standard bottleneck with optional shortcut connection'''
        super().__init__()
        c_  = int(c2 * e) # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1) # first convolution layer
        self.cv2 = Conv(c_, c2, k, 1, g = g) # second convolution layer
        self.add = shortcut and c1 == c2 # whether to add input to output (shortcut connection)
    def forward(self, x):
        # forward propagation : input --> first convolution --> second convolution
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y # add input to output if shortcut connection is enabled
    
class C2f(nn.Module):
    '''faster implementation of CSP bottleneck with 2 convolutions'''
    def __init__(self, c1, c2, n = 1, shortcut = False, g = 1, e = 0.5):
        '''Initialize a CSP bottleneck with 2 convolutions and n bottleneck layers '''
        super().__init__()
        self.c = int(c2 * e) # hidden channels
        self.cv1 = Conv(c1, 2*self.c, 1, 1) # first convolution layer 
        self.cv2 = Conv((2 + n)*self.c, c2, 1) # second convolution layer
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k = ((3, 3), (3, 3)), e = 1.0) for _ in range(n)
        ) # list of bottleneck layers
    
    def forward(self, x):
        # forward propagation : through C2f module
        y = list(self.cv1(x).chunk(2, 1)) # split output of first convolution into 2 parts
        y.extend(m(y[-1]) for m in self.m)
        
        return self.cv2(torch.cat(y, 1)) # concatenate outputs and pass through second convolution
    
    def forward_split(self, x):
        # forward propagation with split output (for inference optimization)
        y = list(self.cv1(x).split((self.c, self.c), 1)) # split output of first convolution into 2 parts
        y.extend(m(y[-1]) for m in self.m) # pass second part through bottleneck layers
        return self.cv2(torch.cat(y, 1)) # concatenate outputs and pass through second convolution
    
class C3(nn.Module):
    '''CSP bottleneck with 3 convolutions'''
    
    def __init__(self, c1, c2, n = 1, shortcut = True, g = 1, e = 0.5):
        """Initializes a CSP bottleneck with 3 convolutions and n bottleneck layers for faster processing"""
        super().__init__()
        c_ = int(c2 * e) # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1) # first convolution layer
        self.cv2 = Conv(c1, c_, 1, 1,) # second convolution layer
        self.cv3 = Conv(2*c_, c2, 1) # third convolution layer
        self.m = nn.Sequential(
            *(Bottleneck(c_, c_, shortcut, g, k = ((3, 3), (3, 3)), e = 1.0) for _ in range(n))
        )
        
    def forward(self, x):
        # forward propagation : through c3 module
        y1 = self.cv1(x) # first convolution on input
        y2 = self.cv2(x) # second convolution on input
        y1 = self.m(y1) # pass first convolution output through bottleneck layers
        return self.cv3(torch.cat((y1, y2), dim=1)) # concatenate outputs and pass through third convolution    

class C3k(C3):
    '''C3k is a csp bottleneck module with customizable kernel size'''
    
    def __init__(self, c1, c2, n = 1, shortcut = True, g = 1, e = 0.5, k = 3):
        """Initializes a C3k module, a CSP bottleneck with customizable kernel size and n bottleneck layers for faster processing
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e) # hidden channels
        self.m = nn.Sequential(
            *(Bottleneck(c_, c_, shortcut, g, k = (k, k), e = 1.0) for _ in range(n))
        )
  
        
class C3k2(C2f):
    '''faster implementation of CSP bottleneck with 2 convolutions
    '''
    
    def __init__(self, c1, c2, n = 1, c3k = False, e = 0.5, g = 1, shortcut = True):
        """Initializes a C3k2 module, a faster CSP bottleneck with 2 convolutions and optional C3k block for faster processing
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c,2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )
    
class SPPF(nn.Module):
    '''Spatial pyramid pooling - fast with 5x less memory, compatible with ONNX Runtime'''
    
    def __init__(self, c1, c2, k = 5):
        
        '''
        Initializes the SPPF module with given input/output channels and kernel size
        This module is equivalent to SPP(k = (5, 9, 13)) but optimized for faster processing and reduced memory usage, making it compatible with ONNX Runtime.
        '''
        super().__init__()
        c_ = c1 // 2 # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1) # first convolution layer
        self.cv2 = Conv(c_*4, c2, 1, 1) # second convolution layer
        self.m = nn.MaxPool2d(kernel_size = k, stride = 1, padding = k // 2) # max pooling layer with specified kernel size and padding for same output shape
        
    def forward(self, x):
        # forward propagation : through SPPF module
        y = [self.cv1(x)] # first convolution on input
        y.extend(self.m(y[-1]) for _ in range(3)) # apply max pooling three times to create spatial pyramid
        return self.cv2(torch.cat(y, 1)) # concatenate outputs and pass through second convolution
        

# Functions for neck module
class Attention(nn.Module):
    
    def __init__(self, dim, num_heads = 8, attn_ratio = 0.5):
    
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act= False)
        self.proj = Conv(dim, dim, 1, act = False)
        self.pe = Conv(dim, dim, 3, 1, g = dim, act = False)
        
    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim*2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim = 2
        )
        
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim = -1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x
    
class PSABlock(nn.Module):
    '''This class encapsulates the functionality for applying multi-head attention and feed-forward neural network with optional shortcut connectoin'''
    
    def __init__(self, c, attn_ratio = 0.5, num_heads = 4, shortcut = True) -> None:
        super().__init__()
        
        self.attn = Attention(c, attn_ratio = attn_ratio, num_heads = num_heads) # initialize the attention mechanism with specified parameters
        self.ffn = nn.Sequential(
            Conv(c, c*2, 1), Conv(c*2, c, 1, act = False)
        )
        self.add = shortcut
        
    def forward(self, x):
        """Executes a forward pass through the PSABlock, applying attention and feed-forward operations, and optionally adding a shortcut connection."""
        
        x = x + self.attn(x) if self.add else self.attn(x) # apply attention and add input to output if shortcut is enabled
        x = x + self.ffn(x) if self.add else self.ffn(x)

        return x
    
class C2PSA(nn.Module):
    '''
    This module implements a convolutional block with attention mechanism to enhance
    feature representation. It consists of a convolutional layer followed by a channel-wise
    '''
    
    def __init__(self, c1, c2, n = 1, e = 0.5):
        # initialize the C2SPA module with input channels c1, output channels c2, number of convolutional layers n, and expansion ratio e
        super().__init__()
        assert c1 == c2, "Input and output channels must be the same for C2PSA"
        self.c = int(c1 * e) # calculate the intermediate channel size based on the expansion ratio
        self.cv1 = Conv(c1, 2*self.c, 1)
        self.cv2 = Conv(2*self.c, c1, 1)
        
        self.m = nn.Sequential(
            *(PSABlock(self.c, attn_ratio = 0.5, num_heads = self.c // 64) for _ in range(n))
        )
    
    def forward(self, x):
        """Executes a forward pass through the C2PSA module, applying convolutional transformations and attention mechanisms to enhance feature representation."""
        
        a, b = self.cv1(x).split((self.c, self.c), dim = 1) # split the output of the first convolutional layer into two parts
        b = self.m(b)  # apply the attention mechanism to one part of the split output
        return self.cv2(torch.cat((a,b), dim = 1)) # concatenate the two parts and apply the second convolutional layer to produce the final output
    
class Concat(nn.Module):
    '''
    Concatenate a list of tensors along dimension
    '''
    def __init__(self, dimension = 1):
        super().__init__()
        self.d = dimension
        
    def forward(self, x):
        return torch.cat(x, dim = self.d)
    
# functions for head module detection

class DWConv(Conv):
    """Depth-wise convolution"""
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        """Initialize depth-wise convolution with given parameters"""
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)

class DFL(nn.Module):
    """
    Distribution Focal Loss (DFL) module.
    Converts raw anchor box predictions into actual box coordinates
    using a weighted sum over a discrete distribution of possible values.
    """
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False)  # 1x1 conv to collapse distribution
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = x.view(1, c1, 1, 1)
        self.c1 = c1

    def forward(self, x):
        b, c, a = x.shape  # batch, channels, anchors
        # reshape to [B, 4, reg_max, anchors] then softmax over reg_max dim
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
    
def make_anchors(feats, strides, grid_cell_offset=0.5):
    """
    Generate anchor points for all feature map scales.
    Returns anchor center coordinates and corresponding strides.
    """
    anchor_points, stride_tensor = [], []
    dtype, device = feats[0].dtype, feats[0].device

    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        # grid coordinates shifted by offset (0.5 = cell center)
        sx = torch.arange(w, device=device, dtype=dtype) + grid_cell_offset
        sy = torch.arange(h, device=device, dtype=dtype) + grid_cell_offset
        sy, sx = torch.meshgrid(sy, sx, indexing='ij')
        anchor_points.append(torch.stack((sx, sy), dim=-1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))

    return torch.cat(anchor_points), torch.cat(stride_tensor)

def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """
    Convert anchor-relative distance predictions to bounding boxes.

    distance : [lt, rb] — predicted distances from anchor to box edges
                (left, top, right, bottom)
    anchor_points : anchor center coordinates
    xywh : if True return (cx, cy, w, h), else (x1, y1, x2, y2)
    """
    lt, rb = distance.chunk(2, dim)               # left-top, right-bottom distances
    x1y1 = anchor_points - lt                     # top-left corner
    x2y2 = anchor_points + rb                     # bottom-right corner
    if xywh:
        c_xy = (x1y1 + x2y2) / 2                 # center x, y
        wh   = x2y2 - x1y1                        # width, height
        return torch.cat((c_xy, wh), dim)
    return torch.cat((x1y1, x2y2), dim)

class Detect(nn.Module):
    """YOLO Detect head for detection models"""

    end2end = False
    legacy = False

    def __init__(self, nc=80, ch=()):
        """
        Initializes the Detect head.

        Args:
            nc  : number of classes (e.g. 80 for COCO)
            ch  : tuple of input channels from neck, one per scale
                  e.g. (256, 512, 1024) for P3, P4, P5
        """
        super().__init__()
        self.nc = nc                        # number of classes
        self.nl = len(ch)                   # number of detection layers (3 for P3/P4/P5)
        self.reg_max = 16                   # DFL bins
        self.no = nc + self.reg_max * 4     # outputs per anchor = classes + 4*reg_max
        self.stride = torch.zeros(self.nl)  # filled during build
        self.shape = None
        self.anchors = torch.empty(0)
        self.strides = torch.empty(0)

        # channel sizes for bbox and cls branches
        c2 = max(16, ch[0] // 4, self.reg_max * 4)   # bbox branch hidden channels
        c3 = max(ch[0], min(self.nc, 100))            # cls branch hidden channels

        # ── bbox regression branch (cv2) ──────────────────────────────────────
        # one sub-network per scale: Conv→Conv→Conv2d → 4*reg_max channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c2, 3),
                Conv(c2, c2, 3),
                nn.Conv2d(c2, 4 * self.reg_max, 1)
            )
            for x in ch
        )

        # ── class prediction branch (cv3) ─────────────────────────────────────
        # legacy path = plain convs (v8 style)
        # default path = depthwise convs (v11 style, more efficient)
        self.cv3 = (
            nn.ModuleList(
                nn.Sequential(
                    Conv(x, c3, 3),
                    Conv(c3, c3, 3),
                    nn.Conv2d(c3, self.nc, 1)
                )
                for x in ch
            )
            if self.legacy
            else nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),     # depthwise + pointwise
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),  # depthwise + pointwise
                    nn.Conv2d(c3, self.nc, 1)                            # final class logits
                )
                for x in ch
            )
        )

        # DFL layer to decode bbox distributions into coordinates
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        """
        Forward pass through all detection scales.

        Args:
            x : list of feature maps [P3, P4, P5]
                shapes e.g. [(B,256,80,80), (B,512,40,40), (B,1024,20,20)]

        Returns:
            During training : list of raw predictions per scale
            During inference: decoded (boxes, scores) tensor
        """
        for i in range(self.nl):
            # concatenate bbox and cls predictions along channel dim
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), dim=1)

        if self.training:
            return x  # raw predictions, loss computed externally

        # ── inference decoding ────────────────────────────────────────────────
        shape = x[0].shape  # B, C, H, W
        # flatten spatial dims and concat all scales → [B, no, total_anchors]
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], dim=2)

        # recompute anchors only if input shape changed
        if self.shape != shape:
            self.anchors, self.strides = (
                a.transpose(0, 1) for a in make_anchors(x, self.stride, 0.5)
            )
            self.shape = shape

        # split into bbox distribution and class logits
        box, cls = x_cat.split((self.reg_max * 4, self.nc), dim=1)

        # decode DFL distribution → actual box (cx, cy, w, h)
        dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides.transpose(0, 1)

        # return [boxes + scores] concatenated
        return torch.cat((dbox, cls.sigmoid()), dim=1)

    def decode_bboxes(self, bboxes, anchors):
        """Decode predicted bbox deltas into (x1,y1,x2,y2) format"""
        return dist2bbox(bboxes, anchors, xywh=True, dim=1)

    def bias_init(self):
        """
        Initialize biases for better training stability.
        Sets cls bias based on prior object probability,
        sets bbox bias to reasonable starting values.
        """
        for a, b, s in zip(self.cv2, self.cv3, self.stride):
            # bbox bias: last Conv2d bias in cv2
            a[-1].bias.data[:] = 1.0
            # cls bias: initialise with log-prior so sigmoid(bias) ≈ 0.01
            b[-1].bias.data[:self.nc] = math.log(5 / self.nc / (640 / s) ** 2)
            