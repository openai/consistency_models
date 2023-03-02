# Ported from the model here:
# https://github.com/NVlabs/stylegan3/blob/407db86e6fe432540a22515310188288687858fa/metrics/frechet_inception_distance.py#L22
#
# I have verified that the spatial features and output features are correct
# within a mean absolute error of ~3e-5.

import collections

import torch


class Conv2dLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kh, kw, stride=1, padding=0):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.weight = torch.nn.Parameter(torch.zeros(out_channels, in_channels, kh, kw))
        self.beta = torch.nn.Parameter(torch.zeros(out_channels))
        self.mean = torch.nn.Parameter(torch.zeros(out_channels))
        self.var = torch.nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        x = torch.nn.functional.conv2d(
            x, self.weight.to(x.dtype), stride=self.stride, padding=self.padding
        )
        x = torch.nn.functional.batch_norm(
            x, running_mean=self.mean, running_var=self.var, bias=self.beta, eps=1e-3
        )
        x = torch.nn.functional.relu(x)
        return x


# ----------------------------------------------------------------------------


class InceptionA(torch.nn.Module):
    def __init__(self, in_channels, tmp_channels):
        super().__init__()
        self.conv = Conv2dLayer(in_channels, 64, kh=1, kw=1)
        self.tower = torch.nn.Sequential(
            collections.OrderedDict(
                [
                    ("conv", Conv2dLayer(in_channels, 48, kh=1, kw=1)),
                    ("conv_1", Conv2dLayer(48, 64, kh=5, kw=5, padding=2)),
                ]
            )
        )
        self.tower_1 = torch.nn.Sequential(
            collections.OrderedDict(
                [
                    ("conv", Conv2dLayer(in_channels, 64, kh=1, kw=1)),
                    ("conv_1", Conv2dLayer(64, 96, kh=3, kw=3, padding=1)),
                    ("conv_2", Conv2dLayer(96, 96, kh=3, kw=3, padding=1)),
                ]
            )
        )
        self.tower_2 = torch.nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        "pool",
                        torch.nn.AvgPool2d(
                            kernel_size=3, stride=1, padding=1, count_include_pad=False
                        ),
                    ),
                    ("conv", Conv2dLayer(in_channels, tmp_channels, kh=1, kw=1)),
                ]
            )
        )

    def forward(self, x):
        return torch.cat(
            [
                self.conv(x).contiguous(),
                self.tower(x).contiguous(),
                self.tower_1(x).contiguous(),
                self.tower_2(x).contiguous(),
            ],
            dim=1,
        )


# ----------------------------------------------------------------------------


class InceptionB(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = Conv2dLayer(in_channels, 384, kh=3, kw=3, stride=2)
        self.tower = torch.nn.Sequential(
            collections.OrderedDict(
                [
                    ("conv", Conv2dLayer(in_channels, 64, kh=1, kw=1)),
                    ("conv_1", Conv2dLayer(64, 96, kh=3, kw=3, padding=1)),
                    ("conv_2", Conv2dLayer(96, 96, kh=3, kw=3, stride=2)),
                ]
            )
        )
        self.pool = torch.nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        return torch.cat(
            [
                self.conv(x).contiguous(),
                self.tower(x).contiguous(),
                self.pool(x).contiguous(),
            ],
            dim=1,
        )


# ----------------------------------------------------------------------------


class InceptionC(torch.nn.Module):
    def __init__(self, in_channels, tmp_channels):
        super().__init__()
        self.conv = Conv2dLayer(in_channels, 192, kh=1, kw=1)
        self.tower = torch.nn.Sequential(
            collections.OrderedDict(
                [
                    ("conv", Conv2dLayer(in_channels, tmp_channels, kh=1, kw=1)),
                    (
                        "conv_1",
                        Conv2dLayer(
                            tmp_channels, tmp_channels, kh=1, kw=7, padding=[0, 3]
                        ),
                    ),
                    (
                        "conv_2",
                        Conv2dLayer(tmp_channels, 192, kh=7, kw=1, padding=[3, 0]),
                    ),
                ]
            )
        )
        self.tower_1 = torch.nn.Sequential(
            collections.OrderedDict(
                [
                    ("conv", Conv2dLayer(in_channels, tmp_channels, kh=1, kw=1)),
                    (
                        "conv_1",
                        Conv2dLayer(
                            tmp_channels, tmp_channels, kh=7, kw=1, padding=[3, 0]
                        ),
                    ),
                    (
                        "conv_2",
                        Conv2dLayer(
                            tmp_channels, tmp_channels, kh=1, kw=7, padding=[0, 3]
                        ),
                    ),
                    (
                        "conv_3",
                        Conv2dLayer(
                            tmp_channels, tmp_channels, kh=7, kw=1, padding=[3, 0]
                        ),
                    ),
                    (
                        "conv_4",
                        Conv2dLayer(tmp_channels, 192, kh=1, kw=7, padding=[0, 3]),
                    ),
                ]
            )
        )
        self.tower_2 = torch.nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        "pool",
                        torch.nn.AvgPool2d(
                            kernel_size=3, stride=1, padding=1, count_include_pad=False
                        ),
                    ),
                    ("conv", Conv2dLayer(in_channels, 192, kh=1, kw=1)),
                ]
            )
        )

    def forward(self, x):
        return torch.cat(
            [
                self.conv(x).contiguous(),
                self.tower(x).contiguous(),
                self.tower_1(x).contiguous(),
                self.tower_2(x).contiguous(),
            ],
            dim=1,
        )


# ----------------------------------------------------------------------------


class InceptionD(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.tower = torch.nn.Sequential(
            collections.OrderedDict(
                [
                    ("conv", Conv2dLayer(in_channels, 192, kh=1, kw=1)),
                    ("conv_1", Conv2dLayer(192, 320, kh=3, kw=3, stride=2)),
                ]
            )
        )
        self.tower_1 = torch.nn.Sequential(
            collections.OrderedDict(
                [
                    ("conv", Conv2dLayer(in_channels, 192, kh=1, kw=1)),
                    ("conv_1", Conv2dLayer(192, 192, kh=1, kw=7, padding=[0, 3])),
                    ("conv_2", Conv2dLayer(192, 192, kh=7, kw=1, padding=[3, 0])),
                    ("conv_3", Conv2dLayer(192, 192, kh=3, kw=3, stride=2)),
                ]
            )
        )
        self.pool = torch.nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        return torch.cat(
            [
                self.tower(x).contiguous(),
                self.tower_1(x).contiguous(),
                self.pool(x).contiguous(),
            ],
            dim=1,
        )


# ----------------------------------------------------------------------------


class InceptionE(torch.nn.Module):
    def __init__(self, in_channels, use_avg_pool):
        super().__init__()
        self.conv = Conv2dLayer(in_channels, 320, kh=1, kw=1)
        self.tower_conv = Conv2dLayer(in_channels, 384, kh=1, kw=1)
        self.tower_mixed_conv = Conv2dLayer(384, 384, kh=1, kw=3, padding=[0, 1])
        self.tower_mixed_conv_1 = Conv2dLayer(384, 384, kh=3, kw=1, padding=[1, 0])
        self.tower_1_conv = Conv2dLayer(in_channels, 448, kh=1, kw=1)
        self.tower_1_conv_1 = Conv2dLayer(448, 384, kh=3, kw=3, padding=1)
        self.tower_1_mixed_conv = Conv2dLayer(384, 384, kh=1, kw=3, padding=[0, 1])
        self.tower_1_mixed_conv_1 = Conv2dLayer(384, 384, kh=3, kw=1, padding=[1, 0])
        if use_avg_pool:
            self.tower_2_pool = torch.nn.AvgPool2d(
                kernel_size=3, stride=1, padding=1, count_include_pad=False
            )
        else:
            self.tower_2_pool = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.tower_2_conv = Conv2dLayer(in_channels, 192, kh=1, kw=1)

    def forward(self, x):
        a = self.tower_conv(x)
        b = self.tower_1_conv_1(self.tower_1_conv(x))
        return torch.cat(
            [
                self.conv(x).contiguous(),
                self.tower_mixed_conv(a).contiguous(),
                self.tower_mixed_conv_1(a).contiguous(),
                self.tower_1_mixed_conv(b).contiguous(),
                self.tower_1_mixed_conv_1(b).contiguous(),
                self.tower_2_conv(self.tower_2_pool(x)).contiguous(),
            ],
            dim=1,
        )


# ----------------------------------------------------------------------------


class InceptionV3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            collections.OrderedDict(
                [
                    ("conv", Conv2dLayer(3, 32, kh=3, kw=3, stride=2)),
                    ("conv_1", Conv2dLayer(32, 32, kh=3, kw=3)),
                    ("conv_2", Conv2dLayer(32, 64, kh=3, kw=3, padding=1)),
                    ("pool0", torch.nn.MaxPool2d(kernel_size=3, stride=2)),
                    ("conv_3", Conv2dLayer(64, 80, kh=1, kw=1)),
                    ("conv_4", Conv2dLayer(80, 192, kh=3, kw=3)),
                    ("pool1", torch.nn.MaxPool2d(kernel_size=3, stride=2)),
                    ("mixed", InceptionA(192, tmp_channels=32)),
                    ("mixed_1", InceptionA(256, tmp_channels=64)),
                    ("mixed_2", InceptionA(288, tmp_channels=64)),
                    ("mixed_3", InceptionB(288)),
                    ("mixed_4", InceptionC(768, tmp_channels=128)),
                    ("mixed_5", InceptionC(768, tmp_channels=160)),
                    ("mixed_6", InceptionC(768, tmp_channels=160)),
                    ("mixed_7", InceptionC(768, tmp_channels=192)),
                    ("mixed_8", InceptionD(768)),
                    ("mixed_9", InceptionE(1280, use_avg_pool=True)),
                    ("mixed_10", InceptionE(2048, use_avg_pool=False)),
                    ("pool2", torch.nn.AvgPool2d(kernel_size=8)),
                ]
            )
        )
        self.output = torch.nn.Linear(2048, 1008)

    def forward(
        self,
        img,
        return_features: bool = True,
        use_fp16: bool = False,
        no_output_bias: bool = False,
    ):
        batch_size, channels, height, width = img.shape  # [NCHW]
        assert channels == 3

        # Cast to float.
        x = img.to(torch.float16 if use_fp16 else torch.float32)

        # Emulate tf.image.resize_bilinear(x, [299, 299]), including the funky alignment.
        new_width, new_height = 299, 299
        theta = torch.eye(2, 3, device=x.device)
        theta[0, 2] += theta[0, 0] / width - theta[0, 0] / new_width
        theta[1, 2] += theta[1, 1] / height - theta[1, 1] / new_height
        theta = theta.to(x.dtype).unsqueeze(0).repeat([batch_size, 1, 1])
        grid = torch.nn.functional.affine_grid(
            theta, [batch_size, channels, new_height, new_width], align_corners=False
        )
        x = torch.nn.functional.grid_sample(
            x, grid, mode="bilinear", padding_mode="border", align_corners=False
        )

        # Scale dynamic range from [0,255] to [-1,1[.
        x -= 128
        x /= 128

        # Main layers.
        intermediate = self.layers[:-6](x)
        spatial_features = (
            self.layers[-6]
            .conv(intermediate)[:, :7]
            .permute(0, 2, 3, 1)
            .reshape(-1, 2023)
        )
        features = self.layers[-6:](intermediate).reshape(-1, 2048).to(torch.float32)
        if return_features:
            return features, spatial_features

        # Output layer.
        return self.acts_to_probs(features, no_output_bias=no_output_bias)

    def acts_to_probs(self, features, no_output_bias: bool = False):
        if no_output_bias:
            logits = torch.nn.functional.linear(features, self.output.weight)
        else:
            logits = self.output(features)
        probs = torch.nn.functional.softmax(logits, dim=1)
        return probs

    def create_softmax_model(self):
        return SoftmaxModel(self.output.weight)


class SoftmaxModel(torch.nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = torch.nn.Parameter(weight.detach().clone())

    def forward(self, x):
        logits = torch.nn.functional.linear(x, self.weight)
        probs = torch.nn.functional.softmax(logits, dim=1)
        return probs
