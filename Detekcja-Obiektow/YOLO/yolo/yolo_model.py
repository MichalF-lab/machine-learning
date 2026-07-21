import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNAct(nn.Module):
    def __init__(self, in_c, out_c, k=1, s=1, p=None, g=1):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, groups=g, bias=False)
        self.bn   = nn.BatchNorm2d(out_c, eps=1e-3, momentum=0.03)
        self.act  = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, in_c, out_c, shortcut=True, e=0.5):
        super().__init__()
        mid = int(out_c * e)
        self.cv1 = ConvBNAct(in_c,  mid,   k=3)
        self.cv2 = ConvBNAct(mid,  out_c,  k=3)
        self.add = shortcut and (in_c == out_c)

    def forward(self, x):
        out = self.cv2(self.cv1(x))
        return x + out if self.add else out


class C2f(nn.Module):
    def __init__(self, in_c, out_c, n=1, shortcut=True, e=0.5):
        super().__init__()
        mid = int(out_c * e)
        self.cv1 = ConvBNAct(in_c,  2 * mid, k=1)
        self.cv2 = ConvBNAct((2 + n) * mid, out_c, k=1)
        self.bottlenecks = nn.ModuleList(
            [Bottleneck(mid, mid, shortcut=shortcut, e=1.0) for _ in range(n)]
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, dim=1))
        for bn in self.bottlenecks:
            y.append(bn(y[-1]))
        return self.cv2(torch.cat(y, dim=1))


class SPPF(nn.Module):
    def __init__(self, in_c, out_c, k=5):
        super().__init__()
        mid = in_c // 2
        self.cv1 = ConvBNAct(in_c, mid, k=1)
        self.cv2 = ConvBNAct(mid * 4, out_c, k=1)
        self.pool = nn.MaxPool2d(k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))


class YOLOv8Backbone(nn.Module):
    def __init__(self, depth_mul=0.33, width_mul=0.25):
        super().__init__()
        def c(base): return max(round(base * width_mul), 1)
        def n(base): return max(round(base * depth_mul), 1)

        self.stem = ConvBNAct(3, c(64), k=3, s=2)

        self.stage1 = nn.Sequential(
            ConvBNAct(c(64), c(128), k=3, s=2),
            C2f(c(128), c(128), n=n(3), shortcut=True),
        )

        self.stage2 = nn.Sequential(
            ConvBNAct(c(128), c(256), k=3, s=2),
            C2f(c(256), c(256), n=n(6), shortcut=True),
        )

        self.stage3 = nn.Sequential(
            ConvBNAct(c(256), c(512), k=3, s=2),
            C2f(c(512), c(512), n=n(6), shortcut=True),
        )

        self.stage4 = nn.Sequential(
            ConvBNAct(c(512), c(512), k=3, s=2),
            C2f(c(512), c(512), n=n(3), shortcut=True),
            SPPF(c(512), c(512)),
        )

    def forward(self, x):
        x  = self.stem(x)
        x  = self.stage1(x)
        p3 = self.stage2(x)
        p4 = self.stage3(p3)
        p5 = self.stage4(p4)
        return p3, p4, p5


class YOLOv8Neck(nn.Module):
    def __init__(self, depth_mul=0.33, width_mul=0.25):
        super().__init__()
        def c(base): return max(round(base * width_mul), 1)
        def n(base): return max(round(base * depth_mul), 1)

        self.up      = nn.Upsample(scale_factor=2, mode='nearest')

        self.td_c2f1 = C2f(c(512) + c(512), c(512), n=n(3), shortcut=False)
        self.td_c2f2 = C2f(c(512) + c(256), c(256), n=n(3), shortcut=False)

        self.bu_down1 = ConvBNAct(c(256), c(256), k=3, s=2)
        self.bu_c2f1  = C2f(c(256) + c(512), c(512), n=n(3), shortcut=False)

        self.bu_down2 = ConvBNAct(c(512), c(512), k=3, s=2)
        self.bu_c2f2  = C2f(c(512) + c(512), c(512), n=n(3), shortcut=False)

        self.out_channels = [c(256), c(512), c(512)]

    def forward(self, p3, p4, p5):
        td_p4 = self.td_c2f1(torch.cat([self.up(p5),   p4], dim=1))
        td_p3 = self.td_c2f2(torch.cat([self.up(td_p4), p3], dim=1))

        bu_p4 = self.bu_c2f1(torch.cat([self.bu_down1(td_p3), td_p4], dim=1))
        bu_p5 = self.bu_c2f2(torch.cat([self.bu_down2(bu_p4),  p5],   dim=1))

        return td_p3, bu_p4, bu_p5


class DetectionHead(nn.Module):
    def __init__(self, in_c, num_classes, mid_c=None):
        super().__init__()
        mid_c = mid_c or in_c

        self.reg_branch = nn.Sequential(
            ConvBNAct(in_c,  mid_c, k=3),
            ConvBNAct(mid_c, mid_c, k=3),
            nn.Conv2d(mid_c, 4, kernel_size=1),
        )

        self.cls_branch = nn.Sequential(
            ConvBNAct(in_c,  mid_c, k=3),
            ConvBNAct(mid_c, mid_c, k=3),
            nn.Conv2d(mid_c, num_classes, kernel_size=1),
        )

    def forward(self, x):
        reg = self.reg_branch(x)
        cls = self.cls_branch(x)
        return torch.cat([reg, cls], dim=1)


class YOLOv8(nn.Module):
    VARIANTS = {
        "nano":   (0.33, 0.25),
        "small":  (0.33, 0.50),
        "medium": (0.67, 0.75),
    }

    def __init__(self, num_classes: int, depth_mul=0.33, width_mul=0.25, img_size=640):
        super().__init__()
        self.num_classes = num_classes
        self.img_size    = img_size
        self.strides     = [8, 16, 32]

        self.backbone = YOLOv8Backbone(depth_mul, width_mul)
        self.neck     = YOLOv8Neck(depth_mul, width_mul)

        neck_channels = self.neck.out_channels
        self.heads = nn.ModuleList([
            DetectionHead(ch, num_classes) for ch in neck_channels
        ])

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        prior_prob = 0.01
        bias_val   = -torch.log(torch.tensor((1 - prior_prob) / prior_prob)).item()
        for head in self.heads:
            nn.init.constant_(head.cls_branch[-1].bias, bias_val)

    def forward(self, x):
        p3, p4, p5 = self.backbone(x)
        f3, f4, f5 = self.neck(p3, p4, p5)

        out = []
        for feat, head in zip([f3, f4, f5], self.heads):
            out.append(head(feat))
        return out

    @torch.no_grad()
    def decode_predictions(self, raw_outputs, conf_thres=0.25, img_size=640):
        device    = raw_outputs[0].device
        all_dets  = []

        anchors_per_scale = []
        for stride in self.strides:
            grid_size = img_size // stride
            gy, gx = torch.meshgrid(
                torch.arange(grid_size, device=device),
                torch.arange(grid_size, device=device),
                indexing='ij'
            )
            anchors_per_scale.append(
                torch.stack([gx, gy], dim=-1).reshape(-1, 2).float() * stride + stride / 2
            )

        batch_size = raw_outputs[0].shape[0]

        for b in range(batch_size):
            dets = []
            for scale_idx, (raw, stride) in enumerate(zip(raw_outputs, self.strides)):
                feat = raw[b]
                H, W = feat.shape[1], feat.shape[2]
                feat = feat.permute(1, 2, 0).reshape(-1, 4 + self.num_classes)

                reg = feat[:, :4]
                cls_logits = feat[:, 4:]

                anch = anchors_per_scale[scale_idx][:H*W]
                cx = anch[:, 0] + reg[:, 0] * stride
                cy = anch[:, 1] + reg[:, 1] * stride
                w  = torch.exp(reg[:, 2].clamp(-4, 4)) * stride
                h  = torch.exp(reg[:, 3].clamp(-4, 4)) * stride

                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2

                scores, class_ids = torch.sigmoid(cls_logits).max(dim=-1)
                mask = scores > conf_thres

                if mask.sum() == 0:
                    continue

                dets.append(torch.stack([
                    x1[mask], y1[mask], x2[mask], y2[mask],
                    scores[mask], class_ids[mask].float()
                ], dim=-1))

            if dets:
                all_dets.append(torch.cat(dets, dim=0))
            else:
                all_dets.append(torch.empty((0, 6), device=device))

        return all_dets

    @classmethod
    def from_variant(cls, variant: str, num_classes: int, **kwargs):
        if variant not in cls.VARIANTS:
            raise ValueError(f"Nieznany wariant '{variant}'. Dostępne: {list(cls.VARIANTS)}")
        d, w = cls.VARIANTS[variant]
        return cls(num_classes=num_classes, depth_mul=d, width_mul=w, **kwargs)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)