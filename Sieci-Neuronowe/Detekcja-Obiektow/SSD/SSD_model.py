import torch
import torch.nn as nn
import torchvision


class SSD(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.num_classes = num_classes + 1 # +1 dla tła

        self.anchors_per_cell = [4, 6, 6, 6, 4, 4]


        # CZĘŚĆ 1: SZKIELET VGG16

        vgg = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT)

        # bierzemy 23 pierwsze warstwy modelu (Conv -> ReLU -> Conv -> ReLU -> Pool), wykrywanie małych obiektów (output 38x38x512)
        self.backbone_to_conv4_3 = nn.Sequential(*list(vgg.features)[:23])

        # Reszta (output 19x19x512 (pooling)) 
        self.backbone_conv5 = nn.Sequential(*list(vgg.features)[23:30])


        # Zamiana warstwy VGG na konwolucje (szukamy gdzie a nie czy)
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1)


        # CZĘŚĆ 2: DODATKOWE WARSTWY

        self.extra_layers = nn.ModuleList([
            # conv8: 19x19 -> 10x10
            nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
            ),
            # conv9: 10x10 -> 5x5
            nn.Sequential(
                nn.Conv2d(512, 128, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
            ),
            # conv10: 5x5 -> 3x3
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3),
                nn.ReLU(inplace=True),
            ),
            # conv11: 3x3 -> 1x1
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3),
                nn.ReLU(inplace=True),
            ),
        ])

        # CZĘŚĆ 3: Głowy Predykcyjne

        in_channels = [512, 1024, 512, 256, 256, 256]

        self.loc_heads = nn.ModuleList()
        self.cls_heads = nn.ModuleList()

        for in_ch, n_anchors in zip(in_channels, self.anchors_per_cell):

            # zmiany (cx, cy, w, h)
            self.loc_heads.append(nn.Conv2d(in_ch, n_anchors * 4, kernel_size=3, padding=1))

            # num_classes wartości na anchor (łącznie z tłem)
            self.cls_heads.append(nn.Conv2d(in_ch, n_anchors * self.num_classes, kernel_size=3, padding=1))

    def extract_feature_maps(self, x):
        feature_maps = []

        # Backbone -> feature mapa 38x38
        x = self.backbone_to_conv4_3(x)
        feature_maps.append(x)

        # Conv5
        x = self.backbone_conv5(x)

        # Conv6 + Conv7 -> 19x19
        x = torch.relu(self.conv6(x))
        x = torch.relu(self.conv7(x))
        feature_maps.append(x)

        # Extra feature layers -> 10x10, 5x5, 3x3, 1x1
        for extra in self.extra_layers:
            x = extra(x)
            feature_maps.append(x)

        return feature_maps

    def forward(self, x):
        batch_size = x.size(0)
        feature_maps = self.extract_feature_maps(x)

        all_loc_preds = []
        all_cls_preds = []

        for feat_map, loc_head, cls_head in zip(
            feature_maps, self.loc_heads, self.cls_heads
        ):
            # Predykcja lokalizacji: [B, anchors*4, H, W]
            loc = loc_head(feat_map)
            # Predykcja klas: [B, anchors*num_classes, H, W]
            cls = cls_head(feat_map)

            # [B, anchors*4, H, W] => [B, H, W, anchors*4] => [B, H*W*anchors, 4]
            loc = loc.permute(0, 2, 3, 1).contiguous()
            loc = loc.view(batch_size, -1, 4)

            # [B, anchors*num_classes, H, W] => [B, H*W*anchors, num_classes]
            cls = cls.permute(0, 2, 3, 1).contiguous()
            cls = cls.view(batch_size, -1, self.num_classes)

            all_loc_preds.append(loc)
            all_cls_preds.append(cls)

        loc_preds = torch.cat(all_loc_preds, dim=1)
        cls_preds = torch.cat(all_cls_preds, dim=1)

        return loc_preds, cls_preds


if __name__ == "__main__":
    model = SSD(num_classes=3)
    model.eval()

    with torch.no_grad():
        loc_preds, cls_preds = model(torch.randn(2, 3, 300, 300))

    print(f"Predykcje lokalizacji: {loc_preds.shape}")
    print(f"Predykcje klas: {cls_preds.shape}")
    print(f"Łączna liczba anchorów: {loc_preds.shape[1]}")