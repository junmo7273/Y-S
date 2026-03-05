# custom_yolov11.py
from ultralytics.models.yolo.detect import DetectionModel
from ultralytics.nn.modules import C2f, Conv, Detect
import torch
import torch.nn as nn

# Define C2f-Lite lightweight module (reduce parameters while retaining small target perception capability)
class C2fLite(nn.Module):
    """Lightweight C2f module: reduce the number of bottlenecks to lower computational complexity"""
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # Channel number scaling
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # Simplify the number of bottlenecks
        self.m = nn.ModuleList([
            nn.Sequential(Conv(self.c, self.c, 3, 1, g=g),
                          Conv(self.c, self.c, 3, 1, g=g)) for _ in range(n)
        ])
        self.shortcut = shortcut

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1)) if self.shortcut else self.cv2(torch.cat(y[1:], 1))

# Custom detection head: add a new small target detection branch
class DetectLite(Detect):
    """Enhanced Detect head for small target detection: add high-resolution detection branch"""
    def __init__(self, nc=80, ch=()):
        super().__init__(nc, ch)
        # Add small target branch (for PV panel small defects: cracks, solder joints, etc.)
        self.cv0_small = nn.Sequential(
            Conv(ch[0], ch[0], 3, 1),
            Conv(ch[0], self.no * self.nl, 1, 1)
        )
        # Adjust anchor boxes: adapt to PV panel small targets (smaller anchor boxes)
        self.anchors = torch.tensor([
            # Small target anchor boxes (adapt to PV panel defects)
            [[10,13], [16,30], [33,23]],
            [[30,61], [62,45], [59,119]],
            [[116,90], [156,198], [373,326]]
        ])

    def forward(self, x):
        # Original detection branch
        shape = x[0].shape
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        
        # Add new small target branch (high-resolution feature map)
        small_feat = self.cv0_small(x[0])  # Detect small targets with highest resolution feature map
        x[0] = x[0] + small_feat  # Feature fusion
        
        # Subsequent logic remains consistent with the original Detect
        if self.training:
            return x
        else:
            return self.inference(x, shape)

# Custom YOLOv11 model
class CustomYOLOv11(DetectionModel):
    def __init__(self, cfg='yolov11n.yaml', ch=3, nc=None, verbose=True):
        super().__init__(cfg, ch, nc, verbose)
        # Replace all C2f modules with C2f-Lite
        for i, m in enumerate(self.model.model):
            if isinstance(m, C2f):
                self.model.model[i] = C2fLite(
                    c1=m.cv1.conv.in_channels,
                    c2=m.cv2.conv.out_channels,
                    n=1,  # Lightweight: reduce the number of bottlenecks
                    shortcut=m.shortcut,
                    g=m.g,
                    e=m.e
                )
        # Replace detection head with enhanced small target version
        for i, m in enumerate(self.model.model):
            if isinstance(m, Detect):
                self.model.model[i] = DetectLite(
                    nc=self.nc,
                    ch=m.ch
                )