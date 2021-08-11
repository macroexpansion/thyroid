import torch.nn.functional as F
import torch


class FixedSizePadding(torch.nn.Module):
    def __init__(self, max_width: int = 366, max_height: int = 258):
        self.MAX_WIDTH = max_width
        self.MAX_HEIGHT = max_height

    def forward(self, image: torch.Tensor):
        _, h, w = image.size()
        left_pad, right_pad, top_pad, bot_pad = 0, 0, 0, 0
        if h < self.MAX_HEIGHT:
            diff = self.MAX_HEIGHT - h
            top_pad = diff // 2
            bot_pad = diff // 2 if diff % 2 == 0 else diff // 2 + 1
        if w < self.MAX_WIDTH:
            diff = self.MAX_WIDTH - w
            left_pad = diff // 2
            right_pad = diff // 2 if diff % 2 == 0 else diff // 2 + 1

        image = F.pad(image, (left_pad, right_pad, top_pad, bot_pad), "constant", 0)

        return image
