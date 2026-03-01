import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet101_Weights


class VisualExtractor(nn.Module):
    def __init__(self, args):
        super(VisualExtractor, self).__init__()
        self.visual_extractor = args.visual_extractor
        self.pretrained = args.visual_extractor_pretrained
        if self.visual_extractor == 'resnet101':
            weights_to_load = ResNet101_Weights.DEFAULT if self.pretrained else None
            model = models.resnet101(weights=weights_to_load)
        else:
            
            model = getattr(models, self.visual_extractor)(pretrained=self.pretrained)
            print(
                f"Warning: The visual extractor '{self.visual_extractor}' might still use 'pretrained' parameter if it's not resnet101. Please check its specific 'weights' enum if warnings persist.")


        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)
        self.avg_fnt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

    def forward(self, images):
        patch_feats = self.model(images)
        avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1))
        batch_size, feat_size, _, _ = patch_feats.shape
        patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
        return patch_feats, avg_feats
