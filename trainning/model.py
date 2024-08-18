import torch.nn as nn
import torchvision.models as models
## ResNet50 (CNN Encoder)

class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.ResNet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        self.ResNet50.fc = nn.Sequential(
                            nn.Linear(2048, 1024),
                            nn.ReLU(),
                            nn.Dropout(0.4),
                            nn.Linear(1024, 2048),
                            nn.ReLU(),
                           )
        
        for k,v in self.ResNet50.named_parameters(recurse=True):
          if 'fc' in k:
            v.requires_grad = True
          else:
            v.requires_grad = False

    def forward(self,x):
        return self.ResNet50(x)        

## lSTM (Decoder)



## ImgCap



