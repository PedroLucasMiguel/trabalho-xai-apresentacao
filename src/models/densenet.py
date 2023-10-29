import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

# Modelo densenet201 utilizando encoder-decoder
class DenseNet201EncoderDecoder(nn.Module):

    def __init__(self, backbone:nn.Module, n_classes=2) -> None:
        super().__init__()

        self.gradient = None

        # Enconder-decoder model
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 128, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(2),
        )

        self.decoder = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(128, 3, 3, padding=1, stride=1),
            nn.ReLU(),
        )
        
        self.features = backbone.features
        self.classifier = nn.Linear(1920, n_classes)

    def gradient_hook(self, gradient):
        self.gradient = gradient

    def get_activations_gradient(self):
        return self.gradient

    def get_activations(self, x: Tensor):
        out = self.encoder(x)
        out = self.decoder(out)
        out = self.features(out)

        return out

    def forward(self, x: Tensor) -> Tensor:
        out = self.encoder(x)
        out = self.decoder(out)
        out = self.features(out)
        
        # Grad-cam
        out = F.relu(out, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        if out.requires_grad:
            h = out.register_hook(self.gradient_hook)

        out = torch.flatten(out, 1)
        out = self.classifier(out)
        
        return out

# Modelo densenet201 para obtenção do grad-cam 
class DenseNet201GradCam(nn.Module):
    def __init__(self, backbone:nn.Module, n_classes=2) -> None:
        super().__init__()

        self.gradient = None

        self.features = backbone.features
        self.classifier = nn.Linear(1920, n_classes)

    def gradient_hook(self, gradient):
        self.gradient = gradient

    def get_activations_gradient(self):
        return self.gradient

    def get_activations(self, x: Tensor):
        out = self.features(x)
        return out

    def forward(self, x: Tensor) -> Tensor:
        out = self.features(x)
        
        # Grad-cam
        out = F.relu(out, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        if out.requires_grad:
            h = out.register_hook(self.gradient_hook)

        out = torch.flatten(out, 1)
        out = self.classifier(out)
        
        return out

# Modelo densenet utilizando a camada de GAP
class DenseNet201GAP(nn.Module):
    def __init__(self, backbone:nn.Module, n_classes=2) -> None:
        super().__init__()

        self.features = backbone.features[:-1] # Removendo última batch norm
        
        # Essa modificação é necessária para obter um número de pesos
        # e ativações análogos a quantidade de classes no dataset
        self.modification = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(1920, n_classes, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_classes, n_classes, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.features(x)
        out = self.modification(out)
        
        # GAP
        out = torch.mean(out, dim=(2,3))

        return out