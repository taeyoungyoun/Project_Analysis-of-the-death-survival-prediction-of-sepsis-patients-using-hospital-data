import torch.nn as nn
from torch.nn import Transformer
import torch


class EncoderPredictionModel(nn.Module):
    def __init__(self, output_dim):
        super(EncoderPredictionModel, self).__init__()
        self.embedding = torch.load('pretrained_model_embedding.pth', map_location=torch.device('cpu'))
        self.encoder = torch.load('pretrained_model_encoder.pth', map_location=torch.device('cpu'))

        
        # embedding의 out_features로부터 d_model 추출
        d_model = self.embedding.out_features
        
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, src):
        out = self.embedding(src)  # Embedding Layer를 통과
        out = self.encoder(out)
        out = self.fc(out[-1, :, :])
        return torch.sigmoid(out)
