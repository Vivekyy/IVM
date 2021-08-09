import torch.nn as nn

class Model(nn.Module):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        
        self.classifier=nn.Sequential(
            nn.Linear(input_shape),
            nn.ReLU(),
            nn.Linear(1200),
            nn.ReLU(),
            nn.Linear(1600),
            nn.ReLU(),
            nn.Linear(1200),
            nn.ReLU(),
            nn.Linear(600),
            nn.ReLU(),
            nn.Linear(output_shape)
        )

    def forward(self, x):
        return(self.classifier(x))
