from common import *

class Autoencoder(nn.Module):    
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 5, 5, stride=1, padding=0),  # out: 76x76x5
            nn.Tanh(),
            nn.MaxPool2d(2, stride=2),  # out: 38x38x5
            nn.Conv2d(5, 2, 5, stride=1, padding=1),  # out: 36x36x2
            nn.Tanh(),
            nn.MaxPool2d(2,stride=2)  # out: 18x18x2
            )
        # torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2, 5, 4, stride=2, padding=0),  # out: 38x38x5
            nn.Tanh(),
            nn.ConvTranspose2d(5, 1, 6, stride=2, padding=0),  # out: 80x80x1
            nn.Tanh(),
            )   

    def forward(self, x):
        x = self.encoder(x) 
        x = self.decoder(x)
        return x

class Autoencoder_bottle(nn.Module):    
    def __init__(self, code_size):
        super(Autoencoder_bottle, self).__init__()
        
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=1, padding=0),  # out: 76x76x16
            nn.ELU(),
            nn.MaxPool2d(2, stride=2),  # out: 38x38x16
            nn.Conv2d(16, 8, 5, stride=1, padding=1),  # out: 36x36x8
            nn.ELU(),
            nn.MaxPool2d(2,stride=2)  # out: 18x18x8
            )
        # torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 4, stride=2, padding=0),  # out: 38x38x16
            nn.ELU(),
            nn.ConvTranspose2d(16, 1, 6, stride=2, padding=0),  # out: 80x80x16
            nn.ELU(),
            )   
        self.bottleNeck = nn.Sequential(
            nn.Linear(18*18*8,code_size),
            nn.Linear(code_size,18*18*8)
            )
    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0),-1)
        x = self.bottleNeck(x)
        x = x.view([-1,8,18,18])     
        x = self.decoder(x)
        return x


class Autoencoder_bottle_wide(nn.Module):    
    def __init__(self, bottlerange):
        super(Autoencoder_bottle_wide, self).__init__()
        
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=1, padding=0),  # out: 76x76x16
            nn.ELU(),
            nn.MaxPool2d(2, stride=2),  # out: 38x38x16
            nn.Conv2d(16, 8, 5, stride=1, padding=1),  # out: 36x36x8
            nn.ELU(),
            nn.MaxPool2d(2,stride=2)  # out: 18x18x8
            )
        # torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 4, stride=2, padding=0),  # out: 38x38x16
            nn.ELU(),
            nn.ConvTranspose2d(16, 1, 6, stride=2, padding=0),  # out: 80x80x16
            nn.ELU(),
            )   
        self.bottleNeck = nn.Sequential(
            nn.Linear(18*18*8,bottlerange*18*18*8),
            nn.Linear(bottlerange*18*18*8,18*18*8)
            )
    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0),-1)
        x = self.bottleNeck(x)
        x = x.view([-1,8,18,18])     
        x = self.decoder(x)
        return x


class ConvolutionalAutoEncoder(nn.Module):
    def __init__(self):
        super(ConvolutionalAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.MaxPool2d(2, stride=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 16, 2, stride=3, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 2, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=0),
            nn.Tanh(),
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x