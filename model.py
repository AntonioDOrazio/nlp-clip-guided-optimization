import torch
from torch import nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.encoder = nn.Sequential (
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=2),
            nn.ReLU(True),
            nn.Flatten(),
        )

        linear_in = self.encoder(torch.randn(1, 3, 256,128)).reshape(-1).shape[0]
        self.dense = nn.Linear(linear_in, 4096)

    def forward(self, x):
        y = self.encoder(x)
        y = self.dense(y)
        return y


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(True),            
            nn.ConvTranspose2d(32, 3, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(3),
            nn.Sigmoid(),
        )

    def forward(self, x):
        output = x
        for module in self.decoder:
            output = module(output)
        return output


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # Encoder
        self.encoder = Encoder()

        # Latent space parameters
        dummy_input = self.encoder(torch.randn(1, 3, 256,128)).reshape(-1)
        print("dummy input shape ", dummy_input.shape)

        self.fc_mu = nn.Linear(dummy_input.shape[0], 1024)  # Adjust input size based on your encoder's output size
        self.fc_logvar = nn.Linear(dummy_input.shape[0], 1024)
        self.fc_decode = nn.Sequential(nn.Linear(1024, 128*32*16))
        self.decoder = nn.Sequential(
  
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(64),
        
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),

            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),

            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(3),

            nn.Sigmoid()
        )

    def encode_only(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return x
    
    def get_mu_logvar(self, x):
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = mu + eps*std
        return z

    def decode(self, z):
        z = self.fc_decode(z)
        z = z.reshape(z.shape[0], 128, 16, 32)
        z = self.decoder(z)
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar



