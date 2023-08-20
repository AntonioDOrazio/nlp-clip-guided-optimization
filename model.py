import torch
import torch.nn.functional as F
from torch import nn
import torchvision
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
    
    def forward(self, x):
        # Flatten input tensor
        x_flat = x.view(-1, self.embedding_dim)
        
        # Compute distances between input and embeddings
        distances = torch.sum(x_flat**2, dim=1, keepdim=True) + \
                    torch.sum(self.embedding.weight**2, dim=1) - \
                    2 * torch.matmul(x_flat, self.embedding.weight.t())
        
        # Find nearest embeddings
        indices = torch.argmin(distances, dim=1)
        quantized = self.embedding(indices).view(x.size())
        
        return quantized, indices



'''
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        # TODO nn.SyncBatchNorm
        self.conv1 = nn.Sequential( 
            nn.Conv2d(padding=1, in_channels=3, out_channels=64, kernel_size=3, stride=1), 
            nn.ReLU())
        self.conv2_1 = nn.Sequential( 
            nn.Conv2d(padding=1, in_channels=64, out_channels=128, kernel_size=3, stride=2), 
            nn.ReLU())
        self.conv2_2 = nn.Sequential( 
            nn.Conv2d(padding=1, in_channels=128, out_channels=128, kernel_size=3, stride=1), 
            nn.ReLU())
        self.conv3_1 = nn.Sequential( 
            nn.Conv2d(padding=1, in_channels=128, out_channels=256, kernel_size=3, stride=2), 
            nn.ReLU())
        self.conv3_2 = nn.Sequential( 
            nn.Conv2d(padding=1, in_channels=256, out_channels=256, kernel_size=3, stride=1), 
            nn.ReLU())
        self.conv4_1 = nn.Sequential( 
            nn.Conv2d(padding=1, in_channels=256, out_channels=512, kernel_size=3, stride=2), 
            nn.ReLU())
        self.conv4_2 = nn.Sequential( 
            nn.Conv2d(padding=1, in_channels=512, out_channels=512, kernel_size=3, stride=1), 
            nn.ReLU())
        self.conv5_1 = nn.Sequential( 
            nn.Conv2d(padding=1, in_channels=512, out_channels=1024, kernel_size=3, stride=2), 
            nn.ReLU())
        self.conv5_2 = nn.Sequential( 
            nn.Conv2d(padding=1, in_channels=1024, out_channels=1024, kernel_size=3, stride=1), 
            nn.ReLU())


    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2_1(y)
        y = self.conv2_2(y)
        y = self.conv3_1(y)
        y_3_2 = self.conv3_2(y)
        y = self.conv4_1(y_3_2)
        y_4_2 = self.conv4_2(y)
        y = self.conv5_1(y_4_2)
        y_5_2 = self.conv5_2(y)

        return y_5_2





class VGGEncoder(nn.Module):
    def __init__(self):
        super(VGGEncoder, self).__init__()
        vgg19 = torchvision.models.vgg19(pretrained=True).to(DEVICE)
        
        vgg19.requires_grad_(False)
        
        # Extract only the features part of VGG19
        self.features = vgg19.features.to(DEVICE)
        self.features.requires_grad_(False)
        self.features.eval()
        # Mean and std values for normalization
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEVICE)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEVICE)

        # Layer indices for style loss
        self.layer_indices = [3, 8, 17, 26, 35]


    def forward(self, x):
        x=x.to(DEVICE)
        x = (x - self.mean) / self.std
        
        features = []
        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx in self.layer_indices:
                features.append(x)
    
        y_5_2 = features[4]

        return y_5_2

'''

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        #self.encoder = Encoder()


        # Latent space parameters
        dummy_input = self.encoder(torch.randn(3, 256,128)).reshape(-1)
        print("dummy input shape ", dummy_input.shape)
        self.fc_mu = nn.Linear(dummy_input.shape[0], 512)  # Adjust input size based on your encoder's output size
        self.fc_logvar = nn.Linear(dummy_input.shape[0], 512)

        # Decoder

        self.fc_decode = nn.Sequential(nn.Linear(512, 128*32*16), nn.ReLU())

        self.decoder = nn.Sequential(
  
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=0),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1),
            #nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, kernel_size=2, stride=1, padding=1),
            nn.Sigmoid()
        )

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


