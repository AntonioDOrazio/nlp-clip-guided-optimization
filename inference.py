
import torch
from model_cond import VAE
from torchvision.utils import save_image
import torchvision
from PIL import Image

DEVICE = "cuda"

model = VAE().to(DEVICE)
model.load_state_dict(torch.load("train_results/model_weights_final.pth"))


to_tensor = torchvision.transforms.functional.to_tensor
x = to_tensor(Image.open("dataset/polyhaven_png_128/abandoned_tiled_room_1k.png")).unsqueeze(0).to(DEVICE)
print(x.shape)

def inference(num_examples=1):
    model.eval()
    with torch.no_grad():

            mu, sigma = model.encode(x)
            z = model.reparameterize(mu, sigma)
            out = model.decode(z)
            save_image(out[0], f"generated.png", nrow=1)

inference(num_examples=5)