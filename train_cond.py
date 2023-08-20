import torch
from tqdm import tqdm
from torch import nn, optim
from model_cond import VAE
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Dataset
import os 
from PIL import Image
import random

fine_tune = True

if fine_tune:
    print("FINE TUNING")

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = 784
NUM_EPOCHS = 600 if not fine_tune else 150
BATCH_SIZE = 32 #64 #64 * 8
LR_RATE =  3e-5 #3e-4 * 8
print(BATCH_SIZE)
SAMPLE_DIR = "./dataset/polyhaven_png_128"
GROUND_TRUTH_DIR = "./dataset/polyhaven_png_128"

STEP_SIZE=150
GAMMA = 0.1


# Define the transformations to be applied to the images
transform = transforms.Compose([
        transforms.ToTensor(),
    ])

transform_label = transforms.Compose([
        transforms.ToTensor(),
    ])

transform_normalize = transforms.Compose([
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])


# Create a custom dataset that loads both sample and ground truth images
class PairedImageDataset(Dataset):
    def __init__(self, sample_dir, ground_truth_dir, x_transform=None, y_transform=None):
        self.sample_files = self.get_image_files(ground_truth_dir) # metti ground sample_dr
        self.ground_truth_files = self.get_image_files(ground_truth_dir)
        self.x_transform = x_transform
        self.y_transform = y_transform

    def __getitem__(self, index):
        sample_file = self.sample_files[index]
        ground_truth_file = self.ground_truth_files[index]

        sample_image = self.load_image(sample_file)
        ground_truth_image = self.load_image(ground_truth_file)

        if self.x_transform is not None:
            sample_image = self.x_transform(sample_image)
        if self.y_transform is not None:
            ground_truth_image = self.y_transform(ground_truth_image)

        return sample_image, ground_truth_image

    def __len__(self):
        return len(self.sample_files)

    def get_image_files(self, directory):
        image_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if self.is_image_file(file):
                    image_files.append(os.path.join(root, file))
        return image_files

    def is_image_file(self, filename):
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
        return any(filename.lower().endswith(ext) for ext in image_extensions)

    def load_image(self, file):
        with Image.open(file) as img:
            return img.convert("RGB")

# Create the dataset for loading both sample and ground truth images
paired_dataset = PairedImageDataset(SAMPLE_DIR, GROUND_TRUTH_DIR, x_transform=transform, y_transform=transform)

# Create the DataLoader for loading both sample and ground truth images
paired_loader = DataLoader(paired_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Dataset Loading

def initialize_weights(model):
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

model = VAE().to(DEVICE)
#vgg = VGGFeatures().to(DEVICE)

initialize_weights(model)

optimizer = optim.Adam(model.parameters(), lr=LR_RATE)
for name, parameter in model.named_parameters():
    if parameter.requires_grad:
        print(name)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)



if fine_tune:
    weights = torch.load("train_results/model_weights_final.pth")
    model.load_state_dict(weights)


save_dir = "train_results" if not fine_tune else "train_results_finetune"
os.makedirs(save_dir, exist_ok=True)
#clip_loss = ClipImageEmbeddingLoss()

for epoch in range(NUM_EPOCHS):
    loop = tqdm(enumerate(paired_loader))
    for i, (x, y_true) in loop:

        # Avoid the gradients exploding at the first stages, then switch back to a higher LR
        if epoch == 5:
            for g in optimizer.param_groups:
                g['lr'] = 3e-4

        x = x.to(DEVICE)
        y_reconstructed, y_mu, y_sigma = model(transform_normalize(x))

        if not fine_tune:

            alpha_bce = 1e6 
            alpha_kld = 1e-1
            alpha_content = 1e2
            alpha_style  = 1e4#3

            if epoch>=500:
                alpha_kld=1e1

        else:

            alpha_bce = 1e5 
            alpha_kld = 1e2


        bce_loss_value = alpha_bce*(torch.nn.functional.binary_cross_entropy_with_logits(y_reconstructed, x, reduction='mean'))
        kld_loss_value = alpha_kld * (-0.5 * torch.mean(1 + y_sigma - y_mu.pow(2) - y_sigma.exp()))

        loss = kld_loss_value+bce_loss_value#+content_loss_value+style_loss_value

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        loop.set_postfix(loss=loss.item(), bce=bce_loss_value.item(),  kld=kld_loss_value.item(), epoch=epoch, lr=optimizer.param_groups[0]['lr'])

    scheduler.step()

    with torch.no_grad():
        random_indices = random.sample(range(len(y_reconstructed)), k=min(15, len(y_reconstructed)))
        epoch_dir = os.path.join(save_dir, f"epoch_{epoch+1}")
        os.makedirs(epoch_dir, exist_ok=True)
        for idx in random_indices:
            image = y_reconstructed[idx].cpu()
            save_image(image, os.path.join(epoch_dir, f"image_{idx}.png"))
            image = x[idx].cpu()
            save_image(image, os.path.join(epoch_dir, f"image_{idx}_x.png"))

    if epoch % 50 == 0:

        weights_path = os.path.join(save_dir, f"model_weights_{epoch}.pth")
        torch.save(model.state_dict(), weights_path)
    if epoch+1 == NUM_EPOCHS:
            if not fine_tune:
                weights_path = os.path.join(save_dir, f"model_weights_final.pth")
            else:
                weights_path = os.path.join(save_dir, f"model_weights_final_finetune.pth")

            torch.save(model.state_dict(), weights_path)   

print("TRAINING COMPLETE!")
print(f"Model architecture and weights saved at {save_dir}")
