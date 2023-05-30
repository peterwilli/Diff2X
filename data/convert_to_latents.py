import argparse
from diffusers import AutoencoderKL
from PIL import Image
from pathlib import Path
from torchvision import transforms

def normalize(x):
    x = (x - x.min()) / (x.max() - x.min())
    return x

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', type=str)
    args = parser.parse_args()

    vae = AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-2-1-base",
        subfolder="vae",
        device_map='auto'
    )

    path = Path(args.path + "/hr_128")
    tf = transforms.Compose([
        #transforms.RandomRotation(degrees=[-90, 90]),  # Randomly rotate the image
        #transforms.ColorJitter(hue=[-0.2, 0.2]),  # Randomly shift the hue
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    to_pil = transforms.ToPILImage()

    # Loop through each file in the folder
    for file_path in path.glob("*.png"):
        print("test")
        image = Image.open(file_path)
        image = tf(image)
        image_source = vae.encode(image.unsqueeze(0)).latent_dist.sample()
        image_source = vae.decode(image_source).sample.squeeze(0)
        image_source = to_pil(normalize(image_source))
        image_source.save(f"{args.path}/sr_32_128/{file_path.name}")
        image_source.resize((32, 32)).save(f"{args.path}/lr_32/{file_path.name}")