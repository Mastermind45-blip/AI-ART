import os
from torch import autocast
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

save_path = os.path.join(os.environ['HOME'], 'Documents', 'Hasil')

if not os.path.exists(save_path):
    os.mkdir(save_path)

def uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + '(' + str(counter) + ')' + extension
        counter += 1

    return path

prompt = "minecraft"

print(f"Characters in prompt {len(prompt)}, limit 200")

# Mengurangi batch size
batch_size = 1  # Ganti dengan nilai yang lebih kecil

pipline = pipeline.to('cuda')
pipline.model.config.max_batch_size = batch_size

with autocast('cuda'):
    image = pipeline(prompt).image[0]

image_path = uniquify(os.path.join(save_path, (prompt[:25] + '...') if len(prompt) > 25 else prompt) + '.png')

print(image_path)

image.save(image_path)
