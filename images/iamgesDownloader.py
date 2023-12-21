import os
import requests

# List of animals
animals = ["hen", "duck", "cow", "sheep", "ox", "bull", "horse"]

# Number of images to download for each animal
num_images_per_animal = 5

# Directory to save downloaded images
download_path = 'images_unsplash'
os.makedirs(download_path, exist_ok=True)

# Function to download images using Unsplash API
def download_images_unsplash(query, num_images, download_path):
    for i in range(1, num_images + 1):
        url = f'https://source.unsplash.com/600x600/?{query}'
        response = requests.get(url)

        # Save the image
        image_path = os.path.join(download_path, f'{query}_{i}.jpg')
        with open(image_path, 'wb') as f:
            f.write(response.content)

        print(f'Downloaded {query} image {i}/{num_images}')

# Download images for each animal
for animal in animals:
    download_images_unsplash(animal, num_images_per_animal, download_path)
