import os
import requests
from ddgs import DDGS
from PIL import Image
from io import BytesIO

def download_images(query, save_folder, max_images=5):
    os.makedirs(save_folder, exist_ok=True)
    count = 0
    with DDGS() as ddgs:
        for result in ddgs.images(query, max_results=max_images):
            try:
                url = result['image']
                response = requests.get(url, timeout=10)
                img = Image.open(BytesIO(response.content)).convert('RGB')
                img = img.resize((128, 128))
                filename = f"{query.replace(' ', '_')}_{count+1}.jpg"
                img.save(os.path.join(save_folder, filename))
                print(f"✅ Saved: {filename}")
                count += 1
                if count >= max_images:
                    break
            except Exception as e:
                print(f"❌ Failed image {count+1}: {e}")

# Define image categories and queries
categories = {
    'recyclable': ['plastic bottle', 'cardboard box', 'glass jar', 'metal can'],
    'non_recyclable': ['banana peel', 'styrofoam plate', 'used tissue', 'diaper']
}

# Run download for each category
for label, queries in categories.items():
    for query in queries:
        download_images(query, f"dataset/{label}", max_images=5)
