import requests
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed 
import os
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm  # Import tqdm for progress bar

# Load the dataset
ds = load_dataset("ChristophSchuhmann/MS_COCO_2017_URL_TEXT")

import requests
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed  # Use ThreadPoolExecutor for I/O-bound tasks
import os

def is_valid_image_url(url, session):
    try:
        response = session.get(url, timeout=5, stream=True) 
        response.raise_for_status()
        img = Image.open(BytesIO(response.raw.read(1024)))  # Read only the first 1KB of the image
        return True
    except (requests.exceptions.RequestException, IOError):
        return False

def filter_valid_urls(data, feature):
    valid_data = []

    # Use ThreadPoolExecutor for I/O-bound parallelism
    max_workers = os.cpu_count() * 2  # Experiment with more threads if network is the bottleneck

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        with requests.Session() as session:  # Use a session for connection pooling
            future_to_url = {executor.submit(is_valid_image_url, item[feature], session): item for item in data}
        
            for future in tqdm(as_completed(future_to_url), total=len(future_to_url), desc="Filtering URLs"):
                item = future_to_url[future]
                try:
                    if future.result():
                        valid_data.append(item)
                except Exception as e:
                    print(f"Error processing URL {item[feature]}: {e}")

    return valid_data

