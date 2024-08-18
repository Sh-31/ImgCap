import os
import shutil
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from datasets.features import Image  

def sample_data(data_path, data_sample_path, delimiter='|', frac=0.1, random_state=31):
    df = pd.read_csv(data_path,  delimiter=delimiter)
    sampled_df = df.sample(frac=frac, random_state=random_state)  
    sampled_df = sampled_df.reset_index(drop=True)
    sampled_df.to_csv(data_sample_path)


def prepare_data_for_huggingface(df, image_dir):
    dataset_dict = {
        'image': [os.path.join(image_dir, img_name) for img_name in df['image_name']],
        'comment_number': df[' comment_number'].tolist(),
        'comment': df[' comment'].tolist()
    }
    dataset = Dataset.from_dict(dataset_dict)
    dataset = dataset.cast_column('image', Image())
    return dataset


def upload_data_to_huggingface(image_dir, sampled_df):
    dataset = prepare_data_for_huggingface(sampled_df, image_dir)
    dataset_dict = DatasetDict({'train': dataset})
    dataset.push_to_hub('shredder-31/Flickr30_sample') 

    return dataset_dict

if _name__ == "__main__":
    
    sampled_df = pd.read_csv('ImgCap/data/sample_data/Flickr30_Sample.csv')

    image_dir = 'sampled_images'
    os.makedirs(image_dir, exist_ok=True)
    for img_name in sampled_df['image_name']:
        src_path = os.path.join('ImgCap/data/Flickr30/imges', img_name)  # Replace 'path_to_images' with your image folder path
        dst_path = os.path.join(image_dir, img_name)
        shutil.copy(src_path, dst_path)

    dataset_dict = upload_data_to_huggingface(image_dir, sampled_df)

    print("Dataset uploaded to Hugging Face.")




