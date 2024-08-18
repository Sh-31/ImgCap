import pandas as pd

def sample_data(data_path, data_sample_path, delimiter='|', frac=0.1, random_state=31):
    df = pd.read_csv(data_path,  delimiter=delimiter)
    sampled_df = df.sample(frac=frac, random_state=random_state)  
    sampled_df = sampled_df.reset_index(drop=True)
    sampled_df.to_csv(data_sample_path)