import os
import pickle
import torch
import random
import numpy as np
import concurrent.futures
from functools import partial
from torch.utils.data import random_split, DataLoader
import torchvision.transforms as T
import torch.multiprocessing as mp

import sys
sys.path.append('../')
from data_utils import Flickr30, collate_fn
from trainning import ImgCap, generate_caption, decoder, beam_search_caption
from utils import load_checkpoint
from eval_utils import *

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def generate_captions_for_batch(model, images, vocab, decoder, beam_width, device="cuda"):
    model.to(device)
    model.eval()
    all_generated_captions = []
    for i in range(images.size(0)):
        single_image = images[i].unsqueeze(0)
        generated_caption = beam_search_caption(model, single_image, vocab, decoder, device=device, beam_width=beam_width)
        all_generated_captions.append(generated_caption)
    return all_generated_captions

def eval_data_loader(model, data_loader, vocab, beam_width=3, device="cuda"):
    model.to(device)
    model.eval()

    all_candidates = []
    all_references = []

    num_workers = os.cpu_count()
    
    with torch.no_grad():
        partial_func = partial(generate_captions_for_batch, model, vocab=vocab, decoder=decoder, beam_width=beam_width, device=device)

        for images, captions in data_loader:
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                future = executor.submit(partial_func, images)
                generated_captions = future.result()

            reference_captions = eval_decode_batch(captions, decoder, vocab)
            all_candidates.extend(generated_captions)
            all_references.extend(reference_captions)

    avg_bleu1 = eval_bleu_score(all_candidates, all_references, n_gram=1)
    avg_bleu2 = eval_bleu_score(all_candidates, all_references, n_gram=2)
    avg_bleu3 = eval_bleu_score(all_candidates, all_references, n_gram=3)
    avg_bleu4 = eval_bleu_score(all_candidates, all_references, n_gram=4)
    avg_cider, _ = eval_CIDEr(all_candidates, all_references)

    return avg_bleu1, avg_bleu2, avg_bleu3, avg_bleu4, avg_cider

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)  # Set the multiprocessing start method to 'spawn'

    root_path = '/teamspace/studios/this_studio/ImgCap'
    Flickr30_image_path = f"{root_path}/data/Flickr30/imges"
    Flickr30_labels_path = f"{root_path}/data/Flickr30/results.csv"
    vocab_path = f"{root_path}/vocab.pkl"

    seed = 31  
    set_seed(seed)

    transforms = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    dataset = Flickr30(Flickr30_image_path, Flickr30_labels_path, vocab=vocab, transform=transforms)

    train_size = int(0.90 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # train_data_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True, collate_fn=collate_fn)
    val_data_loader = DataLoader(val_dataset, batch_size=2048, shuffle=True, collate_fn=collate_fn)

    model = ImgCap(cnn_feature_size=1024, lstm_hidden_size=1024, embedding_dim=1024, num_layers=2, vocab_size=len(vocab))

    checkpoint_path = f"{root_path}/trainning/checkpoints/checkpoint_epoch_40.pth"

    model, optimizer, epoch, train_loss, val_loss, bleu_score, cider_score = load_checkpoint(checkpoint_path=checkpoint_path, model=model)
    print(f"Loaded Model Checkpoint Epoch {epoch}")

    beam_width = 5

    avg_bleu1, avg_bleu2, avg_bleu3, avg_bleu4, avg_cider = eval_data_loader(model, val_data_loader, vocab, beam_width=beam_width, device="cuda")

    print("\n" + "="*60)
    print(f"Evaluation Metrics at Epoch {epoch}")
    print("="*60)
    print(f"BLEU-1 Score: {avg_bleu1:.4f}")
    print(f"BLEU-2 Score: {avg_bleu2:.4f}")
    print(f"BLEU-3 Score: {avg_bleu3:.4f}")
    print(f"BLEU-4 Score: {avg_bleu4:.4f}")
    print(f"CIDEr Score: {avg_cider:.4f}")
    print("="*60)

    ####################################################
    ### Evaluation Metrics at Epoch 40 #################
    ### BLEU-1 Score: 0.37 #############################
    ### BLEU-2 Score: 0.22 #############################
    ### BLEU-3 Score: 0.14 #############################
    ### BLEU-4 Score: 0.09 #############################
    ### CIDEr  Score: 0.41 #############################

