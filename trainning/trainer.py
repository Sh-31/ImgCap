import os
import time
import pickle
import numpy as np
import torch
import random
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as T
from torch.utils.data import random_split, DataLoader
from model import ImgCap
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.append('../')
from data_utils import Flickr30, collate_fn
from utils import load_checkpoint
from eval_utils import *


def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, writer):
    model.train()
    loss_running = []
    norm_avg = []

    for batch_idx, (images, captions) in enumerate(train_loader):
        images, captions = images.to(device), captions.to(device)
        optimizer.zero_grad()

        with autocast(dtype=torch.float16):
            outputs = model(images, captions)
            try:
                loss = criterion(outputs.view(-1, outputs.size(2)), captions.contiguous().view(-1))  # outputs -> (B*S, vocab_size) , captions -> (B * S)
            except Exception as e:
                print(f"Error during training: {e}")
                print(f"Batch size: {captions.shape[0]}, Sequence length: {captions.shape[1]}")
                print(f"Outputs shape: {outputs.shape}, Captions shape: {captions.shape}")
                exit()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        norm_avg.append(norm.item())
        loss_running.append(loss.item())

        writer.add_scalar('Train/Loss', loss.item(), epoch * len(train_loader) + batch_idx)
        writer.add_scalar('Train/Gradient Norm', norm.item(), epoch * len(train_loader) + batch_idx)

    return np.mean(loss_running), np.mean(norm_avg)


def validate_model(model, val_loader, criterion, vocab, decoder, device, writer, epoch):
    model.eval()
    val_loss = []
    bleu_scores = []
    cider_scores = []

    with torch.no_grad():
        for images, captions in val_loader:
            images, captions = images.to(device), captions.to(device)

            with autocast(dtype=torch.float16):
                outputs = model(images, captions)
                loss = criterion(outputs.view(-1, outputs.size(2)), captions.contiguous().view(-1))
            
                generated_captions = model.generate_caption(images, vocab, decoder, device)
                decoded_captions = eval_decode_batch(captions, decoder, vocab)

                bleu4_score = eval_bleu_score(candidates=generated_captions, references=decoded_captions)
                cider_score, _ = eval_CIDEr(candidates=generated_captions, references=decoded_captions)
                
                val_loss.append(loss.item())
                bleu_scores.append(bleu4_score)
                cider_scores.append(cider_score)

    avg_val_loss = np.mean(val_loss)
    avg_bleu = np.mean(bleu_scores)
    avg_cider = np.mean(cider_scores)

    writer.add_scalar('Validation/Loss', avg_val_loss, epoch)
    writer.add_scalar('Validation/BLEU', avg_bleu, epoch)
    writer.add_scalar('Validation/CIDEr', avg_cider, epoch)

    return avg_val_loss, avg_bleu, avg_cider, generated_captions, decoded_captions


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, scaler, num_epochs, start_epoch, device, vocab, decoder, checkpoint_path, log_dir):
    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(start_epoch, start_epoch + num_epochs):
        start_time = time.time()

        train_loss, norm = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, writer)
        epoch_time = time.time() - start_time

        if (epoch + 1) % 5 == 0:
            val_loss, avg_bleu, avg_cider, generated_captions, decoded_captions = validate_model(
                model, val_loader, criterion, vocab, decoder, device, writer, epoch
            )

            epoch_time = time.time() - start_time  

            scheduler.step(val_loss)  # Scheduler step based on validation loss
            current_lr = scheduler.get_last_lr()[0]
            writer.add_scalar('Train/Learning_Rate', current_lr, epoch)
    
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'bleu_score': avg_bleu,
                'cider_score': avg_cider,
            }

            torch.save(checkpoint, f'{checkpoint_path}/checkpoint_epoch_{epoch + 1}.pth')

            print(f"Epoch [{epoch + 1}/{num_epochs}], "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"BLEU Score: {avg_bleu:.4f}, "
                  f"CIDEr Score: {avg_cider:.4f}, "
                  f"Norm: {norm:.2f}, "
                  f"Epoch Time: {epoch_time:.2f} sec")
                  
            print("-" * 120)
            print(f"Generated Caption Example: {generated_captions[0]}")
            print(f"Ground Truth Caption Example: {decoded_captions[0]}")
            print("-" * 120)

        else:
            print(f"Epoch [{epoch + 1}/{num_epochs}], "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Norm: {norm:.2f}, "
                  f"Epoch Time: {epoch_time:.2f} sec")

            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
            }
            
            torch.save(checkpoint, f'{checkpoint_path}/checkpoint_epoch_{epoch + 1}.pth')
      

    writer.close()
    return model, optimizer, train_loss, val_loss

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    ## for 
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

    root_path = '/teamspace/studios/this_studio/ImgCap'
    Flickr30_image_path = f"{root_path}/data/Flickr30/imges"
    Flickr30_labels_path = f"{root_path}/data/Flickr30/results.csv"
    vocab_path = f"{root_path}/vocab.pkl"

    seed = 31  
    set_seed(seed)

    transforms = T.Compose([
    T.ToPILImage(),    
    T.RandomApply([
        T.RandomHorizontalFlip(),
        T.RandomRotation(degrees=15),
        T.RandomCrop(size=(110, 110)),  
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    ], p=0.8),  
    T.Resize(size=(224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    dataset = Flickr30(Flickr30_image_path, Flickr30_labels_path, vocab=vocab, transform=transforms)

    train_size = int(0.90 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_data_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)
    val_data_loader = DataLoader(val_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    decoder = dataset.decoder
    vocab = dataset.vocab
    
    # model = ImgCap(cnn_feature_size=1024, lstm_hidden_size=1024, embedding_dim=1024, num_layers=2, vocab_size=len(vocab)) # Non attantion version
    
    model = ImgCap(feature_size=2048, lstm_hidden_size=1024, embedding_dim=1024, num_layers=2, vocab_size=len(vocab)).to(device) # attantion version
    
    criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])
    optimizer = optim.AdamW(model.parameters(), lr=4e-4,  weight_decay=1e-4)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    scaler = GradScaler()

    #### Load checkpoint ####
    checkpoint_path = f"{root_path}/trainning/checkpoints/attention/checkpoint_epoch_44.pth"
    model, optimizer, epoch, train_loss, val_loss, bleu_score, cider_score = load_checkpoint(checkpoint_path=checkpoint_path, model=model, optimizer=optimizer, device=device)
    print(f"Load Model Checkpoint Epoch {epoch}")
    start_epoch = epoch

    optimizer = optim.SGD(model.parameters(), lr=1e-4,  weight_decay=1e-4)

    model = torch.compile(model)
    model.to(device)

    num_epochs = 250
    checkpoint_dir = f"{root_path}/trainning/checkpoints/attention"
    log_dir = f"{root_path}/trainning/logs/attention"

    model, optimizer, train_loss, val_loss = train_model(
        model, train_data_loader, val_data_loader, criterion, optimizer, scheduler, scaler, num_epochs, start_epoch, device, vocab, decoder,
        checkpoint_dir, log_dir
    )
    # tensorboard --logdir '/trainning/logs/attention'
