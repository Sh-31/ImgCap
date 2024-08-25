import time
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as T
from torch.utils.data import random_split, DataLoader
from model import ImgCap
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.append('../')
from data_utils import Flickr30, collate_fn
from eval_utils import eval_bleu_score, eval_CIDEr


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
                val_loss.append(loss.item())

                generated_captions = model.generate_caption(images, vocab, decoder, device)
                decoded_captions = eval_decode_batch(captions, decoder)

                bleu4_score = eval_bleu_score(candidates=generated_captions, references=decoded_captions)
                cider_score, _ = eval_CIDEr(candidates=generated_captions, references=decoded_captions)
                bleu_scores.append(bleu4_score)
                cider_scores.append(cider_score)

    avg_val_loss = np.mean(val_loss)
    avg_bleu = np.mean(bleu_scores)
    avg_cider = np.mean(cider_scores)

    writer.add_scalar('Validation/Loss', avg_val_loss, epoch)
    writer.add_scalar('Validation/BLEU', avg_bleu, epoch)
    writer.add_scalar('Validation/CIDEr', avg_cider, epoch)

    return avg_val_loss, avg_bleu, avg_cider, generated_captions, decoded_captions


def train_model(model, train_loader, val_loader, criterion, optimizer, scaler, num_epochs, device, vocab, decoder, checkpoint_path, log_dir):
    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss, norm = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, writer)
        epoch_time = time.time() - start_time

        if (epoch + 1) % 10 == 0:
            val_loss, avg_bleu, avg_cider, generated_captions, decoded_captions = validate_model(
                model, val_loader, criterion, vocab, decoder, device, writer, epoch
            )

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


if __name__ == "__main__":

    root_path = '/teamspace/studios/this_studio/ImgCap'
    Flickr30_image_path = f"{root_path}/data/Flickr30/imges"
    Flickr30_labels_path = f"{root_path}/data/Flickr30/results.csv"
    vocab_path = f"{root_path}/vocab.pkl"

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

    train_data_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, collate_fn=collate_fn)
    val_data_loader = DataLoader(val_dataset, batch_size=256, shuffle=True, collate_fn=collate_fn)

    decoder = dataset.decoder
    vocab = dataset.vocab

    criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = GradScaler()

    model = ImgCap(cnn_feature_size=1024, lstm_hidden_size=1024, embedding_dim=1024, num_layers=2, vocab_size=len(vocab))

    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    model.to(device)

    num_epochs = 250

    checkpoint_path = f"{root_path}/trainning/checkpoints"
    log_dir = f"{root_path}/trainning/logs"

    model, optimizer, train_loss, val_loss = train_model(
        model, train_data_loader, val_data_loader, criterion, optimizer, scaler, num_epochs, device, vocab, decoder,
        checkpoint_path, log_dir
    )
