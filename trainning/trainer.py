###Tranning loop###

import time
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import torch.nn as nn
import numpy as np

def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device):
    model.train()
    loss_running = []
    norm_avg = []

    for images, captions in train_loader:
        images, captions = images.to(device), captions.to(device)
        optimizer.zero_grad()

        with autocast(dtype=torch.float16):
            outputs = model(images, captions)
            try:
                loss = criterion(outputs.view(-1, outputs.size(2)), captions.contiguous().view(-1)) # outputs -> (B*S, vocab_size) , captions -> (B * S)
            except Exception as e:
                print(f"Error during training: {e}")
                print(f"Batch size: {captions.shape[0]}, Sequence length: {captions.shape[1]}")
                print(f"Outputs shape: {outputs.shape},  Captions shape: {captions.shape}")
                exit()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        norm_avg.append(norm.item())
        loss_running.append(loss.item())

    return np.mean(loss_running), np.mean(norm_avg)


def validate_model(model, val_loader, criterion, vocab, decoder, device):
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

    return np.mean(val_loss), np.mean(bleu_scores), np.mean(cider_scores), generated_captions, decoded_captions


def train_model(model, train_loader, val_loader, criterion, optimizer, scaler, num_epochs, device, vocab, decoder):
    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss, norm = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        epoch_time = time.time() - start_time

        if (epoch + 1) % 5 == 0:
            val_loss, avg_bleu, avg_cider, generated_captions, decoded_captions = validate_model(
                model, val_loader, criterion, vocab, decoder, device
            )

            epoch_time = time.time() - start_time


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

    return model, optimizer, train_loss, val_loss


# Initialize training components
decoder = dataset.decoder
vocab = dataset.vocab

criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = GradScaler()

model = ImgCap(cnn_feature_size=1024, lstm_hidden_size=512, embedding_dim=512, num_layers=1, vocab_size=1024)
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
model.to(device)

num_epochs = 35

model, optimizer, train_loss, val_loss = train_model(
    model, train_data_loader, val_data_loader, criterion, optimizer, scaler, num_epochs, device, vocab, decoder
)
