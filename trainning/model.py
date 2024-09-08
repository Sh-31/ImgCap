import torch
import torch.nn as nn
import torchvision.models as models

## ResNet50 (CNN Encoder)
class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.ResNet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.features = nn.Sequential(*list(self.ResNet50.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))  
        
        for param in self.ResNet50.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.features(x)  
        x = self.avgpool(x)  
        B, C, H, W = x.size()
        x = x.view(B, C, -1)    # Flatten spatial dimensions: (B, 2048, 49)
        x = x.permute(0, 2, 1)  # (B, 49, 2048) - 49 spatial locations
        return x

class Attention(nn.Module):
    def __init__(self, feature_size, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Linear(feature_size + hidden_size, hidden_size)
        self.attn_weights = nn.Linear(hidden_size, 1)
    
    def forward(self, features, hidden_state): # features: (B, 49, 2048), hidden_state: (B, hidden_size)
        hidden_state = hidden_state.unsqueeze(1).repeat(1, features.size(1), 1)  # (B, 49, hidden_size)
        combined = torch.cat((features, hidden_state), dim=2)  # (B, 49, feature_size + hidden_size)
        attn_hidden = torch.tanh(self.attention(combined))  # (B, 49, hidden_size)
        attention_logits = self.attn_weights(attn_hidden).squeeze(2)  # (B, 49)  
        attention_weights = torch.softmax(attention_logits, dim=1)  # (B, 49)
        context = (features * attention_weights.unsqueeze(2)).sum(dim=1)  # (B, 2048)
        return context, attention_weights

# Attention without learnable paramters:
# logits = torch.matmul(features, hidden_state.unsqueeze(2))  # (B, 49, 1) - Batch Matriax
# attention_weights = torch.softmax(logits, dim=1).squeeze(2)  # (B, 49)
# context = (features * attention_weights.unsqueeze(2)).sum(dim=1)  # (B, 2048)

class lstm(nn.Module):
    def __init__(self, feature_size, hidden_size, number_layers, embedding_dim, vocab_size):
        super(lstm, self).__init__()

        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.attention = Attention(feature_size, hidden_size)
        
        self.lstm = nn.LSTM(
            input_size=hidden_size + feature_size,  # input: concatenated context and word embedding
            hidden_size=hidden_size,
            num_layers=number_layers,
            dropout=0.5,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions=None, max_seq_len=None, teacher_forcing_ratio=0.90):
    
        batch_size = features.size(0)
        max_seq_len = max_seq_len if max_seq_len is not None else captions.size(1)
        h, c = self.init_hidden_state(batch_size)
        
        outputs = torch.zeros(batch_size, max_seq_len, self.fc.out_features).to(features.device)
        word_input = torch.tensor(2, dtype=torch.long).expand(batch_size).to(features.device) # vocab["<sos>"] ---> 2

        for t in range(1, max_seq_len):
            embeddings = self.embedding(word_input) 
            context, _ = self.attention(features, h[-1])  
            lstm_input_step = torch.cat([embeddings, context], dim=1).unsqueeze(1)  # Combine context + word embedding

            out, (h, c) = self.lstm(lstm_input_step, (h, c))  
            output = self.fc(out.squeeze(1))
            outputs[:, t, :] = output

            top1 = output.argmax(1)

            if captions is not None and torch.rand(1).item() < teacher_forcing_ratio:
                word_input = captions[:, t]  
            else:
                word_input = top1  

        return outputs

    def init_hidden_state(self, batch_size):
        device = next(self.parameters()).device
        h0 = torch.zeros(self.lstm.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.lstm.num_layers, batch_size, self.hidden_size).to(device)
        return (h0, c0)


class ImgCap(nn.Module):
    def __init__(self, feature_size, lstm_hidden_size, num_layers, vocab_size, embedding_dim):
        super(ImgCap, self).__init__()
        self.cnn = ResNet50()
        self.lstm = lstm(feature_size, lstm_hidden_size, num_layers, embedding_dim, vocab_size)

    def forward(self, images, captions):
        cnn_features = self.cnn(images)
        output = self.lstm(cnn_features, captions) 
        return output

    def generate_caption(self, images, vocab, decoder, device="cpu", start_token="<sos>", end_token="<eos>", max_seq_length=100):
        self.eval()

        with torch.no_grad():
            start_index = vocab[start_token]
            end_index = vocab[end_token]
            images = images.to(device)
            batch_size = images.size(0)

            captions = [[start_index,] for _ in range(batch_size)]
            end_token_appear = [False] * batch_size

            cnn_features = self.cnn(images)  # (B, 49, 2048)

            h, c = self.lstm.init_hidden_state(batch_size)
       
            word_input = torch.full((batch_size,), start_index, dtype=torch.long).to(device)

            for t in range(max_seq_length):
               
                embeddings = self.lstm.embedding(word_input) 
                context, _ = self.lstm.attention(cnn_features, h[-1])  # Attention context
                lstm_input_step = torch.cat([embeddings, context], dim=1).unsqueeze(1)  # Combine context + word embedding

                out, (h, c) = self.lstm.lstm(lstm_input_step, (h, c))  
              
                output = self.lstm.fc(out.squeeze(1))  # (B, vocab_size)
                
                # Get the predicted word (greedy search)
                predicted_word_indices = torch.argmax(output, dim=1)  # (B,)
                word_input = predicted_word_indices
                
              
                for i in range(batch_size):
                    if not end_token_appear[i]:
                        predicted_word = vocab.lookup_token(predicted_word_indices[i].item())
                        if predicted_word == end_token:
                            captions[i].append(predicted_word_indices[i].item())
                            end_token_appear[i] = True
                        else:
                             captions[i].append(predicted_word_indices[i].item())
           
                   
                    if all(end_token_appear):  # Stop if all captions have reached the <eos> token
                        break

            captions = [decoder(caption) for caption in captions]

        return captions

    def beam_search_caption(self, images, vocab, decoder, device="cpu",
                       start_token="<sos>", end_token="<eos>",
                       beam_width=3, max_seq_length=100):
        self.eval()

        with torch.no_grad():
            start_index = vocab[start_token]
            end_index = vocab[end_token]
            images = images.to(device)
            batch_size = images.size(0)

            # Ensure batch_size is 1 for beam search (one image at a time)
            if batch_size != 1:
                raise ValueError("Beam search currently supports batch_size=1.")

            cnn_features = self.cnn(images)  # (B, 49, 2048)
            h, c = self.lstm.init_hidden_state(batch_size)
            word_input = torch.full((batch_size,), start_index, dtype=torch.long).to(device)

            embeddings = self.lstm.embedding(word_input) 
            context, _ = self.lstm.attention(cnn_features, h[-1])
            lstm_input = torch.cat([embeddings, context], dim=1).unsqueeze(1)  


            sequences = [([start_index], 0.0, lstm_input, (h, c))]  # List of tuples: (sequence, score, input, state)

            completed_sequences = []

            for _ in range(max_seq_length):
                all_candidates = []

                for seq, score, lstm_input, (h,c) in sequences:
                    if seq[-1] == end_index:
                        completed_sequences.append((seq, score))
                        continue

                    lstm_out, (h_new, c_new) = model.lstm.lstm(lstm_input, (h, c))  # lstm_out: (1, 1, 1024)

                    output = model.lstm.fc(lstm_out.squeeze(1))  # Shape: (1, vocab_size)

                    log_probs = F.log_softmax(output, dim=1)  # Shape: (1, vocab_size)

                    top_log_probs, top_indices = log_probs.topk(beam_width, dim=1)  # Each of shape: (1, beam_width)

                    for i in range(beam_width):
                        token = top_indices[0, i].item()
                        token_log_prob = top_log_probs[0, i].item()

                        new_seq = seq + [token]
                        new_score = score + token_log_prob

                        token_tensor = torch.tensor([token], device=device)
                        embeddings = self.lstm.embedding(token_tensor) 
                        context, _ = self.lstm.attention(cnn_features, h_new[-1])
                        new_lstm_input = torch.cat([embeddings, context], dim=1).unsqueeze(1)  

                        if h_new is not None and c_new is not None:
                            h_new, c_new = (h_new.clone(), c_new.clone())
                        else:
                            h_new, c_new = None, None

                        all_candidates.append((new_seq, new_score, new_lstm_input, (h_new, c_new) ))

                if not all_candidates:
                    break

                ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)

                sequences = ordered[:beam_width]

                if len(completed_sequences) >= beam_width:
                    break

            if len(completed_sequences) == 0:
                completed_sequences = sequences
            
            best_seq = max(completed_sequences, key=lambda x: x[1])
            best_caption = decoder(best_seq[0], vocab)

        return best_caption
