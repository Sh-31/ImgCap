import torch
import torch.nn as nn
import torchvision.models as models

## ResNet50 (CNN Encoder)

class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.ResNet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        self.ResNet50.fc = nn.Sequential(
                            nn.Linear(2048, 1024),
                            nn.ReLU(),
                            nn.Dropout(0.5),
                            nn.Linear(1024, 1024),
                            nn.ReLU(),
                           )

        for k,v in self.ResNet50.named_parameters(recurse=True):
          if 'fc' in k:
            v.requires_grad = True
          else:
            v.requires_grad = False

    def forward(self,x):
        return self.ResNet50(x)       

## lSTM (Decoder)

class lstm(nn.Module):
    def __init__(self, input_size, hidden_size, number_layers, embedding_dim, vocab_size):
        super(lstm, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.number_layers = number_layers
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.projection = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=number_layers,
            dropout=0.5,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, captions):
        projected_image = self.projection(x).unsqueeze(dim=1)
        embeddings = self.embedding(captions[:, :-1])
    
        # Concatenate the image feature as frist step with word embeddings
        lstm_input = torch.cat((projected_image, embeddings), dim=1)
        # print(torch.all(projected_image[:, 0, :] == lstm_input[:, 0, :])) # check
        
        lstm_out, _ = self.lstm(lstm_input)
        logits = self.fc(lstm_out)

        return logits

## ImgCap

class ImgCap(nn.Module):
    def __init__(self, cnn_feature_size, lstm_hidden_size, num_layers, vocab_size, embedding_dim):
        super(ImgCap, self).__init__()

        self.cnn = ResNet50()

        self.lstm = lstm(input_size=cnn_feature_size,
                         hidden_size=lstm_hidden_size,
                         number_layers=num_layers,
                         embedding_dim=embedding_dim,
                         vocab_size=vocab_size)

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

            end_token_appear = {i: False for i in range(batch_size)}
            captions = [[] for _ in range(batch_size)]

            cnn_feature = self.cnn(images)
            lstm_input = self.lstm.projection(cnn_feature).unsqueeze(1)  # (B, 1, hidden_size)

            state = None

            for i in range(max_seq_length):
                lstm_out, state = self.lstm.lstm(lstm_input, state)
                output = self.lstm.fc(lstm_out.squeeze(1))
                predicted_word_indices = torch.argmax(output, dim=1)
                lstm_input = self.lstm.embedding(predicted_word_indices).unsqueeze(1)  # (B, 1, hidden_size)

                for j in range(batch_size):
                    if end_token_appear[j]:
                        continue

                    word = vocab.lookup_token(predicted_word_indices[j].item())
                    if word == end_token:
                        end_token_appear[j] = True

                    captions[j].append(predicted_word_indices[j].item())

            captions = [decoder(caption) for caption in captions]

        return captions


### Other Implementations FOR LSTM ###


  # def forward(self, x, captions):
  #     batch, seq_len = captions.shape
  #     h0 = torch.zeros(batch, self.hidden_size).to(x.device)
  #     c0 = torch.zeros(batch, self.hidden_size).to(x.device)

  #     logits = torch.zeros(batch, seq_len, self.vocab_size).to(x.device)

  #     h, c = self.lstm(x, (h0, c0))
  #     output = self.fc(h)
  #     logits[:, 0, :] = output

  #     for t in range(1, seq_len):
  #       word_embedding = self.embedding(captions[:, t-1])
  #       h, c = self.lstm(word_embedding, (h, c))
  #       output = self.fc(h)
  #       logits[:, t, :] = output

  #     return logits


    # def forward(self, x, captions):
    #     # Implementation without Teacher Forcing and using the word probability
    #     batch_size = x.shape[0]
    #     seq_len = captions.size(1)

    #     h0 = torch.zeros(batch_size, self.hidden_size).to(x.device)
    #     c0 = torch.zeros(batch_size, self.hidden_size).to(x.device)

    #     logits = torch.zeros(batch_size, seq_len, self.vocab_size).to(x.device)

    #     h, c = self.lstm(x, (h0, c0))
    #     logit = self.fc(h)
    #     probs = F.softmax(logit, dim=1)
    #     predicted_word_index = torch.argmax(probs, dim=1)

    #     logits[:, 0, :] = logit

    #     for t in range(1, seq_len):
    #         word_embedding = self.embedding(predicted_word_index)
    #         h, c = self.lstm(word_embedding, (h, c))
    #         h = self.dropout(h)
    #         logit = self.fc(h)

    #         probs = F.softmax(logit, dim=1)
    #         predicted_word_index = torch.argmax(probs, dim=1)
    #         logits[:, t, :] = logit

    #     return logits



# class Lstm(nn.Module):
#     def __init__(self, input_size, hidden_size, number_layers, embedding_dim, vocab_size):
#         super(Lstm, self).__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.number_layers = number_layers
#         self.embedding_dim = embedding_dim
#         self.vocab_size = vocab_size

#         self.embedding = nn.Embedding(vocab_size, embedding_dim)

#         # First LSTMCell
#         self.lstm1 = nn.LSTMCell(input_size=embedding_dim, hidden_size=hidden_size)

#         # Second LSTMCell
#         self.lstm2 = nn.LSTMCell(input_size=hidden_size, hidden_size=hidden_size)

#         self.fc = nn.Linear(hidden_size, vocab_size)
#         self.dropout = nn.Dropout(0.2)

#     def init_hidden_and_cell_states(self, batch_size, device):
#         h = torch.zeros(batch_size, self.hidden_size).to(device)
#         c = torch.zeros(batch_size, self.hidden_size).to(device)
#         return h, c

#     def forward(self, x, captions):
#         batch, seq_len = captions.shape

#         h1, c1 = self.init_hidden_and_cell_states(batch, x.device)
#         h2, c2 = self.init_hidden_and_cell_states(batch, x.device)

#         logits = torch.zeros(batch, seq_len, self.vocab_size).to(x.device)

#         x = self.dropout(x)

#         h1, c1 = self.lstm1(x, (h1, c1))
#         h2, c2 = self.lstm2(h1, (h2, c2))

#         output = self.fc(h2)
#         logits[:, 0, :] = output

#         for t in range(1, seq_len):
#             word_embedding = self.embedding(captions[:, t-1])
#             word_embedding = self.dropout(word_embedding)

#             # First LSTMCell
#             h1, c1 = self.lstm1(word_embedding, (h1, c1))
#             # Second LSTMCell
#             h2, c2 = self.lstm2(h1, (h2, c2))

#             output = self.fc(h2)
#             logits[:, t, :] = output

#         return logits
