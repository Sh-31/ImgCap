import torch.nn as nn
import torchvision.models as models
## ResNet50 (CNN Encoder)

class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.ResNet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        self.ResNet50.fc = nn.Sequential(
                            nn.Linear(2048, 512),
                            nn.ReLU(),
                            nn.Linear(512, 1024),
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
            dropout=0.2,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, captions):
        projected_image = self.projection(x).unsqueeze(dim=1)
        embeddings = self.embedding(captions[:, :-1])
        

        # Concatenate the image feature as frist step with word embeddings
        lstm_input = torch.cat((projected_image, embeddings), dim=1)
        # print(torch.all(projected_image[:, 0, :] == lstm_input[:, 0, :]))
    
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


