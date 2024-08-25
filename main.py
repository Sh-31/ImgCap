import pickle
import cv2
import torchvision.transforms as T
from utils import load_checkpoint
from trainning import ImgCap
import torch
import string

with open("/teamspace/studios/this_studio/ImgCap/vocab.pkl", 'rb') as f:
        vocab = pickle.load(f)


def generate_caption(model, images, vocab, decoder, device="cpu", start_token="<sos>", end_token="<eos>", max_seq_length=100):
    model.eval()

    with torch.no_grad():
        start_index = vocab[start_token]
        end_index = vocab[end_token]
        images = images.to(device)
        batch_size = images.size(0)

        end_token_appear = {i: False for i in range(batch_size)}
        captions = [[] for _ in range(batch_size)]

        cnn_feature = model.cnn(images)
        lstm_input = model.lstm.projection(cnn_feature).unsqueeze(1)  # (B, 1, hidden_size)

        state = None

        for i in range(max_seq_length):
            lstm_out, state = model.lstm.lstm(lstm_input, state)
            output = model.lstm.fc(lstm_out.squeeze(1))
            predicted_word_indices = torch.argmax(output, dim=1)
            lstm_input = model.lstm.embedding(predicted_word_indices).unsqueeze(1)  # (B, 1, hidden_size)

            for j in range(batch_size):
                if end_token_appear[j]:
                    continue

                word = vocab.lookup_token(predicted_word_indices[j].item())
                if word == end_token:
                    end_token_appear[j] = True

                captions[j].append(predicted_word_indices[j].item())

        captions = [decoder(caption) for caption in captions]

    return captions

def decoder(indices):
    tokens = [vocab.lookup_token(idx) for idx in indices]
    words = []
    current_word = []
    for token in tokens:
        if len(token) == 1 and token in string.ascii_lowercase:
            current_word.append(token)
        else:
            if current_word:
                words.append("".join(current_word))
                current_word = []
            words.append(token)

    if current_word:
        words.append(" "+"".join(current_word))

    return "".join(words)


transforms = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])    

if __name__ == "__main__":
    checkpoint_path = "/teamspace/studios/this_studio/ImgCap/trainning/checkpoints/checkpoint_epoch_10.pth"
    
    model = ImgCap(cnn_feature_size=1024, lstm_hidden_size=1024, embedding_dim=1024, num_layers=2, vocab_size=len(vocab))
    model = torch.compile(model)

    model, optimizer, epoch, train_loss, val_loss, bleu_score, cider_score= load_checkpoint(checkpoint_path=checkpoint_path, model=model)

    img_path = "/teamspace/studios/this_studio/ImgCap/imgs/test.png"
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = transforms(image).unsqueeze(0)

    generated_captions = generate_caption(model, image, vocab, decoder)

    print(generated_captions)


