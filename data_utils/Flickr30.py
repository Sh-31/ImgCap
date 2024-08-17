import cv2
import string
import spacy
import torch
import pickle
import pandas as pd
from torch.utils.data import Dataset
from torchtext.vocab import build_vocab_from_iterator


class Flickr30(Dataset):
    def __init__(self, imges_folder_path, labels_path, vocab=None, voacb_path='ImgCap/vocab.pkl' , transform=None):
        self.imges_folder_path = imges_folder_path
        self.labels = pd.read_csv(labels_path, delimiter='|')
        self.transform = transform

        # Initialize spaCy model for tokenization
        self.nlp = spacy.load('en_core_web_sm') # Can use split(' ')

        # Build vocabulary if not provided
        if vocab is None:
            self.vocab = self.build_vocab()
            self.save_vocab(self.vocab, voacb_path)
        else:
            self.vocab = vocab

    def build_vocab(self):
        # Tokenize all captions and build vocabulary
        self.labels.columns = self.labels.columns.str.strip()
        captions = self.labels['comment'].tolist()
        tokenized_captions = [self.tokenize_caption(caption) for caption in captions]

        # 997 (token) + 26 (English alphabet) + space = 1024 tokens
        vocab = build_vocab_from_iterator(tokenized_captions, specials=["<unk>", "<pad>", "<sos>", "<eos>"], max_tokens=997, min_freq=5)
        
        # Add individual English alphabet characters and space to the vocabulary
        for ch in string.ascii_lowercase + ' ':
            if ch not in vocab:
                vocab.append_token(ch)
        
        vocab.set_default_index(vocab["<unk>"])
        return vocab

    def save_vocab(self, vocab, vocab_path):
        with open(vocab_path, 'wb') as f:
            pickle.dump(vocab, f)
        print(f"Vocabulary saved at {vocab_path}")

    def tokenize_caption(self, caption):
        return [f" {token.text.lower()}" for token in self.nlp(caption)]

    def encoder(self, text):
        tokens = self.tokenize_caption(text)  
        indices = [self.vocab["<sos>"]]

        for token in tokens:
            if token in self.vocab:
                indices.append(self.vocab[token])
            else:
                # Split the word into characters if the word is not found in the vocabulary
                for ch in token:
                    indices.append(self.vocab[ch])
        
        indices.append(self.vocab["<eos>"])
        return torch.tensor(indices, dtype=torch.long)

    def decoder(self, indices):
        # Convert indices back to tokens, ignoring <sos> and <eos>
        tokens = [self.vocab.lookup_token(idx) for idx in indices]
    
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
        
        # Add the last collected word if there is one
        if current_word:
            words.append(" "+"".join(current_word))
        
        return "".join(words)


    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        image_id, comment_number, caption = self.labels.iloc[idx]
        image = cv2.imread(f'{self.imges_folder_path}/{image_id}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        # Tokenize and encode the caption
        tensor_caption = self.encoder(caption)

        return image, tensor_caption
