import torch
import string
import torch.nn.functional as F

def decoder(indices, vocab):
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

def beam_search_caption(model, images, vocab, decoder, device="cpu",
                       start_token="<sos>", end_token="<eos>",
                       beam_width=3, max_seq_length=100):
    """
    Generates captions for images using beam search.

    Args:
        model (ImgCap): The image captioning model.
        images (torch.Tensor): Batch of images.
        vocab (Vocab): Vocabulary object.
        decoder (function): Function to decode indices to words.
        device (str): Device to perform computation on.
        start_token (str): Start-of-sequence token.
        end_token (str): End-of-sequence token.
        beam_width (int): Number of beams to keep.
        max_seq_length (int): Maximum length of the generated caption.

    Returns:
        list: Generated captions for each image in the batch.
    """
    model.eval()

    with torch.no_grad():
        start_index = vocab[start_token]
        end_index = vocab[end_token]
        images = images.to(device)
        batch_size = images.size(0)
        
        # Ensure batch_size is 1 for beam search (one image at a time)
        if batch_size != 1:
            raise ValueError("Beam search currently supports batch_size=1.")

        cnn_feature = model.cnn(images)  # Shape: (1, 1024)
        lstm_input = model.lstm.projection(cnn_feature).unsqueeze(1)  # Shape: (1, 1, 1024)
        state = None  # Initial LSTM state

        # Initialize the beam with the start token
        sequences = [([start_index], 0.0, lstm_input, state)]  # List of tuples: (sequence, score, input, state)

        completed_sequences = []

        for _ in range(max_seq_length):
            all_candidates = []

            # Iterate over all current sequences in the beam
            for seq, score, lstm_input, state in sequences:
                # If the last token is the end token, add the sequence to completed_sequences
                if seq[-1] == end_index:
                    completed_sequences.append((seq, score))
                    continue

                # Pass the current input and state through the LSTM
                lstm_out, state_new = model.lstm.lstm(lstm_input, state)  # lstm_out: (1, 1, 1024)

                # Pass the LSTM output through the fully connected layer to get logits
                output = model.lstm.fc(lstm_out.squeeze(1))  # Shape: (1, vocab_size)

                # Compute log probabilities
                log_probs = F.log_softmax(output, dim=1)  # Shape: (1, vocab_size)

                # Get the top beam_width tokens and their log probabilities
                top_log_probs, top_indices = log_probs.topk(beam_width, dim=1)  # Each of shape: (1, beam_width)

                # Iterate over the top tokens to create new candidate sequences
                for i in range(beam_width):
                    token = top_indices[0, i].item()
                    token_log_prob = top_log_probs[0, i].item()

                    # Create a new sequence by appending the current token
                    new_seq = seq + [token]
                    new_score = score + token_log_prob

                    # Get the embedding of the new token
                    token_tensor = torch.tensor([token], device=device)
                    new_lstm_input = model.lstm.embedding(token_tensor).unsqueeze(1)  # Shape: (1, 1, 1024)

                    # Clone the new state to ensure each beam has its own state
                    if state_new is not None:
                        new_state = (state_new[0].clone(), state_new[1].clone())
                    else:
                        new_state = None

                    # Add the new candidate to all_candidates
                    all_candidates.append((new_seq, new_score, new_lstm_input, new_state))

            # If no candidates are left to process, break out of the loop
            if not all_candidates:
                break

            # Sort all candidates by score in descending order
            ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)

            # Select the top beam_width sequences to form the new beam
            sequences = ordered[:beam_width]

            # If enough completed sequences are found, stop early
            if len(completed_sequences) >= beam_width:
                break

        # If no sequences have completed, use the current sequences
        if len(completed_sequences) == 0:
            completed_sequences = sequences

        # Select the sequence with the highest score
        best_seq, best_score = max(completed_sequences, key=lambda x: x[1])

        if best_seq[0] == start_index:
            best_seq = best_seq[1:]

        best_caption = decoder(best_seq, vocab)

    return best_caption


def generate_caption(model, images, vocab, decoder, device="cpu", start_token="<sos>", end_token="<eos>", max_seq_length=100, top_k=2):
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

            top_k_probs, top_k_indices = torch.topk(F.softmax(output, dim=1), top_k, dim=1)
            top_k_probs = top_k_probs / torch.sum(top_k_probs, dim=1, keepdim=True)  
            top_k_samples = torch.multinomial(top_k_probs, 1).squeeze()

            predicted_word_indices = top_k_indices[range(batch_size), top_k_samples]

            lstm_input = model.lstm.embedding(predicted_word_indices).unsqueeze(1)  # (B, 1, hidden_size)

            for j in range(batch_size):
                if end_token_appear[j]:
                    continue

                word = vocab.lookup_token(predicted_word_indices[j].item())
                if word == end_token:
                    end_token_appear[j] = True

                captions[j].append(predicted_word_indices[j].item())

        captions = [decoder(caption, vocab) for caption in captions]

    return captions