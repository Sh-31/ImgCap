<p align="center">
  <img src="https://github.com/user-attachments/assets/3af1aebf-241b-4b79-9634-c26e71f47b04" alt="Background Image" width="40%">
</p>

<h1 align="center">ImgCap</h1>

<p align="center">ImgCap is an image captioning model designed to automatically generate descriptive captions for images. It has two versions CNN + LSTM model and CNN + LSTM + Attention mechanism model.</p>

-----
## Usage
-----
1. **Clone the repository:**
   ```bash
   git clone https://github.com/Sh-31/ImgCap.git
   ```

2. **Install the required dependencies:**
   ```bash
   pip3 install -r requirements.txt
   python3 -q -m spacy download en_core_web_sm
   ```

3. **Download the model checkpoint (manual step):**
   - **ImgCap (CNN + LSTM)**: [Download checkpoint](https://huggingface.co/spaces/shredder-31/ImgCap/blob/main/checkpoint_epoch_40.pth)
   - **ImgCap (CNN + LSTM + Attention)**: (Add link here)
   - Place the model checkpoint in the appropriate directory:  
     - For CNN + LSTM + Attention: `ImgCap/trainning/checkpoints/attention`  
     - For CNN + LSTM: `ImgCap/trainning/checkpoints`

4. **Run the main script (Gradio GUI for inference):**
   ```bash
   python3 main.py
   ```

Alternatively, you can use the model directly on Hugging Face Spaces: [ImgCap on Hugging Face](https://huggingface.co/spaces/shredder-31/ImgCap)

-----
## Sample Output
---
![image](https://github.com/user-attachments/assets/95d9ad9a-8050-48ce-81f1-4da498fa7a65)

---
## Dataset
---
#### Flickr30k:  
The Flickr30k dataset consists of 30,000 images, each accompanied by five captions. It provides a wide variety of scenes and objects, making it ideal for diverse image captioning tasks.

To download the dataset, follow these steps:
1. Enable Kaggle’s public API by following the instructions here: [Kaggle API](https://www.kaggle.com/docs/api).
2. Run the following command to download the dataset:
   ```bash
   kaggle datasets download -d hsankesara/flickr-image-dataset -p /teamspace/studios/this_studio/data/Flickr30
   ```

Additionally, I’ve documented a similar image captioning dataset, which you can review here: [Image Caption Documentation](https://docs.google.com/document/d/1eYDuhT2VLufy9Uk5FZBUsgkt6_HCcqjquekvh2XyJzk/edit?usp=sharing).

---
## Model Architecture Comparison and Details
---
The model architectures compared in this report consist of two versions of the **ImgCap** model, each with different configurations. The models were trained using **Float16** precision and optimized with **torch.compile** for improved training efficiency on an **L4 24GB RAM GPU**. 

### Key Differences

1. **Number of Parameters:**
   - **ImgCap with Attention**: This model incorporates an additional attention mechanism that increases the parameter count. Specifically, the attention layer adds about **3.15M** parameters, bringing the total to **85.79M**. Out of these, **36.72M** are trainable, with the rest being frozen in the ResNet50 encoder.
   - **ImgCap without Attention**: The model without the attention mechanism has **52.89M** total parameters, with **29.38M** being trainable, as it simplifies the decoder by removing the attention layers.

2. **CNN Encoder (ResNet50) Freezing Strategy:**
   - Both models use **ResNet50** as the CNN encoder. The convolutional layers in ResNet50 are **frozen** to reduce computational overhead and focus training on the LSTM-based decoder. Only the fully connected layers at the end of ResNet50 are trainable in both models.

3. **LSTM Decoder and Embedding:**
   - Both models use an LSTM-based decoder with trainable embedding layers. The LSTM decoder with attention concatenates the context vectors obtained from the attention mechanism, while the non-attention model directly processes image features via projection layers.
   - The embedding dimension, hidden size, and number of layers in the LSTM remain consistent across both models.
   - Both models use same Vocab size 4096
4. **Vocabulary Construction**:
   - **Captions Tokenization**: Captions are tokenized using spaCy, which splits captions into tokens. These tokens are then used to build the vocabulary (vocab size 4096).
   - **Vocabulary Content**: The vocabulary includes special tokens (`<unk>`, `<pad>`, `<sos>`, `<eos>`) and tokens derived from the captions. Individual English alphabet characters and spaces are also added to the vocabulary to handle out-of-vocabulary words or character-level tokenization.
   - **Tokenization**: Each caption is tokenized into tokens that are then mapped to indices in the vocabulary.
   - **Encoding**: Tokens are converted to indices, starting with `<sos>`, followed by the token indices, and ending with `<eos>`. This encoding helps the LSTM decoder understand the sequence of words.

5. **[Teacher Forcing](https://machinelearningmastery.com/teacher-forcing-for-recurrent-neural-networks/):**
   - **Teacher Forcing Ratio**: During training, both models used a **teacher forcing ratio of 0.90**, meaning that 90% of the time, the ground truth caption tokens were fed into the decoder during sequence generation, while the remaining 10% relied on the model's predictions.

6. **Training Configuration:**
   - Both models were trained using **mixed precision (Float16)** to improve memory efficiency and training speed.
   - The training was executed on an **L4 24GB RAM GPU** using **torch.compile** for improved runtime optimizations, enabling faster convergence and better GPU utilization.

### Parameter Comparison

| Component                  | ImgCap with Attention | ImgCap without Attention |
|-----------------------------|-----------------------|--------------------------|
| **Total Parameters**        | 85.79M                | 52.89M                   |
| **Trainable Parameters**    | 36.72M                | 29.38M                   |
| **Non-trainable Parameters**| 49.07M                | 23.51M                   |

### ImgCap with Attention 
![image](https://github.com/user-attachments/assets/75f74bfd-819d-40ff-afa5-44756d3b340d)


### ImgCap without Attention 
![image](https://github.com/user-attachments/assets/c8adcaa6-7dba-4805-bf0d-142d210a8389)

- The model with attention has more trainable parameters and introduces a more complex mechanism for context generation, leading to improved performance in captioning tasks, as seen in the evaluation metrics. However, due to the larger number of parameters, the model may require longer training time to fully converge. 
---
## Model Evaluation
---

| Model                    | Epoch | Beam Width | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | CIDEr  |
|---------------------------|-------|------------|--------|--------|--------|--------|--------|
| **ImgCap without Attention**  | 40    | 5          | 0.37   | 0.22   | 0.14   | 0.09   | 0.41   |
| **ImgCap with Attention**     | 30    | 5          | 0.3959 | 0.2464 | 0.1619 | 0.1077 | 0.6213 | 

Note: The models are still undertrained, and in theory, their accuracy is expected to improve with further training. Extending the number of epochs could lead to higher BLEU and CIDEr scores, particularly for the attention-based model, which already shows a performance boost.

---
## Future Work
---
In the next phase, I plan to explore the **Vision Transformer (ViT)** architecture to develop a new variant of the **ImgCap** model. This variant will scale more effectively for complex visual understanding tasks. Additionally, I aim to expand the model's capabilities by training it for **multilingual captioning** in both **English** and **Arabic**.
