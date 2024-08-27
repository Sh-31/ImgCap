import cv2
import pickle
import torch
import gradio as gr
import torchvision.transforms as T
from utils import load_checkpoint
from trainning import ImgCap, beam_search_caption, decoder

def ImgCap_inference(img, beam_width):
    root_path = "/teamspace/studios/this_studio"
    with open(f"{root_path}/ImgCap/vocab.pkl", 'rb') as f:
        vocab = pickle.load(f)

    transforms = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]) 

    checkpoint_path = f"{root_path}/ImgCap/trainning/checkpoints/checkpoint_epoch_40.pth"
    
    model = ImgCap(cnn_feature_size=1024, lstm_hidden_size=1024, embedding_dim=1024, num_layers=2, vocab_size=len(vocab))
    model, _, _, _, _, _, _ = load_checkpoint(checkpoint_path=checkpoint_path, model=model)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = transforms(img).unsqueeze(0)

    generated_caption = beam_search_caption(model, img, vocab, decoder, beam_width=beam_width) 

    return generated_caption

if __name__ == "__main__":
    footer_html = "<p style='text-align: center; font-size: 16px;'>Developed by Sherif Ahmed</p>"

    interface = gr.Interface(
        fn=ImgCap_inference,
        inputs=[
            'image', 
            gr.Slider(minimum=1, maximum=5, step=1, label="Beam Width")
        ], 
        outputs=gr.Textbox(label="Generated Caption"),
        title="ImgCap",
        article=footer_html
    )
    interface.launch()
