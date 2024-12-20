import streamlit as st
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
from torch import nn

class TransformerClassifier(nn.Module):
    def __init__(self, lm, n_classes):
        super(TransformerClassifier, self).__init__()
        self.transformer = lm
        layer_size = self.transformer.config.hidden_size

        self.classifer = nn.Sequential(
            nn.Linear(layer_size, n_classes),
            nn.Softmax(dim=1)
        )


    def forward(self, x, attention_mask):
        with torch.no_grad():
            x = self.transformer(input_ids=x, attention_mask=attention_mask)
        x = x.last_hidden_state[:, 0, :]
        x = self.classifer(x)
        return x




language_model_name = "FacebookAI/roberta-base"
tokenizer = AutoTokenizer.from_pretrained(language_model_name)
lm_config = AutoConfig.from_pretrained(language_model_name)
lm = AutoModel.from_config(lm_config)
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
model = TransformerClassifier(lm, 2).to(device)
model.load_state_dict(torch.load(f"model/ai_text_detection.pth", weights_only=True))
model.eval()


labels = ["written by a human", "AI generated"]

def tokenize(text):
    tokens = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        padding=True
    )
    return tokens['input_ids'].to(device), tokens['attention_mask'].to(device)

def detect_ai_text(text):
    tokens, attention_mask = tokenize(text)
    pred = model(tokens, attention_mask)
    return pred[0][1].item(), pred.argmax(1).item()


st.title("AI Text Detector")
st.write("Enter text to analyze:")

text_input = st.text_area("Input Text", height=200)

if st.button("Analyze"):
    if text_input:
        p, result = detect_ai_text(text_input)
        st.write("This text is", labels[result])
        st.write(f'AI generation probability: {p*100:.2f}%')

    else:
        st.warning("Please enter some text.")