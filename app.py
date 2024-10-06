from src.summarization.model.model import *
import torch
import datasets
import gradio as gr
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5TokenizerFast as T5Tokenizer,
    BartForConditionalGeneration,
    BartTokenizerFast as BartTokenizer
)

checkpoint = torch.load("bart.ckpt", map_location=torch.device('cpu') )

model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")
t5_model = NewsSummaryModel.load_from_checkpoint("t5.ckpt", model = model)
# t5_model = NewsSummaryModel(model = model)

t5_tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-base", model_max_length=512)

bart_tokenizer = BartTokenizer.from_pretrained("lucadiliello/bart-small", model_max_length=512)
model = BartForConditionalGeneration.from_pretrained("lucadiliello/bart-small")
bart_model = NewsSummaryModel.load_from_checkpoint("bart.ckpt", model = model)

# test_data = datasets.load_dataset("ccdv/cnn_dailymail", "3.0.0", split="test")

# text = test_data[0]['article']

def summarizeText(text, model_name="T5"):
    if model_name == "T5":
        model = t5_model
        tokenizer = t5_tokenizer
    elif model_name == "Bart":
        model = bart_model
        tokenizer = bart_tokenizer
    
    text_encoding = tokenizer(
        text,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors='pt'
    )
    generated_ids = model.model.generate(
        input_ids=text_encoding['input_ids'],
        attention_mask=text_encoding['attention_mask'],
        max_length=150,
        num_beams=2,
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True
    )

    preds = [
            tokenizer.decode(gen_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for gen_id in generated_ids
    ]
    return "".join(preds)

def greetMe(articleInput, model_name):
    summarizedArticle = summarizeText(articleInput, model_name)
    return summarizedArticle

myapp = gr.Interface( greetMe,
                     ["textbox", gr.Dropdown(choices=["T5", "Bart"], label="Model")],
                     "textbox",
                     examples=[
                         ["enter the original text that is to be summarized", "T5"]
                     ],
                     title="Text Summarizer using Transformers",
                     description="Choose between T5 and Bart models for text summarization.",
                     allow_flagging='never')

if __name__=="__main__":
    myapp.launch(show_api=False)