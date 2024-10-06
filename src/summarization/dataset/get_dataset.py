from torch.utils.data import Dataset
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5TokenizerFast as T5Tokenizer
)
import datasets
from transformers import  BartTokenizer

class NewsSummaryDataset(Dataset):
  def __init__(self,
               data: datasets.arrow_dataset.Dataset,
               tokenizer,
               text_max_token_len: int = 512,
               summary_max_token_len: int = 128
              ):
    self.tokenizer = tokenizer
    self.data = data
    self.text_max_token_len = text_max_token_len
    self.summary_max_token_len = summary_max_token_len

  def encoding_plus(self, 
                    text:str, 
                    max_len: int):
    return self.tokenizer(
        text,
        max_length = max_len,
        padding = "max_length",
        truncation = True,
        add_special_tokens = True,
        return_attention_mask = True,
        return_tensors = 'pt'
    )

  def __getitem__(self, index: int):
    data_row = self.data[index]
    text, summary = data_row['article'], data_row['highlights']
    
    text_encoding = self.encoding_plus(text, self.text_max_token_len)
    summary_encoding = self.encoding_plus(summary, self.summary_max_token_len)

    labels = summary_encoding["input_ids"]
    labels[labels == 0] = -100

    return dict (
        text = text,
        summary = summary,
        text_input_ids = text_encoding["input_ids"].flatten(),
        text_attention_mask = text_encoding["attention_mask"].flatten(),
        labels = labels.flatten(),
        labels_attention_mask = summary_encoding["attention_mask"].flatten(),
    )
  
  def __len__(self):
    return len(self.data)