from src.summarization.model.model import *
import datasets
import torch 
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5TokenizerFast as T5Tokenizer,
    BartForConditionalGeneration,
    BartTokenizerFast as BartTokenizer
)
from transformers import PegasusForConditionalGeneration, PegasusTokenizer


class ModelEvaluation:
    ROUGE = datasets.load_metric("rouge")
    DEVICE = "cuda"
    def __init__(self,
               model: NewsSummaryModel,
               tokenizer: T5Tokenizer,
               batch_size: int = 32,
               text_max_token_len: int = 512,
               summary_min_token_len: int = 60,
               summary_max_token_len: int = 150,
               num_beams: int = 2,

            ):
        self.model = model
        self.model.eval()
        self.model.to(self.DEVICE)

        self.tokenizer = tokenizer

        self.batch_size = batch_size
        self.text_max_token_len = text_max_token_len
        self.summary_min_token_len = summary_min_token_len
        self.summary_max_token_len = summary_max_token_len
        self.num_beams = num_beams

    def _generate_summary_in_batch(self, batch):
    # Tokenizer will automatically set [BOS] <text> [EOS]
    # cut off at BERT max length 512
        text_encodings = self.tokenizer(
        batch["article"],
        max_length=self.text_max_token_len,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="pt"
    )
        input_ids = text_encodings['input_ids'].to(self.DEVICE)
        attention_mask =  text_encodings['attention_mask'].to(self.DEVICE)
        
        outputs = self.model.model.generate(
            input_ids = input_ids,
            attention_mask=attention_mask,
            max_length = self.summary_max_token_len,
            min_length = self.summary_min_token_len,
            num_beams = self.num_beams,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True
        )

        output_preds = self.tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)


        batch["pred"] = output_preds

        return batch
  
    def evaluate(self, test_data: datasets.arrow_dataset.Dataset):
        results = test_data.map(self._generate_summary_in_batch, batched=True, batch_size=self.batch_size, remove_columns=["article", "id"])
        pred_strings = results["pred"]
        label_strings = results["highlights"]

        rouge_output = self.ROUGE.compute(predictions=pred_strings, references=label_strings, rouge_types=['rouge1', 'rouge2', 'rougeL'])
        return rouge_output
    
    
checkpoint = torch.load("t5.ckpt", map_location=torch.device('cpu') )

model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-base")
t5_model = NewsSummaryModel(model = model)
t5_model.freeze()
t5_tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-base", model_max_length=512)

bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base", model_max_length=512)
bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
bart_model = NewsSummaryModel(model = model)
bart_model.freeze()

pegasus_tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-cnn_dailymail", model_max_length=512)
model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-cnn_dailymail")
pegasus_model = NewsSummaryModel(model=model)
bart_model.freeze()


test_data = datasets.load_dataset("ccdv/cnn_dailymail", "3.0.0", split="test")

# t5_eval = ModelEvaluation(t5_model, t5_tokenizer)
# t5_rouge_output = t5_eval.evaluate(test_data)
# t5_rouge_output_mid = {k: v.mid for k,v  in t5_rouge_output.items()}
# print(t5_rouge_output_mid)

# bart_eval = ModelEvaluation(bart_model, bart_tokenizer)
# bart_rouge_output = bart_eval.evaluate(test_data)
# bart_rouge_output_mid = {k: v.mid for k,v  in bart_rouge_output.items()}
# print(bart_rouge_output_mid)

pegasus_eval = ModelEvaluation(pegasus_model, pegasus_tokenizer)
pegasus_rouge_output = pegasus_eval.evaluate(test_data)
pegasus_rouge_output_mid = {k: v.mid for k,v  in pegasus_rouge_output.items()}
print(pegasus_rouge_output_mid)