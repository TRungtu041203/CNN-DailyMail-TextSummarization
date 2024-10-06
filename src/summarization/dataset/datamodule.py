import pytorch_lightning as pl
import datasets
from transformers import T5TokenizerFast as T5Tokenizer
from torch.utils.data import DataLoader
from .get_dataset import *


class NewsSummaryDataModule(pl.LightningDataModule):
  def __init__(
      self, 
      train_df: datasets.arrow_dataset.Dataset,
      val_df: datasets.arrow_dataset.Dataset,
      tokenizer: T5Tokenizer,
      batch_size: int = 8,
      text_max_token_len: int = 512,
      summary_max_token_len: int = 128,
      num_workers: int = 4

  ):
    super().__init__()
    self.train_df = train_df
    self.val_df = val_df
    
    self.batch_size = batch_size
    self.tokenizer = tokenizer
    self.text_max_token_len = text_max_token_len
    self.summary_max_token_len = summary_max_token_len
    self.num_workers = num_workers


  def setup(self, stage=None):
    self.train_dataset = NewsSummaryDataset(
        data=self.train_df,
        tokenizer=self.tokenizer,
        text_max_token_len=self.text_max_token_len,
        summary_max_token_len=self.summary_max_token_len
    )
    self.val_dataset = NewsSummaryDataset(
        data=self.val_df,
        tokenizer=self.tokenizer,
        text_max_token_len=self.text_max_token_len,
        summary_max_token_len=self.summary_max_token_len
    )

  def train_dataloader(self):
    return DataLoader(
        self.train_dataset,
        batch_size= self.batch_size,
        shuffle = True,
        num_workers = self.num_workers
    )
  def val_dataloader(self):
    return DataLoader(
        self.val_dataset,
        batch_size= self.batch_size,
        shuffle = False,
        num_workers = self.num_workers
    )
  def test_dataloader(self):
    return DataLoader(
        self.val_dataset,
        batch_size= self.batch_size,
        shuffle = False,
        num_workers = self.num_workers
    )
     