''' Script to calculate scores for all news, tweets and reddit posts that
reflect their bullishness for the corresponsing cryptocurrency (Bitcoin or
Ethereum) using a pretrained BART-Large-MNLI model from Huggingface. '''

import torch
import pandas as pd
import logging
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from scipy.special import softmax
from tqdm import tqdm

import sys
sys.path.append('../../')

from utils.wrappers import timeit, telegram_notify

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)

class BartMNLI():
    ''' Class to calculate zero-shot probabilities with the BART-Large-MNLI
    model from Huggingface. '''

    def __init__(self, softmax=True):
        ''' Initialises the BART-Large-MNLI model from the Huggingface Hub. 

        Args:
            softmax (bool, optional): If true, softmax function is applied to
                model output and the returned sentiment score is bound by
                [-1,1]. Otherwise it's bound by (-infty,infty). Defaults to True.
        '''
        MODEL = 'facebook/bart-large-mnli'

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'Using device: {self.device}')
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        self.model = (AutoModelForSequenceClassification
                      .from_pretrained(MODEL)
                      .to(self.device))
        
        self.softmax = softmax

    def get_score_btc(self, text) -> float:
        ''' This function outputs a 'Bullish for Bitcoin' score for a given
        input text.
        
        Args:
            text: Input string of type `str` or `List[str]` or
                `List[List[str]]`.
            
        Returns:
            float: Value in the range (-infty,infty) or [-1,1], depending on
                softmax setting, that reflects how likely the hypothesis `This
                example is bullish for Bitcoin.` is true for the input string.
            
        Raises:
            ValueError: If input is not of type `str` or `List[str]` or
                `List[List[str]]`.
        '''
        hypothesis = 'This example is bullish for Bitcoin.'

        encoded_inputs = (self.tokenizer
                          .encode(text,
                                  hypothesis,
                                  return_tensors='pt',
                                  truncation=True,
                                  max_length=1024)
                          .to(self.device))
        with torch.no_grad():
            output = self.model(encoded_inputs)[0][0].detach().cpu().numpy()

        if self.softmax:
            output = softmax(output)
        else:
            pass

        # The score is the difference of the positive value and the negative
        # value returned by the model
        score = output[2] - output[0]

        return score

@timeit
@telegram_notify
def main():
    logging.info('Loading BTC news data')
    btc_news = pd.read_parquet('../data/btc_news_processed.parquet.gzip')
    logging.info(f'Data loaded: {len(btc_news):,} rows')
    
    logging.info('Initialising BART-MNLI model')
    model = BartMNLI(softmax=True)

    tqdm.pandas()

    variable_name = 'bart_mnli_bullish_score'
    logging.info('Calculating BART-MNLI bullishness scores')
    btc_news[variable_name] = btc_news.title.progress_apply(model.get_score_btc)
    logging.info('Sentiment calculation finished')

    logging.info('Dropping text column')
    btc_news = btc_news.drop(columns=['title'])
    
    logging.info('Saving parquet file')
    btc_news.to_parquet('btc_news_bart_mnli.parquet.gzip',
                        compression='gzip')
    logging.info('File saved successfully')

if __name__=='__main__':
    main()
