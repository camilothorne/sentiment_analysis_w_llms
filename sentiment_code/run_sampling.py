# Import logger
import logging

# Other imports
from sentiment_analysis import *
from torchinfo import summary
from datasets import load_dataset
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import time, os

# Mute warnings
import warnings
warnings.filterwarnings("ignore")

# Get current directory
dirname = os.path.dirname(__file__)

# Set logger
logger = logging.getLogger(__name__)


'''
In this file, we reproduce one of methods (M4 - sampling with optimized parameters)
presented in the notebook

    ../notebooks/analysis_notebook.ipynb

and incorporate logging to track execution.
'''


if __name__ == "__main__":

    '''
    Start logger
    '''

    logging.basicConfig(filename = os.path.join(dirname, '../results/log-sampling.log'), 
                        level=logging.INFO, filemode='w')
    logger.info('Started')  

    '''
    Set up device - one of

        - CUDA (NVIDIA GPU)
        - MPS (Silicon GPU)
        - CPU
    '''

    if torch.cuda.is_available():
        logger.info('Using NVIDIA / CUDA GPU')
        torch.device('cuda')
    elif torch.backends.mps.is_available():
        logger.info('Using Apple Silicon / MPS GPU')
        torch.device('mps')
    else:
        logger.info(f'Using {os.cpu_count()} CPUs')
        torch.device('cpu')

    '''
    Download and prepare the models and tokenizer
    '''

    model_checkpoint_medium_x = "Qwen/Qwen2.5-1.5B-Instruct"
    model_checkpoint_small_x  = "Qwen/Qwen2.5-0.5B-Instruct"

    model_m = AutoModelForCausalLM.from_pretrained(
        model_checkpoint_medium_x,
        device_map="auto", # use automatic device placement (mps:0 for MPS)
        torch_dtype="auto"  # optimize for mixed precision (FP16)
    )

    model_s = AutoModelForCausalLM.from_pretrained(
        model_checkpoint_small_x,
        torch_dtype="auto", # optimize for mixed precision (FP16)
        device_map="auto" # use automatic device placement (mps:0 for MPS)
    )

    logger.info(f'1.5B checkpoint: {model_checkpoint_medium_x}')
    logger.info("\n" + str(summary(model_m, verbose=0)))

    logger.info(f'500M model: {model_checkpoint_small_x}')
    logger.info("\n" + str(summary(model_s, verbose=0)))

    tokenizer_m = AutoTokenizer.from_pretrained(model_checkpoint_medium_x)
    tokenizer_s = AutoTokenizer.from_pretrained(model_checkpoint_small_x)

    logger.info(f'The 1.5B model has a vocabulary of {tokenizer_m.vocab_size} tokens.')
    logger.info(f'The 500M model has a vocabulary of {tokenizer_s.vocab_size} tokens.')

    '''
    Download and prepare the IMDB movie reviews dataset
    '''

    logger.info('Deriving a balanced random sample of 5%/500 reviews of IMDB the test set for evaluation')
    logger.info('Downloading and reading IMDB dataset...')
    
    dataset = load_dataset("ajaykarthick/imdb-movie-reviews")
    
    # We're only interested in the test set, as we don't plan to train any model
    df_imdb_test            = pd.DataFrame(dataset['test'])
    df_sample               = df_imdb_test.groupby(['label']).apply(lambda f: f.sample(frac=0.05))
    df_sample['sentiment']  = df_sample['label'].map({0:'negative', 1:'positive'})
    
    logger.info('Test set statistics:')
    _, _, total_words, _, _ = corpus_stats(list(df_sample.review.values))

    '''
    Sampling learning example
    '''

    # We plan to consider 6x6 = 36 possible combinations
    temp_values =  [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    top_k_values = [1000, 800, 600, 400, 200, 100]

    # We'll use a (balanced) sample of 40 examples to optimize parameters:
    df_small_sample = generate_sample_data(dataset, 20)

    # Grid for 1.5B model
    start = time.time()
    grid_med = grid_search(df_small_sample, tokenizer_m, model_m, top_k_values, temp_values)
    end = time.time()
    logger.info(f'Time taken: {end-start}s')
    temp_m, top_k_m  = parse_grid_args(list(grid_med.keys())[-1])
    logger.info(f'[1.5B] Maximum accuracy with (temp, top_k) {list(grid_med.keys())[-1]}: {grid_med[list(grid_med.keys())[-1]]}')

    # Grid for 500M model
    start = time.time()
    grid_small = grid_search(df_small_sample, tokenizer_s, model_s, top_k_values, temp_values)
    end = time.time()
    logger.info(f'Time taken: {end-start}s')
    temp_s, top_k_s  = parse_grid_args(list(grid_small.keys())[-1])
    logger.info(f'[500M] Maximum accuracy with (temp, top_k) {list(grid_small.keys())[-1]}: {grid_small[list(grid_small.keys())[-1]]}')

    # We plug grid values (1.5B)
    start = time.time()
    df_sample['response_m'] = df_sample.review.apply(
        lambda x: generate_response_with_sampling(str(x), 
                                                tokenizer_m, 
                                                model_m, 
                                                top_k=top_k_m, 
                                                temp=temp_m
                                                ))
    end = time.time()
    logger.info(f'Time taken: {end-start}s')
    med_time = end-start

    # We plug grid values (500M)
    start = time.time()
    df_sample['response_s'] = df_sample.review.apply(
        lambda x: generate_response_with_sampling(str(x), 
                                                tokenizer_s, 
                                                model_s, 
                                                top_k=top_k_s, 
                                                temp=temp_s
                                                ))
    end = time.time()
    logger.info(f'Time taken: {end-start}s')
    small_time = end-start

    logger.info(f'Time per word for 1.5B model: {total_words / med_time} words per second')
    logger.info(f'Time per word for 500 model : {total_words / small_time} words per second')

    logger.info(f'Performance for 1.5B model:\n')
    performance_report(df_sample.sentiment, 
                       df_sample.response_m, 
                       name='Sampling (1.5B model)')
    
    logger.info(f'Performance for 500M model:\n')
    performance_report(df_sample.sentiment, 
                       df_sample.response_s, 
                       name='Sampling (500M model)')

    logger.info(f'Rendering confusion matrix for 1.5B model')
    confusion_matrix(df_sample.sentiment, 
                     df_sample.response_m,
                     name='Sampling (1.5B model)')
    
    logger.info(f'Rendering confusion matrix for 500M model')
    confusion_matrix(df_sample.sentiment, 
                     df_sample.response_s,
                     name='Sampling (500M model)')
    
    logger.info('Finished')