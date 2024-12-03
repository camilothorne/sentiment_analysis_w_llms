from sentiment_analysis import *
from torchinfo import summary
from datasets import load_dataset
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import time, os


# Mute warnings
import warnings
warnings.filterwarnings("ignore")


'''
In this file, we reproduce one of methods (M1) presented in the 
notebook

    ../notebooks/analysis_notebook.ipynb

'''


if __name__ == "__main__":


    '''
    Set up device - one of

        - CUDA (NVIDIA GPU)
        - MPS (Silicon GPU)
        - CPU
    '''

    if torch.cuda.is_available():
        print('Using NVIDIA / CUDA GPU')
        torch.device('cuda')
    elif torch.backends.mps.is_available():
        print('Using Silicon / MPS GPU')
        torch.device('mps')
    else:
        print(f'Using {os.cpu_count()} CPUs')
        torch.device('cpu')

    '''
    Download and prepare the models and tokenizer
    '''

    model_checkpoint_medium_x = "Qwen/Qwen2.5-1.5B-Instruct"
    model_checkpoint_small_x  = "Qwen/Qwen2.5-0.5B-Instruct"

    model_m = AutoModelForCausalLM.from_pretrained(
        model_checkpoint_medium_x,
        device_map="auto",
        torch_dtype="auto"
    )
    model_s = AutoModelForCausalLM.from_pretrained(
        model_checkpoint_small_x,
        torch_dtype="auto",
        device_map="auto"
    )
    print(summary(model_m, verbose=0))
    print(summary(model_s, verbose=0))

    tokenizer_m = AutoTokenizer.from_pretrained(model_checkpoint_medium_x)
    tokenizer_s = AutoTokenizer.from_pretrained(model_checkpoint_small_x)
    print(f'The 1.5B model has a vocabulary of {tokenizer_m.vocab_size} tokens.')
    print(f'The 500M model has a vocabulary of {tokenizer_s.vocab_size} tokens.')

    '''
    Download and prepare the IMDB movie reviews dataset
    '''

    print('Deriving a balanced random sample of 5%/500 reviews of IMDB the test set for evaluation')
    dataset = load_dataset("ajaykarthick/imdb-movie-reviews")
    # We're only interested in the test set, as we don't plan to train any model
    df_imdb_test            = pd.DataFrame(dataset['test'])
    df_sample               = df_imdb_test.groupby(['label']).apply(lambda f: f.sample(frac=0.05))
    df_sample['sentiment']  = df_sample['label'].map({0:'negative', 1:'positive'})
    print('Test set statistics:')
    _, _, total_words, _, _ = corpus_stats(list(df_sample.review.values))

    '''
    Zero-shot learning example
    '''

    start = time.time()
    df_sample['response_m'] = df_sample.review.apply(lambda x: 
                                    generate_response_zero(str(x), 
                                            tokenizer_m, model_m))
    end = time.time()
    print(f'Time taken for 1.5B model: {end-start}s')
    med_time = end-start

    start = time.time()
    df_sample['response_s'] = df_sample.review.apply(lambda x: 
                                    generate_response_zero(str(x), 
                                            tokenizer_s, model_s))
    end = time.time()
    print(f'Time taken for 500M model: {end-start}s')
    small_time = end-start

    print(f'Time per word for 1.5B model: {total_words / med_time} words per second')
    print(f'Time per word for 500 model : {total_words / small_time} words per second')

    print(f'Performance for 1.5B model:\n')
    performance_report(df_sample.sentiment, 
                       df_sample.response_m, 
                       name='Zero-shot learning (1.5B model)')
    
    print(f'Performance for 1.5B model:\n')
    performance_report(df_sample.sentiment, 
                       df_sample.response_s, 
                       name='Zero-shot learning (500M model)')

    print(f'Rendering confusion matrix for 1.5B model')
    confusion_matrix(df_sample.sentiment, 
                     df_sample.response_m)
    
    print(f'Rendering confusion matrix for 500M model')
    confusion_matrix(df_sample.sentiment, 
                     df_sample.response_s)
