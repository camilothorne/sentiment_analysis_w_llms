import logging
import transformers
import os
import torch
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score as accuracy
import pandas as pd
import sklearn.metrics as metrics

# Set up logging
logger = logging.getLogger(__name__)

# We mute deprecation warnings
import warnings
warnings.filterwarnings("ignore")

# Get current directory
dirname = os.path.dirname(__file__)


'''
In this module, we clean up the functions and procedures used in the notebook

    ../notebooks/analysis_notebook.ipynb

but incorporating logging and expanding on comments for better readability.
'''


def corpus_stats(corpus:list[str])->tuple[int,float,int,int,int]:
    '''
    Corpus statistics (whitespace tokenization).
    
    Arguments:
        - corpus: list of strings
    Returns:
        - tot_sents:  number of sentences
        - avg_words:  average number of tokens per sentence
        - tot_words:  total number of words
        - max_words:  max number of tokens per sentence
        - min_words:  min number of tokens per sentence
    '''
    tot_sents  = len(corpus)
    n_tokens   = []
    tot_words  = 0
    for sent in corpus:
        n_tokens.append(len(sent.split(' ')))
        tot_words += len(sent.split(' '))
    # Print stats
    logger.info(f'Number of sentences:                  {tot_sents}')
    logger.info(f'Average number of words per sentence: {sum(n_tokens)/tot_sents}')
    logger.info(f'Max number of words per sentence:     {max(n_tokens)}')
    logger.info(f'Min number of words per sentence:     {min(n_tokens)}')
    logger.info(f'Total number of words:                {tot_words}')
    return tot_sents, sum(n_tokens)/tot_sents, tot_words, max(n_tokens), min(n_tokens)


def encode_input_zero(input_prompt:str,
                 tokenizer:transformers.AutoTokenizer,
                 model:transformers.AutoModelForCausalLM)->torch.Tensor:
    '''
    Function for parsing prompts embedded in conversation messages (Zero shot learning).

    Arguments:
        - input_prompt:  input query string
        - tokenizer:     AutoTokenizer object
        - model:         AutoModelForCausalLM object
    Returns:
        - tokenization:  a PyTorch tensor
    '''
    # The 'user' message is used to query the model with a sentiment prompt.
    messages = [
        {"role": "system", "content":
                "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content":
                "What is the sentiment of the following text, reply 'positive' or 'negative': " + input_prompt}
    ]
    # Transforms message JSON into a plain text string in the model's input format.
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    # Returns a tensor.
    return tokenizer([text],
                      return_tensors = 'pt'
                    ).to(model.device)


def generate_response_zero(input_prompt:str,
                      tokenizer:transformers.AutoTokenizer,
                      model:transformers.AutoModelForCausalLM,
                      length:int=1)->str:
    '''
    Function for generating responses (Zero shot learning).

    Arguments:
        - input_prompt:  query string
        - tokenizer:     AutoTokenizer object
        - model:         AutoModelForCausalLM object
        - length:        integer used to cap the length of the output sequence (1 token or word by default)
    Returns:
        - response:      predicted string (label)
    '''
    # We encode inputs into a tensor.
    model_inputs = encode_input_zero(input_prompt, tokenizer, model)
    # We use greedy decoding to generate the labels.
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=length
    )
    # We decode the returned tensor into a string and
    # Return it lower-cased.
    generated_ids = [
        output_ids[len(input_ids):] for input_ids,
            output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response.lower()


def encode_sentiment_logit(input_prompt:str,
                    tokenizer:transformers.AutoTokenizer,
                    model:transformers.AutoModelForCausalLM,
                    sentiment:str)->torch.Tensor:
    '''
    Function for parsing labelled prompts (likelihood queries).

    Arguments:
        - input_prompt:  input query string
        - tokenizer:     AutoTokenizer object
        - model:         AutoModelForCausalLM object
        - sentiment:     sentiment label - 'positive' or 'negative' (string)
    Returns:
        - tokenization:  a PyTorch tensor
    '''
    messages = [
        {"role": "system", "content":
                "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content":
                "Is the labelling of this prompt correct:" + input_prompt + " " + sentiment + "."}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return tokenizer([text],
                      return_tensors = 'pt'
                    ).to(model.device)

def get_sentiment_logit(input_prompt:str,
                      tokenizer:transformers.AutoTokenizer,
                      model:transformers.AutoModelForCausalLM,
                      sentiment:str,
                      length:int=1)->float:
    '''
    Function for generating responses with likelihoods (likelihood queries).

    Arguments:
        - input_prompt:  query string
        - tokenizer:     AutoTokenizer object
        - model:         AutoModelForCausalLM object
        - sentiment:     sentiment label - 'positive' or 'negative' (string)
    Returns:
        - score:         logit score (float)
    '''
    # Note that we add a sentiment label here.
    model_inputs = encode_sentiment_logit(input_prompt, tokenizer, model, sentiment)
    # Here, instead of generating a label, or text, we only focus
    # on returning the likelihood of the predicted (best) next token (a float).
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=length,
        output_scores=True,
        do_sample=False,
        return_dict_in_generate=True,
        return_legacy_cache=True
    )
    return generated_ids.scores[0][0][0]

def get_sentiment_from_logit(input_prompt:str,
                  tokenizer:transformers.AutoTokenizer,
                  model:transformers.AutoModelForCausalLM)->str:
    '''
    Function to classify sentiment (likelihood queries).

    Arguments:
        - input_prompt:  query string
        - tokenizer:     AutoTokenizer object
        - model:         AutoModelForCausalLM object
    Returns:
        - label:         string (sentiment label)
    '''
    pos = get_sentiment_logit(input_prompt, tokenizer, model, 'positive')
    neg = get_sentiment_logit(input_prompt, tokenizer, model, 'negative')
    # We return the label that gave rise to the highest log-likelihood.
    if pos > neg:
        return 'positive'
    else:
        return 'negative'
    

def encode_input_with_context(input_prompt:str,
                 tokenizer:transformers.AutoTokenizer,
                 model:transformers.AutoModelForCausalLM,
                 in_context:str)->torch.Tensor:
    '''
    Function for parsing prompts embedded in conversation messages, with
    a variable storing in-context examples (K-shot learning).

    Arguments:
        - input_prompt:  input query string
        - tokenizer:     AutoTokenizer object
        - model:         AutoModelForCausalLM object
        - in_context:    string with K examples
    Returns:
        - tokenization:  a PyTorch tensor
    '''
    # Note the use of contexts.
    messages = [
        {"role": "system", "content":
                "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content":
                "What is the sentiment of the following text, reply 'positive' or 'negative': " + in_context + input_prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return tokenizer([text],
                      return_tensors = 'pt'
                    ).to(model.device)


def generate_response_with_context(input_prompt:str,
                      tokenizer:transformers.AutoTokenizer,
                      model:transformers.AutoModelForCausalLM,
                      in_context:str,
                      length:int=1)->str:
    '''
    Function for generating responses, with a variable 
    storing in-context examples (K-shot learning).

    Arguments:
        - input_prompt:  query string
        - tokenizer:     AutoTokenizer object
        - model:         AutoModelForCausalLM object
        - in_context:    string with K examples
    Returns:
        - response:      predicted string (label)
    '''
    # Note the use of contexts.
    model_inputs = encode_input_with_context(input_prompt, tokenizer, model, in_context)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=length
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids,
            output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response.lower()


def generate_context(data:dict, num_per_class:int)->str:
    '''
    Generates K positive and K negative examples (shortest examples) 
    from a given dataset (K-shot learning).

    Arguments:
        - dataset:         a datasets.DatasetDict object
        - num_per_class:   an integer (count of examples per class)
    Returns:
        - result:          a string with K positive and K negative examples (shortest examples)
    '''
    # We use the train partition, to ensure that the examples are disjoint
    # from our test set:
    df_train = pd.DataFrame(data['train'])
    # We sort positives and negatives by length, and pick always the shortest:
    df_train['len'] = df_train.review.apply(lambda x: len(str(x)))
    df_pos = df_train[df_train.label==1].sort_values(by='len').head(num_per_class)
    df_neg = df_train[df_train.label==0].sort_values(by='len').head(num_per_class)
    # We interate over the examples to form the in context string:
    result = ""
    for _, row in df_pos.iterrows():
        result = result + str(row['review']) + " Positive.\n"
    for _, row in df_neg.iterrows():
        result = result + str(row['review']) + " Negative.\n"
    # We return the final context a single string:
    logger.info(f"{num_per_class}-shot ICL context:\n\n{result}")
    return result


def generate_sample_data(data:dict, num_per_class:int)->str:
    '''
    Generates sample for parameter optimization.

    Arguments:
        - dataset:         a dict object
        - num_per_class:   an integer (count of examples per class)
    Returns:
        - res:             a DataFrame object
    '''
    # We use the train partition, to ensure that the examples are disjoint
    # from our test set:
    df_train = pd.DataFrame(data['train'])
    # We sort positives and negatives by length, and pick always the shortest:
    df_train['len'] = df_train.review.apply(lambda x: len(str(x)))
    df_pos = df_train[df_train.label==1].sort_values(by='len').head(num_per_class)
    df_neg = df_train[df_train.label==0].sort_values(by='len').head(num_per_class)
    # We return a dataframe
    res = pd.concat([df_pos, df_neg], ignore_index=True)
    res['sentiment'] = res['label'].map({0:'negative', 1:'positive'})
    return res


def generate_response_with_sampling(input_prompt:str,
                      tokenizer:transformers.AutoTokenizer,
                      model:transformers.AutoModelForCausalLM,
                      top_k:float,
                      temp:float,
                      length:int=1
                     )->str:
    '''
    Function for generating responses.

    Arguments:
        - input_prompt:  query string
        - tokenizer:     AutoTokenizer object
        - model:         AutoModelForCausalLM object
        - top_k:         top k value (float)
        - temp:          temperature value (float)
        - length:        integer used to cap the length of the output sequence (1 token or word by default)
    Returns:
        - response:      predicted string (label)
    '''
    # We encode inputs into a tensor. We re-use the zero-shot function.
    model_inputs = encode_input_zero(input_prompt, tokenizer, model)
    # We use greedy decoding to generate the labels.
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=length,
        temperature=temp,
        top_k=top_k
    )
    # We decode the returned tensor into a string and
    # Return it lower-cased and stripped of whitespaces.
    generated_ids = [
        output_ids[len(input_ids):] for input_ids,
            output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response.lower().strip()


def grid_search(sample:pd.DataFrame, 
                tokenizer:transformers.AutoTokenizer,
                model:transformers.AutoModelForCausalLM,               
                top_k_v:list, 
                temp_v:list)->dict:
    '''
    Grid search over |temperatures| x |top_k_values| combinations.

    Arguments:
        - tokenizer:   tokenizer object as above
        - model:       model object as above
        - temp_v:      list of temperatures
        - top_k_v:     list of top_k values
    Returns:
        - results:     dictionary of scores
    '''
    # We save results in a dictionary (for a given model and sample 
    # combination) with entries of form:
    #
    #     (temp, top_k): accuracy
    # 
    results = {}
    for temp in temp_v:
        for top_k in top_k_v:
            key = str((temp, top_k))
            result = sample.review.apply(lambda x: 
                            generate_response_with_sampling(str(x), 
                                                            tokenizer, 
                                                            model, 
                                                            top_k, 
                                                            temp))
            acc = accuracy(sample.sentiment, result)
            results[key] = acc
    # We return the dictionary sorted (asc. order) by accuracy,
    # with **the last entry being the optimal combination**.
    results = dict(sorted(results.items(), key=lambda item: item[1]))
    return results


def parse_grid_args(grid_args:str)->tuple[float,int]:
    '''
    Parse the results of grid search, returning the top_k and temperature values.

    Input:
        - grid_args:        values to parse (string)
    Returns:
        - temp, top_k:     temperature (float) and top (integer) values (tuple)
    '''
    grid_tup = grid_args.replace('(','').replace(')','').strip().split(',')
    logger.info(f"Top K: {grid_tup[1]}, Temperature: {grid_tup[0]}")
    return float(grid_tup[0]), int(grid_tup[1])


def process_input_cot(input_prompt:str,
                 tokenizer:transformers.AutoTokenizer,
                 model:transformers.AutoModelForCausalLM,
                 length:int=10)->torch.Tensor:
    '''
    Function for COT sentiment inference.

    Arguments:
        - input_prompt:  input query string
        - tokenizer:     AutoTokenizer object
        - model:         AutoModelForCausalLM object
        - length:        max length of generation (integer)
    Returns:
        - result:        a generated string
    '''

    # 1st element of chain
    message_1 = [
        {"role": "system", "content":
           "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content":
         "Given the text: '" + input_prompt + 
         "' Which specific aspect is being mentioned? The specific aspect is"}
    ]
    text_1 = tokenizer.apply_chat_template(
        message_1,
        tokenize=False,
        add_generation_prompt=True
    )
    tokens_1 = tokenizer([text_1],
                      return_tensors = 'pt'
                    ).to(model.device)
    generated_ids_1 = model.generate(
        **tokens_1,
        max_new_tokens=length
    )
    generated_ids_1 = [
        output_ids_1[len(input_ids_1):] for input_ids_1,
            output_ids_1 in zip(tokens_1.input_ids, generated_ids_1)
    ]
    response_1 = tokenizer.batch_decode(generated_ids_1, skip_special_tokens=True)[0]

    # 2nd element of chain
    message_2 = [
        {"role": "system", "content":
           "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content":
            "Given the text: '"  + input_prompt + 
            "' and the aspect '" + response_1   + 
            "' Based on common sense, what is the implicit opinion towards this aspect? The opinion is"}
    ]
    text_2 = tokenizer.apply_chat_template(
        message_2,
        tokenize=False,
        add_generation_prompt=True
    )
    tokens_2 = tokenizer([text_2],
                      return_tensors = 'pt'
                    ).to(model.device)
    generated_ids_2 = model.generate(
        **tokens_2,
        max_new_tokens=length
    )
    generated_ids_2 = [
        output_ids_2[len(input_ids_2):] for input_ids_2,
            output_ids_2 in zip(tokens_2.input_ids, generated_ids_2)
    ]
    response_2 = tokenizer.batch_decode(generated_ids_2, skip_special_tokens=True)[0]

    # Final element of chain
    message_3 = [
        {"role": "system", "content":
           "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content":
            "Given the text: '"   + input_prompt + 
            "' the apect '"       + response_1   +
            "' and the opinion '" + response_2   +
            "' What would be the sentiment expressed? Negative or Positive?"}
    ]
    text_3 = tokenizer.apply_chat_template(
        message_3,
        tokenize=False,
        add_generation_prompt=True
    )
    tokens_3 = tokenizer([text_3],
                      return_tensors = 'pt'
                    ).to(model.device)
    generated_ids_3 = model.generate(
        **tokens_3,
        max_new_tokens=length
    )
    generated_ids_3 = [
        output_ids_3[len(input_ids_3):] for input_ids_3,
            output_ids_3 in zip(tokens_3.input_ids, generated_ids_3)
    ]
    response_3 = tokenizer.batch_decode(generated_ids_3, skip_special_tokens=True)[0]

    # We return the COT final output
    return response_3


def mine_response_cot(response:str)->str:
    '''
    With CoT, the models may return complete phrases or short sentences.
    We thus implement a method that infers the polarity of the generated
    string.

    Arguments:
        - response:      generated text
    Returns:
        - label:         predicted string (negative, positive or NaN)
    '''
    if 'positive' in response.lower() or 'good' in response.lower() or 'great' in response.lower():
        return 'positive'
    elif 'negative' in response.lower() or 'bad' in response.lower() or 'terrible' in response.lower():
        return 'negative'
    else:
        return 'NaN'


def get_sentiment_cot(input_prompt:str,
                 tokenizer:transformers.AutoTokenizer,
                 model:transformers.AutoModelForCausalLM,
                 length:int=10)->str:
    '''
    This function combines `mine_response_cot` with
    `process_input_cot`
    
    Arguments:
        - input_prompt:  input query string
        - tokenizer:     AutoTokenizer object
        - model:         AutoModelForCausalLM object
        - length:        max length of generation (integer)
    Returns:
        - result:        string label (positive, negative or NaN)
    '''
    raw_response = process_input_cot(input_prompt, tokenizer, model)
    response = mine_response_cot(raw_response)
    return response


def performance_report(gold:pd.Series, 
                       pred:pd.Series,
                       name:str
                       )->None:
    '''
    Function to display classification report.

    Arguments:
        - gold:      gold values (Series object)
        - pred:      predicted values (Series object)
        - name:      name of the model (string)
    '''
    # Print and serialize the classification report
    result = classification_report(gold, 
                                   pred, 
                                   zero_division=0)
    result_dict = classification_report(gold, 
                                   pred, 
                                   zero_division=0,
                                   output_dict=True)
    df_res = pd.DataFrame(result_dict).transpose()
    df_res.to_csv(os.path.join(dirname, 
                               '../results/results_' + name + '.csv'))
    logger.info(result)


def save_predictions(predictions:pd.DataFrame, 
                     name:str)->None:
    '''
    Function to save predictions.

    Arguments:
        - predictions:      model predictions (DataFrame object)
        - name:             name of the model (string)
    '''
    predictions.to_csv(os.path.join(dirname, 
                '../results/predictions_' + name + '.csv'), index=False)
    logger.info('Saving predictions to disk.')


def confusion_matrix(actual:pd.Series, 
                     predicted:pd.Series,
                     name:str)->None:
    '''
    Function to display confusion matrix plot.

    Arguments:
        - actual:    actual values
        - predicted: predicted values
        - name:      name of the model (string)
    '''
    confusion_matrix = metrics.confusion_matrix(actual, predicted)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,
                display_labels=set(predicted.values))
    cm_display.plot()
    plt.savefig(os.path.join(dirname, 
                               '../results/confusion_' + name + '.png'))
    

def plot_accuracy(scores:pd.DataFrame)->None:
    '''
    Function to display accuracy

    Arguments:
        - scores:    DataFrame of scores
    '''
    plt.bar(scores.model, scores.acc)
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.xticks(range(0,scores.shape[0]+10), rotation=70, fontsize=8)
    plt.margins(0.05)
    plt.title('Accuracy Distribution')
    plt.savefig(os.path.join(dirname, 
                               '../results/accuracy_scores.png'))


def plot_agreement(scores:np.array, 
                   labels:list)->None:
    '''
    Function to display agreement scores

    Arguments:
        - scores:    array of scores
        - labels:    list of labels
    '''
    plt.imshow(scores) 
    plt.yticks(range(0,len(labels)), labels, fontsize=8)
    plt.xticks(range(0,len(labels)), labels, fontsize=8, rotation=90)
    # Loop over data dimensions to paste scores.
    for i in range(len(labels)):
        for j in range(i+1,len(labels)):
            text = plt.text(j, i, "%.2f" % scores[i, j],
                           ha="center", va="center", color="w", fontsize=8)
    plt.margins(0.05)
    plt.title('Model Pairwise Agreement Scores')
    plt.savefig(os.path.join(dirname, 
                               '../results/agreement_scores.png'))
