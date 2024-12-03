import transformers
import os


from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score as accuracy
import pandas as pd
import sklearn.metrics as metrics


# We mute deprecation warnings
import warnings
warnings.filterwarnings("ignore")


# Get current directory
dirname = os.path.dirname(__file__)


def corpus_stats(corpus:list[str])->tuple[int,float,int,int,int]:
    '''
    Corpus statistics (whitespace tokenization)
    
    Arguments:
        - corpus: list of strings
    Returns:
        - number of sentences
        - average number of tokens per sentence
        - total number of words
        - max number of tokens per sentence
        - min number of tokens per sentence
    '''
    tot_sents  = len(corpus)
    n_tokens   = []
    tot_words  = 0
    for sent in corpus:
        n_tokens.append(len(sent.split(' ')))
        tot_words += len(sent.split(' '))
    # Print stats
    print(f'Number of sentences:                  {tot_sents}')
    print(f'Average number of words per sentence: {sum(n_tokens)/tot_sents}')
    print(f'Max number of words per sentence:     {max(n_tokens)}')
    print(f'Min number of words per sentence:     {min(n_tokens)}')
    print(f'Total number of words:                {tot_words}')
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
        - tensor
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
        - string (label)
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
        - tensor
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
        - logit (float)
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
        - string (sentiment label)
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
        - tensor
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
    Function for generating responses, with a variable storing in-context examples (K-shot learning).

    Arguments:
        - input_prompt:  query string
        - tokenizer:     AutoTokenizer object
        - model:         AutoModelForCausalLM object
        - in_context:    string with K examples
    Returns:
        - string (label)
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
    Generates K positive and K negative examples (shortest examples) from a given dataset (K-shot learning).

    Inputs:
        - dataset:         a datasets.DatasetDict object
        - num_per_class:   an integer (count of examples per class)
    Returns:
        - a string with K positive and K negative examples (shortest examples)
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
    for index, row in df_pos.iterrows():
        result = result + str(row['review']) + " Positive.\n"
    for index, row in df_neg.iterrows():
        result = result + str(row['review']) + " Negative.\n"
    # We return the final context a single string:
    return result


def generate_sample_data(data:dict, num_per_class:int)->str:
    '''
    Inputs:
        - dataset:         a dict object
        - num_per_class:   an integer (count of examples per class)
    Returns:
        - a dataframe
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
    Function for generating responses

    Arguments:
        - input_prompt:  query string
        - tokenizer:     AutoTokenizer object
        - model:         AutoModelForCausalLM object
        - top_k:         top k value (float)
        - temp:          temperature value (float)
        - length:        integer used to cap the length of the output sequence (1 token or word by default)
    Returns:
        - string (label)
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
        - dictionary of scores
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
                               '../plots/results_' + name + '.csv'))
    print(result)


def confusion_matrix(actual:pd.Series, 
                    predicted:pd.Series, 
                    name:str)->None:
    '''
    Function to display confusion matrix plot.

    Arguments:
        - actual:    actual values
        - predicted: predicted values
        - name:      name of the model
    '''
    confusion_matrix = metrics.confusion_matrix(actual, predicted)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,
                display_labels=set(predicted.values))
    cm_display.margins(0.2)
    cm_display.title(f"Confusion Matrix for {name}")
    cm_display.savefig(os.path.join(dirname, 
                                    '../plots/confusion_atrix' + name + '.png'))
    cm_display.close()