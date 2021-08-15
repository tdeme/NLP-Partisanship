from transformers import TFAutoModel, AutoTokenizer, AutoModelForSequenceClassification, DistilBertForSequenceClassification
from newspaper import Article
from torch import nn

'''
This script can be used to classify the party either of plain text,
or of the text found at a given url.
'''

def prepare_text(article_url, split=True):
    '''
    This function uses the Article object from the newspaper module
    to scrape the webpage at the given url. The split parameter
    determines whether the function will return the text split into
    paragraphs, or simply the original text stripped of newlines.
    (Split is set to True by default).
    '''

    article = Article(article_url)
    article.download()
    article.parse()
    paras = article.text.split('\n\n')
    if split:
        return paras
    else:
        text = ''
        for para in paras:
            text+=para
        return text
  

def get_score(pt_outputs):
    '''
    The pt_predictions are the outputs of the model. To get a more
    concrete prediction, we use the softmax function to get 
    probability-based predictions that are more intuitive to interpret.
    The function returns the average of the probability-based predictions.
    '''

    pt_predictions = nn.functional.softmax(pt_outputs.logits, dim=-1)
    sum = 0
    for prediction in pt_predictions.tolist():
        sum+=prediction[1]
    overall_score = sum/len(pt_predictions)
    return overall_score


def run_tests(user_input, is_url):
    #Much of this syntax comes from the Hugging Face documentation.
    
    my_tokenizer = AutoTokenizer.from_pretrained('../recent_twitter_model')
    my_bert_model = DistilBertForSequenceClassification.from_pretrained('../recent_twitter_model')

    if is_url:
        my_paras = prepare_text(user_input)
    else:
        my_paras = user_input.split('\n\n')

    my_batch = my_tokenizer(
        [para for para in my_paras],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    my_outputs = my_bert_model(**my_batch)

    return get_score(my_outputs)


def main():
    import os

    #If you are hit with an error, you can try uncommenting the line below.
    #os.environ['KMP_DUPLICATE_LIB_OK']='True'

    if eval(input('Would you like to enter a url to the chosen text, or enter the text directly? (1 for url, 2 for direct input) '))==1:
        url = input('Ok, please enter the url: ')
        print(f'The model gave the text a score of {run_tests(url, True)}. ' \
            +'The score should be interpreted as a predicted probability that the text is right-leaning.')

    else:
        text = input('Ok, please enter the text here: ')
        print(f'The model gave the text a score of {run_tests(text, False)}. ' \
            +'The score should be interpreted as a predicted probability that the text is right-leaning.')       


if __name__ == '__main__':
    main()