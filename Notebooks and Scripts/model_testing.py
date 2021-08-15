from transformers import TFAutoModel, AutoTokenizer, AutoModelForSequenceClassification, DistilBertForSequenceClassification
from newspaper import Article
from torch import nn

'''
This script will test the model previously trained, and compare
its performance with that of a similar existing model that was 
found on the Hugging Face model hub.
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


def evaluate(score, tally, party):
    '''
    The party parameter was included in the urls object so that a
    0 indicates a left-leaning article and a 1 indicates a right-
    leaning article. If the score predicted the correct leaning,
    a 1 will be appended to the corresponding tally list. Otherwise,
    a 0 will be appended, indicating that the prediction was false.
    '''

    if score<0.5:
        if not party:
            tally.append(1)
        else:
            tally.append(0)
    else:
        if party:
            tally.append(1)
        else:
            tally.append(0)


def run_tests(urls):
    #Much of this syntax comes from the Hugging Face documentation.
    
    my_tokenizer = AutoTokenizer.from_pretrained('../recent_twitter_model')
    my_bert_model = DistilBertForSequenceClassification.from_pretrained('../recent_twitter_model')
    my_tally = []

    control_tokenizer = AutoTokenizer.from_pretrained('spencerh/rightpartisan')
    control_bert_model = DistilBertForSequenceClassification.from_pretrained('spencerh/rightpartisan')
    control_tally = []

    for url in urls:
    
        my_paras = prepare_text(url[0])
        control_paras = prepare_text(url[0], False)

        my_batch = my_tokenizer(
            [para for para in my_paras],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        my_outputs = my_bert_model(**my_batch)

        my_score = get_score(my_outputs)

        evaluate(my_score, my_tally, url[1])

        control_batch = control_tokenizer(
            [para for para in control_paras],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        control_outputs = control_bert_model(**control_batch)

        control_score = get_score(control_outputs)

        evaluate(control_score, control_tally, url[1])

    return my_tally, control_tally


def main():
    import os

    os.environ['KMP_DUPLICATE_LIB_OK']='True'

    urls = (('https://www.cnn.com/2021/03/06/opinions/biden-gop-relief-bill-zelizer/index.html',0),
    ('https://www.cnn.com/2021/03/06/opinions/tweets-gop-zoe-lofgren-ghitis/index.html',0),
    ('https://www.cnn.com/2021/03/05/opinions/pandemic-lessons-preparedness-besser/index.html',0),
    ('https://www.cnn.com/2021/03/04/opinions/joe-bidens-big-chance-sachs/index.html',0),
    ('https://www.cnn.com/2021/03/04/opinions/texas-covid-restrictions-science-mehnert/index.html',0),
    ('https://www.foxnews.com/opinion/afghanistan-leadership-rep-michael-waltz',1),
    ('https://www.foxnews.com/opinion/afghanistan-mission-impossible-biden-fantasyland-endgame-k-t-mcfarland',1),
    ('https://www.foxnews.com/opinion/afghanistan-biden-team-us-pivot-mike-pompeo',1),
    ('https://www.foxnews.com/opinion/gov-cuomo-resigned-cnn-chris-cuomo-tim-graham',1),
    ('https://www.foxnews.com/opinion/biden-broken-border-policies-crisis-arizona-ag-mark-brnovich',1),
    )

    print(run_tests(urls))


if __name__ == '__main__':
    main()
