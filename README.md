# Overview

## Project Goals 
1. Collect large datasets of tweets judged to be partisan
2. Conduct an initial bag-of-words analysis on these datasets
3. Train a transformer NLP model on the collected data
4. Verify the performance of the model and test on other data
5. Create a tool with the model that can be used by a non-programmer

## Part 1
The first part of this project was collecting the data. The notebook that was used to collect the data for the project can be found at `Notebooks and Scripts/tweet_scraping.ipynb`. This notebook used the python package [twint](https://github.com/twintproject/twint) to scrape tweets from 20 politicians. The [govtrack ideology scores](https://www.govtrack.us/congress/members/report-cards/2020/senate/ideology) were used to determine which politicians should be included for both left and right-leaning datasets. After scraping the tweets with twint, they were saved in the form of csv files and can be found at `Results and Data/left_tweets.csv` and `Results and Data/right_tweets.csv`. 

## Part 2
Next, a bag-of-words analysis was conducted on the datasets collected in part 1. The analysis can be found at `Notebooks and Scripts/Bag of Words Analysis`, and is inspired by the work done in [this article](https://towardsdatascience.com/detecting-politically-biased-phrases-from-u-s-senators-with-natural-language-processing-tutorial-d6273211d331) (except using the tweets collected in part 1 instead of speeches). 

Using [scikit-learn's](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) `CountVectorizer`, two more csv datasets were created, which show the most frequent two or three word phrases used in left and right leaning tweets. Here is a sample of those results:
<img width="511" alt="Screen Shot 2021-08-18 at 12 34 01 PM" src="https://user-images.githubusercontent.com/81947750/129960865-490567bf-f161-407b-bae2-c7bb4ff5466a.png">

## Part 3

