# Climate Change Education Engine

**Author:** Nabil Abbas

“We are the first generation to be able to **end poverty**, and the last generation that can take steps to avoid the worst impacts of **climate change**.  Future generations will judge us harshly if we fail to uphold our **moral and historical** responsibilities.”   -Ban Ki-Moon, Secretary-General United Nations

## Goals

Create a data pipeline to recommend custom content tailored towards a twitter user dependent on their online activity regarding the topic of “Climate Change”.

The recommendation system is advanced because through the use of topic modeling we know what topics climate change deniers discuss and thus what specific content to cater towards them on a user to user basis.

## Tech Stack
- Python
- Pandas
- Textblob
- SpaCy
- NLTK
- Imblearn
- Scikit-Learn
- Matplotlib
- Plotly
- Seaborn
- Twint
- Pushshift


## Strategy

#### Classifier
Use Pushshift Reddit API tool to obtain 5 years of comment text data from two subreddits, **"Climate"** and **"ClimateSkeptics"**. 

The total number of comment text data accumulated to be 173,000 comments.  After running NLP to clean and process the text data, the data is vectorized and fitted to a classifier.

Text from the **ClimateSkeptics** subreddit will fall under our "Climate Change Denier" class.  While text under the **Climate** subreddit will fall under the "Climate Change Believer" class.

***The Climate Change Denying classification will be our Positive class.***

#### Topic Modeling
Using Latent Dirichlet Allocatino (LDA), train a model to create 15 subtopics from the corpus of the subreddit comment text data.
#### Processing Data for Recommender
Use Twint Twitter webscraping tool to run a search on climate change discussions on Twitter and extract 10 climate change tweets per user for 500 users.  After cleaning the text data using NLP, run the trained classifier and topic modeler on the tweets to prepare the data for the recommender.

#### Fitting Recommender
Fit a collaboratively filtered SVD Recommendation Model trained to recommend specific informational content to a Twitter user based on modeled topic sentiment.

## Understanding Our Class Data - Positive Class 

The bar graphs below will highlight the most important indicators for the **positive** classification as well as the the most frequent key words/phrases for the **positive** classification.

![](/images/positive_feature.png) 
![](/images/denier_word_count.png) 

## Understanding Our Class Data - Negative Class 
The bar graphs below will highlight the most important indicators for the **negative** classification as well as the the most frequent key words/phrases for the **negative** classification.

![](/images/negative_feature.png) 
![](/images/climate_word_count.png) 

## Our Model
Following a train test split of our sample data, I vectorized the tweets, using both Scikit-Learn’s Count Vectorizer and TF-IDF Vectorizer.  To optimize classifier performance I tested GridSearchCV to adjust hyperparameters, and TruncatedSVD to reduce dimensionality. ~10 ML classifiers were implemented and Logistic Regression proved to the be the best classifier.

## Model Performance
The **dummy metric** below provides the followed baseline metrics:

| F1  | Recall  | Precision  |  Accuracy |
|-----|---------|------------|-----------|
| 65% | 66%     | 54%        | 65%       |

These metrics are expected because of a class imbalance that still exists despite undersampling.  Additional undersampling methods will be considered a revisitation of the project.

The **Logistic Regression** model with count vectorized data yielded the most desirable results.

| F1  | Recall  | Precision  |  Accuracy |
|-----|---------|------------|-----------|
| 84% | 78%     | 81%        | 78%       |

The predictions from this model were the most desirable as it accurately predicted the majority of the True Positive class. Additionally the false positive classification is preferred over the false negative classification.  The ROC plot and confusion matrix below display the model performance and how the test samples were classified.


![](/images/log_reg_confusion_matrix.png)
![](/images/ROC_AUC_LR.png)

The classifier is not ideal yet because likely due to the unresolved class imbalance, the classifier appears to be **overfit** to the training data.

| F1  | Recall  | Precision  |  Accuracy |
|-----|---------|------------|-----------|
| 96% | 98%     | 94%        | 96%       |

## Topic Modeling

After using LDA to create topics the following were the topics that I came up with.  This will be fine tuned with a future project revisitation.

- Topics Legend
- Climate Beliefs
- Rising Temperatures
- Thought Processes
- Physics
- Atmospheric Changes
- Science
- Global Warming 
- Government Involvement
- Climate Reports
- Energy
- Water
- Article / Link Discussion
- Internet Conversations
- Polar Ice
- Legal

You can view some of the more distinct topic word clouds below:

![](/images/word_cloud.png)

The distribution of the topics amongst the ~5500 tweets grabbed from 500 users are as follows.

![](/images/Tweet_topic_counts.png)

I did not include the distribution of topic appearances on a per class basis because the topics are distributed similarly to the bar graph above. You may find this information in the presentation deck I've also included in my github.


## Takeaways
The classifier, although not fully optimal, serves the purpose of the project task.  Topic Modeling grouped key words into relevant topic groups allowing for the topics to group the Twitter text data.
## Future Consideration
- Finish data wrangling and fit Recommender.
- Find content to recommend for the Recommender.
- Compile Front End using Flask or Streamlit.
- Validate classifier by performing a text analysis project to compare tweets and Reddit comments.
- Explore additional undersampling techniques.
- Test different dimensionality for Count Vectorizer.
- Get additional tweet data for Recommender.

## Sources

- Pushshift.io
- Twint
