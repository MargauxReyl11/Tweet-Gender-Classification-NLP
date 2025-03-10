#application #task #dataset

Robert Wang, Olivia Joergens, & Margaux Reyl

# Abstract 

Our project aimed to improve the accuracy of machine learning models in predicting user demographics, specifically gender, from Twitter data. By analyzing both tweet content and publicly available user information, we focused on constructing a model that can more accurately determine these demographic traits relative to pre existing methods. Our results indicated that models in which usernames and bios were incorporated in training increased performance relative to those that rely solely on tweet text, while location data was found to introduce noise and did not increase performance. We additionally explored the application of synthetic tweet generation in enhancing model robustness, finding that the addition of synthetic tweets did not improve model performance.  Our findings shed light on how language and user metadata influence demographic classification, posing applications to a wide range of disciplines including social science research, digital advertising, and content personalization. 

# What This Project is About

Social media profiles pose a valuable source for data from which demographic information can be extracted, supporting research across various fields. While many models rely solely on the tweets themselves, this project aimed to explore how classification accuracy can be improved, bias reduced, and model robustness enhanced through the incorporation of additional data beyond tweet content. Using publicly available data, such as profile bios and usernames, we explored how the incorporation of these further data points might further refine the accuracy of classification tasks and mitigate bias in providing a more extensive, comprehensive representation of users across demographic groups. 

In order to develop a model that enhances demographic classification, we integrated these features alongside textual content and reviewed whether they improve classification accuracy and model robustness. We additionally explored how certain data augmentation techniques, specifically synthetic tweet generation, might further improve our model’s performance. 
To achieve this, we used publicly available Twitter datasets that contain labeled demographic information to train our model. Preceding analysis, we preprocessed the data, removing irrelevant content and standardizing text, in order to ensure consistency and accuracy. Our approach involved fine-tuning transformer-based language model, BERT, incorporating an additional classification layer that integrates both tweet content and user profile data. 

The model's performance was evaluated using standard classification metrics such as accuracy, precision, recall, and F1-score. Additionally, we conducted an error analysis to identify patterns in misclassification and assess whether integrating metadata and synthetic data generation leads to meaningful improvements in prediction quality. Our findings help determine whether language models can more effectively classify user demographics when provided with a richer set of features beyond just textual content. Ultimately, this project contributes to a better understanding of the relationship between language, online behavior, and demographic identity, with potential applications in a range of disciplines, including social science research, digital advertising, and content personalization.

# Progress 
We cleaned, trained, and expanded our dataset and models. We first refined our dataset, isolating existing, complete profiles and excluding tweets in which the confidence level of gender classification fell below 100%. We then trained a BERT-based model for demographic classification. At this stage, we only used tweet texts. After establishing this baseline, we began to incorporate user metadata, including profile descriptions, names, and locations and repeated the training process on this enhanced dataset. We additionally generated synthetic tweets and profile metadata using the OpenAI gpt-4o-mini model, allowing us to create additional, diverse data for experimentation.

# Approach 

Our project utilized BERT in two stages. We first fine-tuned a tweet-only model, training BERT using only tweet content as input. The dataset was preprocessed, tokenized using the BERT tokenizer, and split into 80% training and 20% validation sets. The model was trained using a cross-entropy loss to classify tweets as either male, female, or brand. Tweets lacking a gender designation were removed during preprocessing.
After establishing this initial model, we incorporated user metadata, including profile descriptions, names, and locations, and repeated the training process on this enhanced dataset in order to assess whether additional features improved classification tasks. To enhance the model’s performance further, we generated synthetic tweets and profile metadata using OpenAI's gpt-4o-mini to create additional data points that aligned with the preexisting dataset and aided in addressing imbalances.
In order to evaluate model performance, we compared results between two baselines. In the first experiment, the tweet-only BERT model served as a benchmark to determine how well tweet text alone could classify demographics. We would then use the tweet + metadata BERT model to measure whether adding profile information improved classification accuracy. In the second experiment, we augmented the training data with 2,000 synthetic tweets. Afterward, we trained the model on both no metadata and all the metadata used in the first experiment. We then compared the performance of the model with and without fake tweets. We primarily evaluated the model’s performance using accuracy, but we also logged the precision, recall, and F1-score.

# Experiment

## Dataset

This project used the Twitter User Gender Classification, available publicly on Kaggle, containing 20,050 rows of labeled demographic information including gender classification (male, female, and brand), username, bios, location, and various engagement metrics. After preprocessing the dataset by removing all rows that do not have a 100% confidence in gender, as well as rows that did not have a 100% confidence that the profile existed, we ended with a dataset with 13,804 rows. This dataset was appropriate for our task because it provided a wide-range of profile information (further information is not available in publicly available datasets due to privacy restrictions) that can serve to understand what kind of variables are most influential in improving classification accuracy. Using this dataset, we derive demographic classification using tweet content as our baseline, to which we compare additional profile metadata metrics in order to evaluate and further improve our model. For now, we only use 5,000 rows of the dataset to train/validate the model and another 2,500 rows to test the model.

## Evaluation Method

We used standard metrics of classification in measuring model performance: accuracy (reflective of correct classifications across total predictions), precision (reflective of the proportion of correctly predicted positive cases out of all predicted positive cases), recall (reflective of the proportion of actual positive cases correctly identified by the model), and F1-score (reflective of the harmonic mean of precision and recall, balancing both measures).

## Experimental Setup

Our project utilized BERT (Bidirectional Encoder Representations from Transformers) in two stages in order to classify user demographics based on Twitter data. We first fine-tuned a tweet-only model, training BERT using only tweet content as input. The dataset was preprocessed, tokenized using the BERT tokenizer, and split into 80% training and 20% validation sets. The model was then tested on a separate set of tweets outside of training or validation data to obtain further OOS results. Following preprocessing, the model was then trained using a cross-entropy loss function to classify tweets into one of three categories: male, female, or brand (tweets without no gender designation specified are removed during preprocessing). We fine-tuned pretrained BERT using the following specifications:
- Batch Size: 16
- Epochs: 5
- Learning Rate: 2e-5
- Optimizer: AdamW with weight decay
- Loss Function: Cross-entropy loss 
- Training/Validation: 80% training/20% validation
- Hardware: GPU-enabled environment

## Metadata Analysis Setup

For the metadata analysis, we first concatenated the metadata content with the tweet text to create a new column, making sure to label the metadata during concatenation. We then trained the model on the new column with the tweet content and metadata. This process was repeated on the following metadata combinations:
- All metadata (username, bio, location)
- Username
- Bio
- Username and Bio
- Location

## Synthetic Tweets Setup

For the synthetic tweets analysis, we first generated 2,000 fake tweets, which included both tweet content and metadata. We converted the fake tweets into a dataframe, appending it to the 5,000 tweets used in the metadata analysis. The metadata analysis experiment was then run again on the combined 7,000 tweets.

# Results

## Metadata Analysis

<meta charset="utf-8"><b style="font-weight:normal;" id="docs-internal-guid-2c06763c-7fff-6c1a-e6ee-89b3edbf3024"><p dir="ltr" style="line-height:1.38;background-color:#ffffff;margin-top:0pt;margin-bottom:0pt;padding:3pt 0pt 0pt 0pt;"></p><div dir="ltr" style="margin-left:0pt;" align="left">
Metadata Included | In-Sample Accuracy | Out-of-Sample Accuracy
-- | -- | --
Base (no metadata) | 55.10% | 58.00%
All Metadata (username, bio, location) | 79.00% | 60.70%
Username | 71.30% | 74.10%
Bio | 71.00% | 76.00%
Username and Bio | 79.50% | 81.20%
Location | 59.80% | 62.30%
</div></b>

![image](https://github.com/user-attachments/assets/3c2c7203-756c-40ab-8a3a-4042f04ec7c2)


- The base model, which relied solely on tweet text, performed modestly—only slightly better than chance for a three-class problem.
- When used individually, both username and bio improved accuracy noticeably compared to the base model. More importantly, when these two metadata fields are combined, they work synergistically, and improve the out-of-sample accuracy to over 80%. This suggests that the linguistic patterns or personal identifiers in usernames and bios were strong indicators of gender in the dataset.
- In contrast, location metadata performed poorly, performing only slightly above the base model. This implies that location either has a weak or non-existent correlation with gender in the dataset or that its signal is too noisy. Moreover, when location was added to username and bio (resulting in "All Metadata"), the in-sample accuracy is decent (79.0%) but the out-of-sample accuracy drops dramatically to 60.7%. This deterioration strongly suggests that location introduced noise.

## Synthetic Tweets Analysis

<meta charset="utf-8"><b style="font-weight:normal;" id="docs-internal-guid-2c06763c-7fff-6c1a-e6ee-89b3edbf3024"><p dir="ltr" style="line-height:1.38;background-color:#ffffff;margin-top:0pt;margin-bottom:0pt;padding:3pt 0pt 0pt 0pt;"></p><div dir="ltr" style="margin-left:0pt;" align="left">
Metadata Included | In-Sample Accuracy | Out-of-Sample Accuracy
-- | -- | --
Base Model (No metadata) | 65.00% | 61.00%
All Metadata (username, bio, location) | 85.50% | 61.00%
Name | 78.20% | 72.00%
Bio | 78.70% | 74.30%
Name and Bio | 76.90% | 69.60%
Location | 64.60% | 63.30%

![IMG_4570](https://github.com/user-attachments/assets/b753c131-138e-49a1-9842-db57f06397c2)

- In the base model and the model with location metadata, the model with fake tweets included performed a few percentage points better.
- In every other model, the model with fake tweets either had no improvement or decreased the performance of the model.
- Thus, we can conclude that adding synthetic data does not appear to meaningfully improve model performance.
- This could indicate that the synthetic tweets were not a good representation of actual tweets and added unnecessary noise to the data.
- Another possibility is that we would need significantly more synthetic tweets to improve the model performance.

## Performance by Gender

No Fake Tweets:

<google-sheets-html-origin><!--td {border: 1px solid #cccccc;}br {mso-data-placement:same-cell;}-->
male |   |  
-- | -- | --
Precision | Recall | F1-Score
0.585 | 0.648 | 0.614
  |   |  
female |   |  
Precision | Recall | F1-Score
0.684 | 0.642 | 0.662
  |   |  
brand |   |  
Precision | Recall | F1-Score
0.819 | 0.788 | 0.803



Fake Tweets Included:


<google-sheets-html-origin><!--td {border: 1px solid #cccccc;}br {mso-data-placement:same-cell;}-->
male |   |  
-- | -- | --
Precision | Recall | F1-Score
0.613 | 0.641 | 0.626
  |   |  
female |   |  
Precision | Recall | F1-Score
0.693 | 0.669 | 0.681
  |   |  
brand |   |  
Precision | Recall | F1-Score
0.827 | 0.821 | 0.822



- Across all models, the models on average perform worse across all metrics in identifying male tweets. The only exception is the recall of male tweets compared to female tweets in the metadata only model (0.648 vs 0.642).
- Across all models, the models on average perform the best in recognizing brands across all metrics.
- The consistent lower performance in identifying male tweets (with a minor exception in recall in the metadata-only model) suggests that there might be intrinsic differences in the language use or metadata characteristics between male and female users. This finding could point to an underlying bias in the data or the models, warranting further investigation.
- The models’ high performance in recognizing brands across all metrics indicates that language or metadata patterns used by brands are likely more distinct and consistent compared to individual user tweets. This might also imply that demographic prediction tasks for brands could be inherently easier.

# Limitations
We encountered several limitations throughout our conducting of this project: 1) hardware limitations restricted memory, preventing training on the full dataset, 2) dataset-defined genders limited classification to male, female, and "brand"; non-defined genders were removed during preprocessing, and 3) no existing baseline models were found, so we had to create custom baselines, limiting the robustness of our analysis.


# Future Steps

Further work on this project would include retraining the model using a significantly larger dataset to enhance its generalizability and overall performance. While we trained our model on 5,000 tweets, expanding the dataset could provide deeper insights into model accuracy and robustness. Additionally, incorporating a more diverse range of gender identifications and other available demographic metrics, like age, while maintaining privacy considerations would allow for a more representative and inclusive approach to demographic classification. 
Further analysis of metadata and metadata combinations, such as profile pictures and other publicly available engagement metrics, could further improve classification accuracy and reveal additional influential features. Exploring these factors would help refine the model’s predictive capabilities and enable us to assess how different sources of metadata contribute to demographic classification.
