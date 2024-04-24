# -ML-SVM-from-scratch
This is a repository for guidance on how to use Support Vector Machines using Maths only.
## DATA DESCRIPTION
The dataset has been downloaded from the UCI Machine Learning Repository. It offers statistics on 39797 articles published by Mashable over a period of two years. The dataset includes quantitative content elements and metadata for online news stories, with the target variable 'shares' representing the number of times the piece was shared on social media. Because the article's original content cannot be shared outside of Mashable, the dataset provides information such as the number of words, links, photos, polarity and subjectivity measures, and so on that accurately represent the articles.
The dataset link is: https://archive.ics.uci.edu/dataset/332/online+news+popularity

##DATA PRE-PROCESSING 

In the data pre-processing stage, we conducted several steps to enhance the quality and relevance of the dataset.
Null Value Check: Initially, we inspected the dataset for any null values. The data set seems to be clean with no null values in any of the columns.

Dropping Irrelevant Columns: We identified and removed columns deemed irrelevant for analysis, such as 'URL' and 'timedelta', as they didn't contribute significantly to our objectives. Additionally, we consolidated multiple columns representing weekday numbers into a single column, which was subsequently ordinal encoded.

Dropping highly correlated columns: Next, the pairs of highly correlated columns in the dataset were identified with a cut-off of 0.70. Below is the list of highly correlated columns.


To decide which of the two columns in each pair is to be dropped, we utilized outlier detection and feature interpretability in a real-world scenario.
Handling Outliers: To address outliers, particularly among highly correlated columns, we opted to drop those with a greater number of outliers. However, exceptions were made for certain columns like 'data_channel_is_bus', 'data_channel_is_world', and 'data_channel_is_tech', which possess categorical distributions (0 or 1) and hold interpretive significance in real-world scenarios.

## Skewed Distribution Treatment: Recognizing right-skewed distributions among continuous variables, we further examined the distribution of highly skewed features. We noted that the 'number of shares' distribution exhibited heavy right skewness. 
We applied the cubed root transformation to a dataset, particularly to variables with right-skewed distributions to make data more symmetrical and approximately normally distributed. Log transformation was not applied due to the presence of 0 values in the dataset. Data transformations will handle the outliers to some extent instead of dropping them.  This avoids the loss of data.

### kw_min_max_spread: This feature represents the spread between the minimum and maximum occurrences of keywords in the content. It indicates the range of keyword occurrences, which may reflect the diversity or focus of the content. 
### kw_avg_spread: Similar to kw_min_max_spread, this feature represents the spread between the average minimum and maximum keyword occurrences. It provides another perspective on the distribution of keyword occurrences in the content. 
### num_links_to_content_length: This feature represents the ratio of the number of links to other content (e.g., articles, websites) in the content to the content length. It reflects the richness of external references in the content. 
### num_self_links_to_content_length: This feature represents the ratio of the number of links to other content within the same website or domain to the content length. It indicates the internal linking structure within the content. 
### average_token_length_times_num_keywords: This feature represents the product of the average token length in the content and the number of keywords. It provides a measure of the overall complexity or informativeness of the content.
### average_lda_topic_score: This feature represents the average score of two LDA topics (LDA_01 and LDA_03). LDA (Latent Dirichlet Allocation) is a topic modeling technique used to discover the topics present in a corpus of text. This feature reflects the overall topic distribution in the content. 
### average_positive_polarity: This feature represents the average positive polarity score, which is a measure of the sentiment polarity (positivity) of the content. It indicates the overall positivity of the content. 
### average_negative_polarity: This feature represents the average negative polarity score, which is a measure of the sentiment polarity (negativity) of the content. It indicates the overall negativity of the content.

# Support Vector Machine 


## Sampling: As SVM’s Dual optimization requires a lot of computational power we have used samples of the data. Converting class labels 0 to -1, and making sure there is no imbalance in data by sampling the +1 and -1 classes equally. Due to computational limitations, we have used only 6000 records from the original data with an 80:20 train-test split ratio.
## Class Initialization: The KernelSVM class is initialized with the training and testing data, as well as parameters like the kernel function, regularization parameter (C), and the maximum number of iterations.
The training and testing data are stored as instance variables, with the target variable 'popularity' separated from the feature variables.

## Kernel Function: The kernel_function method implements the kernel function used in the Kernel SVM. It supports linear, polynomial, and RBF (Radial Basis Function) kernels.
The appropriate kernel function is selected based on the kernel parameter provided during class initialization.
## Kernel SVM Optimization: The fit method performs the Kernel SVM optimization using the Sequential Minimal Optimization (SMO) algorithm.It initializes the Lagrange multipliers (alpha) with random values between -C and C, and the bias term (b) to 0. The optimization process iterates until the maximum number of iterations is reached or the number of changed Lagrange multipliers is less than the number of training samples.
During each iteration, the method updates the Lagrange multipliers and the bias term based on the SMO algorithm.
The cost history is stored during the training process for visualization purposes.
## Cost Function Computation: The compute_cost method calculates the cost function for the Kernel SVM, which is the sum of the hinge loss and the regularization term.
## Prediction: The predict method uses the trained Kernel SVM model to predict the class labels for new input data.
It computes the decision function based on the Lagrange multipliers, bias term, and the kernel function, and returns the sign of the decision function as the predicted class labels.
Metric Calculation: The metric_calculation method computes the classification metrics (accuracy, precision, recall, F1-score) for the Kernel SVM model.
## Classification Report: The classification_report method is responsible for the overall evaluation of the Kernel SVM model. It calls the predict method to obtain the predictions on the training and testing data and then calculates the classification metrics using the metric_calculation method.
It also calls the plotting methods to visualize the cost function, decision boundary, and confusion matrix for both the training and testing data.


