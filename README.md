# Assignment_2

# Step 1: Importing Libraries
The necessary libraries and functions used in the code are imported:
- Libraries: Pandas, Numpy, Seaborn
- Models: LinearSVC, MultinomialNB, BaggingClassifier and LogisticRegression
- Functions: re, unicodedata, inflect, shuffle, CountVectorizer, TfidfTransformer, WordNetLemmatizer, stopwords, pad_sequences, train_test_split,  and cross_val_score from sklearn

# Step 2: Dateset Upload
Uploading the .json dataset using pandas and exploring the data.

# Step 3: Data Exploration
Remapping data labels and balancing the dataset for a fair model training; to avoid biasing towards a specific labeled category

# Step 4: Data Cleaning and Preprocessing
using NLP methods, the data is cleaned:
- non-ascii characters are removed
- all letters are changed into lowercase
- punctuation marks are removed
- verbs are lemmatized to their original forms
- later on in the code, in the vectorzation step, stopwords are also removed

# Step 5: Train-Test Split
Splitting the Data into a train set, used to train the model, and a test set, used to test the model's performance and accuracy before deployment.

# Step 6: Data Preparation for Model Training
Transforming the data into the form required for it to be compatible with the chosen model's input criteria.
- CountVectorizer is used to transform the data into vectors that are later on transformed into arrays for it to be used as train/test sets in the model.
- Tfidf is used to assign the weights to our words, or features, for accurate analysis and predictions

# Step 7: Training and Testing the Models
the Data preprocessed earlier is used to train the ML Models
Cross-Validation is used to check the model's accuracy with new input data each time and to check for overfitting
the test set, and the predictions are compared to the actual labels to calculate accuracy
