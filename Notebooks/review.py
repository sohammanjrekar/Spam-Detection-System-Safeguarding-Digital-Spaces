{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "0e71a8a2-6175-4aea-abd8-f11fa01df8d7",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[nltk_data] Downloading package stopwords to\n",
      "[nltk_data]     C:\\Users\\mrsoh\\AppData\\Roaming\\nltk_data...\n",
      "[nltk_data]   Package stopwords is already up-to-date!\n",
      "[nltk_data] Downloading package punkt to\n",
      "[nltk_data]     C:\\Users\\mrsoh\\AppData\\Roaming\\nltk_data...\n",
      "[nltk_data]   Package punkt is already up-to-date!\n"
     ]
    }
   ],
   "source": [
    "# Import necessary libraries\n",
    "import pandas as pd\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier\n",
    "from sklearn.svm import SVC\n",
    "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score\n",
    "import joblib\n",
    "import pandas as pd\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix\n",
    "from sklearn.model_selection import train_test_split\n",
    "import string\n",
    "from nltk.corpus import stopwords\n",
    "from nltk.tokenize import word_tokenize\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.svm import SVC\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from wordcloud import WordCloud\n",
    "import joblib\n",
    "import pandas as pd\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.ensemble import RandomForestClassifier, VotingClassifier\n",
    "from sklearn.naive_bayes import MultinomialNB\n",
    "from sklearn.svm import SVC\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import joblib\n",
    "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix\n",
    "import pandas as pd\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix\n",
    "from sklearn.model_selection import train_test_split\n",
    "import nltk\n",
    "from nltk.corpus import stopwords\n",
    "from nltk.tokenize import word_tokenize\n",
    "import string\n",
    "from sklearn.ensemble import RandomForestClassifier, VotingClassifier\n",
    "import joblib\n",
    "# Download stopwords and tokenizer data from NLTK\n",
    "nltk.download('stopwords')\n",
    "nltk.download('punkt')\n",
    "\n",
    "# Load NLTK stopwords\n",
    "stop_words = set(stopwords.words('english'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "82b7c797-3265-41df-9665-bd5dfc42ae5e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>User_id</th>\n",
       "      <th>Product_id</th>\n",
       "      <th>Rating</th>\n",
       "      <th>Date</th>\n",
       "      <th>Review</th>\n",
       "      <th>Label</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>923</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>12/8/2014</td>\n",
       "      <td>The food at snack is a selection of popular Gr...</td>\n",
       "      <td>-1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>924</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>5/16/2013</td>\n",
       "      <td>This little place in Soho is wonderful. I had ...</td>\n",
       "      <td>-1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>925</td>\n",
       "      <td>0</td>\n",
       "      <td>4</td>\n",
       "      <td>7/1/2013</td>\n",
       "      <td>ordered lunch for 15 from Snack last Friday. Â...</td>\n",
       "      <td>-1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>926</td>\n",
       "      <td>0</td>\n",
       "      <td>4</td>\n",
       "      <td>7/28/2011</td>\n",
       "      <td>This is a beautiful quaint little restaurant o...</td>\n",
       "      <td>-1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>927</td>\n",
       "      <td>0</td>\n",
       "      <td>4</td>\n",
       "      <td>11/1/2010</td>\n",
       "      <td>Snack is great place for a Â casual sit down l...</td>\n",
       "      <td>-1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   User_id  Product_id  Rating       Date  \\\n",
       "0      923           0       3  12/8/2014   \n",
       "1      924           0       3  5/16/2013   \n",
       "2      925           0       4   7/1/2013   \n",
       "3      926           0       4  7/28/2011   \n",
       "4      927           0       4  11/1/2010   \n",
       "\n",
       "                                              Review  Label  \n",
       "0  The food at snack is a selection of popular Gr...     -1  \n",
       "1  This little place in Soho is wonderful. I had ...     -1  \n",
       "2  ordered lunch for 15 from Snack last Friday. Â...     -1  \n",
       "3  This is a beautiful quaint little restaurant o...     -1  \n",
       "4  Snack is great place for a Â casual sit down l...     -1  "
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\n",
    "# Load the dataset containing labeled messages (spam and non-spam)\n",
    "data = pd.read_csv('data/review.csv')\n",
    "data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "ba108798-b857-453a-beaf-1157205c8dc0",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 359052 entries, 0 to 359051\n",
      "Data columns (total 6 columns):\n",
      " #   Column      Non-Null Count   Dtype \n",
      "---  ------      --------------   ----- \n",
      " 0   User_id     359052 non-null  int64 \n",
      " 1   Product_id  359052 non-null  int64 \n",
      " 2   Rating      359052 non-null  int64 \n",
      " 3   Date        359052 non-null  object\n",
      " 4   Review      359052 non-null  object\n",
      " 5   Label       359052 non-null  int64 \n",
      "dtypes: int64(4), object(2)\n",
      "memory usage: 16.4+ MB\n"
     ]
    }
   ],
   "source": [
    "data.dropna()\n",
    "data.info()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0551333d-22d7-4655-a025-575de7fbca7f",
   "metadata": {},
   "source": [
    "This dataset comprises for reviews about the hotels located in North America. Moreover, this dataset contains the label for a review being either GENUINE (Not-Spam) or FAKE (Spam).\r\n",
    "\r\n",
    "Her a value of 1 in the attribute LABEL is considered as Not-Spam or Genuine reviews. Whereas value -1 means a fake or SPAM review"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "91cc8d37-3b3d-4e78-9e18-2784381a6727",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Text preprocessing function\n",
    "def preprocess_text(text):\n",
    "    # Convert text to lowercase\n",
    "    text = text.lower()\n",
    "    \n",
    "    # Remove punctuation and numbers\n",
    "    text = ''.join([char for char in text if char not in string.punctuation and not char.isdigit()])\n",
    "    \n",
    "    # Tokenization (split text into words)\n",
    "    tokens = word_tokenize(text)\n",
    "    \n",
    "    # Remove stopwords (common words like \"the\", \"and\", \"is\")\n",
    "    tokens = [word for word in tokens if word not in stop_words]\n",
    "    \n",
    "    # Rejoin tokens into a single string\n",
    "    text = ' '.join(tokens)\n",
    "    \n",
    "    return text\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a8fb13a3-68e9-4b07-a2ce-3571f44b958c",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Apply text preprocessing to the 'sms' column\n",
    "data['Review'] = data['Review'].apply(preprocess_text)\n",
    "\n",
    "# Create a TF-IDF vectorizer to convert text data into feature vectors\n",
    "vectorizer = TfidfVectorizer(max_features=1000)\n",
    "X = vectorizer.fit_transform(data['Review'].values)\n",
    "y = data['Label'].values\n",
    "joblib.dump(vectorizer,'reviewTfidfVectorizer.sav')\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9616f35a-1cdd-45c5-9623-f910b857f334",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "# Split the dataset into training and testing sets (70% training, 30% testing)\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ca62e978-4da7-446f-8f27-1c393e483cdc",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "svm = SVC(probability=True,kernel='linear')  \n",
    "\n",
    "svm.fit(X_train, y_train)\n",
    "\n",
    "# Make predictions on the test set for each model\n",
    "svm_predictions = svm.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2de13643-48fe-4248-8fea-8d7e61e44df7",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Calculate evaluation metrics (Accuracy, Precision, Recall, F1-Score, Confusion Matrix) for each model\n",
    "\n",
    "svm_accuracy = accuracy_score(y_test, svm_predictions)\n",
    "svm_precision = precision_score(y_test, svm_predictions)\n",
    "svm_recall = recall_score(y_test, svm_predictions)\n",
    "svm_f1 = f1_score(y_test, svm_predictions)\n",
    "svm_confusion = confusion_matrix(y_test, svm_predictions)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8d09e7de-4a16-4b88-9a55-c7bddfabfae0",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Print evaluation metrics for each individual model\n",
    "\n",
    "\n",
    "print(\"\\nSVM Metrics:\")\n",
    "print(\"Accuracy:\", svm_accuracy)\n",
    "print(\"Precision:\", svm_precision)\n",
    "print(\"Recall:\", svm_recall)\n",
    "print(\"F1-Score:\", svm_f1)\n",
    "print(\"Confusion Matrix:\")\n",
    "print(svm_confusion)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0444e22e-ee9d-445b-81ba-935aa207b1d8",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Visualize confusion matrices using Seaborn\n",
    "def plot_confusion_matrix(cm, labels, title):\n",
    "    plt.figure(figsize=(6, 6))\n",
    "    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)\n",
    "    plt.xlabel('Predicted')\n",
    "    plt.ylabel('Actual')\n",
    "    plt.title(title)\n",
    "\n",
    "plt.figure(figsize=(16, 6))\n",
    "\n",
    "\n",
    "plot_confusion_matrix(svm_confusion, ['Not Spam', 'Spam'], 'SVM')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "572f27c3-c2a6-4213-89c8-39dbdde9ffc1",
   "metadata": {},
   "outputs": [],
   "source": [
    "from wordcloud import WordCloud\n",
    "# Assuming 'sms' column contains your text data\n",
    "text_data = \" \".join(data['Review'])\n",
    "\n",
    "# Create a word cloud\n",
    "wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_data)\n",
    "\n",
    "# Display the word cloud\n",
    "plt.figure(figsize=(10, 5))\n",
    "plt.imshow(wordcloud, interpolation='bilinear')\n",
    "plt.axis('off')\n",
    "plt.title('Word Cloud')\n",
    "plt.show()\n",
    "\n",
    "# Create a bar plot for the most frequent words\n",
    "word_freq = pd.Series(text_data.split()).value_counts()[:20]\n",
    "word_freq.plot(kind='bar', figsize=(12, 6))\n",
    "plt.title('Top 20 Most Frequent Words')\n",
    "plt.xlabel('Words')\n",
    "plt.ylabel('Frequency')\n",
    "plt.xticks(rotation=45)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4f768678-923a-4ac0-8089-e0fe6bcde757",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
