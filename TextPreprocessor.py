import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker
import contractions
from transformers import AutoTokenizer

class TextPreprocessor:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('ibm-granite/granite-embedding-30m-english')
        self.lemmatizer = WordNetLemmatizer()
        self.spell = SpellChecker()

    def preprocess_text(self, text):
        """
        Preprocess text by tokenizing and removing stopwords.

        Args:
            text (str): Input text.

        Returns:
            str: Preprocessed text.
        """
        text = contractions.fix(text)
        text = text.lower()
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        tokens = self.tokenizer.tokenize(text)
        tokens = [self.spell.correction(token) for token in tokens]
        tokens = [token for token in tokens if token not in stopwords.words('english')]
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        text = ' '.join(tokens)
        return text