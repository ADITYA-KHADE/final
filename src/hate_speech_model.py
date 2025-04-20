import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import string
import pickle
import os
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import json
from collections import defaultdict, Counter

# Download required NLTK resources if not already downloaded
try:
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('wordnet', quiet=True)
    nltk.download('stopwords', quiet=True)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load slang dictionary
SLANG_DICT = {
    "af": "as fuck",
    "asl": "as hell",
    "b": "bitch",
    "bf": "boyfriend",
    "bih": "bitch",
    "bruh": "brother",
    "bro": "brother",
    "bs": "bullshit",
    "btw": "by the way",
    "cuz": "because",
    "deadass": "seriously",
    "dm": "direct message",
    "fr": "for real",
    "fam": "family",
    "gf": "girlfriend",
    "hmu": "hit me up",
    "idc": "i don't care",
    "idgaf": "i don't give a fuck",
    "idk": "i don't know",
    "imo": "in my opinion",
    "irl": "in real life",
    "lmao": "laughing my ass off",
    "lmfao": "laughing my fucking ass off",
    "mf": "motherfucker",
    "ngl": "not gonna lie",
    "nigga": "n-word",
    "niggas": "n-word",
    "omg": "oh my god",
    "rn": "right now",
    "smh": "shaking my head",
    "srs": "serious",
    "stfu": "shut the fuck up",
    "tbh": "to be honest",
    "tf": "the fuck",
    "tho": "though",
    "thot": "that ho over there",
    "u": "you",
    "ur": "your",
    "wtf": "what the fuck",
    "ya": "you",
    "yall": "you all",
    "bc": "because",
    "cus": "because",
    "cause": "because",
    "cos": "because",
    "rly": "really",
    "pls": "please",
    "plz": "please",
    "rt": "retweet",
    "w/": "with",
    "w/o": "without",
    "wyd": "what you doing",
    "imma": "i am going to",
    "gonna": "going to",
    "wanna": "want to",
    "gotta": "got to",
    "tryna": "trying to",
    "ima": "i am going to",
    "y'all": "you all",
    "aint": "ain't",
}

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model definition
class LSTM_CNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, lstm_hidden_dim, cnn_hidden_dim, num_classes, dropout=0.5, num_filters=100, filter_sizes=[3, 4, 5]):
        super(LSTM_CNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, lstm_hidden_dim, batch_first=True, bidirectional=True)
        
        # Multiple CNN filters for different n-gram sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(lstm_hidden_dim*2, num_filters, kernel_size=fs, padding=fs//2) 
            for fs in filter_sizes
        ])
        
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)
        
        # Attention mechanism
        self.attention = nn.Linear(lstm_hidden_dim*2, 1)
    
    def forward(self, x):
        # Handle both float and long inputs
        if x.dtype == torch.float32:
            x = x.long()
        
        # Embedding layer
        embedded = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        
        # LSTM layer
        lstm_out, _ = self.lstm(embedded)  # [batch_size, seq_len, lstm_hidden_dim*2]
        
        # Attention mechanism
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        attended_lstm = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Reshape for CNN: [batch_size, lstm_hidden_dim*2, seq_len]
        lstm_out = lstm_out.permute(0, 2, 1)
        
        # Apply multi-filter CNN
        conv_results = []
        for conv in self.convs:
            conv_out = F.relu(conv(lstm_out))  # [batch_size, num_filters, seq_len]
            pooled = self.pool(conv_out).squeeze(-1)  # [batch_size, num_filters]
            conv_results.append(pooled)
        
        # Concatenate all CNN outputs
        concat_output = torch.cat(conv_results, dim=1)  # [batch_size, num_filters * len(filter_sizes)]
        
        # Dropout and final classification
        dropped = self.dropout(concat_output)
        output = self.fc(dropped)  # [batch_size, num_classes]
        
        return output

# Error analysis class
class ErrorAnalyzer:
    def __init__(self, model, vocab):
        self.model = model
        self.vocab = vocab
        self.errors = []
        self.feature_importance = defaultdict(float)
        self.confusion_matrix = [[0, 0], [0, 0]]  # [[TN, FP], [FN, TP]]
        self.class_metrics = {"precision": 0, "recall": 0, "f1": 0}
    
    def analyze_error(self, text, true_label):
        """Analyze a misclassified example"""
        processed_text = preprocess_text(text)
        sequence = text_to_sequence(processed_text, self.vocab, max_len=100)
        sequence_tensor = torch.tensor([sequence], dtype=torch.long).to(device)
        
        with torch.no_grad():
            outputs = self.model(sequence_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        if predicted_class != true_label:
            # Record the error
            error_info = {
                "text": text,
                "processed_text": processed_text,
                "true_label": true_label,
                "predicted_label": predicted_class,
                "confidence": confidence
            }
            self.errors.append(error_info)
            
            # Update confusion matrix
            if true_label == 1 and predicted_class == 0:  # False Negative
                self.confusion_matrix[1][0] += 1
            elif true_label == 0 and predicted_class == 1:  # False Positive
                self.confusion_matrix[0][1] += 1
            
            # Analyze important words in misclassification
            words = processed_text.split()
            for word in words:
                if word in self.vocab and self.vocab[word] > 0:
                    self.feature_importance[word] += 1
            
            return error_info
        return None
    
    def get_error_summary(self, top_n=10):
        """Get a summary of error analysis"""
        if not self.errors:
            return {"message": "No errors recorded"}
        
        # Calculate metrics
        tn, fp = self.confusion_matrix[0]
        fn, tp = self.confusion_matrix[1]
        
        if tp + fp > 0:
            self.class_metrics["precision"] = tp / (tp + fp)
        if tp + fn > 0:
            self.class_metrics["recall"] = tp / (tp + fn)
        if self.class_metrics["precision"] + self.class_metrics["recall"] > 0:
            self.class_metrics["f1"] = 2 * (self.class_metrics["precision"] * self.class_metrics["recall"]) / (self.class_metrics["precision"] + self.class_metrics["recall"])
        
        # Get most common misclassified words
        top_words = dict(sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_n])
        
        # Group errors by type
        fn_examples = [e for e in self.errors if e["true_label"] == 1 and e["predicted_label"] == 0]
        fp_examples = [e for e in self.errors if e["true_label"] == 0 and e["predicted_label"] == 1]
        
        return {
            "total_errors": len(self.errors),
            "false_negatives": len(fn_examples),
            "false_positives": len(fp_examples),
            "metrics": self.class_metrics,
            "top_misclassified_words": top_words,
            "example_false_negatives": fn_examples[:3],
            "example_false_positives": fp_examples[:3]
        }
    
    def save_analysis(self, filepath):
        """Save error analysis to a JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.get_error_summary(), f, indent=2)

# Enhanced preprocess text function
def preprocess_text(text, use_lemmatization=True, handle_slang=True, remove_stopwords=False):
    """
    Advanced text preprocessing with lemmatization and slang handling
    
    Args:
        text (str): Input text to preprocess
        use_lemmatization (bool): Whether to apply lemmatization
        handle_slang (bool): Whether to replace common slang terms
        remove_stopwords (bool): Whether to remove stopwords
        
    Returns:
        str: Preprocessed text
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase first (but keep track of capitalization)
    has_all_caps = bool(re.search(r'\b[A-Z]{3,}\b', text))  # Flag if text has words in ALL CAPS
    exclamation_count = text.count('!')  # Count exclamation marks
    question_count = text.count('?')  # Count question marks
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove mentions (@user)
    text = re.sub(r'@\w+', '', text)
    # Remove hashtags (#hashtag) but keep the text
    text = re.sub(r'#(\w+)', r'\1', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Check for common offensive terms before removing punctuation
    has_profanity = bool(re.search(r'\b(fuck|shit|bitch|ass|damn|cunt|nigga|faggot|retard)\b', text.lower()))
    has_slurs = bool(re.search(r'\b(nigger|fag|spic|kike|chink|tranny|homo)\b', text.lower()))
    
    # Check for religious hate speech patterns
    religious_groups = r'\b(muslim|islam|christian|jew|jewish|hindu|buddhist|sikh|catholic|protestant|mormon|atheist)\w*\b'
    negative_generalizations = r'\b(all|every|always|those|these)\b.{0,30}\b(are|is)\b'
    negative_attributes = r'\b(dangerous|terrorist|evil|threat|bad|violent|radical|extremist|shouldn\'t be trusted|can\'t be trusted)\b'
    
    has_religious_reference = bool(re.search(religious_groups, text.lower()))
    has_negative_generalization = bool(re.search(negative_generalizations, text.lower()))
    has_negative_attribute = bool(re.search(negative_attributes, text.lower()))
    
    # Detecting "All [group] are [negative]" pattern
    has_group_hate_pattern = (has_religious_reference and 
                             (has_negative_generalization or has_negative_attribute))
    
    # Replace repeated characters (e.g., "sooooo" -> "so")
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)  # Keep double letters as they might be intentional
    
    # Save emojis before removing special characters (simplified approach)
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F700-\U0001F77F"  # alchemical symbols
                               u"\U0001F780-\U0001F7FF"  # Geometric Shapes
                               u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                               u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                               u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                               u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                               u"\U00002702-\U000027B0"  # Dingbats
                               u"\U000024C2-\U0001F251" 
                               "]+", flags=re.UNICODE)
    
    emojis = emoji_pattern.findall(text)
    has_emojis = len(emojis) > 0
    
    # Remove punctuation but preserve apostrophes for contractions
    text = text.replace("'", "APOSTROPHE")
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.replace("APOSTROPHE", "'")
    
    # Convert to lowercase
    text = text.lower()
    
    # Handle slang and abbreviations
    if handle_slang:
        words = []
        for word in text.split():
            # Check if word is in slang dictionary
            if word in SLANG_DICT:
                words.append(SLANG_DICT[word])
            else:
                words.append(word)
        text = ' '.join(words)
    
    # Apply lemmatization
    if use_lemmatization:
        words = []
        for word in text.split():
            # Handle contractions
            if "'" in word:
                words.append(word)  # Keep contractions as is
            else:
                words.append(lemmatizer.lemmatize(word))
        text = ' '.join(words)
    
    # Remove stopwords if requested
    if remove_stopwords:
        words = [word for word in text.split() if word.lower() not in stop_words]
        text = ' '.join(words)
    
    # Add back some signals as special tokens
    if has_all_caps:
        text += " _ALLCAPS_"
    if exclamation_count > 3:
        text += " _MANYEXCL_"
    if question_count > 1:
        text += " _MANYQUEST_"
    if has_profanity:
        text += " _PROFANITY_"
    if has_slurs:
        text += " _SLURS_"
    if has_emojis:
        text += " _EMOJI_"
    if has_group_hate_pattern:
        text += " _GROUPHATE_"
    if has_religious_reference and has_negative_attribute:
        text += " _RELIGIOUSHATE_"
        
    return text

# Function to convert text to sequence
def text_to_sequence(text, vocab_dict, max_len=100):
    """Convert text to a sequence of token indices using the vocabulary"""
    tokens = text.split()
    sequence = [vocab_dict.get(token, 0) for token in tokens]  # Use 0 for unknown tokens
    sequence = sequence[:max_len]  # Truncate if longer than max_len
    sequence += [0] * (max_len - len(sequence))  # Pad if shorter than max_len
    return sequence

def load_model_and_vocab(model_path, vocab_path):
    """Load the model and vocabulary from files."""
    try:
        # Load vocabulary
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        
        # Load model
        model_info = torch.load(model_path, map_location=device)
        
        # Check if model has new hyperparameters from tuning
        lstm_hidden_dim = model_info.get('lstm_hidden_dim', 128)
        embed_dim = model_info.get('embed_dim', 100)
        cnn_hidden_dim = model_info.get('cnn_hidden_dim', 128)
        dropout = model_info.get('dropout', 0.5)
        num_filters = model_info.get('num_filters', 100)
        filter_sizes = model_info.get('filter_sizes', [3, 4, 5])
        
        # Create model with parameters from saved file
        model = LSTM_CNN(
            model_info['vocab_size'],
            embed_dim,
            lstm_hidden_dim,
            cnn_hidden_dim,
            model_info['num_classes'],
            dropout,
            num_filters,
            filter_sizes
        )
        
        model.load_state_dict(model_info['state_dict'])
        model.to(device)
        model.eval()
        
        return model, vocab
    except Exception as e:
        print(f"Error loading model or vocabulary: {e}")
        return None, None

def classify_text(model, vocab, text, use_advanced_preprocessing=True):
    """Classify text as hate speech or non-hate speech."""
    if use_advanced_preprocessing:
        processed_text = preprocess_text(text, use_lemmatization=True, handle_slang=True)
    else:
        # Use original preprocessing for backward compatibility
        processed_text = preprocess_text(text, use_lemmatization=False, handle_slang=False)
        
    sequence = text_to_sequence(processed_text, vocab, max_len=100)
    sequence_tensor = torch.tensor([sequence], dtype=torch.long).to(device)
    
    with torch.no_grad():
        outputs = model(sequence_tensor)
        probabilities = F.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    labels = ['Non-Hate Speech', 'Hate Speech']
    return {
        'text': text,
        'processed_text': processed_text,
        'prediction': labels[predicted_class],
        'prediction_code': predicted_class,
        'confidence': confidence,
        'probabilities': probabilities[0].cpu().numpy().tolist()
    }

# Hyperparameter tuning functions
def create_model_with_params(params, vocab_size, num_classes):
    """Create a model with the given hyperparameters"""
    return LSTM_CNN(
        vocab_size=vocab_size,
        embed_dim=params['embed_dim'],
        lstm_hidden_dim=params['lstm_hidden_dim'],
        cnn_hidden_dim=params['cnn_hidden_dim'],
        num_classes=num_classes,
        dropout=params['dropout'],
        num_filters=params['num_filters'],
        filter_sizes=params['filter_sizes']
    )

def save_hyperparameter_tuning_results(results, filepath):
    """Save hyperparameter tuning results to a JSON file"""
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    # Example usage
    model_path = './hate_speech_model_lstm_cnn.pth'
    vocab_path = './vocab.pkl'
    
    if os.path.exists(model_path) and os.path.exists(vocab_path):
        model, vocab = load_model_and_vocab(model_path, vocab_path)
        
        if model and vocab:
            test_texts = [
                "I love how diverse our community is becoming!",
                "I hate all people from that country, they should go back where they came from",
                "The movie was terrible, I hated every minute of it",
                "People of that religion are all terrorists and should be banned",
                "Everyone deserves equal rights regardless of their background"
            ]
            
            # Initialize error analyzer
            error_analyzer = ErrorAnalyzer(model, vocab)
            
            for text in test_texts:
                result = classify_text(model, vocab, text, use_advanced_preprocessing=True)
                print("="*70)
                print(f"Text: {result['text']}")
                print(f"Processed: {result['processed_text']}")
                print(f"Prediction: {result['prediction']} (Confidence: {result['confidence']:.4f})")
                
                # Example error analysis (assuming ground truth for demonstration)
                # In real scenarios, you would get true_label from your labeled dataset
                true_label = 1 if "hate" in text.lower() or "terrorist" in text.lower() else 0
                error_analyzer.analyze_error(text, true_label)
            
            # Save error analysis
            error_summary = error_analyzer.get_error_summary()
            print("\nError Analysis Summary:")
            print(json.dumps(error_summary, indent=2))
