import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
import matplotlib.colors as mcolors
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
import re
import nltk
from nltk.corpus import stopwords
import io
import base64
from hate_speech_model import preprocess_text
from comprehensive_hate_speech import classify_text_with_comprehensive_detection, get_category_probabilities

# Download necessary NLTK resources
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    
class HateSpeechVisualizer:
    """Class for visualizing hate speech datasets and model performance"""
    
    def __init__(self, model=None, vocab=None):
        """Initialize the visualizer with optional model and vocabulary"""
        self.model = model
        self.vocab = vocab
        self.stopwords = set(stopwords.words('english'))
        self.stopwords.update(['rt', 'http', 'https', 'amp', 'com', 'co', 'www'])
        self.colors = list(mcolors.TABLEAU_COLORS.values())
        
        # Add explicit terms to filter out of non-hate speech visualizations
        self.explicit_terms = [
            'fuck', 'fucking', 'fucked', 'fucker', 'motherfucker', 
            'pussy', 'cunt', 'dick', 'cock', 'penis', 'vagina', 
            'ass', 'asshole', 'shit', 'bullshit', 'crap',
            'tits', 'boobs', 'breasts', 'cum', 'jizz', 'semen',
            'sex', 'sexual', 'sexy', 'horny', 'masturbate',
            'oral', 'anal', 'bitch', 'bastard', 'damn'
        ]
        self.explicit_terms_set = set(self.explicit_terms)
        
    def load_data(self, data_path):
        """Load dataset from CSV file"""
        print(f"Loading dataset from {data_path}...")
        
        try:
            # Load the CSV file
            df = pd.read_csv(data_path, encoding='latin1')
            
            # Check if this is Dataset_3.csv format (has 'class', 'tweet', etc.)
            if 'class' in df.columns and 'tweet' in df.columns:
                df['label'] = df['class'].apply(lambda x: 1 if x == 0 else 0)
                df_clean = df[['tweet', 'label']].rename(columns={'tweet': 'text'})
                
                # Add hate speech type columns if available
                if 'hate_speech' in df.columns and 'offensive_language' in df.columns and 'neither' in df.columns:
                    df_clean['hate_speech_votes'] = df['hate_speech']
                    df_clean['offensive_language_votes'] = df['offensive_language']
                    df_clean['neither_votes'] = df['neither']
                    df_clean['total_votes'] = df['count']
            else:
                # Handle other dataset formats
                if 'text' not in df.columns and 'tweet' in df.columns:
                    df = df.rename(columns={'tweet': 'text'})
                
                if 'label' not in df.columns:
                    if 'class' in df.columns:
                        df['label'] = df['class'].apply(lambda x: 1 if x == 0 else 0)
                    else:
                        raise ValueError("Cannot find class or label column in dataset")
                
                df_clean = df[['text', 'label']]
            
            print(f"Dataset loaded with {len(df_clean)} samples")
            return df_clean
        
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None
    
    def plot_class_distribution(self, df, save_path=None):
        """Plot the distribution of hate speech vs non-hate speech in the dataset"""
        plt.figure(figsize=(10, 6))
        
        # Create class distribution
        class_counts = df['label'].value_counts().sort_index()
        class_names = ['Hate Speech', 'Non-Hate Speech'] if len(class_counts) == 2 else ['Hate Speech', 'Offensive', 'Neither']
        
        # Create bar chart
        ax = sns.barplot(x=class_names, y=class_counts.values, palette='viridis')
        
        # Add count labels on top of bars
        for i, count in enumerate(class_counts.values):
            ax.text(i, count + 50, f"{count} ({count/sum(class_counts.values):.1%})", 
                   ha='center', va='bottom', fontweight='bold')
        
        plt.title('Distribution of Classes in Dataset', fontsize=16)
        plt.ylabel('Count', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            return save_path
        else:
            # Convert plot to base64 string for embedding in HTML
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png')
            img_buf.seek(0)
            img_str = base64.b64encode(img_buf.read()).decode('utf-8')
            plt.close()
            return img_str
    
    def plot_word_frequency(self, df, class_column='label', class_value=1, top_n=20, save_path=None):
        """Plot most frequent words for a specific class"""
        # Filter for the target class
        target_texts = df[df[class_column] == class_value]['text'].tolist()
        
        # Process texts and count words
        all_words = []
        for text in target_texts:
            clean_text = preprocess_text(text)
            words = clean_text.split()
            
            # For non-hate speech, also filter out explicit terms
            if class_value == 0:  # Non-hate speech class
                all_words.extend([w for w in words if w.lower() not in self.stopwords 
                                 and len(w) > 2 
                                 and w.lower() not in self.explicit_terms_set])
            else:
                all_words.extend([w for w in words if w.lower() not in self.stopwords and len(w) > 2])
        
        # Count and get top words
        word_counts = Counter(all_words)
        top_words = dict(word_counts.most_common(top_n))
        
        # Plot
        plt.figure(figsize=(12, 8))
        plt.bar(top_words.keys(), top_words.values(), color=self.colors[:len(top_words)])
        plt.xticks(rotation=45, ha='right')
        plt.title(f'Top {top_n} Words in {"Hate Speech" if class_value == 1 else "Non-Hate Speech"} Texts', fontsize=16)
        plt.ylabel('Frequency', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            return save_path
        else:
            # Convert plot to base64 string for embedding in HTML
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png')
            img_buf.seek(0)
            img_str = base64.b64encode(img_buf.read()).decode('utf-8')
            plt.close()
            return img_str
    
    def create_word_cloud(self, df, class_column='label', class_value=1, save_path=None):
        """Generate a word cloud for specific class of text"""
        # Filter for the target class
        target_texts = ' '.join(df[df[class_column] == class_value]['text'].tolist())
        
        # Clean text
        words = preprocess_text(target_texts).split()
        
        # For non-hate speech, filter out explicit terms
        if class_value == 0:  # Non-hate speech class
            filtered_words = [word for word in words 
                             if word.lower() not in self.stopwords 
                             and len(word) > 2
                             and word.lower() not in self.explicit_terms_set]
        else:
            filtered_words = [word for word in words 
                             if word.lower() not in self.stopwords and len(word) > 2]
            
        clean_text = ' '.join(filtered_words)
        
        # Create wordcloud with explicit font path for Windows systems
        try:
            # Try using a common Windows font
            wordcloud = WordCloud(
                background_color='white',
                max_words=150,
                max_font_size=80,
                width=800,
                height=400,
                contour_width=1,
                contour_color='steelblue',
                colormap='viridis',
                font_path=r'C:\Windows\Fonts\Arial.ttf'  # Explicit Windows font path
            ).generate(clean_text)
        except (OSError, FileNotFoundError):
            try:
                # Fallback to another common font
                wordcloud = WordCloud(
                    background_color='white',
                    max_words=150,
                    max_font_size=80,
                    width=800,
                    height=400,
                    contour_width=1,
                    contour_color='steelblue',
                    colormap='viridis',
                    font_path=r'C:\Windows\Fonts\Verdana.ttf'  # Alternative font
                ).generate(clean_text)
            except (OSError, FileNotFoundError):
                # Last resort, try without specifying font path but catch any errors
                try:
                    wordcloud = WordCloud(
                        background_color='white',
                        max_words=150,
                        max_font_size=80,
                        width=800,
                        height=400,
                        contour_width=1,
                        contour_color='steelblue',
                        colormap='viridis'
                    ).generate(clean_text)
                except Exception as e:
                    print(f"WordCloud generation failed: {e}")
                    # Create a dummy image with error message
                    plt.figure(figsize=(16, 8))
                    plt.text(0.5, 0.5, "WordCloud generation failed: Font not found", 
                            fontsize=16, ha='center', va='center')
                    plt.axis('off')
                    plt.tight_layout()
                    
                    # Convert plot to base64 string for embedding in HTML
                    img_buf = io.BytesIO()
                    plt.savefig(img_buf, format='png')
                    img_buf.seek(0)
                    img_str = base64.b64encode(img_buf.read()).decode('utf-8')
                    plt.close()
                    return img_str
        
        # Plot
        plt.figure(figsize=(16, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud for {"Hate Speech" if class_value == 1 else "Non-Hate Speech"} Texts', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            return save_path
        else:
            # Convert plot to base64 string for embedding in HTML
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png')
            img_buf.seek(0)
            img_str = base64.b64encode(img_buf.read()).decode('utf-8')
            plt.close()
            return img_str
    
    def plot_text_length_distribution(self, df, class_column='label', save_path=None):
        """Plot distribution of text lengths by class"""
        # Add text length column
        df['text_length'] = df['text'].apply(lambda x: len(x.split()))
        
        # Plot
        plt.figure(figsize=(12, 6))
        
        # Use class names as labels
        labels = {0: 'Non-Hate Speech', 1: 'Hate Speech'}
        
        # Create violin plot
        ax = sns.violinplot(x=class_column, y='text_length', data=df, 
                           palette='viridis', inner='quartile')
        
        # Overlay box plot for better visualization of quartiles
        sns.boxplot(x=class_column, y='text_length', data=df, 
                   width=0.15, color='white', ax=ax)
        
        # Add jitter points for individual data points
        sns.stripplot(x=class_column, y='text_length', data=df.sample(min(1000, len(df))), 
                     size=3, color='black', alpha=0.3, ax=ax)
        
        plt.title('Distribution of Text Lengths by Class', fontsize=16)
        plt.ylabel('Number of Words', fontsize=14)
        plt.xlabel('Class', fontsize=14)
        
        # Replace numeric labels with text
        plt.xticks(range(len(labels)), [labels[i] for i in sorted(labels.keys())])
        
        # Add textbox with statistics
        stats_text = '\n'.join([
            f"{labels[cls]} Avg Length: {df[df[class_column]==cls]['text_length'].mean():.1f} words" 
            for cls in sorted(labels.keys())
        ])
        plt.annotate(stats_text, xy=(0.05, 0.95), xycoords='axes fraction',
                   bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8),
                   va='top', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            return save_path
        else:
            # Convert plot to base64 string for embedding in HTML
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png')
            img_buf.seek(0)
            img_str = base64.b64encode(img_buf.read()).decode('utf-8')
            plt.close()
            return img_str
    
    def plot_bigram_comparison(self, df, class_column='label', top_n=15, save_path=None):
        """Compare top bigrams between hate speech and non-hate speech"""
        # Process texts for each class
        texts_by_class = {
            'Hate Speech': df[df[class_column] == 1]['text'].tolist(),
            'Non-Hate Speech': df[df[class_column] == 0]['text'].tolist(),
        }
        
        # Function to get bigrams
        def get_top_bigrams(texts, n=top_n):
            # Join and preprocess
            all_text = ' '.join([preprocess_text(text) for text in texts])
            
            # Create bigrams
            words = all_text.split()
            bigrams = [' '.join(words[i:i+2]) for i in range(len(words)-1)]
            
            # Count and return top bigrams
            bigram_counts = Counter(bigrams)
            return dict(bigram_counts.most_common(n))
        
        # Get bigrams for each class
        bigrams_by_class = {cls: get_top_bigrams(texts) for cls, texts in texts_by_class.items()}
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        for i, (cls, bigrams) in enumerate(bigrams_by_class.items()):
            axes[i].barh(list(reversed(list(bigrams.keys()))), 
                        list(reversed(list(bigrams.values()))), 
                        color=self.colors[i])
            axes[i].set_title(f'Top {top_n} Bigrams in {cls}', fontsize=16)
            axes[i].set_xlabel('Frequency', fontsize=14)
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            return save_path
        else:
            # Convert plot to base64 string for embedding in HTML
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png')
            img_buf.seek(0)
            img_str = base64.b64encode(img_buf.read()).decode('utf-8')
            plt.close()
            return img_str
    
    def plot_category_distribution(self, df, sample_size=200, save_path=None):
        """Plot the distribution of different hate speech categories using the comprehensive model"""
        if self.model is None:
            print("Model not loaded. Cannot categorize hate speech.")
            return None
        
        # Select a random sample (or all if fewer than sample_size)
        if len(df) > sample_size:
            sample_df = df.sample(sample_size, random_state=42)
        else:
            sample_df = df
        
        # Get category probabilities for each text
        categories = [
            'Religious', 'Racial/Ethnic', 'Gender-based', 
            'LGBTQ+', 'Disability-related', 'Nationality/Immigrant'
        ]
        
        # Initialize array to store category probabilities
        all_probs = np.zeros((len(sample_df), len(categories)))
        
        # Process each text
        for i, text in enumerate(sample_df['text']):
            try:
                # Get category probabilities
                cat_probs = get_category_probabilities(text)
                
                # Store in array
                for j, cat in enumerate(categories):
                    all_probs[i, j] = cat_probs.get(cat, 0)
            except Exception as e:
                print(f"Error processing text {i}: {e}")
        
        # Calculate mean probabilities for each category
        mean_probs = np.mean(all_probs, axis=0)
        
        # Calculate standard deviation
        std_probs = np.std(all_probs, axis=0)
        
        # Plot
        plt.figure(figsize=(14, 8))
        
        # Create bar chart with error bars
        plt.bar(categories, mean_probs, yerr=std_probs, capsize=10, 
               color='purple', alpha=0.7, ecolor='black')
        
        # Add value labels on top of bars
        for i, v in enumerate(mean_probs):
            plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
        
        plt.title('Average Probability of Different Hate Speech Categories', fontsize=16)
        plt.ylabel('Average Probability', fontsize=14)
        plt.ylim(0, 1.0)
        plt.xticks(rotation=30, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            return save_path
        else:
            # Convert plot to base64 string for embedding in HTML
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png')
            img_buf.seek(0)
            img_str = base64.b64encode(img_buf.read()).decode('utf-8')
            plt.close()
            return img_str
    
    def plot_dataset_comparison(self, dataset_paths, save_path=None):
        """Compare hate speech statistics across different datasets"""
        # Load all datasets
        datasets = {}
        for path in dataset_paths:
            name = path.split('/')[-1].replace('.csv', '')
            df = self.load_data(path)
            if df is not None:
                datasets[name] = df
        
        if not datasets:
            print("No valid datasets found")
            return None
        
        # Extract statistics
        stats = {
            'Dataset': [],
            'Total Samples': [],
            'Hate Speech %': [],
            'Avg Text Length': [],
        }
        
        for name, df in datasets.items():
            stats['Dataset'].append(name)
            stats['Total Samples'].append(len(df))
            
            # Calculate hate speech percentage
            hate_percent = (df['label'] == 1).mean() * 100
            stats['Hate Speech %'].append(hate_percent)
            
            # Calculate average text length
            df['text_length'] = df['text'].apply(lambda x: len(x.split()))
            stats['Avg Text Length'].append(df['text_length'].mean())
        
        # Convert to DataFrame
        stats_df = pd.DataFrame(stats)
        
        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # Plot hate speech percentage
        sns.barplot(x='Dataset', y='Hate Speech %', data=stats_df, ax=axes[0], palette='viridis')
        axes[0].set_title('Percentage of Hate Speech by Dataset', fontsize=16)
        axes[0].set_ylabel('Percentage (%)', fontsize=14)
        axes[0].set_ylim(0, 100)
        
        # Add value labels
        for i, v in enumerate(stats_df['Hate Speech %']):
            axes[0].text(i, v + 2, f"{v:.1f}%", ha='center', fontweight='bold')
        
        # Plot average text length
        sns.barplot(x='Dataset', y='Avg Text Length', data=stats_df, ax=axes[1], palette='viridis')
        axes[1].set_title('Average Text Length by Dataset', fontsize=16)
        axes[1].set_ylabel('Average Number of Words', fontsize=14)
        
        # Add value labels
        for i, v in enumerate(stats_df['Avg Text Length']):
            axes[1].text(i, v + 1, f"{v:.1f}", ha='center', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            return save_path
        else:
            # Convert plot to base64 string for embedding in HTML
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png')
            img_buf.seek(0)
            img_str = base64.b64encode(img_buf.read()).decode('utf-8')
            plt.close()
            return img_str
    
    def plot_confidence_distribution(self, df, sample_size=100, save_path=None):
        """Plot distribution of model confidence scores for predictions"""
        if self.model is None:
            print("Model not loaded. Cannot generate confidence scores.")
            return None
            
        # Select a random sample (or all if fewer than sample_size)
        if len(df) > sample_size:
            sample_df = df.sample(sample_size, random_state=42)
        else:
            sample_df = df
        
        # Get predictions and confidence scores
        predictions = []
        confidence_scores = []
        true_labels = sample_df['label'].tolist()
        
        for text in sample_df['text']:
            try:
                result = classify_text_with_comprehensive_detection(self.model, self.vocab, text)
                pred_label = 1 if result['prediction'] == 'Hate Speech' else 0
                predictions.append(pred_label)
                confidence_scores.append(result['confidence'])
            except Exception as e:
                print(f"Error processing text: {e}")
                predictions.append(None)
                confidence_scores.append(None)
        
        # Create DataFrame for plotting
        results_df = pd.DataFrame({
            'True Label': true_labels,
            'Predicted Label': predictions,
            'Confidence': confidence_scores
        }).dropna()
        
        results_df['Correct'] = results_df['True Label'] == results_df['Predicted Label']
        results_df['True Label'] = results_df['True Label'].map({0: 'Non-Hate Speech', 1: 'Hate Speech'})
        
        # Plot
        plt.figure(figsize=(14, 8))
        
        # Create violin plots
        sns.violinplot(x='True Label', y='Confidence', hue='Correct', 
                     data=results_df, palette=['salmon', 'mediumseagreen'], 
                     inner='quartile', split=True)
        
        plt.title('Distribution of Model Confidence Scores', fontsize=16)
        plt.ylabel('Confidence Score', fontsize=14)
        plt.xlabel('True Label', fontsize=14)
        plt.ylim(0, 1.05)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add legend
        plt.legend(title='Prediction', labels=['Incorrect', 'Correct'])
        
        # Add statistics as text box
        correct_avg = results_df[results_df['Correct']]['Confidence'].mean()
        incorrect_avg = results_df[~results_df['Correct']]['Confidence'].mean()
        
        stats_text = f"Avg. Confidence:\nCorrect Predictions: {correct_avg:.2f}\nIncorrect Predictions: {incorrect_avg:.2f}"
        plt.annotate(stats_text, xy=(0.05, 0.95), xycoords='axes fraction',
                   bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8),
                   va='top', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            return save_path
        else:
            # Convert plot to base64 string for embedding in HTML
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png')
            img_buf.seek(0)
            img_str = base64.b64encode(img_buf.read()).decode('utf-8')
            plt.close()
            return img_str
    
    def visualize_text_embeddings(self, df, sample_size=1000, method='TSNE', save_path=None):
        """Visualize text embeddings using dimensionality reduction"""
        if self.model is None or self.vocab is None:
            print("Model or vocabulary not loaded. Cannot generate embeddings.")
            return None
        
        # Select a random sample (or all if fewer than sample_size)
        if len(df) > sample_size:
            sample_df = df.sample(sample_size, random_state=42)
        else:
            sample_df = df
        
        # Process texts to get feature vectors
        processed_texts = [preprocess_text(text) for text in sample_df['text']]
        
        # Use CountVectorizer to get feature vectors
        vectorizer = CountVectorizer(max_features=1000)
        X = vectorizer.fit_transform(processed_texts)
        
        # Apply dimensionality reduction
        if method == 'TSNE':
            reducer = TSNE(n_components=2, random_state=42)
            X_reduced = reducer.fit_transform(X.toarray())
        elif method == 'PCA':
            reducer = PCA(n_components=2, random_state=42)
            X_reduced = reducer.fit_transform(X.toarray())
        else:
            reducer = TruncatedSVD(n_components=2, random_state=42)
            X_reduced = reducer.fit_transform(X)
        
        # Create DataFrame for plotting
        vis_df = pd.DataFrame({
            'x': X_reduced[:, 0],
            'y': X_reduced[:, 1],
            'label': sample_df['label']
        })
        
        # Plot
        plt.figure(figsize=(12, 10))
        
        # Create scatter plot
        scatter = plt.scatter(vis_df['x'], vis_df['y'], 
                             c=vis_df['label'], 
                             cmap='viridis', 
                             alpha=0.6, 
                             s=50)
        
        # Add legend
        legend_labels = {0: 'Non-Hate Speech', 1: 'Hate Speech'}
        handles, _ = scatter.legend_elements()
        plt.legend(handles, [legend_labels[i] for i in sorted(legend_labels.keys())],
                  title="Class")
        
        plt.title(f'Text Embedding Visualization using {method}', fontsize=16)
        plt.xlabel(f'{method} Component 1', fontsize=14)
        plt.ylabel(f'{method} Component 2', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.5)
        
        # Add centroids
        for label in vis_df['label'].unique():
            centroid_x = vis_df[vis_df['label'] == label]['x'].mean()
            centroid_y = vis_df[vis_df['label'] == label]['y'].mean()
            plt.scatter(centroid_x, centroid_y, 
                       marker='X', s=200, c='red' if label == 1 else 'blue', 
                       edgecolors='black', linewidth=2)
            plt.annotate(f"{legend_labels[label]} Centroid", 
                       (centroid_x, centroid_y),
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle="round,pad=0.5", fc="yellow", ec="black", alpha=0.8),
                       fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            return save_path
        else:
            # Convert plot to base64 string for embedding in HTML
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png')
            img_buf.seek(0)
            img_str = base64.b64encode(img_buf.read()).decode('utf-8')
            plt.close()
            return img_str

    def generate_dashboard_html(self, df, output_path=None):
        """Generate an HTML dashboard with all visualizations"""
        # Generate base64 encoded images for all visualization types
        class_dist_img = self.plot_class_distribution(df)
        word_freq_hate_img = self.plot_word_frequency(df, class_value=1)
        word_freq_nonhate_img = self.plot_word_frequency(df, class_value=0)
        wordcloud_hate_img = self.create_word_cloud(df, class_value=1)
        wordcloud_nonhate_img = self.create_word_cloud(df, class_value=0)
        text_length_img = self.plot_text_length_distribution(df)
        bigram_img = self.plot_bigram_comparison(df)
        
        # Only generate if model is available
        category_dist_img = None
        confidence_img = None
        embedding_img = None
        
        if self.model is not None and self.vocab is not None:
            category_dist_img = self.plot_category_distribution(df)
            confidence_img = self.plot_confidence_distribution(df)
            embedding_img = self.visualize_text_embeddings(df, method='TSNE')
        
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Hate Speech Detection - Data Visualization Dashboard</title>
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap" rel="stylesheet">
            <style>
                :root {{
                    --primary-color: #4285F4;
                    --secondary-color: #34A853;
                    --danger-color: #EA4335;
                    --light-color: #F8F9FA;
                    --dark-color: #202124;
                    --gray-color: #5F6368;
                    --border-radius: 8px;
                    --box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                
                * {{
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }}
                
                body {{ 
                    font-family: 'Roboto', sans-serif;
                    line-height: 1.6;
                    color: var(--dark-color);
                    background-color: #f5f5f5;
                    padding: 0;
                    margin: 0;
                }}
                
                header {{
                    background-color: var(--primary-color);
                    color: white;
                    padding: 1rem;
                    text-align: center;
                    box-shadow: var(--box-shadow);
                }}
                
                .container {{ 
                    max-width: 1200px;
                    margin: 20px auto;
                    padding: 0 20px;
                }}
                
                .dashboard-grid {{
                    display: grid;
                    grid-template-columns: 1fr;
                    gap: 20px;
                }}
                
                @media (min-width: 768px) {{
                    .dashboard-grid {{
                        grid-template-columns: 1fr 1fr;
                    }}
                }}
                
                .card {{
                    background-color: white;
                    border-radius: var(--border-radius);
                    padding: 20px;
                    box-shadow: var(--box-shadow);
                }}
                
                .full-width {{
                    grid-column: 1 / -1;
                }}
                
                .card-header {{
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 15px;
                    border-bottom: 1px solid #eee;
                    padding-bottom: 10px;
                }}
                
                .card-title {{
                    margin: 0;
                    color: var(--primary-color);
                    font-size: 1.25rem;
                }}
                
                .visualization {{
                    width: 100%;
                    height: auto;
                    display: block;
                    margin: 0 auto;
                }}
                
                footer {{
                    margin-top: 40px;
                    text-align: center;
                    padding: 20px;
                    background-color: var(--light-color);
                    color: var(--gray-color);
                    font-size: 0.9rem;
                }}
            </style>
        </head>
        <body>
            <header>
                <h1>Hate Speech Detection - Data Visualization Dashboard</h1>
            </header>
            
            <div class="container">
                <div class="dashboard-grid">
                    <!-- Class Distribution -->
                    <div class="card">
                        <div class="card-header">
                            <h2 class="card-title">Class Distribution</h2>
                        </div>
                        <img class="visualization" src="data:image/png;base64,{class_dist_img}" alt="Class Distribution">
                    </div>
                    
                    <!-- Text Length Distribution -->
                    <div class="card">
                        <div class="card-header">
                            <h2 class="card-title">Text Length Distribution</h2>
                        </div>
                        <img class="visualization" src="data:image/png;base64,{text_length_img}" alt="Text Length Distribution">
                    </div>
                    
                    <!-- Word Frequency - Hate Speech -->
                    <div class="card">
                        <div class="card-header">
                            <h2 class="card-title">Word Frequency - Hate Speech</h2>
                        </div>
                        <img class="visualization" src="data:image/png;base64,{word_freq_hate_img}" alt="Word Frequency - Hate Speech">
                    </div>
                    
                    <!-- Word Frequency - Non-Hate Speech -->
                    <div class="card">
                        <div class="card-header">
                            <h2 class="card-title">Word Frequency - Non-Hate Speech</h2>
                        </div>
                        <img class="visualization" src="data:image/png;base64,{word_freq_nonhate_img}" alt="Word Frequency - Non-Hate Speech">
                    </div>
                    
                    <!-- Word Cloud - Hate Speech -->
                    <div class="card full-width">
                        <div class="card-header">
                            <h2 class="card-title">Word Cloud - Hate Speech</h2>
                        </div>
                        <img class="visualization" src="data:image/png;base64,{wordcloud_hate_img}" alt="Word Cloud - Hate Speech">
                    </div>
                    
                    <!-- Word Cloud - Non-Hate Speech -->
                    <div class="card full-width">
                        <div class="card-header">
                            <h2 class="card-title">Word Cloud - Non-Hate Speech</h2>
                        </div>
                        <img class="visualization" src="data:image/png;base64,{wordcloud_nonhate_img}" alt="Word Cloud - Non-Hate Speech">
                    </div>
                    
                    <!-- Bigram Comparison -->
                    <div class="card full-width">
                        <div class="card-header">
                            <h2 class="card-title">Bigram Comparison</h2>
                        </div>
                        <img class="visualization" src="data:image/png;base64,{bigram_img}" alt="Bigram Comparison">
                    </div>
        """
        
        # Add model-dependent visualizations if available
        if category_dist_img:
            html_content += f"""
                    <!-- Category Distribution -->
                    <div class="card full-width">
                        <div class="card-header">
                            <h2 class="card-title">Hate Speech Category Distribution</h2>
                        </div>
                        <img class="visualization" src="data:image/png;base64,{category_dist_img}" alt="Category Distribution">
                    </div>
            """
        
        if confidence_img:
            html_content += f"""
                    <!-- Confidence Distribution -->
                    <div class="card full-width">
                        <div class="card-header">
                            <h2 class="card-title">Model Confidence Distribution</h2>
                        </div>
                        <img class="visualization" src="data:image/png;base64,{confidence_img}" alt="Confidence Distribution">
                    </div>
            """
        
        if embedding_img:
            html_content += f"""
                    <!-- Text Embeddings -->
                    <div class="card full-width">
                        <div class="card-header">
                            <h2 class="card-title">Text Embeddings Visualization</h2>
                        </div>
                        <img class="visualization" src="data:image/png;base64,{embedding_img}" alt="Text Embeddings">
                    </div>
            """
        
        # Complete HTML
        html_content += """
                </div>
            </div>
            
            <footer>
                <p>Hate Speech Detection Data Visualization Dashboard &copy; 2024</p>
            </footer>
        </body>
        </html>
        """
        
        # Save to file if output path is provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            return output_path
        else:
            return html_content