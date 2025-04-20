from flask import Flask, request, jsonify, render_template_string, redirect, url_for, send_from_directory
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, confusion_matrix, precision_score, recall_score, accuracy_score
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import os
import re
import string
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from werkzeug.utils import secure_filename
from hate_speech_model import LSTM_CNN, preprocess_text, text_to_sequence
from comprehensive_hate_speech import classify_text_with_comprehensive_detection
from data_visualizations import HateSpeechVisualizer

app = Flask(__name__)
CORS(app)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Global variables to store the model and vocabulary
lstm_cnn_model = None
vocab = None
training_status = {'is_training': False, 'progress': 0, 'message': ''}
current_dataset = '../Data/Dataset_3.csv'
available_datasets = [
    {'name': 'Dataset 1', 'path': '../Data/Dataset_1.csv'},
    {'name': 'Dataset 2', 'path': '../Data/Dataset_2.csv'},
    {'name': 'Dataset 3', 'path': '../Data/Dataset_3.csv'},
    {'name': 'Labeled Data', 'path': '../Data/labeled_data.csv'}
]
model_metrics = {
    'accuracy': 0.0,
    'precision': 0.0,
    'recall': 0.0,
    'f1': 0.0,
    'confusion_matrix': None,
    'class_report': None,
    'confusion_matrix_img': None,
    'calculated': False
}

metrics_calculated = False

@app.before_request
def calculate_metrics_before_request():
    global metrics_calculated, lstm_cnn_model, vocab, model_metrics
    
    if not metrics_calculated and lstm_cnn_model is not None and vocab is not None:
        calculate_metrics()
        metrics_calculated = True

def load_models():
    global lstm_cnn_model, vocab
    
    try:
        # First try to load enhanced model
        vocab_path = './vocab_improved.pkl'
        model_path = './hate_speech_model_enhanced.pth'
        
        # Check if enhanced model exists
        if not os.path.exists(model_path):
            # Fall back to improved model
            model_path = './hate_speech_model_improved.pth'
            print(f"Enhanced model not found, falling back to {model_path}")
            
            # If improved model doesn't exist either, fall back to original model
            if not os.path.exists(model_path):
                model_path = './hate_speech_model_lstm_cnn.pth'
                vocab_path = './vocab.pkl'
                print(f"Improved model not found, falling back to {model_path}")
        
        print(f"Loading vocabulary from {vocab_path}")
        if not os.path.exists(vocab_path):
            # If improved vocab doesn't exist, fall back to original
            vocab_path = './vocab.pkl'
            print(f"Improved vocabulary not found, falling back to {vocab_path}")
            
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Vocabulary file not found at {vocab_path}")
            
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        
        if vocab is None:
            raise ValueError("Loaded vocabulary is None")
            
        print(f"Vocabulary loaded with {len(vocab)} entries")
        
        print(f"Loading LSTM-CNN model from {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        model_info = torch.load(model_path, map_location=device)
        
        # Initialize LSTM-CNN model
        lstm_cnn_model = LSTM_CNN(
            model_info['vocab_size'],
            model_info['embed_dim'],
            model_info['lstm_hidden_dim'],
            model_info['cnn_hidden_dim'],
            model_info['num_classes'],
            model_info['dropout']
        )
        lstm_cnn_model.load_state_dict(model_info['state_dict'])
        lstm_cnn_model.to(device)
        lstm_cnn_model.eval()
        
        model_type = "enhanced" if "enhanced" in model_path else ("improved" if "improved" in model_path else "original")
        print(f"LSTM-CNN model ({model_type}) loaded successfully!")
        
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        import traceback
        traceback.print_exc()
        return False

def load_dataset(data_path='../Data/Dataset_3.csv'):
    """Load and preprocess the dataset"""
    print(f"Loading dataset from {data_path}...")
    
    try:
        # Load the CSV file
        df = pd.read_csv(data_path, encoding='latin1')
        print(f"Dataset loaded with {len(df)} samples")
        
        # Convert class to binary labels (1 for hate speech, 0 for non-hate speech)
        df['label'] = df['class'].apply(lambda x: 1 if x == 0 else 0)
        
        # Create clean dataset with only text and label columns
        df_clean = df[['tweet', 'label']].rename(columns={'tweet': 'text'})
        
        # Print class distribution
        class_dist = df_clean['label'].value_counts()
        print(f"Class distribution:\n{class_dist}")
        
        return df_clean
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def build_vocab(texts, max_vocab_size=10000):
    """Build vocabulary from texts"""
    counter = {}
    for text in texts:
        for token in text.split():
            if token in counter:
                counter[token] += 1
            else:
                counter[token] = 1
    
    # Sort by frequency
    sorted_tokens = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    
    # Create vocab dict with top tokens
    vocab_dict = {' ': 0}  # Padding token
    for i, (token, _) in enumerate(sorted_tokens[:max_vocab_size-1]):
        vocab_dict[token] = i + 1
    
    return vocab_dict

def train_model(df, epochs=5, batch_size=32, save_path='hate_speech_model_lstm_cnn.pth'):
    """Train the LSTM-CNN model on the provided dataframe"""
    global lstm_cnn_model, vocab, training_status
    
    try:
        training_status = {'is_training': True, 'progress': 0, 'message': 'Starting training...'}
        
        # Split data
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
        
        # Balance training data
        train_majority = train_df[train_df.label == 0]
        train_minority = train_df[train_df.label == 1]
        
        # Upsample minority class
        from sklearn.utils import resample
        train_minority_upsampled = resample(
            train_minority, 
            replace=True,
            n_samples=len(train_majority),
            random_state=42
        )
        
        # Combine balanced data
        train_df_balanced = pd.concat([train_majority, train_minority_upsampled])
        print(f"Balanced training data: {len(train_df_balanced)} samples")
        
        training_status['message'] = 'Building vocabulary...'
        training_status['progress'] = 10
        
        # Build vocabulary from training data
        all_texts = train_df_balanced['text'].apply(preprocess_text).tolist()
        vocab = build_vocab(all_texts)
        
        print(f"Vocabulary size: {len(vocab)}")
        
        # Save vocabulary
        with open('vocab.pkl', 'wb') as f:
            pickle.dump(vocab, f)
        
        training_status['message'] = 'Converting texts to sequences...'
        training_status['progress'] = 20
        
        # Convert texts to sequences
        X_train = train_df_balanced['text'].apply(lambda x: text_to_sequence(preprocess_text(x), vocab)).tolist()
        y_train = train_df_balanced['label'].tolist()
        
        X_val = val_df['text'].apply(lambda x: text_to_sequence(preprocess_text(x), vocab)).tolist()
        y_val = val_df['label'].tolist()
        
        # Convert to tensors
        X_train = torch.tensor(X_train, dtype=torch.long)
        y_train = torch.tensor(y_train, dtype=torch.long)
        X_val = torch.tensor(X_val, dtype=torch.long)
        y_val = torch.tensor(y_val, dtype=torch.long)
        
        # Create dataloaders
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        training_status['message'] = 'Initializing model...'
        training_status['progress'] = 30
        
        # Initialize model
        vocab_size = len(vocab)
        embed_dim = 100
        lstm_hidden_dim = 128
        cnn_hidden_dim = 128
        num_classes = 2
        dropout = 0.5
        
        lstm_cnn_model = LSTM_CNN(vocab_size, embed_dim, lstm_hidden_dim, cnn_hidden_dim, num_classes, dropout)
        lstm_cnn_model.to(device)
        
        # Define loss function and optimizer
        class_weights = torch.tensor([0.3, 0.7], device=device)  # Adjust for class imbalance
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(lstm_cnn_model.parameters(), lr=1e-3)
        
        # Training loop
        best_val_f1 = 0
        
        for epoch in range(epochs):
            training_status['message'] = f'Training epoch {epoch+1}/{epochs}...'
            training_status['progress'] = 30 + (epoch / epochs) * 60
            
            # Training phase
            lstm_cnn_model.train()
            train_loss = 0.0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = lstm_cnn_model(inputs)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss = train_loss / len(train_loader)
            
            # Validation phase
            lstm_cnn_model.eval()
            val_loss = 0.0
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    outputs = lstm_cnn_model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(labels.cpu().numpy())
            
            val_loss = val_loss / len(val_loader)
            
            # Calculate F1 score
            val_f1 = f1_score(all_labels, all_preds, pos_label=1)
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Training Loss: {train_loss:.4f}")
            print(f"Validation Loss: {val_loss:.4f}, F1 (hate speech): {val_f1:.4f}")
            
            # Save best model
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                
                model_info = {
                    'state_dict': lstm_cnn_model.state_dict(),
                    'vocab_size': vocab_size,
                    'embed_dim': embed_dim,
                    'lstm_hidden_dim': lstm_hidden_dim,
                    'cnn_hidden_dim': cnn_hidden_dim,
                    'num_classes': num_classes,
                    'dropout': dropout,
                    'val_f1': val_f1
                }
                
                torch.save(model_info, save_path)
        
        training_status['message'] = 'Training complete!'
        training_status['progress'] = 100
        training_status['is_training'] = False
        
        print(f"Training complete! Best validation F1: {best_val_f1:.4f}")
        return True
    
    except Exception as e:
        training_status['message'] = f'Error during training: {str(e)}'
        training_status['is_training'] = False
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        return False

def calculate_metrics():
    """Calculate performance metrics for the current model on the validation set"""
    global model_metrics, current_dataset, lstm_cnn_model, vocab
    
    try:
        # Load the dataset
        df = load_dataset(current_dataset)
        if df is None:
            return False
        
        # Create a validation split
        _, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
        
        # Convert texts to sequences
        X_val = val_df['text'].apply(lambda x: text_to_sequence(preprocess_text(x), vocab)).tolist()
        y_val = val_df['label'].tolist()
        
        # Convert to tensors
        X_val = torch.tensor(X_val, dtype=torch.long)
        y_val = torch.tensor(y_val, dtype=torch.long)
        
        # Create dataloader
        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=32)
        
        # Evaluation
        lstm_cnn_model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = lstm_cnn_model(inputs)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        
        # Handle case where only one class is present
        unique_labels = np.unique(np.concatenate(np.array([all_labels, all_preds])))
        
        try:
            precision = precision_score(all_labels, all_preds, pos_label=1, zero_division=0)
            recall = recall_score(all_labels, all_preds, pos_label=1, zero_division=0)
            f1 = f1_score(all_labels, all_preds, pos_label=1, zero_division=0)
        except Exception as e:
            print(f"Warning calculating precision/recall/f1: {e}")
            precision = 0
            recall = 0
            f1 = 0
            
        conf_matrix = confusion_matrix(all_labels, all_preds)
        
        # Use labels parameter to handle datasets with only one class
        class_report = classification_report(
            all_labels, 
            all_preds, 
            target_names=['Non-Hate Speech', 'Hate Speech'], 
            labels=[0, 1],  # Explicitly specify the labels
            output_dict=True,
            zero_division=0  # Handle zero division cases
        )
        
        # Create confusion matrix visualization
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Non-Hate', 'Hate'],
                    yticklabels=['Non-Hate', 'Hate'])
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        
        # Save plot to a base64 string for embedding in HTML
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        img_buf.seek(0)
        img_str = base64.b64encode(img_buf.read()).decode('utf-8')
        plt.close()
        
        # Update global metrics
        model_metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'confusion_matrix': conf_matrix.tolist(),
            'class_report': class_report,
            'confusion_matrix_img': img_str,
            'calculated': True
        }
        
        return True
    
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        import traceback
        traceback.print_exc()
        return False

@app.route('/get_metrics')
def get_metrics():
    global model_metrics
    
    # Check if force recalculation was requested
    force_recalculation = request.args.get('force', 'false').lower() == 'true'
    
    if force_recalculation or not model_metrics['calculated']:
        if not calculate_metrics():
            return jsonify({"success": False, "message": "Failed to calculate metrics"}), 500
    
    return jsonify({"success": True, "metrics": model_metrics})

@app.route('/get_available_datasets')
def get_available_datasets():
    # Only return built-in datasets, without uploaded datasets
    return jsonify({"datasets": available_datasets, "current": current_dataset})

@app.route('/select_dataset', methods=['POST'])
def select_dataset():
    global current_dataset, model_metrics, training_status
    
    try:
        data = request.json
        if not data or 'dataset_path' not in data:
            return jsonify({"success": False, "message": "No dataset path provided"}), 400
        
        dataset_path = data['dataset_path']
        
        # Validate the dataset path - only allow paths from our predefined list
        valid_paths = [ds['path'] for ds in available_datasets]
        if dataset_path not in valid_paths:
            return jsonify({"success": False, "message": "Invalid dataset path"}), 400
        
        current_dataset = dataset_path
        # Reset metrics so they'll be recalculated for the new dataset
        model_metrics['calculated'] = False
        
        # Check if training is already in progress
        if training_status['is_training']:
            return jsonify({
                "success": True, 
                "message": f"Dataset selected: {os.path.basename(dataset_path)}. Training already in progress.",
                "training_started": False
            }), 200
        
        # Start automatic training on the selected dataset
        # Load the dataset
        df = load_dataset(current_dataset)
        if df is None:
            return jsonify({
                "success": True, 
                "message": f"Dataset selected: {os.path.basename(dataset_path)}, but failed to load for training.",
                "training_started": False
            }), 200
        
        # Use default 5 epochs for automatic training
        epochs = 5
        
        # Start training in a separate thread to not block the server
        import threading
        training_thread = threading.Thread(target=train_model, args=(df, epochs))
        training_thread.daemon = True
        training_thread.start()
        
        return jsonify({
            "success": True, 
            "message": f"Dataset selected: {os.path.basename(dataset_path)}. Training automatically started with {epochs} epochs.",
            "training_started": True
        }), 200
    
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    # Get text from request
    data = request.json
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400
    
    text = data['text']
    
    try:
        if lstm_cnn_model is None or vocab is None:
            return jsonify({"error": "LSTM-CNN model not loaded"}), 500
            
        # Use comprehensive classification with hate speech detection
        result = classify_text_with_comprehensive_detection(lstm_cnn_model, vocab, text)
        
        # Prepare response
        response = {
            'label': result['prediction'],
            'score': float(result['confidence']),
            'model': 'LSTM-CNN'
        }
        
        return jsonify(response)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/train', methods=['POST'])
def train():
    global training_status, model_metrics
    
    if training_status['is_training']:
        return jsonify({"success": False, "message": "Training is already in progress"})
    
    try:
        data = request.json
        epochs = data.get('epochs', 5)
        
        # Load the dataset
        df = load_dataset(current_dataset)
        if df is None:
            return jsonify({"success": False, "message": "Failed to load dataset"})
        
        # Start training in a separate thread to not block the server
        import threading
        training_thread = threading.Thread(target=train_model, args=(df, epochs))
        training_thread.daemon = True
        training_thread.start()
        
        # Reset metrics so they'll be recalculated after training
        model_metrics['calculated'] = False
        
        return jsonify({"success": True, "message": "Training started"})
    
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})

@app.route('/training_status')
def get_training_status():
    global training_status
    return jsonify(training_status)

@app.route('/error_analysis', methods=['GET', 'POST'])
def error_analysis():
    global lstm_cnn_model, vocab, current_dataset
    
    if request.method == 'POST':
        try:
            data = request.json
            num_samples = data.get('num_samples', 100)
            
            # Create output directory if it doesn't exist
            error_dir = os.path.join(base_dir, 'error_analysis')
            os.makedirs(error_dir, exist_ok=True)
            output_path = os.path.join(error_dir, 'error_analysis.json')
            
            # Import error analyzer
            from hate_speech_model import ErrorAnalyzer
            
            # Check if model and vocab are loaded
            if lstm_cnn_model is None or vocab is None:
                return jsonify({"success": False, "message": "Model or vocabulary not loaded"}), 500
                
            # Load dataset
            df = load_dataset(current_dataset)
            if df is None:
                return jsonify({"success": False, "message": "Failed to load dataset"}), 500
                
            # Initialize error analyzer
            error_analyzer = ErrorAnalyzer(lstm_cnn_model, vocab)
            
            # Sample examples for analysis
            if len(df) > num_samples:
                df_sample = df.sample(num_samples, random_state=42)
            else:
                df_sample = df
                
            # Analyze each example
            for _, row in df_sample.iterrows():
                text = row['text']
                true_label = row['label']
                error_analyzer.analyze_error(text, true_label)
                
            # Save and return error analysis
            error_summary = error_analyzer.get_error_summary()
            with open(output_path, 'w') as f:
                json.dump(error_summary, f, indent=2)
                
            return jsonify({"success": True, "message": f"Error analysis complete with {num_samples} samples", 
                           "summary": error_summary})
                           
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"success": False, "message": str(e)}), 500
    
    # GET request returns the form
    return jsonify({"message": "Use POST method with JSON data to perform error analysis"})

@app.route('/start_hyperparameter_tuning', methods=['POST'])
def start_hyperparameter_tuning():
    global current_dataset
    
    try:
        data = request.json
        num_epochs = data.get('epochs', 5)
        
        # Create output directory
        output_dir = os.path.join(base_dir, 'hyperparameter_tuning_results')
        os.makedirs(output_dir, exist_ok=True)
        
        # Start tuning in a separate thread
        import threading
        import hyperparameter_tuning
        
        def run_tuning():
            try:
                hyperparameter_tuning.run_hyperparameter_tuning(
                    current_dataset,
                    output_dir=output_dir,
                    vocab_path='./vocab_improved.pkl' if os.path.exists('./vocab_improved.pkl') else './vocab.pkl',
                    use_advanced_preprocessing=True
                )
            except Exception as e:
                print(f"Error during hyperparameter tuning: {e}")
                import traceback
                traceback.print_exc()
        
        # Start the tuning thread
        tuning_thread = threading.Thread(target=run_tuning)
        tuning_thread.daemon = True
        tuning_thread.start()
        
        return jsonify({
            "success": True,
            "message": f"Hyperparameter tuning started with dataset {os.path.basename(current_dataset)}. This may take a while."
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/visualizations')
def visualizations_dashboard():
    """Render the visualization dashboard or generate new visualizations"""
    # Define paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, 'visualizations')
    
    # Create visualizations directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a visualizer from our loaded model
    visualizer = HateSpeechVisualizer(model=lstm_cnn_model, vocab=vocab)
    
    # Extract dataset name from path
    dataset_name = os.path.basename(current_dataset).replace('.csv', '')
    dashboard_path = os.path.join(output_dir, dataset_name, 'dashboard.html')
    
    # Check if a dashboard already exists for this dataset
    if os.path.exists(dashboard_path):
        with open(dashboard_path, 'r', encoding='utf-8') as f:
            dashboard_html = f.read()
        return dashboard_html
    
    # If no dashboard exists, generate visualizations
    # Load the dataset
    df = load_dataset(current_dataset)
    if df is None:
        return jsonify({"error": "Failed to load dataset for visualization"}), 500
    
    # Create directory for dataset visualizations
    dataset_viz_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_viz_dir, exist_ok=True)
    
    # Generate HTML dashboard
    dashboard_html = visualizer.generate_dashboard_html(df, output_path=dashboard_path)
    
    return dashboard_html

@app.route('/start_visualizations', methods=['POST'])
def start_visualizations():
    """Start the visualization generation process in the background"""
    global current_dataset, lstm_cnn_model, vocab
    
    try:
        # Create a new thread to generate visualizations
        def generate_visualizations():
            current_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(current_dir, 'visualizations')
            os.makedirs(output_dir, exist_ok=True)
            
            # Create a visualizer
            visualizer = HateSpeechVisualizer(model=lstm_cnn_model, vocab=vocab)
            
            # Find all CSV files in the Data directory
            data_dir = os.path.join(os.path.dirname(current_dir), 'Data')
            dataset_paths = []
            for file in os.listdir(data_dir):
                if file.endswith('.csv'):
                    dataset_paths.append(os.path.join(data_dir, file))
            
            # Process each dataset
            for dataset_path in dataset_paths:
                dataset_name = os.path.basename(dataset_path).replace('.csv', '')
                print(f"Processing dataset: {dataset_name}")
                
                # Load dataset
                try:
                    df = visualizer.load_data(dataset_path)
                    if df is None:
                        continue
                    
                    # Create directory for dataset visualizations
                    dataset_viz_dir = os.path.join(output_dir, dataset_name)
                    os.makedirs(dataset_viz_dir, exist_ok=True)
                    
                    # Generate HTML dashboard
                    dashboard_path = os.path.join(dataset_viz_dir, 'dashboard.html')
                    visualizer.generate_dashboard_html(df, output_path=dashboard_path)
                    print(f"Dashboard generated: {dashboard_path}")
                except Exception as e:
                    print(f"Error processing dataset {dataset_name}: {e}")
                    continue
            
            # Compare datasets
            if len(dataset_paths) > 1:
                comparison_path = os.path.join(output_dir, 'dataset_comparison.png')
                visualizer.plot_dataset_comparison(dataset_paths, save_path=comparison_path)
        
        # Start the thread
        import threading
        viz_thread = threading.Thread(target=generate_visualizations)
        viz_thread.daemon = True
        viz_thread.start()
        
        return jsonify({
            "success": True, 
            "message": "Visualization generation started in the background. This might take a few minutes.",
            "view_url": "/visualizations"
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/visualizations/<path:filename>')
def visualization_files(filename):
    """Serve visualization files"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    viz_dir = os.path.join(current_dir, 'visualizations')
    return send_from_directory(viz_dir, filename)

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Hate Speech Detection</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary-color: #4285F4;
            --secondary-color: #34A853;
            --danger-color: #EA4335;
            --light-color: #F8F9FA;
            --dark-color: #202124;
            --gray-color: #5F6368;
            --border-radius: 8px;
            --box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body { 
            font-family: 'Roboto', sans-serif;
            line-height: 1.6;
            color: var(--dark-color);
            background-color: #f5f5f5;
            padding: 0;
            margin: 0;
        }
        
        header {
            background-color: var(--primary-color);
            color: white;
            padding: 1rem;
            text-align: center;
            box-shadow: var(--box-shadow);
        }
        
        .container { 
            max-width: 1200px;
            margin: 20px auto;
            padding: 0 20px;
        }
        
        .grid {
            display: grid;
            grid-template-columns: 1fr;
            gap: 20px;
        }
        
        @media (min-width: 768px) {
            .grid {
                grid-template-columns: 1fr 1fr;
            }
        }
        
        .full-width {
            grid-column: 1 / -1;
        }
        
        .card {
            background-color: white;
            border-radius: var(--border-radius);
            padding: 20px;
            box-shadow: var(--box-shadow);
        }
        
        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            border-bottom: 1px solid #eee;
            padding-bottom: 10px;
        }
        
        .card-title {
            margin: 0;
            color: var(--primary-color);
            font-size: 1.25rem;
        }
        
        .dataset-info {
            background-color: var(--light-color);
        }
        
        .dataset-selector {
            display: flex;
            flex-direction: column;
            gap: 10px;
            margin-top: 15px;
        }
        
        textarea, select, input[type="text"], input[type="number"] { 
            width: 100%; 
            padding: 12px;
            border: 1px solid #ddd;
            border-radius: var(--border-radius);
            font-family: inherit;
            font-size: 14px;
            margin-top: 5px;
        }
        
        textarea {
            height: 120px;
            resize: vertical;
        }
        
        button { 
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 10px 20px; 
            background-color: var(--primary-color); 
            color: white; 
            border: none; 
            cursor: pointer; 
            border-radius: var(--border-radius);
            font-weight: 500;
            transition: background-color 0.2s;
        }
        
        button:hover { background-color: #3367D6; }
        
        button.success { background-color: var(--secondary-color); }
        button.success:hover { background-color: #2E9547; }
        
        button.danger { background-color: var(--danger-color); }
        button.danger:hover { background-color: #D93025; }
        
        .btn-group {
            display: flex;
            gap: 10px;
            margin-top: 15px;
        }
        
        .result { 
            margin-top: 20px; 
            padding: 15px; 
            border-radius: var(--border-radius);
            display: none;
        }
        
        .hate-speech { 
            background-color: rgba(234, 67, 53, 0.1); 
            border-left: 4px solid var(--danger-color);
        }
        
        .non-hate-speech { 
            background-color: rgba(52, 168, 83, 0.1); 
            border-left: 4px solid var(--secondary-color);
        }
        
        .spinner { 
            display: none; 
            margin-left: 10px;
            vertical-align: middle;
            border: 3px solid rgba(0, 0, 0, 0.1);
            border-radius: 50%;
            border-top: 3px solid var(--primary-color);
            width: 16px;
            height: 16px;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .progress-container {
            width: 100%;
            background-color: #f3f3f3;
            border-radius: var(--border-radius);
            margin-top: 15px;
            overflow: hidden;
            height: 20px;
        }
        
        .progress-bar {
            height: 100%;
            background-color: var(--primary-color);
            text-align: center;
            color: white;
            line-height: 20px;
            font-size: 12px;
            transition: width 0.3s;
        }
        
        .section {
            display: none;
            margin-top: 15px;
        }
        
        .toggle-section {
            cursor: pointer;
            color: var(--primary-color);
            display: inline-flex;
            align-items: center;
            font-weight: 500;
        }
        
        .toggle-section::after {
            content: '▼';
            margin-left: 5px;
            font-size: 10px;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        
        .metric-card {
            background-color: var(--light-color);
            border-radius: var(--border-radius);
            padding: 15px;
            text-align: center;
        }
        
        .metric-value {
            font-size: 1.8rem;
            font-weight: bold;
            color: var(--primary-color);
        }
        
        .metric-label {
            font-size: 0.9rem;
            color: var(--gray-color);
        }
        
        .visualization {
            margin-top: 20px;
            text-align: center;
        }
        
        .visualization img {
            max-width: 100%;
            height: auto;
            border-radius: var(--border-radius);
            box-shadow: var(--box-shadow);
        }
        
        .status-message {
            margin-top: 10px;
            padding: 10px;
            border-radius: var(--border-radius);
        }
        
        .success-msg { 
            background-color: rgba(52, 168, 83, 0.1); 
            color: var(--secondary-color);
        }
        
        .error-msg { 
            background-color: rgba(234, 67, 53, 0.1); 
            color: var(--danger-color);
        }
        
        .badge {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 50px;
            font-size: 12px;
            font-weight: 500;
            margin-left: 10px;
        }
        
        .badge-light {
            background-color: var(--light-color);
            color: var(--gray-color);
        }
        
        footer {
            margin-top: 40px;
            text-align: center;
            padding: 20px;
            background-color: var(--light-color);
            color: var(--gray-color);
            font-size: 0.9rem;
        }
        
        .feature-icon {
            font-size: 2rem;
            margin-bottom: 10px;
            color: var(--primary-color);
        }
        
        .viz-features {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        
        .viz-feature {
            text-align: center;
            padding: 15px;
            background-color: var(--light-color);
            border-radius: var(--border-radius);
        }
    </style>
</head>
<body>
    <header>
        <h1>Hate Speech Detection</h1>
    </header>
    
    <div class="container">
        <div class="grid">
            <div class="card dataset-info">
                <div class="card-header">
                    <h2 class="card-title">Dataset Information</h2>
                    <span class="badge badge-light" id="current-dataset-badge">Dataset_3.csv</span>
                </div>
                <p>This model detects hate speech and offensive language in text content using a hybrid LSTM-CNN architecture.</p>
                
                <div class="toggle-section" onclick="toggleSection('dataset-section')">Change Dataset</div>
                <div id="dataset-section" class="section">
                    <div class="dataset-selector">
                        <label for="dataset-select">Select a dataset:</label>
                        <select id="dataset-select">
                            <!-- Will be populated via JavaScript -->
                        </select>
                        <button onclick="selectDataset()">Use Selected Dataset</button>
                    </div>
                    <div id="dataset-status" class="status-message"></div>
                </div>
            </div>
            
            <div class="card">
                <div class="card-header">
                    <h2 class="card-title">Model Performance</h2>
                    <button onclick="refreshMetrics()">Refresh Metrics</button>
                </div>
                <div id="metrics-container">
                    <div class="metrics-grid">
                        <div class="metric-card">
                            <div class="metric-value" id="accuracy-value">-</div>
                            <div class="metric-label">Accuracy</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value" id="precision-value">-</div>
                            <div class="metric-label">Precision</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value" id="recall-value">-</div>
                            <div class="metric-label">Recall</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value" id="f1-value">-</div>
                            <div class="metric-label">F1 Score</div>
                        </div>
                    </div>
                    
                    <div class="visualization">
                        <h3>Confusion Matrix</h3>
                        <div id="confusion-matrix">
                            <p>Loading visualization...</p>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <div class="card-header">
                    <h2 class="card-title">Test the Model</h2>
                </div>
                <div>
                    <label for="text-input">Enter text to analyze:</label>
                    <textarea id="text-input" placeholder="Type or paste text here..."></textarea>
                </div>
                <div class="btn-group">
                    <button onclick="detectHateSpeech()">Analyze Text</button>
                    <span class="spinner" id="loading"></span>
                </div>
                <div id="result" class="result"></div>
            </div>
            
            <div class="card">
                <div class="card-header">
                    <h2 class="card-title">Model Training</h2>
                </div>
                <div class="toggle-section" onclick="toggleSection('training-section')">Advanced Options</div>
                <div id="training-section" class="section">
                    <p>Retrain the LSTM-CNN model on the currently selected dataset.</p>
                    <div style="margin-top: 15px;">
                        <label for="epochs">Number of Training Epochs:</label>
                        <input type="number" id="epochs" min="1" max="20" value="5">
                    </div>
                    <div class="btn-group">
                        <button onclick="trainModel()" class="success">Start Training</button>
                    </div>
                    <div id="training-status" style="display: none; margin-top: 15px;">
                        <p id="training-message">Training in progress...</p>
                        <div class="progress-container">
                            <div id="training-progress" class="progress-bar" style="width:0%">0%</div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Data Visualization Card -->
            <div class="card full-width">
                <div class="card-header">
                    <h2 class="card-title">Data Visualizations</h2>
                </div>
                <p>Explore deeper insights into your hate speech datasets with interactive visualizations.</p>
                
                <div class="viz-features">
                    <div class="viz-feature">
                        <div class="feature-icon">📊</div>
                        <h3>Class Distribution</h3>
                        <p>Analyze the balance between hate speech and non-hate speech classes.</p>
                    </div>
                    <div class="viz-feature">
                        <div class="feature-icon">☁️</div>
                        <h3>Word Clouds</h3>
                        <p>Visualize most common words in hate speech and non-hate speech.</p>
                    </div>
                    <div class="viz-feature">
                        <div class="feature-icon">📏</div>
                        <h3>Text Length Analysis</h3>
                        <p>Compare text length distributions between classes.</p>
                    </div>
                    <div class="viz-feature">
                        <div class="feature-icon">🔄</div>
                        <h3>Compare Datasets</h3>
                        <p>View statistics across multiple datasets.</p>
                    </div>
                </div>
                
                <div class="btn-group">
                    <button onclick="openVisualizations()">View Visualizations</button>
                    <button onclick="generateVisualizations()" class="success">Generate New Visualizations</button>
                </div>
                <div id="visualization-status" class="status-message"></div>
            </div>
        </div>
    </div>
    
    <footer>
        <p>LSTM-CNN Hate Speech Detection Model &copy; 2024</p>
    </footer>

    <script>
        document.addEventListener('DOMContentLoaded', function() {
            loadAvailableDatasets();
            loadMetrics();
        });
        
        function toggleSection(sectionId) {
            const section = document.getElementById(sectionId);
            section.style.display = section.style.display === 'none' || section.style.display === '' ? 'block' : 'none';
        }
        
        function loadAvailableDatasets() {
            fetch('/get_available_datasets')
            .then(response => response.json())
            .then(data => {
                const select = document.getElementById('dataset-select');
                select.innerHTML = '';
                
                data.datasets.forEach(dataset => {
                    const option = document.createElement('option');
                    option.value = dataset.path;
                    option.textContent = dataset.name;
                    if (dataset.path === data.current) {
                        option.selected = true;
                    }
                    select.appendChild(option);
                });
                
                // Update current dataset display
                document.getElementById('current-dataset-badge').textContent = data.current.split('/').pop();
            })
            .catch(error => {
                console.error('Error loading datasets:', error);
            });
        }
        
        function selectDataset() {
            const select = document.getElementById('dataset-select');
            const statusDiv = document.getElementById('dataset-status');
            
            fetch('/select_dataset', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ dataset_path: select.value })
            })
            .then(response => response.json())
            .then(data => {
                statusDiv.textContent = data.message;
                statusDiv.className = data.success ? 'status-message success-msg' : 'status-message error-msg';
                
                if (data.success) {
                    document.getElementById('current-dataset-badge').textContent = select.value.split('/').pop();
                    
                    // If training automatically started, show the training status
                    if (data.training_started) {
                        // Show the training section
                        document.getElementById('training-section').style.display = 'block';
                        
                        // Show training progress
                        const statusDiv = document.getElementById('training-status');
                        const progressBar = document.getElementById('training-progress');
                        const messageElem = document.getElementById('training-message');
                        
                        statusDiv.style.display = 'block';
                        progressBar.style.width = '0%';
                        progressBar.textContent = '0%';
                        messageElem.textContent = 'Training automatically started...';
                        
                        // Start checking training status
                        checkTrainingStatus();
                    }
                    
                    // Refresh metrics for the new dataset
                    loadMetrics();
                }
            })
            .catch(error => {
                statusDiv.textContent = 'Error: ' + error;
                statusDiv.className = 'status-message error-msg';
            });
        }
        
        function loadMetrics() {
            const accuracyElem = document.getElementById('accuracy-value');
            const precisionElem = document.getElementById('precision-value');
            const recallElem = document.getElementById('recall-value');
            const f1Elem = document.getElementById('f1-value');
            const confusionMatrixElem = document.getElementById('confusion-matrix');
            
            // Show loading state
            accuracyElem.textContent = '-';
            precisionElem.textContent = '-';
            recallElem.textContent = '-';
            f1Elem.textContent = '-';
            confusionMatrixElem.innerHTML = '<p>Loading visualization...</p>';
            
            fetch('/get_metrics')
            .then(response => response.json())
            .then(data => {
                if (data.success && data.metrics.calculated) {
                    // Update metric values
                    accuracyElem.textContent = (data.metrics.accuracy * 100).toFixed(1) + '%';
                    precisionElem.textContent = (data.metrics.precision * 100).toFixed(1) + '%';
                    recallElem.textContent = (data.metrics.recall * 100).toFixed(1) + '%';
                    f1Elem.textContent = (data.metrics.f1 * 100).toFixed(1) + '%';
                    
                    // Display confusion matrix image
                    if (data.metrics.confusion_matrix_img) {
                        confusionMatrixElem.innerHTML = `<img src="data:image/png;base64,${data.metrics.confusion_matrix_img}" alt="Confusion Matrix">`;
                    } else {
                        confusionMatrixElem.innerHTML = '<p>Visualization not available</p>';
                    }
                } else {
                    confusionMatrixElem.innerHTML = '<p>Failed to load metrics. Please try again.</p>';
                }
            })
            .catch(error => {
                console.error('Error loading metrics:', error);
                confusionMatrixElem.innerHTML = '<p>Error loading metrics: ' + error + '</p>';
            });
        }
        
        function refreshMetrics() {
            // Force recalculation of metrics
            fetch('/get_metrics?force=true')
            .then(response => response.json())
            .then(data => {
                loadMetrics();
            })
            .catch(error => {
                console.error('Error refreshing metrics:', error);
            });
        }
        
        function detectHateSpeech() {
            const text = document.getElementById('text-input').value.trim();
            if (!text) {
                alert('Please enter some text to analyze');
                return;
            }
            
            // Show loading indicator
            document.getElementById('loading').style.display = 'inline-block';
            
            // Clear previous result
            const resultDiv = document.getElementById('result');
            resultDiv.style.display = 'none';
            resultDiv.className = 'result';
            
            // Send request to API
            fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text: text })
            })
            .then(response => response.json())
            .then(data => {
                // Hide loading indicator
                document.getElementById('loading').style.display = 'none';
                
                // Display result
                resultDiv.style.display = 'block';
                resultDiv.className = data.label === 'Hate Speech' ? 
                                     'result hate-speech' : 'result non-hate-speech';
                
                resultDiv.innerHTML = `
                    <h3>Analysis Result:</h3>
                    <p><strong>Classification:</strong> ${data.label}</p>
                    <p><strong>Confidence:</strong> ${(data.score * 100).toFixed(2)}%</p>
                    <p><strong>Model:</strong> LSTM-CNN</p>
                `;
            })
            .catch(error => {
                document.getElementById('loading').style.display = 'none';
                alert('Error: ' + error);
            });
        }
        
        function trainModel() {
            const epochs = document.getElementById('epochs').value;
            const statusDiv = document.getElementById('training-status');
            const progressBar = document.getElementById('training-progress');
            const messageElem = document.getElementById('training-message');
            
            statusDiv.style.display = 'block';
            progressBar.style.width = '0%';
            progressBar.textContent = '0%';
            messageElem.textContent = 'Starting training...';
            
            fetch('/train', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ epochs: parseInt(epochs) })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    checkTrainingStatus();
                } else {
                    messageElem.textContent = 'Failed to start training: ' + data.message;
                }
            })
            .catch(error => {
                messageElem.textContent = 'Error: ' + error;
            });
        }
        
        function checkTrainingStatus() {
            const progressBar = document.getElementById('training-progress');
            const messageElem = document.getElementById('training-message');
            
            fetch('/training_status')
            .then(response => response.json())
            .then(data => {
                progressBar.style.width = data.progress + '%';
                progressBar.textContent = data.progress + '%';
                messageElem.textContent = data.message;
                
                if (data.is_training) {
                    setTimeout(checkTrainingStatus, 1000);
                } else if (data.progress === 100) {
                    messageElem.textContent = 'Training complete! Refreshing metrics...';
                    setTimeout(() => {
                        loadMetrics();
                        messageElem.textContent = 'Training complete! Metrics updated.';
                    }, 2000);
                }
            })
            .catch(error => {
                messageElem.textContent = 'Error checking status: ' + error;
            });
        }
        
        function openVisualizations() {
            // Open visualizations in a new tab
            window.open('/visualizations', '_blank');
        }
        
        function generateVisualizations() {
            const statusDiv = document.getElementById('visualization-status');
            statusDiv.textContent = 'Generating visualizations...';
            statusDiv.className = 'status-message';
            
            fetch('/start_visualizations', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    statusDiv.textContent = data.message;
                    statusDiv.className = 'status-message success-msg';
                } else {
                    statusDiv.textContent = 'Error: ' + data.message;
                    statusDiv.className = 'status-message error-msg';
                }
            })
            .catch(error => {
                statusDiv.textContent = 'Error: ' + error;
                statusDiv.className = 'status-message error-msg';
            });
        }
    </script>
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

if __name__ == '__main__':
    # Load models at startup
    if load_models():
        print("Models loaded successfully!")
    else:
        print("Failed to load models. Please check model files.")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
