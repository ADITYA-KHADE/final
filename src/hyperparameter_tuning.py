import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import json
import os
import time
import pickle
from itertools import product
import pandas as pd
from hate_speech_model import LSTM_CNN, preprocess_text, text_to_sequence, ErrorAnalyzer

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_dataset(data_path, use_advanced_preprocessing=True):
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
        
        # Apply advanced preprocessing if requested
        if use_advanced_preprocessing:
            print("Applying advanced preprocessing...")
            df_clean['processed_text'] = df_clean['text'].apply(lambda x: preprocess_text(x, 
                                                                             use_lemmatization=True,
                                                                             handle_slang=True))
        else:
            print("Applying standard preprocessing...")
            df_clean['processed_text'] = df_clean['text'].apply(preprocess_text)
        
        return df_clean
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
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

def prepare_data_for_tuning(data_path, vocab=None, test_size=0.2, random_state=42, max_vocab_size=10000, 
                           use_advanced_preprocessing=True):
    """Prepare data for hyperparameter tuning"""
    # Load and preprocess data
    df = load_dataset(data_path, use_advanced_preprocessing)
    if df is None:
        return None, None, None, None, None
    
    # Split data into train and validation sets
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df['label'])
    
    # Build vocabulary if not provided
    if vocab is None:
        vocab = build_vocab(train_df['processed_text'], max_vocab_size)
        
    # Convert texts to sequences
    X_train = train_df['processed_text'].apply(lambda x: text_to_sequence(x, vocab)).tolist()
    y_train = train_df['label'].tolist()
    
    X_val = val_df['processed_text'].apply(lambda x: text_to_sequence(x, vocab)).tolist()
    y_val = val_df['label'].tolist()
    
    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.long)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_val = torch.tensor(X_val, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)
    
    return X_train, y_train, X_val, y_val, vocab

def train_model_with_params(params, X_train, y_train, X_val, y_val, vocab_size, num_classes=2, 
                           epochs=3, batch_size=32, early_stopping_patience=2):
    """Train a model with specific hyperparameters"""
    # Create model with these parameters
    model = LSTM_CNN(
        vocab_size=vocab_size,
        embed_dim=params['embed_dim'],
        lstm_hidden_dim=params['lstm_hidden_dim'],
        cnn_hidden_dim=params['cnn_hidden_dim'],
        num_classes=num_classes,
        dropout=params['dropout'],
        num_filters=params.get('num_filters', 100),
        filter_sizes=params.get('filter_sizes', [3, 4, 5])
    )
    model.to(device)
    
    # Create dataloaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    
    # Training loop
    best_val_loss = float('inf')
    best_val_f1 = 0.0
    patience_counter = 0
    train_losses = []
    val_losses = []
    val_f1s = []
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss = train_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)
        
        # Calculate F1 score
        from sklearn.metrics import f1_score
        val_f1 = f1_score(all_labels, all_preds, pos_label=1, average='binary')
        val_f1s.append(val_f1)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}, F1: {val_f1:.4f}")
        
        # Check for early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Calculate final metrics
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.metrics import precision_score, recall_score, accuracy_score
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, pos_label=1, zero_division=0)
    recall = recall_score(all_labels, all_preds, pos_label=1, zero_division=0)
    f1 = f1_score(all_labels, all_preds, pos_label=1, zero_division=0)
    conf_matrix = confusion_matrix(all_labels, all_preds).tolist()
    
    # Save model state
    model_state = {
        'state_dict': model.state_dict(),
        'vocab_size': vocab_size,
        'embed_dim': params['embed_dim'],
        'lstm_hidden_dim': params['lstm_hidden_dim'],
        'cnn_hidden_dim': params['cnn_hidden_dim'],
        'num_classes': num_classes,
        'dropout': params['dropout'],
        'num_filters': params.get('num_filters', 100),
        'filter_sizes': params.get('filter_sizes', [3, 4, 5]),
        'val_f1': best_val_f1
    }
    
    # Return results
    results = {
        'params': params,
        'metrics': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'best_val_f1': float(best_val_f1),
            'train_loss': train_losses,
            'val_loss': val_losses,
            'val_f1': val_f1s,
            'confusion_matrix': conf_matrix,
            'training_time': training_time
        },
        'model_state': model_state
    }
    
    return results

def run_hyperparameter_tuning(data_path, output_dir='./hyperparameter_tuning_results', 
                             vocab_path=None, use_advanced_preprocessing=True):
    """Run hyperparameter tuning and save results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Define hyperparameter grid
    param_grid = {
        'embed_dim': [50, 100, 200],
        'lstm_hidden_dim': [64, 128, 256],
        'cnn_hidden_dim': [64, 128, 256],
        'dropout': [0.3, 0.5, 0.7],
        'learning_rate': [1e-4, 1e-3, 5e-3],
        'num_filters': [50, 100, 150],
        'filter_sizes': [[3], [3, 4, 5], [2, 3, 4, 5]]
    }
    
    # Load vocab if path provided
    vocab = None
    if vocab_path and os.path.exists(vocab_path):
        try:
            with open(vocab_path, 'rb') as f:
                vocab = pickle.load(f)
            print(f"Loaded vocabulary with {len(vocab)} entries")
        except Exception as e:
            print(f"Error loading vocabulary: {e}")
    
    # Prepare data
    X_train, y_train, X_val, y_val, data_vocab = prepare_data_for_tuning(
        data_path, 
        vocab=vocab,
        use_advanced_preprocessing=use_advanced_preprocessing
    )
    
    # If vocab wasn't loaded, use the one created during data preparation
    if vocab is None:
        vocab = data_vocab
        
    if X_train is None:
        print("Error preparing data. Exiting.")
        return
    
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")
    
    # Generate parameter combinations
    # To limit combinations, we'll select a subset of meaningful combinations
    # rather than testing all possible combinations
    selected_params = [
        # Baseline
        {'embed_dim': 100, 'lstm_hidden_dim': 128, 'cnn_hidden_dim': 128, 'dropout': 0.5, 
         'learning_rate': 1e-3, 'num_filters': 100, 'filter_sizes': [3, 4, 5]},
        
        # Smaller model
        {'embed_dim': 50, 'lstm_hidden_dim': 64, 'cnn_hidden_dim': 64, 'dropout': 0.3, 
         'learning_rate': 1e-3, 'num_filters': 50, 'filter_sizes': [3]},
        
        # Larger model
        {'embed_dim': 200, 'lstm_hidden_dim': 256, 'cnn_hidden_dim': 256, 'dropout': 0.5, 
         'learning_rate': 1e-3, 'num_filters': 150, 'filter_sizes': [2, 3, 4, 5]},
        
        # Different learning rates
        {'embed_dim': 100, 'lstm_hidden_dim': 128, 'cnn_hidden_dim': 128, 'dropout': 0.5, 
         'learning_rate': 1e-4, 'num_filters': 100, 'filter_sizes': [3, 4, 5]},
        
        # Higher regularization
        {'embed_dim': 100, 'lstm_hidden_dim': 128, 'cnn_hidden_dim': 128, 'dropout': 0.7, 
         'learning_rate': 1e-3, 'num_filters': 100, 'filter_sizes': [3, 4, 5]},
    ]
    
    # Run hyperparameter tuning
    all_results = []
    best_f1 = 0
    best_params = None
    best_model_state = None
    
    for i, params in enumerate(selected_params):
        print("\n" + "="*60)
        print(f"Testing parameter combination {i+1}/{len(selected_params)}")
        print(json.dumps(params, indent=2))
        
        try:
            result = train_model_with_params(
                params, X_train, y_train, X_val, y_val, vocab_size,
                epochs=5, batch_size=32, early_stopping_patience=2
            )
            
            all_results.append(result)
            
            # Check if this is the best model so far
            current_f1 = result['metrics']['f1']
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_params = params
                best_model_state = result['model_state']
                
                # Save the best model so far
                best_model_path = os.path.join(output_dir, 'best_model_so_far.pth')
                torch.save(best_model_state, best_model_path)
                print(f"New best model with F1 score: {best_f1:.4f}, saved to {best_model_path}")
                
        except Exception as e:
            print(f"Error with params {params}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save all results
    with open(os.path.join(output_dir, 'all_tuning_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Save best model
    if best_model_state is not None:
        best_model_path = os.path.join(output_dir, 'best_tuned_model.pth')
        torch.save(best_model_state, best_model_path)
        
        # Also save the vocabulary
        vocab_path = os.path.join(output_dir, 'vocab.pkl')
        with open(vocab_path, 'wb') as f:
            pickle.dump(vocab, f)
        
        print(f"\nBest model saved to {best_model_path}")
        print(f"Best parameters: {json.dumps(best_params, indent=2)}")
        print(f"Best F1 score: {best_f1:.4f}")
    
    return {
        'best_params': best_params,
        'best_f1': best_f1,
        'all_results': all_results,
        'best_model_path': best_model_path if best_model_state is not None else None
    }

def analyze_errors_on_dataset(model_path, vocab_path, data_path, output_path, num_samples=100):
    """Analyze errors made by a model on a dataset"""
    # Load model and vocab
    from hate_speech_model import load_model_and_vocab, ErrorAnalyzer
    model, vocab = load_model_and_vocab(model_path, vocab_path)
    
    if model is None or vocab is None:
        print("Failed to load model or vocabulary")
        return
    
    # Load dataset
    df = load_dataset(data_path)
    if df is None:
        print("Failed to load dataset")
        return
    
    # Initialize error analyzer
    error_analyzer = ErrorAnalyzer(model, vocab)
    
    # Sample examples for analysis
    if len(df) > num_samples:
        df_sample = df.sample(num_samples, random_state=42)
    else:
        df_sample = df
    
    print(f"Analyzing {len(df_sample)} examples...")
    
    # Analyze each example
    for _, row in df_sample.iterrows():
        text = row['text']
        true_label = row['label']
        error_analyzer.analyze_error(text, true_label)
    
    # Save error analysis
    error_summary = error_analyzer.get_error_summary()
    with open(output_path, 'w') as f:
        json.dump(error_summary, f, indent=2)
    
    print(f"Error analysis saved to {output_path}")
    return error_summary

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for hate speech detection model')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Tuning command
    tune_parser = subparsers.add_parser('tune', help='Run hyperparameter tuning')
    tune_parser.add_argument('--data', type=str, required=True, help='Path to training data CSV')
    tune_parser.add_argument('--output-dir', type=str, default='./hyperparameter_tuning_results', 
                            help='Directory to save results')
    tune_parser.add_argument('--vocab', type=str, default=None, help='Path to existing vocabulary pickle file')
    tune_parser.add_argument('--no-advanced-preprocessing', action='store_true', 
                            help='Disable advanced preprocessing')
    
    # Error analysis command
    error_parser = subparsers.add_parser('analyze-errors', help='Analyze model errors')
    error_parser.add_argument('--model', type=str, required=True, help='Path to model file')
    error_parser.add_argument('--vocab', type=str, required=True, help='Path to vocabulary file')
    error_parser.add_argument('--data', type=str, required=True, help='Path to dataset CSV')
    error_parser.add_argument('--output', type=str, required=True, help='Path to save error analysis')
    error_parser.add_argument('--samples', type=int, default=100, help='Number of samples to analyze')
    
    args = parser.parse_args()
    
    if args.command == 'tune':
        print("Running hyperparameter tuning...")
        run_hyperparameter_tuning(
            args.data,
            output_dir=args.output_dir,
            vocab_path=args.vocab,
            use_advanced_preprocessing=not args.no_advanced_preprocessing
        )
    elif args.command == 'analyze-errors':
        print("Running error analysis...")
        analyze_errors_on_dataset(
            args.model,
            args.vocab,
            args.data,
            args.output,
            num_samples=args.samples
        )
    else:
        parser.print_help()