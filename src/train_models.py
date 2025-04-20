import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import os

# Import LSTM+CNN model
from hate_speech_model import LSTM_CNN, preprocess_text, text_to_sequence

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_dataset(data_path):
    """Load and preprocess the dataset"""
    print(f"Loading dataset from {data_path}...")
    
    # Load the CSV file
    try:
        df = pd.read_csv(data_path, encoding='latin1')
        print(f"Dataset loaded with {len(df)} samples")
        
        # Identify columns
        if 'class' in df.columns:
            # Convert class to binary labels (1 for hate speech, 0 for non-hate speech)
            df['label'] = df['class'].apply(lambda x: 1 if x == 0 else 0)
        
        # Identify text column
        text_col = None
        if 'tweet' in df.columns:
            text_col = 'tweet'
        elif 'text' in df.columns:
            text_col = 'text'
        else:
            # Try to identify the text column
            text_cols = [col for col in df.columns if 'text' in col.lower() or 'tweet' in col.lower()]
            if text_cols:
                text_col = text_cols[0]
        
        if text_col is None:
            raise ValueError("Could not identify text column in dataset")
            
        # Create clean dataset with only text and label columns
        df_clean = df[[text_col, 'label']].rename(columns={text_col: 'text'})
        
        # Print class distribution
        class_dist = df_clean['label'].value_counts()
        print(f"Class distribution:\n{class_dist}")
        
        return df_clean
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def train_lstm_cnn_model(df, epochs=5, batch_size=32, save_path='hate_speech_model_lstm_cnn.pth'):
    """Train the LSTM-CNN model"""
    print("\n====== Training LSTM-CNN Model ======")
    
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
    
    # Build vocabulary from training data
    print("Building vocabulary...")
    all_texts = train_df_balanced['text'].apply(preprocess_text).tolist()
    
    # Count word frequencies for more consistent vocabulary
    counter = {}
    for text in all_texts:
        words = text.split()
        for word in words:
            if word in counter:
                counter[word] += 1
            else:
                counter[word] = 1
    
    # Sort by frequency for consistent vocabulary between runs
    sorted_words = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    
    # Create vocabulary dictionary
    vocab = {' ': 0}  # Padding token
    for i, (word, _) in enumerate(sorted_words):
        vocab[word] = i + 1
    
    print(f"Vocabulary size: {len(vocab)}")
    
    # Save vocabulary
    with open('vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)
    print("Vocabulary saved to vocab.pkl")
    
    # Convert texts to sequences
    print("Converting texts to sequences...")
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
    
    # Initialize model
    vocab_size = len(vocab)
    embed_dim = 100
    lstm_hidden_dim = 128
    cnn_hidden_dim = 128
    num_classes = 2
    dropout = 0.5
    
    model = LSTM_CNN(vocab_size, embed_dim, lstm_hidden_dim, cnn_hidden_dim, num_classes, dropout)
    model.to(device)
    
    # Define loss function and optimizer
    class_weights = torch.tensor([0.3, 0.7], device=device)  # Adjust for class imbalance
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Training loop
    best_val_f1 = 0
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        
        for inputs, labels in tqdm(train_loader, desc="Training"):
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
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        val_loss = val_loss / len(val_loader)
        
        # Calculate F1 score
        val_f1 = f1_score(all_labels, all_preds, pos_label=1)
        
        print(f"Training Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, F1 (hate speech): {val_f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=['Non-Hate', 'Hate']))
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            print("New best model found! Saving...")
            
            model_info = {
                'state_dict': model.state_dict(),
                'vocab_size': vocab_size,
                'embed_dim': embed_dim,
                'lstm_hidden_dim': lstm_hidden_dim,
                'cnn_hidden_dim': cnn_hidden_dim,
                'num_classes': num_classes,
                'dropout': dropout,
                'val_f1': val_f1
            }
            
            torch.save(model_info, save_path)
    
    print(f"\nLSTM-CNN Training complete! Best validation F1: {best_val_f1:.4f}")
    print(f"Model saved to {save_path}")
    return model, vocab

if __name__ == "__main__":
    # Set the data path to Dataset_3.csv
    data_path = '../Data/Dataset_3.csv'
    
    # Load dataset
    df = load_dataset(data_path)
    
    if df is not None:
        # Train LSTM-CNN model
        lstm_cnn_model, vocab = train_lstm_cnn_model(df, epochs=10)
        
        print("\n======== Training Complete ========")
        print("LSTM-CNN model successfully trained on Dataset_3.csv")
        print("Model saved to: hate_speech_model_lstm_cnn.pth")
        print("Vocabulary saved to: vocab.pkl")
    else:
        print("Failed to load dataset. Please check the file path and format.")
