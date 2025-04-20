import os
import sys
import torch
import pickle
from data_visualizations import HateSpeechVisualizer
from hate_speech_model import HateSpeechDetector

def main():
    """
    Generate data visualizations from hate speech datasets and create an interactive dashboard
    """
    # Directory paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(current_dir), 'Data')
    output_dir = os.path.join(current_dir, 'visualizations')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model and vocabulary (optional, for advanced visualizations)
    model_path = os.path.join(current_dir, 'hate_speech_model_lstm_cnn.pth')
    vocab_path = os.path.join(current_dir, 'vocab.pkl')
    
    model = None
    vocab = None
    
    try:
        # Load model
        model = HateSpeechDetector()
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        
        # Load vocabulary
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        
        print("Model and vocabulary loaded successfully!")
    except Exception as e:
        print(f"Warning: Could not load model or vocabulary: {e}")
        print("Some advanced visualizations will not be available.")
    
    # Initialize visualizer
    visualizer = HateSpeechVisualizer(model=model, vocab=vocab)
    
    # Find all CSV files in the Data directory
    dataset_paths = []
    for file in os.listdir(data_dir):
        if file.endswith('.csv'):
            dataset_paths.append(os.path.join(data_dir, file))
    
    # If no datasets found, exit
    if not dataset_paths:
        print("No CSV datasets found in the Data directory.")
        return
    
    print(f"Found {len(dataset_paths)} datasets: {[os.path.basename(path) for path in dataset_paths]}")
    
    # Process each dataset
    for dataset_path in dataset_paths:
        dataset_name = os.path.basename(dataset_path).replace('.csv', '')
        print(f"\nProcessing dataset: {dataset_name}")
        
        # Load dataset
        df = visualizer.load_data(dataset_path)
        if df is None:
            print(f"Failed to load dataset: {dataset_path}")
            continue
        
        # Create directory for dataset visualizations
        dataset_viz_dir = os.path.join(output_dir, dataset_name)
        os.makedirs(dataset_viz_dir, exist_ok=True)
        
        # Generate individual visualizations
        print("Generating visualizations...")
        
        # Basic visualizations
        visualizer.plot_class_distribution(df, save_path=os.path.join(dataset_viz_dir, 'class_distribution.png'))
        visualizer.plot_text_length_distribution(df, save_path=os.path.join(dataset_viz_dir, 'text_length_distribution.png'))
        visualizer.plot_word_frequency(df, class_value=1, save_path=os.path.join(dataset_viz_dir, 'word_freq_hate.png'))
        visualizer.plot_word_frequency(df, class_value=0, save_path=os.path.join(dataset_viz_dir, 'word_freq_nonhate.png'))
        visualizer.create_word_cloud(df, class_value=1, save_path=os.path.join(dataset_viz_dir, 'wordcloud_hate.png'))
        visualizer.create_word_cloud(df, class_value=0, save_path=os.path.join(dataset_viz_dir, 'wordcloud_nonhate.png'))
        visualizer.plot_bigram_comparison(df, save_path=os.path.join(dataset_viz_dir, 'bigram_comparison.png'))
        
        # Advanced visualizations (model-dependent)
        if model is not None and vocab is not None:
            try:
                visualizer.plot_category_distribution(df, sample_size=100, save_path=os.path.join(dataset_viz_dir, 'category_distribution.png'))
                visualizer.plot_confidence_distribution(df, sample_size=100, save_path=os.path.join(dataset_viz_dir, 'confidence_distribution.png'))
                visualizer.visualize_text_embeddings(df, sample_size=500, method='TSNE', save_path=os.path.join(dataset_viz_dir, 'text_embeddings.png'))
            except Exception as e:
                print(f"Warning: Could not generate advanced visualizations: {e}")
        
        # Generate HTML dashboard
        dashboard_path = os.path.join(dataset_viz_dir, 'dashboard.html')
        visualizer.generate_dashboard_html(df, output_path=dashboard_path)
        print(f"Dashboard generated: {dashboard_path}")
    
    # Compare datasets
    if len(dataset_paths) > 1:
        print("\nGenerating dataset comparison visualization...")
        comparison_path = os.path.join(output_dir, 'dataset_comparison.png')
        visualizer.plot_dataset_comparison(dataset_paths, save_path=comparison_path)
        print(f"Dataset comparison saved to: {comparison_path}")
    
    print("\nVisualization generation complete!")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    main()