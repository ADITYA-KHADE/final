# Hate Speech Detection System

## Overview

This is a comprehensive hate speech detection system that uses machine learning to identify and categorize different types of hate speech in text data. The system includes:

- Text classification models (LSTM + CNN)
- Comprehensive hate speech detection with multiple categories
- Religious hate speech detection
- Data visualization dashboards
- Web interface for analysis and visualization

## Project Structure

```
├── Data/                     # Dataset files
│   ├── Dataset_1.csv         # Twitter dataset
│   ├── Dataset_2.csv         # Social media dataset
│   ├── Dataset_3.csv         # Multi-class labeled dataset
│   └── labeled_data.csv      # Main labeled dataset
├── src/                      # Source code
│   ├── app.py                # Flask web application
│   ├── comprehensive_hate_speech.py  # Comprehensive hate speech detector
│   ├── data_visualizations.py        # Visualization utilities
│   ├── generate_visualizations.py    # Script to generate dashboards
│   ├── hate_speech_model.py          # ML model implementation
│   ├── religious_hate_speech.py      # Religious hate speech detector
│   ├── train_models.py               # Model training script
│   ├── hyperparameter_tuning.py      # Model tuning script
│   └── visualizations/               # Generated visualization dashboards
```

## System Requirements

- Python 3.7+
- CUDA-capable GPU (recommended for training)
- 8GB+ RAM
- Windows/Linux/macOS

## Installation Instructions

1. Clone this repository:
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   ```

3. Activate the virtual environment:
   - Windows:
     ```
     venv\Scripts\activate
     ```
   - Linux/macOS:
     ```
     source venv/bin/activate
     ```

4. Install the required packages:
   ```
   pip install -r src/requirements.txt
   ```

## Running the Application

### Start the Web Application

1. Navigate to the src directory:
   ```
   cd src
   ```

2. Run the Flask application:
   ```
   python app.py
   ```

3. Open your web browser and go to:
   ```
   http://localhost:8000
   ```

### Generate Visualizations

To generate visualization dashboards for all datasets:

```
python src/generate_visualizations.py
```

This will create interactive HTML dashboards in the `src/visualizations` directory for each dataset.

### Train Models

To train the hate speech detection model:

```
python src/train_models.py
```

## Using the System

### Web Interface

The web interface allows you to:

1. Analyze text for hate speech
2. View model predictions and confidence scores
3. Browse visualization dashboards for datasets
4. Compare different datasets
5. Explore different categories of hate speech

### API Endpoints

- `/` - Home page
- `/analyze` - Text analysis page
- `/visualizations` - Visualization dashboards
- `/api/analyze` - API endpoint for text analysis

## Model Details

The system uses a hybrid LSTM+CNN architecture for text classification:

- Embedding layer for word representations
- LSTM layer to capture sequential information
- CNN layer for feature extraction
- Dense layers for classification

The model is enhanced with comprehensive hate speech detection that recognizes multiple categories:
- Religious hate speech
- Racial/ethnic hate speech
- Gender-based hate speech
- LGBTQ+ hate speech
- Disability-related hate speech
- Nationality/immigrant hate speech
- Sexual/explicit content

## Troubleshooting

- **WordCloud Generation Issues**: If you encounter issues with WordCloud generation on Windows, ensure you have a compatible font available (e.g., Arial, Verdana).
  
- **CUDA/GPU Issues**: If you encounter CUDA-related errors, check your PyTorch installation matches your CUDA version, or use CPU mode by modifying the device setting in the code.

- **Memory Errors**: For large datasets, you may need to increase the available memory or process datasets in smaller batches.

## Acknowledgements

This project uses several datasets and libraries:

- Twitter hate speech dataset
- PyTorch for deep learning
- NLTK for text processing
- Flask for the web interface
- Matplotlib and Seaborn for visualizations

## License

[Your License Information]

## Contact

[Your Contact Information]
