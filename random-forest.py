import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def load_dataset(dataset_path):
    """Load the preprocessed soil dataset"""
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    return dataset

def train_and_save_model(dataset_path, model_save_path):
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(dataset_path)
    
    features = dataset['features']
    labels = dataset['labels']
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )
    
    # Initialize and train model
    print("Training Random Forest model...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    )
    
    rf_model.fit(X_train, y_train)
    
    # Evaluate model
    train_score = rf_model.score(X_train, y_train)
    test_score = rf_model.score(X_test, y_test)
    
    print(f"\nTraining accuracy: {train_score:.4f}")
    print(f"Testing accuracy: {test_score:.4f}")
    
    # Generate and plot confusion matrix
    y_pred = rf_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model and label names
    print("\nSaving model...")
    model_data = {
        'model': rf_model,
        'label_names': dataset['metadata']['label_names']
    }
    
    with open(model_save_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Model saved to: {model_save_path}")

if __name__ == "__main__":
    # Paths
    dataset_path = '/content/drive/MyDrive/ML/tanah/soil_dataset.pkl'
    model_save_path = '/content/drive/MyDrive/ML/tanah/soil_classifier_rf.pkl'
    
    # Train and save model
    train_and_save_model(dataset_path, model_save_path)