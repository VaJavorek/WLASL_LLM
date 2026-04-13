import pandas as pd
import argparse
import os
from datetime import datetime

def calculate_top1_accuracy(csv_file_path):
    """Calculate top-1 accuracy from WLASL predictions CSV file."""
    
    # Check if file exists
    if not os.path.exists(csv_file_path):
        print(f"Error: File '{csv_file_path}' not found!")
        return None
    
    # Load the CSV file
    print(f"Loading predictions from: {csv_file_path}")
    df = pd.read_csv(csv_file_path)
    
    # Display basic info
    print(f"Total predictions: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    # Remove error entries (predictions that start with "ERROR:")
    error_mask = df['predicted_gloss'].astype(str).str.startswith('ERROR:')
    error_count = error_mask.sum()
    
    if error_count > 0:
        print(f"Found {error_count} error entries, excluding from accuracy calculation")
        df_clean = df[~error_mask].copy()
    else:
        df_clean = df.copy()
    
    print(f"Valid predictions for evaluation: {len(df_clean)}")
    
    if len(df_clean) == 0:
        print("No valid predictions found!")
        return None
    
    # Calculate exact match accuracy (case-insensitive)
    df_clean['gt_normalized'] = df_clean['ground_truth_gloss'].str.lower().str.strip()
    df_clean['pred_normalized'] = df_clean['predicted_gloss'].str.lower().str.strip()
    
    exact_matches = (df_clean['gt_normalized'] == df_clean['pred_normalized']).sum()
    top1_accuracy = exact_matches / len(df_clean)
    
    # Calculate some additional statistics
    avg_processing_time = df_clean['processing_time'].mean()
    total_processing_time = df_clean['processing_time'].sum()
    
    # Get unique glosses info
    unique_gt_glosses = df_clean['ground_truth_gloss'].nunique()
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Top-1 Accuracy: {top1_accuracy:.4f} ({top1_accuracy*100:.2f}%)")
    print(f"Correct predictions: {exact_matches}/{len(df_clean)}")
    print(f"Error rate: {error_count}/{len(df)} ({error_count/len(df)*100:.2f}%)")
    print(f"Unique ground truth glosses: {unique_gt_glosses}")
    print(f"Average processing time: {avg_processing_time:.2f} seconds")
    print(f"Total processing time: {total_processing_time:.1f} seconds")
    
    # Show some examples of correct and incorrect predictions
    print("\n" + "-"*30)
    print("SAMPLE CORRECT PREDICTIONS:")
    print("-"*30)
    correct_predictions = df_clean[df_clean['gt_normalized'] == df_clean['pred_normalized']]
    if len(correct_predictions) > 0:
        for i, row in correct_predictions.head(5).iterrows():
            print(f"GT: '{row['ground_truth_gloss']}' | Pred: '{row['predicted_gloss']}' | Video: {row['video_id']}")
    else:
        print("No correct predictions found.")
    
    print("\n" + "-"*30)
    print("SAMPLE INCORRECT PREDICTIONS:")
    print("-"*30)
    incorrect_predictions = df_clean[df_clean['gt_normalized'] != df_clean['pred_normalized']]
    if len(incorrect_predictions) > 0:
        for i, row in incorrect_predictions.head(5).iterrows():
            print(f"GT: '{row['ground_truth_gloss']}' | Pred: '{row['predicted_gloss']}' | Video: {row['video_id']}")
    else:
        print("All predictions are correct!")
    
    # Analyze most common prediction errors
    print("\n" + "-"*30)
    print("MOST FREQUENT GROUND TRUTH GLOSSES:")
    print("-"*30)
    gt_counts = df_clean['ground_truth_gloss'].value_counts().head(10)
    for gloss, count in gt_counts.items():
        accuracy_for_gloss = (df_clean[df_clean['ground_truth_gloss'] == gloss]['gt_normalized'] == 
                             df_clean[df_clean['ground_truth_gloss'] == gloss]['pred_normalized']).mean()
        print(f"{gloss}: {count} videos, {accuracy_for_gloss:.2f} accuracy")
    
    return {
        'top1_accuracy': top1_accuracy,
        'correct_predictions': exact_matches,
        'total_valid_predictions': len(df_clean),
        'total_predictions': len(df),
        'error_count': error_count,
        'unique_glosses': unique_gt_glosses,
        'avg_processing_time': avg_processing_time,
        'total_processing_time': total_processing_time
    }

def main():
    parser = argparse.ArgumentParser(description='Evaluate WLASL predictions CSV file')
    parser.add_argument('csv_file', nargs='?', help='Path to the predictions CSV file')
    
    args = parser.parse_args()
    
    print(f"WLASL Predictions Evaluation")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"="*50)
    
    results = calculate_top1_accuracy(args.csv_file)
    
    if results:
        print(f"\nEvaluation completed successfully!")
        print(f"Results saved in terminal output above.")
    else:
        print(f"Evaluation failed!")

if __name__ == "__main__":
    main() 