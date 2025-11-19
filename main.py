# === DIABETES PREDICTION ML SCRIPT ===
# This script will:
# 1. Automatically download the dataset from Kaggle if not found.
# 2. Train 3 ML models (Logistic Regression, Random Forest, XGBoost).
# 3. Generate the outputs for Table 1, Table 2, and Figure 1 for your report.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import os
import zipfile

# --- Kaggle Dataset Details ---
KAGGLE_DATASET_NAME = 'mohankrishnathalla/diabetes-health-indicators-dataset'
DATASET_FILENAME = 'diabetes_health_indicators_dataset.csv'
FIGURE_FILENAME = 'diabetes_figure1.png'

def download_dataset_if_needed():
    """
    Checks if the dataset CSV exists. If not, downloads and extracts it from Kaggle.
    """
    if os.path.exists(DATASET_FILENAME):
        print(f"'{DATASET_FILENAME}' found locally.")
        return True

    print(f"'{DATASET_FILENAME}' not found. Attempting to download from Kaggle...")
    
    try:
        # Try to import kaggle, this is the main dependency
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("\n" + "="*50)
        print("ERROR: 'kaggle' library not found.")
        print("Please install it by running: pip install kaggle")
        print("="*50 + "\n")
        return False
    except Exception as e:
        print(f"\n" + "="*50)
        print(f"ERROR: Could not initialize Kaggle API. Is your 'kaggle.json' file in the right place?")
        print(f"Please download it from your Kaggle account (Settings > API > Create New Token)")
        print(f"And place it in: ~/.kaggle/kaggle.json (Linux/Mac) or C:\\Users\\<Your-Username>\\.kaggle\\kaggle.json (Windows)")
        print(f"Error details: {e}")
        print("="*50 + "\n")
        return False

    # If kaggle library is present, authenticate and download
    try:
        api = KaggleApi()
        api.authenticate()
        
        print(f"Downloading '{KAGGLE_DATASET_NAME}'...")
        # This dataset is a single zip file that contains two CSVs.
        api.dataset_download_files(KAGGLE_DATASET_NAME, path='.', quiet=False, unzip=True)
        
        # After unzipping, the file we want should be present.
        if os.path.exists(DATASET_FILENAME):
            print(f"Successfully downloaded and extracted '{DATASET_FILENAME}'.")
            
            # Clean up the *other* CSV file we don't need
            other_csv = 'diabetes_012_health_indicators_BRFSS2015.csv'
            if os.path.exists(other_csv):
                os.remove(other_csv)
                print(f"Cleaned up '{other_csv}'.")
                
            return True
        else:
            print(f"Error: Kaggle download did not produce the expected file: {DATASET_FILENAME}")
            # Check for the zip file in case unzip failed
            zip_filename = KAGGLE_DATASET_NAME.split('/')[1] + '.zip'
            if os.path.exists(zip_filename):
                print(f"Found zip file '{zip_filename}'. Attempting to extract...")
                with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
                    zip_ref.extractall('.')
                if os.path.exists(DATASET_FILENAME):
                    print("Extraction successful.")
                    os.remove(zip_filename) # Clean up zip
                    # Clean up the *other* CSV file we don't need
                    other_csv = 'diabetes_012_health_indicators_BRFSS2015.csv'
                    if os.path.exists(other_csv):
                        os.remove(other_csv)
                        print(f"Cleaned up '{other_csv}'.")
                    return True
                else:
                    print("Extraction failed to produce the file.")
                    return False
            return False

    except Exception as e:
        print(f"\n" + "="*50)
        print(f"An error occurred during download/extraction: {e}")
        print("Please check your Kaggle API setup and dataset name.")
        print("="*50 + "\n")
        return False

def run_analysis():
    """
    Main function to run the ML analysis after ensuring data is present.
    """
    print("Loading Diabetes Health Indicators data...")
    try:
        df = pd.read_csv(DATASET_FILENAME)
    except FileNotFoundError:
        print(f"Error: '{DATASET_FILENAME}' not found even after download attempt.")
        print("Please manually download the file from Kaggle and place it in this folder.")
        return

    print(f"Data loaded. Shape: {df.shape}")

    # === STEP 2: [OUTPUT FOR REPORT: TABLE 1] ===
    # This section generates the data for the first placeholder in your report.
    print("\n" + "="*30)
    print("OUTPUT FOR REPORT: TABLE 1")
    print("="*30)
    # Determine the target column (support common variants) for Table 1 display
    target_col = None
    if 'Diabetes_binary' in df.columns:
        target_col = 'Diabetes_binary'
    elif 'diagnosed_diabetes' in df.columns:
        target_col = 'diagnosed_diabetes'
    elif 'diabetes_binary' in df.columns:
        target_col = 'diabetes_binary'
    elif 'diabetes_stage' in df.columns:
        # Map stages to binary: 'No Diabetes' -> 0, everything else -> 1
        df['__diabetes_stage_mapped'] = df['diabetes_stage'].apply(lambda x: 0 if str(x).strip().lower() in ['no diabetes', 'no', 'none', 'nan', ''] else 1)
        target_col = '__diabetes_stage_mapped'

    if target_col is None:
        candidates = [c for c in df.columns if 'diabet' in c.lower()]
        if candidates:
            target_col = candidates[0]

    if target_col is None:
        print("Error: Could not find a target column like 'Diabetes_binary' or 'diagnosed_diabetes' in the CSV.")
        print("Columns found:\n", df.columns.tolist())
        return

    imbalance_report = df[target_col].value_counts(normalize=True) * 100
    imbalance_counts = df[target_col].value_counts()

    print("Class Balance Report:")
    # Safely handle labels 0/1 or other encodings
    try:
        no_count = imbalance_counts[0]
        has_count = imbalance_counts[1]
        no_pct = imbalance_report[0]
        has_pct = imbalance_report[1]
        print(f"No Diabetes (0): {no_count} samples ({no_pct:.2f}%)")
        print(f"Has Diabetes (1): {has_count} samples ({has_pct:.2f}%)")
    except Exception:
        # Fallback: print all value counts
        print(imbalance_counts)
    print("\n--> COPY the counts and percentages above into Table 1 of your report.")
    print("="*30 + "\n")


    # === STEP 3: FEATURE (X) & TARGET (y) DEFINITION ===
    print("Defining features (X) and target (y)...")

    # Determine the target column (support common variants)
    target_col = None
    if 'Diabetes_binary' in df.columns:
        target_col = 'Diabetes_binary'
    elif 'diagnosed_diabetes' in df.columns:
        target_col = 'diagnosed_diabetes'
    elif 'diabetes_binary' in df.columns:
        target_col = 'diabetes_binary'
    elif 'diabetes_stage' in df.columns:
        # Map stages to binary: 'No Diabetes' -> 0, everything else -> 1
        df['__diabetes_stage_mapped'] = df['diabetes_stage'].apply(lambda x: 0 if str(x).strip().lower() in ['no diabetes', 'no', 'none', 'nan', ''] else 1)
        target_col = '__diabetes_stage_mapped'

    if target_col is None:
        # Try to find any column name containing 'diabet'
        candidates = [c for c in df.columns if 'diabet' in c.lower()]
        if candidates:
            target_col = candidates[0]

    if target_col is None:
        print("Error: Could not find a target column like 'Diabetes_binary' or 'diagnosed_diabetes' in the CSV.")
        print("Columns found:\n", df.columns.tolist())
        return

    # 'target_col' should now point to our Y (1 = Has Diabetes, 0 = No Diabetes)
    y = df[target_col].astype(int)

    # X is all numeric columns EXCEPT the target
    X = df.drop(columns=[target_col], errors='ignore')
    # Keep only numeric features (the downloaded CSV contains many non-numeric columns)
    X = X.select_dtypes(include=[np.number])

    # This dataset file may contain an alternate target column we don't need; drop if present
    if 'Diabetes_012' in X.columns:
        X = X.drop('Diabetes_012', axis=1)

    all_feature_names = X.columns.tolist()
    print(f"Using {len(all_feature_names)} features for prediction.")


    # === STEP 4: PRE-PROCESSING (SCALING) ===
    # All features are numeric (binary 0/1 or scales 1-5, 1-13).
    # We will standardize them all for the models.
    print("Scaling all features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Convert back to DataFrame to keep feature names
    X = pd.DataFrame(X_scaled, columns=all_feature_names)


    # === STEP 5: TRAIN-TEST SPLIT ===
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, # 20% for testing
        random_state=42,
        stratify=y # CRITICAL: Ensures our 84%/16% split is in both sets
    )
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size:  {X_test.shape[0]} samples")


    # === STEP 6: TRAIN MODELS ===
    # We will use class weighting to handle the 84/16 imbalance.
    print("Training models... (This may take a minute or two)")

    # Model 1: Logistic Regression (Baseline)
    # 'balanced' mode automatically adjusts for imbalance.
    print("Fitting Logistic Regression...")
    model_lr = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    model_lr.fit(X_train, y_train)

    # Model 2: Random Forest
    # 'balanced' mode automatically adjusts for imbalance.
    print("Fitting Random Forest...")
    model_rf = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
    model_rf.fit(X_train, y_train)

    # Model 3: XGBoost
    # 'scale_pos_weight' is how XGBoost handles imbalance.
    # It's (Count of negatives / Count of positives)
    scale_pos_weight = (y == 0).sum() / (y == 1).sum()
    print(f"Using XGBoost scale_pos_weight: {scale_pos_weight:.2f}")
    
    print("Fitting XGBoost...")
    model_xgb = XGBClassifier(
        random_state=42, 
        n_jobs=-1, 
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight
    )
    model_xgb.fit(X_train, y_train)

    print("All models trained.")


    # === STEP 7: [OUTPUT FOR REPORT: TABLE 2] ===
    # This section generates the data for the second placeholder in your report.
    print("\n" + "="*30)
    print("OUTPUT FOR REPORT: TABLE 2")
    print("="*30)

    models = {
        "Logistic Regression": model_lr,
        "Random Forest": model_rf,
        "XGBoost": model_xgb
    }

    report_data = []

    for name, model in models.items():
        print(f"\n--- Classification Report for {name} ---")
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, target_names=['No Diabetes (0)', 'Has Diabetes (1)'], output_dict=True)
        
        # Print the full report for you to see
        print(classification_report(y_test, y_pred, target_names=['No Diabetes (0)', 'Has Diabetes (1)']))
        
        # Store the data for your report's table
        report_data.append({
            "Model": name,
            "Accuracy": f"{report['accuracy'] * 100:.2f}%",
            "F1-Score (Class 0)": f"{report['No Diabetes (0)']['f1-score']:.2f}",
            "F1-Score (Class 1)": f"{report['Has Diabetes (1)']['f1-score']:.2f}",
            "Macro Avg F1-Score": f"{report['macro avg']['f1-score']:.2f}"
        })

    print("\n--> COPY THE DATA BELOW INTO Table 2 of your report:")
    print_friendly_table = pd.DataFrame(report_data).set_index('Model')
    print(print_friendly_table.to_markdown(numalign="left", stralign="left"))
    print("="*30 + "\n")


    # === STEP 8: [OUTPUT FOR REPORT: FIGURE 1] ===
    # This section generates the data for the third placeholder in your report.
    print("\n" + "="*30)
    print("OUTPUT FOR REPORT: FIGURE 1")
    print("="*30)

    print("Generating Feature Importance plot (from Random Forest)...")

    # We'll use the Random Forest for feature importance as it's robust
    importances = pd.DataFrame(
        data={'feature': all_feature_names, 'importance': model_rf.feature_importances_}
    ).sort_values(by='importance', ascending=False)

    print("\nTop Features:")
    print(importances.to_markdown(index=False, numalign="left", stralign="left"))

    # Plot the horizontal bar chart
    plt.figure(figsize=(12, 10)) # Made figure taller to fit all 21 features
    sns.barplot(
        x='importance',
        y='feature',
        data=importances,
        palette='viridis'
    )
    plt.title('Feature Importance for Predicting Diabetes (Random Forest)', fontsize=16)
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()

    # Save the figure
    plt.savefig(FIGURE_FILENAME, dpi=300)

    print(f"\n--> SUCCESS: '{FIGURE_FILENAME}' has been saved.")
    print("--> This is the bar chart for Figure 1 in your report.")
    print("="*30 + "\n")

    print("=== SCRIPT FINISHED ===")


# --- This runs the main script ---
if __name__ == "__main__":
    # First, try to download the data if it's missing
    if download_dataset_if_needed():
        # If download is successful (or file already exists), run the analysis
        run_analysis()
    else:
        print("Script aborted because dataset could not be found or downloaded.")