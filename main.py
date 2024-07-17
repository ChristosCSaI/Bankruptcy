import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from scipy.stats import ttest_ind

# Load dataset
data_path = 'Dataset2Use_Assignment2.xlsx'  # Change this to your dataset path
data = pd.read_excel(data_path)

# Data Preprocessing
X = data.iloc[:, :-2]  # Features
y = data.iloc[:, -2].apply(lambda x: 1 if x == 2 else 0)  # Recode target for 1 as bankrupt (2 in original data)

# Visualizations
# Figure 1: Number of healthy and bankrupt companies per year
plt.figure(figsize=(10, 6))
data.groupby(['Year', 'Status']).size().unstack().plot(kind='bar', stacked=True)
plt.title('Number of Healthy and Bankrupt Companies per Year')
plt.xlabel('Year')
plt.ylabel('Number of Companies')
plt.legend(['Healthy', 'Bankrupt'])
plt.show()

# Figure 2: Min, Max, Average values for each indicator
fig, axes = plt.subplots(2, 1, figsize=(14, 12))

# Healthy companies
data_healthy = data[data['Status'] == 1]
data_healthy.describe().loc[['min', 'max', 'mean'], 'Indicator1':'Indicator8'].transpose().plot(ax=axes[0])
axes[0].set_title('Healthy Companies: Min, Max, Average Values for Each Indicator')
axes[0].set_xlabel('Indicators')
axes[0].set_ylabel('Values')

# Bankrupt companies
data_bankrupt = data[data['Status'] == 2]
data_bankrupt.describe().loc[['min', 'max', 'mean'], 'Indicator1':'Indicator8'].transpose().plot(ax=axes[1])
axes[1].set_title('Bankrupt Companies: Min, Max, Average Values for Each Indicator')
axes[1].set_xlabel('Indicators')
axes[1].set_ylabel('Values')

plt.tight_layout()
plt.show()

# Normalize the feature data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Setting up Stratified K-Fold
skf = StratifiedKFold(n_splits=4)
unbalanced_results = []
balanced_results = []

# Defining models
models = {
    "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Trees": DecisionTreeClassifier(),
    "Random Forests": RandomForestClassifier(),
    "k-Nearest Neighbors": KNeighborsClassifier(),
    "Naïve Bayes": GaussianNB(),
    "Support Vector Machines": SVC(probability=True),
    "Extra Trees Classifier": RandomForestClassifier(n_estimators=100)  # Example of an additional model
}

# Function to collect results
def collect_results(model_name, y_true, y_pred, y_proba, dataset_type, balanced_status, fold):
    cm = confusion_matrix(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)
    return {
        'Classifier Name': model_name,
        'Training or Test Set': dataset_type,
        'Balanced or Unbalanced Train Set': balanced_status,
        'Fold': fold,
        'TP': cm[1, 1],
        'TN': cm[0, 0],
        'FP': cm[0, 1],
        'FN': cm[1, 0],
        'F1 Score': f1,
        'AUC ROC': auc
    }

# Model training and evaluation
for name, model in models.items():
    for fold, (train_index, test_index) in enumerate(skf.split(X_scaled, y), start=1):
        X_train, X_test = X_scaled.iloc[train_index], X_scaled.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Unbalanced Training
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        y_train_proba = model.predict_proba(X_train)[:, 1]
        y_test_proba = model.predict_proba(X_test)[:, 1]
        
        unbalanced_results.append(collect_results(name, y_train, y_train_pred, y_train_proba, 'Train', 'Unbalanced', fold))
        unbalanced_results.append(collect_results(name, y_test, y_test_pred, y_test_proba, 'Test', 'Unbalanced', fold))
        
        # Balancing the training set
        Xy_train = pd.concat([X_train, y_train], axis=1)
        bankrupt = Xy_train[Xy_train['Status'] == 1]
        non_bankrupt = Xy_train[Xy_train['Status'] == 0]
        non_bankrupt_downsampled = resample(non_bankrupt, replace=False, n_samples=len(bankrupt) * 3, random_state=42)
        balanced_train_df = pd.concat([bankrupt, non_bankrupt_downsampled])
        X_train_balanced = balanced_train_df.drop('Status', axis=1)
        y_train_balanced = balanced_train_df['Status']
        
        model.fit(X_train_balanced, y_train_balanced)
        y_train_balanced_pred = model.predict(X_train_balanced)
        y_test_pred = model.predict(X_test)
        y_train_balanced_proba = model.predict_proba(X_train_balanced)[:, 1]
        y_test_proba = model.predict_proba(X_test)[:, 1]
        
        balanced_results.append(collect_results(name, y_train_balanced, y_train_balanced_pred, y_train_balanced_proba, 'Train', 'Balanced', fold))
        balanced_results.append(collect_results(name, y_test, y_test_pred, y_test_proba, 'Test', 'Balanced', fold))

# Convert results to DataFrames
unbalanced_results_df = pd.DataFrame(unbalanced_results)
balanced_results_df = pd.DataFrame(balanced_results)

# Save results to CSV
unbalanced_results_df.to_csv('unbalancedDataOutcomes.csv', index=False)
balanced_results_df.to_csv('balancedDataOutcomes.csv', index=False)

# Combine results into one Excel file
combined_results_path = 'CombinedResults.xlsx'
with pd.ExcelWriter(combined_results_path) as writer:
    unbalanced_results_df.to_excel(writer, sheet_name='Unbalanced')
    balanced_results_df.to_excel(writer, sheet_name='Balanced')

# Load combined results for final analysis
combined_df = pd.read_excel(combined_results_path, sheet_name=None)
unbalanced_df = combined_df['Unbalanced']
balanced_df = combined_df['Balanced']

# Add performance metrics
for df in [unbalanced_df, balanced_df]:
    df['Accuracy'] = (df['TP'] + df['TN']) / (df['TP'] + df['TN'] + df['FP'] + df['FN'])
    df['Precision'] = df['TP'] / (df['TP'] + df['FP'])
    df['Recall'] = df['TP'] / (df['TP'] + df['FN'])
    df['Specificity'] = df['TN'] / (df['TN'] + df['FP'])

# Save combined results with additional metrics
with pd.ExcelWriter(combined_results_path) as writer:
    unbalanced_df.to_excel(writer, sheet_name='Unbalanced', index=False)
    balanced_df.to_excel(writer, sheet_name='Balanced', index=False)

# Final Analysis and Plots
# Unbalanced vs Balanced Dataset Impact
plt.figure(figsize=(12, 6))
plt.title('Impact of Unbalanced vs Balanced Dataset on Model Performance')
plt.plot(unbalanced_df.groupby('Classifier Name')['F1 Score'].mean(), label='Unbalanced F1 Score', marker='o')
plt.plot(balanced_df.groupby('Classifier Name')['F1 Score'].mean(), label='Balanced F1 Score', marker='o')
plt.legend()
plt.xlabel('Classifier Name')
plt.ylabel('F1 Score')
plt.show()

# Train vs Test Performance
plt.figure(figsize=(12, 6))
plt.title('Train vs Test Performance (Balanced Data)')
train_balanced = balanced_df[balanced_df['Training or Test Set'] == 'Train']
test_balanced = balanced_df[balanced_df['Training or Test Set'] == 'Test']
plt.plot(train_balanced.groupby('Classifier Name')['F1 Score'].mean(), label='Train F1 Score', marker='o')
plt.plot(test_balanced.groupby('Classifier Name')['F1 Score'].mean(), label='Test F1 Score', marker='o')
plt.legend()
plt.xlabel('Classifier Name')
plt.ylabel('F1 Score')
plt.show()

# Check if any model satisfies the performance constraints
performance_constraints = {
    'TP Rate': 0.6,  # True Positive Rate (Recall) for bankrupt companies
    'TN Rate': 0.7   # True Negative Rate (Specificity
    ) for non-bankrupt companies
}

best_models = balanced_df[(balanced_df['Recall'] >= performance_constraints['TP Rate']) &
                          (balanced_df['Specificity'] >= performance_constraints['TN Rate'])]

if not best_models.empty:
    print("Models satisfying the performance constraints:")
    print(best_models[['Classifier Name', 'F1 Score', 'AUC ROC', 'Recall', 'Specificity']])
else:
    print("No model satisfies the performance constraints.")

# Identifying the best model based on test set results
test_set_results = balanced_df[balanced_df['Training or Test Set'] == 'Test']
best_model = test_set_results.loc[test_set_results['F1 Score'].idxmax()]
print("Best Model based on test set F1 Score:")
print(best_model[['Classifier Name', 'F1 Score', 'AUC ROC', 'Recall', 'Specificity']])

# Statistical test to show significant difference between the best model and others
best_model_scores = test_set_results[test_set_results['Classifier Name'] == best_model['Classifier Name']]['F1 Score']
other_model_scores = test_set_results[test_set_results['Classifier Name'] != best_model['Classifier Name']]['F1 Score']

t_stat, p_val = ttest_ind(best_model_scores, other_model_scores)
print(f"T-test between best model and other models: t_stat={t_stat}, p_val={p_val}")

# Save final results with analysis
combined_results_path_final = 'FinalCombinedResults.xlsx'
with pd.ExcelWriter(combined_results_path_final) as writer:
    unbalanced_df.to_excel(writer, sheet_name='Unbalanced', index=False)
    balanced_df.to_excel(writer, sheet_name='Balanced', index=False)
    best_models.to_excel(writer, sheet_name='Best Models', index=False)

print(f"Final combined results saved to {combined_results_path_final}")
