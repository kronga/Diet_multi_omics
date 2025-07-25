import os
import re
import traceback
import warnings
import numpy as np
import pandas as pd
from LabUtils.addloglevels import sethandlers
from LabQueue.qp import qp, fakeqp
from LabData import config_global as config
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, GroupKFold
from sklearn.metrics import (
    mean_squared_error,
    roc_auc_score,
    f1_score,
    accuracy_score,
    log_loss,
    r2_score,
)
from scipy.stats import pearsonr, spearmanr
import optuna
from lightgbm import LGBMRegressor, LGBMClassifier
import shap
import matplotlib.pyplot as plt
import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from itertools import product

# Constants
NUM_TRIALS = 20
METABO_DIR_ROOT = '/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Analyses/davidkro/batch_normalizations'
# data_dir = '/net/mraid20/export/genie/LabData/Analyses/Metabolomics_repeat/diatery_binary/'
optuna.logging.set_verbosity(optuna.logging.ERROR)


class OptunaTrainer:
    """Simplified Optuna-based model trainer"""

    def __init__(self, task='classification', only_AGB=False):
        self.task = task
        self.only_AGB = only_AGB
        self.model_class = LGBMClassifier if task == 'classification' else LGBMRegressor

    def _create_objective(self, X_train, y_train, X_val, y_val):
        """Create Optuna objective function"""

        def objective(trial):
            params = {
                'num_threads': NUM_THREADS,
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 4),
                'min_child_samples': trial.suggest_int('min_child_samples', 15, 100),
                'max_depth': trial.suggest_int('max_depth', -1, 7),
                'learning_rate': trial.suggest_float('learning_rate', 1e-6, 0.1),
                'n_estimators': trial.suggest_int('n_estimators', 1000, 2000, step=500),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-6, 10.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-6, 10.0),
                # 'lambda_l1': trial.suggest_float('lambda_l1', 1e-6, 10.0),
                # 'lambda_l2': trial.suggest_float('lambda_l2', 1e-6, 10.0),
                'verbose': -1
            }

            # params = {
            #     'num_threads': NUM_THREADS,  # Keep this fixed for consistency.
            #
            #     # Learning rate: Focus on a moderately small range to ensure stable convergence.
            #     'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.05),
            #
            #     # Number of boosting rounds: Fewer options when trials are limited.
            #     'n_estimators': trial.suggest_int('n_estimators', 500, 2500, step=500),
            #
            #     # Maximum depth: Restricting tree depth helps prevent overly complex trees.
            #     'max_depth': trial.suggest_int('max_depth', 3, 7),
            #
            #     # Minimum child samples: Controls overfitting by requiring a minimum number of data points in a leaf.
            #     'min_child_samples': trial.suggest_int('min_child_samples', 20, 50),
            #
            #     # Feature fraction: Use a subset of features per tree to help manage high-dimensional data.
            #     'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 0.9),
            #
            #     # Regularization: Tune one regularization parameter (L1) while keeping others fixed.
            #     'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
            #     'reg_lambda': 1.0,  # Fixed to reduce search space complexity.
            #     'verbose': -1
            # }

            # Set feature_fraction to 1.0 if only_AGB is True, otherwise use trial suggestion
            if self.only_AGB:
                params['feature_fraction'] = 1.0

            else:
                params['feature_fraction'] = trial.suggest_float('feature_fraction', 0.5, 0.9)

            if self.task == 'classification':
                params.update({'objective': 'binary', 'is_unbalance': True})

            model = self.model_class(**params)
            model.fit(
                X_train, y_train,
            )

            if self.task == 'classification':
                # Check if validation set has both classes
                if len(np.unique(y_val)) < 2:
                    # Return a very low score to indicate this is not a good solution
                    logging.warning("Only one class present in validation set")
                    return float('-inf')
                else:
                    y_pred = model.predict_proba(X_val)[:, 1]
                    score = roc_auc_score(y_val, y_pred)
            else:
                y_pred = model.predict(X_val)
                mse = -mean_squared_error(y_val, y_pred)
                # Calculate R2 score and add it as a user attribute.
                r2 = r2_score(y_val, y_pred)
                trial.set_user_attr("r2", r2)
                trial.set_user_attr("mse", mse)
                score = r2

            return score

        return objective

    def evaluate_metrics(self, y_true, y_pred, y_pred_proba=None):
        """Calculate performance metrics"""
        if self.task == 'classification':
            return {
                'auc': roc_auc_score(y_true, y_pred_proba),
                'f1': f1_score(y_true, y_pred),
                'accuracy': accuracy_score(y_true, y_pred),
                'logloss': log_loss(y_true, y_pred_proba)
            }
        else:
            return {
                'mse': mean_squared_error(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'r2': r2_score(y_true, y_pred),
                'pearson_r': pearsonr(y_true, y_pred)[0],
                'spearman_r': spearmanr(y_true, y_pred)[0]
            }


def train_model(
        X: pd.DataFrame,
        Y: pd.Series,
        col_name: str,
        output_dir: str = 'results',
        gender: Optional[str] = None,
        only_AGB: bool = False,
        no_optimization: bool = False,
        n_splits: int = 5
) -> Dict:
    """
    Train and evaluate model for a single target

    Parameters:
        X: Input features DataFrame
        Y: Target variable Series
        col_name: Name of the target column
        output_dir: Directory to save results
        gender: Optional gender filter
        only_AGB: Whether to use only AGB models
        no_optimization: If True, uses predefined hyperparameters instead of optimization
        n_splits: Number of splits for cross-validation
    """
    # Setup logging
    os.makedirs(output_dir, exist_ok=True)
    results_dir = os.path.join(output_dir, col_name)
    os.makedirs(results_dir, exist_ok=True)

    # drop nan values from Y and their corresponding rows in X
    Y = Y.dropna()
    X = X.loc[Y.index]

    # Get the root logger
    logger = logging.getLogger()

    # Add a file handler for this specific target
    target_log_file = os.path.join(results_dir, f'training_{col_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s',
                                  datefmt='%Y-%d-%m %H:%M:%S')
    file_handler = logging.FileHandler(target_log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    try:
        logging.info(f"Starting training for {col_name}" + (f" ({gender})" if gender else ""))
        logging.info(f"Using {'predefined hyperparameters' if no_optimization else 'hyperparameter optimization'}")

        # Transform continuous two-class values first
        unique_values = np.unique(Y)
        if len(unique_values) == 2 and not set(unique_values).issubset({0, 1}):
            logging.info(f"Transforming {col_name} from {unique_values} to binary")
            Y = pd.Series((Y == max(unique_values)).astype(int), index=Y.index)

        # Determine task type
        task = 'classification' if len(np.unique(Y)) == 2 else 'regression'
        trainer = OptunaTrainer(task=task, only_AGB=only_AGB)

        # Define default hyperparameters for no_optimization case, based on top performing trials
        default_params = {
            'classification': {
                'num_threads': NUM_THREADS,
                'bagging_fraction': 0.8,  # median: 0.948
                'bagging_freq': 3,  # median: 3
                'feature_fraction': 0.75,  # median: 0.735
                'min_child_samples': 70,  # median: 70
                'max_depth': 5,  # median: 4
                'learning_rate': 0.03,  # median: 0.0557
                'n_estimators': 1000,  # median: 1500
                'reg_alpha': 4.5,  # median: 4.503
                'reg_lambda': 5.8,  # median: 5.783
                'lambda_l1': 1.1,  # median: 1.093
                'lambda_l2': 7.0,  # median: 7.012,
                'is_unbalance': True,
                'verbose': -1
            },
            'regression': {
                'num_threads': NUM_THREADS,
                'bagging_fraction': 0.8,
                'bagging_freq': 3,
                'feature_fraction': 0.7,
                'min_child_samples': 70,
                'max_depth': 5,
                'learning_rate': 0.056,
                'n_estimators': 1000,
                'reg_alpha': 4.5,
                'reg_lambda': 5.8,
                'lambda_l1': 1.1,
                'lambda_l2': 7.0,
                'verbose': -1
            }
        }

        # For classification tasks, check class balance and minimum samples
        if task == 'classification':
            class_counts = Y.value_counts()
            prevalence = Y.mean()
            min_class_size = class_counts.min()

            logging.info(f"Target prevalence: {prevalence:.3f}")
            logging.info(f"Class distribution: {dict(class_counts)}")

            # Check if we have enough samples in each class for k-fold CV
            min_samples_per_fold = 2  # Minimum samples of each class needed per fold
            if min_class_size < min_samples_per_fold * n_splits:
                raise ValueError(
                    f"Insufficient samples for {n_splits}-fold CV. Minimum class has {min_class_size} samples, "
                    f"but need at least {min_samples_per_fold * n_splits} samples per class."
                )
        else:
            prevalence = Y[Y > Y.min()].count() / Y.count()

        # Initialize containers for results
        cv_results = []
        all_predictions = []
        all_true_values = []
        all_indices = []
        optuna_trials_data = []

        valid_seed = None

        RANDOM_SEEDS = [42, 123, 456, 789, 101112, 131415, 161718, 192021, 222324, 252627]
        strat_column = Y
        from sklearn.model_selection import StratifiedGroupKFold

        # ------------------------------------------------------------------
        # 1. decide once whether we have groups
        # ------------------------------------------------------------------
        groups_present = 'RegistrationCode' in X.columns
        if groups_present:
            groups = X['RegistrationCode'].copy()  # keep for Group/StratifiedGroupKFold
            X = X.drop(columns=['RegistrationCode'])  # ? NEW: remove from feature matrix
        else:
            groups = None
        strat_column = None  # will be set later
        valid_seed = None

        # ------------------------------------------------------------------
        # 2. find a seed that gives goodfolds
        # ------------------------------------------------------------------
        for seed in RANDOM_SEEDS:
            try:
                if task == 'classification':
                    # ??????????????????????????? classification ??????????????????????????
                    cv = (
                        StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
                        if groups_present
                        else StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
                    )

                    all_folds_valid = True
                    for fold, (tr_idx, te_idx) in enumerate(cv.split(X, Y, groups=groups)):
                        if len(np.unique(Y.iloc[te_idx])) < 2:  # class-balance check
                            logging.info(f"Seed {seed}: Fold {fold + 1} missing a class ? skipping")
                            all_folds_valid = False
                            break

                else:
                    # ??????????????????????????? regression  ????????????????????????????
                    if groups_present:
                        cv = GroupKFold(n_splits=n_splits)  # ? NEW (no shuffle arg)
                    elif 'age_bin' in X.columns:  # keep your age-bin strat
                        strat_column = X['age_bin'].copy()
                        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
                        X = X.drop(columns=['age_bin'])
                    else:
                        cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

                    all_folds_valid = True  # nothing special to verify for regression

                if all_folds_valid:
                    valid_seed = seed
                    logging.info(f"Found valid seed: {seed}")
                    break

            except Exception as e:
                logging.warning(f"Seed {seed} threw an error: {e}")
                continue

        if valid_seed is None:
            raise ValueError("Could not find a valid stratification / grouping seed "
                             f"after trying {RANDOM_SEEDS}")

        # ------------------------------------------------------------------
        # 3. re-instantiate CV with the chosen seed (if needed)
        # ------------------------------------------------------------------
        if task == 'classification':
            cv = (
                StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=valid_seed)
                if groups_present
                else StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=valid_seed)
            )
        else:
            if groups_present:
                cv = GroupKFold(n_splits=n_splits)
            elif strat_column is not None:
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=valid_seed)
            else:
                cv = KFold(n_splits=n_splits, shuffle=True, random_state=valid_seed)

        # ------------------------------------------------------------------
        # 4. outer cross-validation loop
        # ------------------------------------------------------------------
        def _iter_folds():
            """helper so we can call cv.split with the right arguments"""
            if isinstance(cv, (StratifiedKFold, StratifiedGroupKFold)):
                return cv.split(X, Y if task == 'classification' else strat_column, groups=groups)
            elif isinstance(cv, GroupKFold):
                return cv.split(X, groups=groups)
            else:  # plain KFold
                return cv.split(X)

        for fold, (train_idx, test_idx) in enumerate(_iter_folds()):
            logging.info(f"\nProcessing fold {fold + 1}/{n_splits}")

            # Split data
            X_train_full, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train_full, y_test = Y.iloc[train_idx], Y.iloc[test_idx]

            # Create validation set with stratification
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_full, y_train_full,
                test_size=0.2,
                stratify=y_train_full if task == 'classification' else None,
                random_state=42
            )

            # Verify class distribution in validation set for classification tasks
            if task == 'classification':
                val_classes = np.unique(y_val)
                if len(val_classes) < 2:
                    logging.warning(f"Initial split resulted in imbalanced validation set. Attempting resampling.")

                    # Calculate original class proportions
                    class_counts = np.bincount(y_train_full)
                    class_ratio = class_counts[1] / len(y_train_full)  # proportion of class 1

                    # Get indices for each class
                    class_0_idx = np.where(y_train_full == 0)[0]
                    class_1_idx = np.where(y_train_full == 1)[0]

                    # Calculate validation set sizes maintaining original ratio
                    total_val_size = int(len(y_train_full) * 0.2)  # 20% for validation
                    val_size_class_1 = int(total_val_size * class_ratio)
                    val_size_class_0 = total_val_size - val_size_class_1

                    # Sample from each class for validation set
                    val_idx_0 = np.random.choice(class_0_idx, size=val_size_class_0, replace=False)
                    val_idx_1 = np.random.choice(class_1_idx, size=val_size_class_1, replace=False)
                    val_idx = np.concatenate([val_idx_0, val_idx_1])

                    # Remaining indices go to training
                    train_idx = np.array(list(set(range(len(y_train_full))) - set(val_idx)))

                    # Split the data using these indices
                    X_train = X_train_full.iloc[train_idx]
                    X_val = X_train_full.iloc[val_idx]
                    y_train = y_train_full.iloc[train_idx]
                    y_val = y_train_full.iloc[val_idx]

                    logging.info(f"Original class ratio: {class_ratio:.3f}")
                    logging.info(f"After resampling - Training class distribution: {np.bincount(y_train)}")
                    logging.info(f"After resampling - Validation class distribution: {np.bincount(y_val)}")
                    logging.info(f"After resampling - Validation class ratio: {np.mean(y_val):.3f}")

            if no_optimization:
                # Use predefined hyperparameters
                best_params = default_params[task]
                logging.info(f"Using predefined hyperparameters: {best_params}")
            else:
                # Optimize hyperparameters
                study_name = f"{col_name}_{fold}"
                study = optuna.create_study(direction='maximize', study_name=study_name)
                objective = trainer._create_objective(X_train, y_train, X_val, y_val)
                study.optimize(objective, n_trials=NUM_TRIALS)
                best_params = study.best_params

                # Store Optuna trials data
                trials_df = study.trials_dataframe()
                trials_df['fold'] = fold
                optuna_trials_data.append(trials_df)

            # Train model with best/default parameters
            best_model = trainer.model_class(**best_params)
            best_model.fit(X_train_full, y_train_full)

            # Generate predictions
            if task == 'classification':
                test_preds = best_model.predict(X_test)
                test_preds_proba = best_model.predict_proba(X_test)[:, 1]

                # Store the indices of the test set
                test_indices = X_test.index

                # Check if test set has at least 2 unique classes before calculating AUC
                n_classes = len(np.unique(y_test))
                if n_classes >= 2:
                    metrics = trainer.evaluate_metrics(y_test, test_preds, test_preds_proba)
                else:
                    logging.warning(f"Fold {fold + 1} has only {n_classes} class in test set. Setting AUC to NaN.")
                    metrics = trainer.evaluate_metrics(y_test, test_preds, test_preds_proba)
                    metrics['auc'] = np.nan

                all_predictions.extend(test_preds_proba)
            else:
                test_preds = best_model.predict(X_test)

                # Store the indices of the test set
                test_indices = X_test.index

                metrics = trainer.evaluate_metrics(y_test, test_preds)

                # Calculate adjusted R² for each fold for regression tasks
                n_samples = len(y_test)
                n_features = X.shape[1]  # Number of features/predictors
                r2 = metrics.get('r2', 0)  # Get R² value, default to 0 if not present
                adjusted_r2 = 1 - ((1 - r2) * (n_samples - 1) / (n_samples - n_features - 1))
                metrics['adjusted_r2'] = adjusted_r2

                all_predictions.extend(test_preds)

            # Extend indices and true values for later saving
            all_true_values.extend(y_test)
            all_indices.extend(test_indices)

            metrics['fold'] = fold
            cv_results.append(metrics)

            logging.info(f"Fold {fold + 1} metrics: {metrics}")

            if not no_optimization:
                # Clean up Optuna study file for this fold
                for file in os.listdir():
                    if file.startswith(f'optuna_{study_name}') and file.endswith('.db'):
                        try:
                            os.remove(file)
                        except Exception as e:
                            logging.warning(f"Could not remove Optuna file {file}: {str(e)}")

        # Calculate combined metrics across all folds
        logging.info("\nCalculating combined metrics across all folds")
        if task == 'classification':
            # Check if combined test set has at least 2 unique classes
            n_classes_all = len(np.unique(all_true_values))
            if n_classes_all >= 2:
                combined_auc = roc_auc_score(all_true_values, all_predictions)
                logging.info(f"Combined AUC across all folds: {combined_auc:.3f}")
            else:
                combined_auc = np.nan
                logging.warning("Combined test set has less than 2 classes. Setting combined AUC to NaN.")
        else:
            combined_r2 = r2_score(all_true_values, all_predictions)
            pearson_r = pearsonr(all_true_values, all_predictions)[0]

            # Calculate overall adjusted R²
            n_samples_total = len(all_true_values)
            n_features = X.shape[1]
            combined_adjusted_r2 = 1 - ((1 - combined_r2) * (n_samples_total - 1) / (n_samples_total - n_features - 1))

            logging.info(f"Combined R² across all folds: {combined_r2:.3f}")
            logging.info(f"Combined Adjusted R² across all folds: {combined_adjusted_r2:.3f}")

        # Train final model on all data
        logging.info("\nTraining final model on all data")
        if no_optimization:
            final_params = default_params[task]
        else:
            X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2)
            final_study = optuna.create_study(direction='maximize')
            final_study.optimize(trainer._create_objective(X_train, y_train, X_val, y_val), n_trials=NUM_TRIALS)
            final_params = final_study.best_params

        final_model = trainer.model_class(**final_params)
        final_model.fit(X, Y)

        # Save results
        cv_df = pd.DataFrame(cv_results)
        cv_df.to_csv(os.path.join(results_dir, f'cv_results_{col_name}.csv'), index=False)

        if not no_optimization:
            # Save optuna trials data
            trials_df = pd.concat(optuna_trials_data)
            trials_df.to_csv(os.path.join(results_dir, f'optuna_trials_{col_name}.csv'), index=False)

        # Save predictions with indices
        pred_df = pd.DataFrame({
            'index': all_indices,
            'true_values': all_true_values,
            'predictions': all_predictions
        })
        pred_df.set_index('index', inplace=True)
        pred_df.to_csv(os.path.join(results_dir, f'predictions_{col_name}.csv'))

        # Calculate final metrics
        mean_metrics = {f'mean_{k}': v for k, v in pd.DataFrame(cv_results).mean().items() if k != 'fold'}

        # Add combined metrics to the final results
        if task == 'classification':
            mean_metrics['combined_auc'] = combined_auc
            mean_metrics['prevalence'] = prevalence
        else:
            mean_metrics['combined_r2'] = combined_r2
            mean_metrics['combined_adjusted_r2'] = combined_adjusted_r2  # Add combined adjusted R²
            mean_metrics['pearson_r'] = pearson_r
            mean_metrics['prevalence'] = prevalence
        mean_metrics['n_samples'] = len(Y)

        # Generate and save SHAP plot
        try:
            explainer = shap.TreeExplainer(final_model)
            shap_values = explainer.shap_values(X)
            if task == 'classification':
                shap_values = shap_values[1] if isinstance(shap_values, list) else shap_values

            # Calculate mean absolute SHAP values for feature importance ranking
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': mean_abs_shap
            }).sort_values('importance', ascending=False)

            # Get top 20 features
            top_features = feature_importance.head(20)['feature'].tolist()

            # Save SHAP values for top 20 features
            top_shap_values = pd.DataFrame(
                shap_values[:, [list(X.columns).index(feat) for feat in top_features]],
                columns=top_features
            )
            top_shap_values.to_csv(os.path.join(results_dir, f'top_20_shap_values_{col_name}.csv'))


            # Save feature importance ranking
            feature_importance.to_csv(os.path.join(results_dir, f'feature_importance_{col_name}.csv'), index=False)

            # Generate SHAP summary plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X, show=False)
            plt.savefig(os.path.join(results_dir, f'shap_summary_{col_name}.png'))
            plt.close()
        except Exception as e:
            logging.error(f"Error generating SHAP plot: {str(e)}")

        return {
            'target': col_name,
            'task': task,
            'optimization': 'none' if no_optimization else 'optuna',
            'n_splits': n_splits,
            **mean_metrics
        }

    finally:
        # Remove the target-specific file handler
        logger.removeHandler(file_handler)
        file_handler.close()

def filter_metabolite_columns(df, identified_metabolites_path):
    """
    Filter DataFrame columns based on identified metabolites, matching column names
    without the last underscore segment, while keeping demographic columns.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame with metabolite columns
    identified_metabolites_path : str
        Path to CSV file containing identified metabolites in a single column

    Returns:
    --------
    pandas.DataFrame
        DataFrame containing demographic columns and columns that match the identified metabolites
    """
    # Define demographic columns to always keep
    demographic_cols = ['age', 'gender', 'bmi']

    # Read the identified metabolites
    identified = pd.read_csv(identified_metabolites_path)
    identified_list = identified.iloc[:, 0].tolist()

    # Process column names to match the format in identified metabolites
    processed_cols = {}
    for col in df.columns:
        # Skip demographic columns
        if col.lower() in demographic_cols:
            continue
        # Split on underscore and rejoin all parts except the last one
        processed_name = '_'.join(col.split('_')[:-1])
        processed_cols[col] = processed_name

    # Filter columns that match identified metabolites
    keep_cols = [col for col, processed in processed_cols.items()
                 if processed in identified_list]

    # Add demographic columns to the list of columns to keep
    # Case-insensitive check for demographic columns
    existing_demographic_cols = [col for col in df.columns
                                 if any(demo.lower() == col.lower()
                                        for demo in demographic_cols)]
    final_cols = existing_demographic_cols + keep_cols

    # Return DataFrame with matching columns and demographic columns
    return df[final_cols]


def load_data(x_file: str, y_file: Optional[str] = None, y_cols: Optional[List[str]] = None, only_AGB: bool = False,
              drop_AGB: bool = False, filter_identified: bool = False) -> Tuple[
    pd.DataFrame, pd.DataFrame]:
    """
    Load and preprocess feature and target data.

    Args:
        x_file: Path to features file
        y_file: Optional path to targets file
        y_cols: Optional list of target columns if they're in x_file
        only_AGB: If True, keep only age, gender, and BMI features

    Returns:
        Tuple of (features DataFrame, targets DataFrame)
    """
    x = pd.read_csv(os.path.join(data_dir, x_file), index_col=0)

    # Extract target columns if they're in the features file
    if y_file is None:
        y = x[y_cols].copy()
        x.drop(columns=y_cols, inplace=True)
    else:
        y = pd.read_csv(os.path.join(data_dir, y_file), index_col=0)

    if filter_identified:
        identified_metabolites_path = os.path.join(METABO_DIR_ROOT, 'analysis/identified_metabolites.csv')
        x = filter_metabolite_columns(x, identified_metabolites_path)

    # Clean column names
    x = x.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '_', x))
    y = y.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '_', x))

    # Filter for only age, gender, BMI if specified
    if only_AGB:
        agb_columns = [col for col in x.columns if col.lower() in ['age', 'gender', 'bmi', 'age_bin']]
        # if not all(feature.lower() in [col.lower() for col in x.columns] for feature in ['age', 'gender', 'bmi']):
        #     raise ValueError("Not all required features (age, gender, BMI) found in the dataset")
        x = x[agb_columns]
        # keep all columns in y except for agb_columns in y
        y = y.drop(columns=[col for col in agb_columns if col in y.columns])
    elif drop_AGB:
        agb_columns = [col for col in x.columns if col.lower() in ['age', 'gender', 'bmi', 'age_bin']]
        # check if any of the agb_columns are in x and remove them
        x = x.drop(columns=[col for col in agb_columns if col in x.columns])
    else:
        agb_columns = [col for col in x.columns if col.lower() in ['age', 'gender', 'bmi']]
        # check if any of the agb_columns are in y and remove them
        y = y.drop(columns=[col for col in agb_columns if col in y.columns])

    # Handle missing and infinite values
    for df in [x, y]:
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        # column_means = df.mean()
        # df.fillna(column_means, inplace=True)

    return x, y


def transform_to_binary(Y):
    """
    Transform continuous variables to binary classification tasks using quartiles.
    Keep only top and bottom quartiles, middle values are dropped.
    Skip columns that are already binary.

    Args:
        Y (pd.DataFrame): Input dataframe with continuous variables

    Returns:
        pd.DataFrame: Transformed dataframe with binary values (0 for bottom quartile, 1 for top quartile)
    """
    Y_binary = pd.DataFrame()

    for col in Y.columns:
        # Check if column is already binary
        unique_values = Y[col].unique()
        if len(unique_values) == 2 and set(unique_values).issubset({0, 1}):
            logging.info(f"Column {col} is already binary, keeping as is")
            Y_binary[col] = Y[col]
            continue

        # Check if column has enough unique values to be considered continuous
        if len(unique_values) < 5:
            logging.warning(
                f"Column {col} has less than 5 unique values, might not be continuous. Skipping transformation.")
            Y_binary[col] = Y[col]
            continue

        bottom_quartile = Y[col].quantile(0.25)
        top_quartile = Y[col].quantile(0.75)

        # Create binary column
        binary_col = Y[col].copy()
        binary_col = binary_col[(Y[col] <= bottom_quartile) | (Y[col] >= top_quartile)]
        binary_col = (binary_col >= top_quartile).astype(int)

        Y_binary[col] = binary_col

    return Y_binary


def upload_jobs(q, x_file, y_file=None, y_cols=None, split_by_gender=False, only_AGB=False, drop_bmi=False,
                drop_AGB=False, filter_identified=False, binary_classification=False, no_optimization=False,
                bottom_10_percent=False, top_10_percent=False, n_splits=5):
    """
    Upload and manage training jobs

    Args:
        ...
        no_optimization (bool): If True, uses predefined hyperparameters instead of optimization
        n_splits (int): Number of splits for cross-validation
    """
    logging.info("Starting upload_jobs process")
    results = []

    try:
        # Load data
        X, Y = load_data(x_file, y_file, y_cols, only_AGB=only_AGB, filter_identified=filter_identified, drop_AGB=drop_AGB)
        if drop_bmi:
            X = X.drop(columns=['bmi']) if 'bmi' in X.columns else X

        # Transform Y to binary classification if requested
        if binary_classification:
            Y = transform_to_binary(Y)
            # Remove samples that were dropped during binary transformation
            X = X.loc[Y.index]

        # transform to classification task where we compare bottom 10% to the rest
        elif bottom_10_percent:
            Y = Y.copy()
            Y = Y.rank(pct=True)
            Y = (Y <= 0.1).astype(int)
        elif top_10_percent:
            Y = Y.copy()
            Y = Y.rank(pct=True)
            Y = (Y >= 0.9).astype(int)

        logging.info(f"Loaded data - X shape: {X.shape}, Y shape: {Y.shape}")
        logging.info(f"Using {'predefined hyperparameters' if no_optimization else 'hyperparameter optimization'}")

        if split_by_gender and 'gender' in X.columns:
            for gender, gender_value in [('male', 1), ('female', 0)]:
                X_gender = X[X['gender'] == gender_value].drop(columns=['gender'])
                Y_gender = Y.loc[X_gender.index]

                for col in Y_gender.columns:
                    logging.info(f"Submitting job for {col} ({gender})")
                    ticket = q.method(train_model, kwargs={
                        'X': X_gender,
                        'Y': Y_gender[col],
                        'col_name': col,
                        'gender': gender,
                        'only_AGB': only_AGB,
                        'no_optimization': no_optimization,
                        'n_splits': n_splits
                    })
                    results.append((ticket, col, gender))
        else:
            for col in Y.columns:
            # for col in Y[:1]:
                logging.info(f"Submitting job for {col}")
                ticket = q.method(train_model, kwargs={
                    'X': X,
                    'Y': Y[col],
                    'col_name': col,
                    'only_AGB': only_AGB,
                    'no_optimization': no_optimization,
                    'n_splits': n_splits
                })
                results.append((ticket, col, None))

        # Process results
        final_results = []
        for ticket, col, gender in results:
            try:
                result = q.waitforresult(ticket)
                if result:
                    final_results.append(result)
            except Exception as e:
                logging.error(f"Error processing {col} ({gender}): {str(e)}")

        # Save final results
        if final_results:
            pd.DataFrame(final_results).to_csv('final_results.csv', index=False)

    except Exception as e:
        logging.error(f"Error in upload_jobs: {str(e)}")
        logging.error(traceback.format_exc())
        raise


def main(x_files, y_files, split_by_gender=False, only_AGB=False, drop_bmi=False, drop_AGB=False,
         binary_classification=False, no_optimization=False, n_splits=5, bottom_10_percent=False,
         top_10_percent=False, mem_gb='10'):
    """
    Run analysis for all combinations of x_files and y_files.

    Args:
        x_files (list): List of x file base names (without .csv extension)
        y_files (list): List of y file base names (without .csv extension)
        split_by_gender (bool): Whether to split analysis by gender
        only_AGB (bool): Whether to use only AGB data
        drop_bmi (bool): Whether to drop BMI from analysis
        binary_classification (bool): Whether to use binary classification
        no_optimization (bool): If True, uses predefined hyperparameters instead of optimization
        n_splits (int): Number of splits for cross-validation
    """
    # Convert single strings to lists for consistency
    if isinstance(x_files, str):
        x_files = [x_files]
    if isinstance(y_files, str):
        y_files = [y_files]

    # Generate all possible combinations of x and y files
    file_combinations = list(product(x_files, y_files))

    for x_base, y_base in file_combinations:
        # Add .csv extension for actual file operations
        x_file = f"{x_base}.csv"
        y_file = f"{y_base}.csv"

        # Set up the base path depending on gender split
        if split_by_gender:
            base_path = os.path.join(OPTUNA_DIR, 'split_by_gender', x_base)
        else:
            base_path = os.path.join(OPTUNA_DIR, x_base)

        # Add y file subdirectory
        path_dir = os.path.join(base_path, y_base)

        # Add AGB subdirectory if needed
        if only_AGB:
            path_dir = os.path.join(path_dir, 'only_AGB')
        elif drop_AGB:
            path_dir = os.path.join(path_dir, 'drop_AGB')

        # Add binary classification suffix if needed
        if binary_classification:
            path_dir = os.path.join(path_dir, 'binary_quartiles')

        if bottom_10_percent:
            path_dir = os.path.join(path_dir, 'bottom_10_percent')
        elif top_10_percent:
            path_dir = os.path.join(path_dir, 'top_10_percent')

        # Add no_optimization suffix if needed
        if no_optimization:
            path_dir = os.path.join(path_dir, 'no_optimization')

        # Create directory if it doesn't exist
        os.makedirs(path_dir, exist_ok=True)

        # Store original directory
        original_dir = os.getcwd()

        try:
            # Change to the created directory
            os.chdir(path_dir)

            # Start QP
            # qp = fakeqp
            with qp(jobname='optuna', q=['himem7.q'], _mem_def=mem_gb, _trds_def=NUM_THREADS,
                    _specific_nodes='plink'
                    ) as q:
                q.startpermanentrun()
                upload_jobs(
                    q=q,
                    x_file=x_file,
                    y_file=y_file,
                    split_by_gender=split_by_gender,
                    only_AGB=only_AGB,
                    drop_bmi=drop_bmi,
                    drop_AGB=drop_AGB,
                    filter_identified=False,
                    binary_classification=binary_classification,
                    bottom_10_percent=bottom_10_percent,
                    top_10_percent=top_10_percent,
                    no_optimization=no_optimization,
                    n_splits=n_splits
                )

        finally:
            # Always return to original directory
            os.chdir(original_dir)

        print(f'Completed analysis for X: {x_file} and Y: {y_file}')


if __name__ == '__main__':
    NUM_THREADS = 20
    mem_gb = '16'
    data_dir = '/net/mraid20/export/genie/LabData/Analyses/Metabolomics_repeat/final_for_diet_paper/Genetics'
    OPTUNA_DIR = '/net/mraid20/export/genie/LabData/Analyses/Metabolomics_repeat/predict_MS/Optuna/Diet_paper/Genetics/rerun'
    # OPTUNA_DIR = '/net/mraid20/export/genie/LabData/Analyses/Metabolomics_repeat/predict_MS/Optuna/Diet_paper/Clustered_Reruns36/'

    # Define your lists of files
    x_files = [
        # 'x_ms_matched'
        # 'x_mpa_matched',
        # 'x_NMR_matched',
        # 'x_oral_matched',

        # 'x_ms_all_diet'
        # 'x_ms_all_diet_signs'

        # 'x_ms_matched_pre',
        # 'x_NMR_matched_pre',
        # 'x_mpa_matched_pre',
        # 'x_oral_matched_pre',

        'rna_baseline',

        # 'x_ms_mpa_matched',
        # 'x_ms_NMR_matched',
        # 'x_ms_oral_matched',
        # 'x_ms_mpa_NMR_matched'
        # 'x_ms_mpa_NMR_oral_matched',

        # 'x_diet_features',
        # 'x_phipseq_merged_50_diet',
        # 'x_medical_conditions_consolidated',
        # 'x_blood'
        # 'x_hr',
        # 'x_hr_clean_ranked',
        # 'x_hr_clean_ranked_filtered_wfpb_1000',
        # 'x_hr_clean_ranked_filtered_wfpb_1900',
        # 'diet_features_baseline',
        # 'diet_scores_baseline',


        # 'x_prs_hr_ranked',
        # 'x_prs_data_sig_diet',
        # 'x_prs_data_sig_diet_nominal',
        # 'x_prs_data_sig_nominal',
        # 'x_prs_data_sig',

        # 'baseline_ms_diet',
        # 'followup_ms_diet',
        # 'diff_ms_diet_agb',
        # 'combined_base_diff',

        # 'ms_breast_cancer_delta_x',
        # 'predicted_macros_alloc_x',
        # 'predicted_macros_alloc_followup_x',
    ]

    y_files = [
        # 'y_blood',
        # 'y_diet_adherence_scores',
        # 'y_nova_scores',
        # 'y_diet_macros',
        # 'y_nutrients',
        # 'y_food_categories',
        # 'y_foods',
        # 'y_lipid_ratios',
        # 'y_plant_ratios',

        # 'y_ms_all_diet_scores'
        # 'y_ms_all_diet_signs'

        # 'rna_diet_scores_baseline_hc_disease'
        # 'rna_food_categories_baseline_hc_disease',
        'rna_nutrients_baseline_hc_disease',
        # 'rna_agb_baseline',


        # 'y_max_consumption',
        # 'y_min_consumption',
        # 'y_lifestyle_matched',
        # 'y_never_eaters',
        # 'y_oil_types',
        # 'y_cereal_types',
        # 'y_bread_types',

        # 'y_medical_conditions_consolidated',
        # 'y_hr',
        # 'y_hr_clean_ranked',
        # 'y_filtered_wfpb_1000_diet',
        # 'agb_baseline',
        # 'y_prs_hr_ranked',
        # 'y_prs_diet_scores',
        # 'y_prs_nutrients',
        # 'y_prs_agb_height',

        # 'diff_diet_scores'
        # 'ms_breast_cancer_delta_y',
        # 'predicted_macros_alloc_y',
        # 'predicted_macros_alloc_followup_y'
    ]

    start = datetime.now()
    sethandlers(file_dir=config.log_dir)

    # Run with different combinations of parameters
    main(
        x_files=x_files,
        y_files=y_files,
        split_by_gender=False,
        only_AGB=False,
        drop_bmi=False,
        drop_AGB=False,
        binary_classification=False,
        no_optimization=False,
        bottom_10_percent=False,
        top_10_percent=False,
        n_splits=5,
        mem_gb=mem_gb
    )

    print(f'Total execution time: {datetime.now() - start}')