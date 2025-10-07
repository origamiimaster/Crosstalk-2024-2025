from process_dataset import MyDataset, TestDataset
import wandb
from sklearn.model_selection import cross_val_score, StratifiedKFold
from cuml import RandomForestRegressor as cuRFR
from cuml import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
import numpy as np
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet


def evaluate_model_with_wandb(model, X, y, cv_folds=5, project_name="crosstalk-sebastian-ivy"):
    """
    Evaluates a model using cross-validation and logs results to wandb.

    Args:
        model: Scikit-learn compatible model.
        X (array-like): Feature data.
        y (array-like): Target labels.
        cv_folds (int): Number of cross-validation folds.
        project_name (str): Name of the wandb project.
    """

    # Initialize wandb run
    wandb.init(project=project_name, config={
        "model": model.__class__.__name__,
        "cv_folds": cv_folds,
        "n_samples": len(X),
        "n_features": X.shape[1] if hasattr(X, 'shape') else 'unknown'
    })

    # Cross-validation
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores1 = cross_val_score(model, X, y, cv=cv, scoring='r2')
    
    # scores2 = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    # scores3 = cross_val_score(model, X, y, cv=cv, scoring='top_k_accuracy')

    # Log metrics
    metrics = {
        "mean_r2": np.mean(scores1),
        "std_r2": np.std(scores1),
        "min_r2": np.min(scores1),
        "max_r2": np.max(scores1), 
        # "mean_accuracy": np.mean(scores2),
        # "std_accuracy": np.std(scores2),
        # "min_accuracy": np.min(scores2),
        # "max_accuracy": np.max(scores2), 
    }
    wandb.log(metrics)

    print(f"Cross-validation results for {model.__class__.__name__}:")

    for val in metrics:
        print(f"{val}: {metrics[val]:.4f}")
    # print(f"Mean Accuracy: {np.mean(scores):.4f}")
    # print(f"Std Accuracy: {np.std(scores):.4f}")
    # print(f"Min Accuracy: {np.min(scores):.4f}")
    # print(f"Max Accuracy: {np.max(scores):.4f}")
    

    # Finish wandb run
    wandb.finish()




if __name__ == "__main__":
    train_dataset = MyDataset(filename="crosstalk_train.parquet", x_cols=('ECFP4', 'FCFP4')) 
    # train_dataset = HashedHashDataset(filename="crosstalk_train.parquet", x_cols=('ECFP4', 'FCFP4')) 

    # model = RandomForestRegressor(random_state=0)
    r1 = cuRFR()
    r2 = cuRFR()
    r3 = cuRFR()
    
    r4 = LinearRegression()
    # r5 = LogisticRegression()
    r6 = Ridge()
    r7 = Lasso()
    r8 = ElasticNet()

    r4 = LinearRegression()
    r7 = Lasso()
    r8 = ElasticNet()

    
    model = VotingRegressor([
        ("tree1", r1), 
        ("tree2", r2), 
        ("tree3", r3), 
        ("lr", r4), 
        ("en", r8),
        ("lasso", r7),
        ("ridge", r6)
        ], n_jobs=1)
    

    evaluate_model_with_wandb(model, train_dataset.X, train_dataset.y)

    test_dataset = TestDataset(filename="crosstalk_test_20250305_inputs.parquet", x_cols=('ECFP4', 'FCFP4'))
    print("Datasets Loaded")

    model.fit(train_dataset.X, train_dataset.y)

    output = model.predict(test_dataset.X)
    print(output)
    
    with open("big_votes.txt", "w") as f:
        for val in output:
            f.write(str(val) + "\n")
    
    