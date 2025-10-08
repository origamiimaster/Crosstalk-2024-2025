from process_dataset import MyDataset, TestDataset
import wandb
from sklearn.model_selection import cross_val_score, StratifiedKFold
from cuml import RandomForestRegressor as cuRFR
from cuml import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
import numpy as np
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
import pandas as pd


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
    }
    wandb.log(metrics)

    print(f"Cross-validation results for {model.__class__.__name__}:")

    for val in metrics:
        print(f"{val}: {metrics[val]:.4f}")

    # Finish wandb run
    wandb.finish()


if __name__ == "__main__":
    train_dataset = MyDataset(filename="crosstalk_train.parquet", x_cols=('ECFP4', 'FCFP4'))

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

    # test_dataset = TestDataset(filename="crosstalk_test_20250305_inputs.parquet", x_cols=('ECFP4', 'FCFP4'))
    # print("Datasets Loaded")

    # model.fit(train_dataset.X, train_dataset.y)

    # output = model.predict(test_dataset.X)
    # print(output)

    # with open("big_votes.txt", "w") as f:
    #     for val in output:
    #         f.write(str(val) + "\n")

    screen_dataset = pd.read_csv("smiles.csv")

    from rdkit import Chem
    from tqdm import tqdm
    from rdkit.Chem import AllChem
    from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray
    import numpy as np

    def generate_fingerprints(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None  # Invalid molecule
        # ECFP4 (Morgan fingerprint, radius=2, binary)
        ecfp4 = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        # FCFP4 (feature-based Morgan fingerprint, radius=2)
        fcfp4 = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048, useFeatures=True)

        # Convert to numpy arrays (optional, depending on how you want to store)
        ecfp4_array = np.zeros((2048,), dtype=int)
        fcfp4_array = np.zeros((2048,), dtype=int)
        ConvertToNumpyArray(ecfp4, ecfp4_array)
        ConvertToNumpyArray(fcfp4, fcfp4_array)

        return ecfp4_array, fcfp4_array

    tqdm.pandas(desc="Generating fingerprints")
    screen_dataset[['ECFP4', 'FCFP4']] = screen_dataset['smiles'].progress_apply(lambda x: pd.Series(generate_fingerprints(x)))
    screen_dataset['Combined'] = screen_dataset.apply(lambda row: np.concatenate([row['ECFP4'], row['FCFP4']]), axis=1)

    # screen_X = np.stack(screen_dataset['Combined'].values)
    # print("Shape of X:", screen_X.shape)

    # print(screen_dataset)

    # output = model.predict(screen_X)
    # print(output)

    # with open("screen_results.txt", "w") as f:
    #     for val in output:
    #         f.write(str(val) + "\n")


    BATCH_SIZE = 1000  # Adjust based on your system’s memory

    tqdm.pandas(desc="Processing batches")

    with open("screen_results.txt", "w") as f:
        for start_idx in tqdm(range(0, len(screen_dataset), BATCH_SIZE), desc="Batching predictions"):
            end_idx = min(start_idx + BATCH_SIZE, len(screen_dataset))
            batch = screen_dataset.iloc[start_idx:end_idx]

            # Generate fingerprints for the batch
            fingerprints = batch['smiles'].apply(lambda x: pd.Series(generate_fingerprints(x)))
            batch[['ECFP4', 'FCFP4']] = fingerprints

            # Combine fingerprints
            combined = np.stack([
                np.concatenate([row['ECFP4'], row['FCFP4']])
                for _, row in batch.iterrows()
            ])

            # Predict for this batch
            batch_output = model.predict(combined)

            # Write results immediately
            for val in batch_output:
                f.write(str(val) + "\n")
