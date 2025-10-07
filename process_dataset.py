import torch
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.feature_extraction import FeatureHasher

def process_row(row, x_cols, feature_length):
    """Parse and concatenate the float arrays from x_cols in a single row."""
    row_features = np.empty(feature_length, dtype=np.float32)
    for a, x_col in enumerate(x_cols):
        array = np.fromstring(row[x_col], sep=",", dtype=np.float32)
        row_features[len(array) * a:len(array) * (a + 1)] = array
    return row_features


@dataclass
class MyDataset:
    """Basic dataset class holding a dataset."""

    x_cols: tuple
    filename: str
    y_col: str = "DELLabel"
    X: np.ndarray = None
    y: np.ndarray = None

    def __post_init__(self):
        # Read data from parquet file
        df = pd.read_parquet(self.filename, columns=list(self.x_cols) + [self.y_col])

        # Process y values
        self.y = df[self.y_col].values.astype(np.float32)
        df = df.drop(columns=[self.y_col])

        # Optional: check for invalid y values (like NaNs)
        if np.isnan(self.y).any():
            invalid_labels = np.where(np.isnan(self.y))[0]
            raise ValueError(f"Found {len(invalid_labels)} invalid (NaN) labels in y.")

        # Determine feature length
        first_row = np.fromstring(df[self.x_cols[0]].iloc[0], sep=",", dtype=np.float32)
        feature_length = len(first_row) * len(self.x_cols)

        num_cores = multiprocessing.cpu_count()
        print(f"Using {num_cores} CPU cores for parallel processing.")

        # Parallel processing of rows
        results = Parallel(n_jobs=num_cores)(
            delayed(process_row)(row, self.x_cols, feature_length)
            for _, row in df.iterrows()
        )

        self.X = np.vstack(results)

        # Check for NaN values
        if np.isnan(self.X).any():
            invalid_rows = np.where(np.isnan(self.X).any(axis=1))[0]
            print(f"Warning: Found {len(invalid_rows)} invalid rows in dataset")
            self.X = self.X[~invalid_rows]
            self.y = self.y[~invalid_rows]
        
        if np.isnan(self.X).any():
            invalid_rows = np.where(np.isnan(self.X).any(axis=1))[0]
            print(f"Warning: Found {len(invalid_rows)} invalid rows in dataset")
            self.X = self.X[~invalid_rows]
            self.y = self.y[~invalid_rows]



        del df



import hashlib


@dataclass
class HashedHashDataset:
    x_cols: tuple
    filename: str
    y_col: str = "DELLabel"
    X: np.ndarray = None
    y: np.ndarray = None
    n_features: int = 2048

    def __post_init__(self):
        df = pd.read_parquet(self.filename, columns=list(self.x_cols) + [self.y_col])

        self.y = df[self.y_col].values
        df = df.drop(columns=[self.y_col])

        if not np.all(np.isin(self.y, [0, 1])):
            raise ValueError("y must contain only binary labels (0 or 1)")

        self.X = np.empty((len(df), self.n_features), dtype=np.float32)

        for i, row in enumerate(df.itertuples(index=False)):
            all_values = []
            for x_col in self.x_cols:
                array = np.fromstring(getattr(row, x_col), sep=",", dtype=np.float32)
                all_values.extend(array)

            combined_array = np.array(all_values, dtype=np.float32)
            
            # Hash the combined array
            hash_digest = hashlib.sha512(combined_array.tobytes()).digest()

            # Map hash digest to 2048 floats (use repetition or truncation as needed)
            floats_needed = self.n_features
            repeats = (floats_needed * 4) // len(hash_digest) + 1
            extended_digest = (hash_digest * repeats)[:floats_needed * 4]

            self.X[i] = np.frombuffer(extended_digest, dtype=np.float32)

        if np.isnan(self.X).any():
            invalid_rows = np.where(np.isnan(self.X).any(axis=1))[0]
            print(f"Warning: Found {len(invalid_rows)} invalid rows in dataset")

        del df


from dataclasses import dataclass
import numpy as np
import pandas as pd
import hashlib
from joblib import Parallel, delayed
import multiprocessing


def hash_row(row, x_cols, n_features):
    all_values = []
    for x_col in x_cols:
        array = np.fromstring(row[x_col], sep=",", dtype=np.float32)
        all_values.extend(array)

    combined_array = np.array(all_values, dtype=np.float32)
    
    # Hash the combined array
    hash_digest = hashlib.sha512(combined_array.tobytes()).digest()
    
    # Repeat digest to fill 2048 floats
    floats_needed = n_features
    repeats = (floats_needed * 4) // len(hash_digest) + 1
    extended_digest = (hash_digest * repeats)[:floats_needed * 4]
    
    return np.frombuffer(extended_digest, dtype=np.float32)




@dataclass
class JobHashedHashDataset:
    x_cols: tuple
    filename: str
    y_col: str = "DELLabel"
    X: np.ndarray = None
    y: np.ndarray = None
    n_features: int = 2048

    def __post_init__(self):
        df = pd.read_parquet(self.filename, columns=list(self.x_cols) + [self.y_col])

        self.y = df[self.y_col].values.astype(np.float32)
        df = df.drop(columns=[self.y_col])

        # Optional: check for invalid y values (like NaNs)
        if np.isnan(self.y).any():
            invalid_labels = np.where(np.isnan(self.y))[0]
            raise ValueError(f"Found {len(invalid_labels)} invalid (NaN) labels in y.")

        num_cores = multiprocessing.cpu_count()
        print(f"Using {num_cores} CPU cores for parallel processing.")

        results = Parallel(n_jobs=num_cores)(
            delayed(hash_row)(row, self.x_cols, self.n_features)
            for _, row in df.iterrows()
        )

        self.X = np.vstack(results)

        if np.isnan(self.X).any():
            invalid_rows = np.where(np.isnan(self.X).any(axis=1))[0]
            print(f"Warning: Found {len(invalid_rows)} invalid rows in dataset")

        del df
        
        
@dataclass
class TestDataset:
    x_cols: tuple
    filename: str
    X: np.ndarray = None

    def __post_init__(self):
        # Read data from parquet file
        df = pd.read_parquet(self.filename, columns=list(self.x_cols))

        # Determine feature length
        first_row = np.fromstring(df[self.x_cols[0]].iloc[0], sep=",", dtype=np.float32)
        feature_length = len(first_row) * len(self.x_cols)

        num_cores = multiprocessing.cpu_count()
        print(f"Using {num_cores} CPU cores for parallel processing.")

        # Parallel processing of rows
        results = Parallel(n_jobs=num_cores)(
            delayed(process_row)(row, self.x_cols, feature_length)
            for _, row in df.iterrows()
        )

        self.X = np.vstack(results)

        # Check for NaN values
        if np.isnan(self.X).any():
            invalid_rows = np.where(np.isnan(self.X).any(axis=1))[0]
            print(f"Warning: Found {len(invalid_rows)} invalid rows in dataset")

        del df
