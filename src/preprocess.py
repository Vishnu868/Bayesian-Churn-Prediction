"""Data loading, cleaning, encoding, scaling and PCA for the Telco churn dataset."""
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop(columns=["customerID"], errors="ignore")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna().reset_index(drop=True)          # 7043 -> 7032 rows
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = LabelEncoder().fit_transform(df[col])
    return df


def build_features(df: pd.DataFrame, variance: float = 0.95, test_size: float = 0.2, seed: int = 42):
    X = df.drop(columns=["Churn"])
    y = df["Churn"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    scaler = StandardScaler().fit(X_train)
    pca = PCA(n_components=variance, random_state=seed).fit(scaler.transform(X_train))
    Z_train = pca.transform(scaler.transform(X_train))
    Z_test = pca.transform(scaler.transform(X_test))
    return {
        "feature_names": list(X.columns),
        "X_train": Z_train, "X_test": Z_test,
        "y_train": y_train, "y_test": y_test,
        "scaler": scaler, "pca": pca,
    }
