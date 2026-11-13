"""Models (D12). Each entry: a scikit-learn pipeline and a small grid tuned on the validation season."""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from . import config
from .features import CATEGORICAL, NUMERIC


def _onehot():
    return make_pipeline(SimpleImputer(strategy="constant", fill_value="missing"),
                         OneHotEncoder(handle_unknown="ignore"))


def logistic():
    # Interpretable: coefficients show which drivers matter. Needs imputation and scaling.
    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), NUMERIC),
        ("cat", _onehot(), CATEGORICAL)])
    model = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=5000))])
    grid = {"clf__C": [0.01, 0.1, 1.0]}
    return model, grid


def random_forest():
    # Non-linear effects with little tuning; robust reference model.
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median", add_indicator=True), NUMERIC),
        ("cat", _onehot(), CATEGORICAL)])
    model = Pipeline([("pre", pre), ("clf", RandomForestClassifier(
        n_estimators=500, n_jobs=-1, random_state=config.RANDOM_STATE))])
    grid = {"clf__min_samples_leaf": [5, 20, 50], "clf__max_features": ["sqrt", 0.5]}
    return model, grid


def gradient_boosting():
    # Captures interactions (age x position x competition for the spot). Handles missing values natively.
    pre = ColumnTransformer([
        ("num", "passthrough", NUMERIC),
        ("cat", make_pipeline(SimpleImputer(strategy="constant", fill_value="missing"),
                              OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
         CATEGORICAL)])
    n_num = len(NUMERIC)
    clf = HistGradientBoostingClassifier(
        max_iter=300, random_state=config.RANDOM_STATE,
        categorical_features=list(range(n_num, n_num + len(CATEGORICAL))))
    model = Pipeline([("pre", pre), ("clf", clf)])
    grid = {"clf__learning_rate": [0.03, 0.1], "clf__max_leaf_nodes": [7, 15], "clf__min_samples_leaf": [20, 50]}
    return model, grid


MODELS = {"logistic": logistic, "random_forest": random_forest, "gradient_boosting": gradient_boosting}
