import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from config import TRAIN_DATA_PATH, MODEL_DIR, MODEL_PATH, REPORT_DIR, METRICS_PATH, TARGET, RANDOM_STATE
from features import create_features

def main():
    df = pd.read_csv(TRAIN_DATA_PATH)
    df = create_features(df)

    X = df.drop(columns=[TARGET, "PassengerId", "Name", "Ticket", "Cabin"])
    y = df[TARGET]

    numeric_features = ["Age", "Fare", "SibSp", "Parch", "FamilySize", "Fare"]
    categorical_features = ["Pclass", "Sex", "Embarked", "Title", "IsAlone", "CabinKnown"]

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy='most_frequent')),
        ("encoder", OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
        ],
        sparse_threshold=0
    
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
        "Neural Network": MLPClassifier(
            max_iter=1000,
            random_state=RANDOM_STATE,
            early_stopping=True
        )
    }

    param_grids = {
        "Logistic Regression": {
            "model__C": [0.01, 0.1, 1, 10]
        },

        "Random Forest": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [4, 6, 8, None],
            "model__min_samples_split": [2, 5, 10]
        },

        "Neural Network": {
            "model__hidden_layer_sizes": [(32,), (64,), (64, 32)],
            "model__activation": ["relu", "tanh"],
            "model__alpha": [0.0001, 0.001, 0.01],
            "model__learning_rate": ["constant", "adaptive"],
            "model__learning_rate_init": [0.001, 0.01]
        }
    }

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE
    )

    best_model = None
    best_score = 0
    results = {}

    cv= StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])

        grid = GridSearchCV(
            pipeline,
            param_grids[name],
            cv=cv,
            scoring='f1',
            n_jobs=-1,
            error_score='raise'
        )

        grid.fit(X_train, y_train)

        preds = grid.predict(X_val)
        probs = grid.predict_proba(X_val)[:, 1]

        accuracy = accuracy_score(y_val, preds)
        f1 = f1_score(y_val, preds)
        roc_auc = roc_auc_score(y_val, probs)

        results[name] = {
            "accuracy": round(accuracy, 4),
            "f1": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
            "best_params": grid.best_params_
        }

        if f1 > best_score:
            best_score = f1
            best_model = grid.best_estimator_

    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    joblib.dump(best_model, MODEL_PATH)

    with open(METRICS_PATH, "w") as f:
        json.dump(results, f, indent=4)

    print("Training Complete.")
    print(f"Best model saved to: {MODEL_PATH}")
    print(results)


if __name__ == "__main__":
    main()