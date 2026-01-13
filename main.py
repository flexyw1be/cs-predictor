from data_loader import load_data, split_time_series_data
from features import feature_engineering
from ga_optimizer import run_genetic_optimization
from sklearn.metrics import accuracy_score, classification_report


def main():
    raw_df = load_data()

    X, y = feature_engineering(raw_df)

    X_train, X_test, y_train, y_test = split_time_series_data(X, y)

    best_rf_model = run_genetic_optimization(X_train, y_train)

    predictions = best_rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("-" * 30)
    print(f"Final Model Accuracy on Test Set: {accuracy:.4f}")
    print("-" * 30)
    print("Classification Report:")
    print(classification_report(y_test, predictions))

    import pandas as pd
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_rf_model.feature_importances_
    }).sort_values(by='importance', ascending=False)

    print("\nTop 10 Important Features:")
    print(feature_importance.head(10))


if __name__ == "__main__":
    main()