import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')
import seaborn as sns

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, roc_auc_score, brier_score_loss,
    confusion_matrix, classification_report, precision_score, recall_score, f1_score
)

# Импорт Байесовского оптимизатора
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer


def train_model():
    # --- 1. ЗАГРУЗКА ДАННЫХ ---
    train_path, val_path, test_path = 'data/processed/train.csv', 'data/processed/val.csv', 'data/processed/test.csv'
    if not os.path.exists(train_path):
        print("Ошибка: Сначала запусти data_processor.py")
        return

    df_train, df_val, df_test = pd.read_csv(train_path), pd.read_csv(val_path), pd.read_csv(test_path)

    feature_cols = [
        'map', 'rank_diff', 'abs_rank_diff', 'picked_by_is_A',
        'is_decider', 'map_winrate_A', 'map_winrate_B',
        'recent_form_A', 'recent_form_B', 'elo_diff', 'map_elo_diff'
    ]
    target = 'winner_is_A'

    X_train, y_train = df_train[feature_cols], df_train[target]
    X_val, y_val = df_val[feature_cols], df_val[target]
    X_test, y_test = df_test[feature_cols], df_test[target]
    groups_train = df_train['match_id']

    # --- 2. ПРЕПРОЦЕССИНГ ---
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), [c for c in feature_cols if c != 'map']),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), ['map'])
    ])
    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)

    # --- 3. БАЙЕСОВСКАЯ ОПТИМИЗАЦИЯ ---
    print("Запуск Байесовской оптимизации параметров...")
    rf_base = RandomForestClassifier(random_state=85)

    # Пространство поиска (используем распределения вместо списков)
    search_spaces = {
        'n_estimators': Integer(100, 500),
        'max_depth': Integer(5, 30),
        'min_samples_split': Integer(2, 10),
        'min_samples_leaf': Integer(1, 5),
        'max_features': Categorical(['sqrt', 'log2'])
    }

    cv_strategy = GroupKFold(n_splits=5)

    opt = BayesSearchCV(
        estimator=rf_base,
        search_spaces=search_spaces,
        n_iter=32,  # Количество итераций (обычно 30-50 достаточно)
        cv=cv_strategy,
        scoring='roc_auc',
        n_jobs=-1,
        random_state=85,
        verbose=1
    )

    # В BayesSearchCV groups передается напрямую в fit
    opt.fit(X_train_proc, y_train, groups=groups_train)

    print(f"\nЛучшие параметры (Bayes): {opt.best_params_}")

    # --- 4. КАЛИБРОВКА ---
    print("Калибровка вероятностей...")
    X_calib_combined = np.vstack([X_train_proc, X_val_proc])
    y_calib_combined = np.concatenate([y_train, y_val])

    model = CalibratedClassifierCV(estimator=opt.best_estimator_, cv=5, method='sigmoid')
    model.fit(X_calib_combined, y_calib_combined)

    # --- 5. РАСЧЕТ ТОЧНОСТИ ---
    y_prob = model.predict_proba(X_test_proc)[:, 1]
    y_pred = model.predict(X_test_proc)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print("\n" + "=" * 40)
    print(f"{'ИТОГОВЫЕ МЕТРИКИ (BAYES OPT)':^40}")
    print("=" * 40)
    print(f"Accuracy (Точность):      {acc:.2%}")
    print(f"Precision (Прогноз):      {prec:.2%}")
    print(f"Recall (Полнота):         {rec:.2%}")
    print(f"F1-Score (Баланс):        {f1:.4f}")
    print(f"ROC-AUC:                  {roc_auc:.4f}")
    print(f"Brier Score:              {brier_score_loss(y_test, y_prob):.4f}")
    print("=" * 40)

    # --- 6. СОХРАНЕНИЕ ---
    if not os.path.exists('models'): os.makedirs('models')
    joblib.dump(model, 'models/rf_model.pkl')
    joblib.dump(preprocessor, 'models/preprocessor.pkl')

    # --- 7. ГРАФИКИ ---
    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Oranges')
    plt.title(f'Confusion Matrix\nAccuracy: {acc:.2%}')
    plt.show()


if __name__ == "__main__":
    train_model()