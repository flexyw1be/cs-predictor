import pandas as pd
import numpy as np
import os
import joblib
import argparse
import matplotlib
matplotlib.use('Agg')  # Используем бэкенд без GUI
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, roc_auc_score, brier_score_loss,
    confusion_matrix, classification_report, precision_score, recall_score, f1_score
)
from catboost import CatBoostClassifier

from config import get_feature_cols


def train_model(use_kaggle=False):
    """
    Обучение модели.
    
    Параметры:
    ----------
    use_kaggle : bool
        Использовать Kaggle датасет (45K+ матчей) вместо основного
    """
    # --- 1. ЗАГРУЗКА ДАННЫХ ---
    if use_kaggle:
        data_dir = 'data/processed_kaggle'
        dataset_type = 'kaggle'
        print("="*60)
        print("🎮 ТРЕНИРОВКА НА KAGGLE ДАТАСЕТЕ (45K+ матчей)")
        print("="*60)
    else:
        data_dir = 'data/processed'
        dataset_type = 'main'
        print("="*60)
        print("🎮 ТРЕНИРОВКА НА ОСНОВНОМ ДАТАСЕТЕ")
        print("="*60)
    
    train_path = f'{data_dir}/train.csv'
    val_path = f'{data_dir}/val.csv'
    test_path = f'{data_dir}/test.csv'
    
    if not os.path.exists(train_path):
        if use_kaggle:
            print("Ошибка: Сначала запусти process_kaggle.py")
        else:
            print("Ошибка: Сначала запусти data_processor.py")
        return

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)
    
    print(f"\n📊 Размеры данных:")
    print(f"   Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")

    # Получаем признаки из config
    all_features = get_feature_cols(dataset_type)
    
    # Фильтруем только доступные в данных
    feature_cols = [c for c in all_features if c in df_train.columns]
    
    print(f"\n📋 Используется {len(feature_cols)} признаков из {len(all_features)}")
    
    target = 'winner_is_A'

    X_train, y_train = df_train[feature_cols].fillna(0), df_train[target]
    X_val, y_val = df_val[feature_cols].fillna(0), df_val[target]
    X_test, y_test = df_test[feature_cols].fillna(0), df_test[target]
    
    # Объединяем train + val для обучения
    X_train_full = pd.concat([X_train, X_val], ignore_index=True)
    y_train_full = pd.concat([y_train, y_val], ignore_index=True)
    
    groups_train = df_train['match_id'] if 'match_id' in df_train.columns else None

    # --- 2. CATBOOST (лучше работает с категориями и небольшими данными) ---
    print("\n" + "="*60)
    print("Обучение CatBoost (параметры из GA оптимизации)...")
    print("="*60)
    
    # Категориальные признаки
    cat_features = [c for c in ['map'] if c in feature_cols]
    
    # CatBoost с оптимизированными параметрами
    cb_model = CatBoostClassifier(
        iterations=884,
        depth=5,
        learning_rate=0.036,
        l2_leaf_reg=1.0,
        bagging_temperature=0.11,
        random_strength=0.21,
        random_seed=42,
        early_stopping_rounds=50,
        verbose=100,
        eval_metric='AUC',
        use_best_model=True,
    )
        
    cb_model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        cat_features=cat_features,
    )
    
    # Предсказания CatBoost
    y_prob_cb = cb_model.predict_proba(X_test)[:, 1]
    y_pred_cb = cb_model.predict(X_test)
    
    acc_cb = accuracy_score(y_test, y_pred_cb)
    roc_auc_cb = roc_auc_score(y_test, y_prob_cb)
    
    print(f"\nCatBoost Test Accuracy: {acc_cb:.2%}")
    print(f"CatBoost Test ROC-AUC: {roc_auc_cb:.4f}")

    # --- 3. RANDOM FOREST ДЛЯ СРАВНЕНИЯ ---
    print("\n" + "="*60)
    print("Обучение RandomForest (параметры из GA оптимизации)...")
    print("="*60)
    
    # Препроцессинг для RF
    num_features = [c for c in feature_cols if c not in ['map']]
    
    if cat_features:
        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_features)
        ])
    else:
        # Для Kaggle датасета (без категорий)
        preprocessor = StandardScaler()
    
    X_train_proc = preprocessor.fit_transform(X_train_full)
    X_test_proc = preprocessor.transform(X_test)

    # RandomForest с параметрами из GA оптимизации
    rf_model = RandomForestClassifier(
        n_estimators=379,
        max_depth=18,
        min_samples_leaf=4,
        min_samples_split=11,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_proc, y_train_full)
    
    y_prob_rf = rf_model.predict_proba(X_test_proc)[:, 1]
    y_pred_rf = rf_model.predict(X_test_proc)
    
    acc_rf = accuracy_score(y_test, y_pred_rf)
    roc_auc_rf = roc_auc_score(y_test, y_prob_rf)
    
    print(f"RandomForest Test Accuracy: {acc_rf:.2%}")
    print(f"RandomForest Test ROC-AUC: {roc_auc_rf:.4f}")

    # --- 4. ENSEMBLE (усреднение вероятностей) ---
    y_prob_ensemble = (y_prob_cb + y_prob_rf) / 2
    y_pred_ensemble = (y_prob_ensemble > 0.5).astype(int)
    
    acc_ens = accuracy_score(y_test, y_pred_ensemble)
    roc_auc_ens = roc_auc_score(y_test, y_prob_ensemble)
    
    # Выбираем лучшую модель
    best_model = 'CatBoost' if roc_auc_cb >= roc_auc_rf else 'RandomForest'
    if roc_auc_ens > max(roc_auc_cb, roc_auc_rf):
        best_model = 'Ensemble'
        y_prob = y_prob_ensemble
        y_pred = y_pred_ensemble
        acc = acc_ens
        roc_auc = roc_auc_ens
    elif best_model == 'CatBoost':
        y_prob = y_prob_cb
        y_pred = y_pred_cb
        acc = acc_cb
        roc_auc = roc_auc_cb
    else:
        y_prob = y_prob_rf
        y_pred = y_pred_rf
        acc = acc_rf
        roc_auc = roc_auc_rf

    # --- 5. ИТОГОВЫЕ МЕТРИКИ ---
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n" + "=" * 60)
    print(f"{'ИТОГОВЫЕ МЕТРИКИ':^60}")
    print("=" * 60)
    print(f"{'Модель':<25}: {best_model}")
    print(f"{'CatBoost Accuracy':<25}: {acc_cb:.2%}")
    print(f"{'RandomForest Accuracy':<25}: {acc_rf:.2%}")
    print(f"{'Ensemble Accuracy':<25}: {acc_ens:.2%}")
    print("-" * 60)
    print(f"{'Лучшая Accuracy':<25}: {acc:.2%}")
    print(f"{'Precision':<25}: {prec:.2%}")
    print(f"{'Recall':<25}: {rec:.2%}")
    print(f"{'F1-Score':<25}: {f1:.4f}")
    print(f"{'ROC-AUC':<25}: {roc_auc:.4f}")
    print(f"{'Brier Score':<25}: {brier_score_loss(y_test, y_prob):.4f}")
    print("=" * 60)
    
    # --- 6. FEATURE IMPORTANCE ---
    print("\nFeature Importance (CatBoost):")
    importances = sorted(zip(feature_cols, cb_model.feature_importances_), key=lambda x: -x[1])
    for name, imp in importances[:15]:
        print(f"  {name:25s}: {imp:.2f}")

    # --- 7. СОХРАНЕНИЕ ---
    if not os.path.exists('models'): 
        os.makedirs('models')
    cb_model.save_model('models/catboost_model.cbm')
    joblib.dump(rf_model, 'models/rf_model.pkl')
    joblib.dump(preprocessor, 'models/preprocessor.pkl')
    joblib.dump(feature_cols, 'models/feature_cols.pkl')
    
    print(f"\nМодели сохранены в папку models/")

    # --- 8. ГРАФИКИ ---
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Confusion Matrix
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', 
                   cmap='Blues', ax=axes[0])
        axes[0].set_title(f'Confusion Matrix ({best_model})\nAccuracy: {acc:.2%}')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        
        # Feature Importance
        imp_df = pd.DataFrame(importances[:10], columns=['Feature', 'Importance'])
        sns.barplot(data=imp_df, x='Importance', y='Feature', ax=axes[1], palette='viridis')
        axes[1].set_title('Top 10 Feature Importance')
        
        plt.tight_layout()
        plt.savefig('models/training_results.png', dpi=150)
        plt.show()
    except Exception as e:
        print(f"Не удалось создать графики: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Обучение модели для предсказания CS')
    parser.add_argument('--kaggle', action='store_true', 
                        help='Использовать Kaggle датасет (45K+ матчей)')
    args = parser.parse_args()
    
    train_model(use_kaggle=args.kaggle)