"""
Обучение CatBoost с Grid Search оптимизацией гиперпараметров.
Использует встроенный метод CatBoost для поиска лучших параметров.
"""
import pandas as pd
import os
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from config import get_feature_cols


def train_cb_gridsearch(data_dir='data/processed', use_kaggle=False):
    """
    Тренировка CatBoost с Grid Search оптимизацией.
    
    Параметры:
    ----------
    data_dir : str
        Путь к директории с данными
    use_kaggle : bool
        Использовать ли Kaggle датасет
    """
    # Загрузка данных
    if use_kaggle:
        data_dir = 'data/processed_kaggle'
        dataset_type = 'kaggle'
    else:
        dataset_type = 'main'
    
    print(f"📂 Загрузка данных из {data_dir}...")
    df_train = pd.read_csv(f'{data_dir}/train.csv')
    df_val = pd.read_csv(f'{data_dir}/val.csv')
    df_test = pd.read_csv(f'{data_dir}/test.csv')
    
    print(f"   Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")
    
    # Получаем полный список признаков из config
    all_features = get_feature_cols(dataset_type)
    
    # Фильтруем только доступные в данных
    feature_cols = [c for c in all_features if c in df_train.columns]
    print(f"\n📋 Используется {len(feature_cols)} признаков из {len(all_features)}")
    
    # Определяем категориальные признаки для CatBoost
    cat_features = [c for c in feature_cols if c == 'map']
    print(f"   Категориальные: {cat_features}")
    
    # Подготовка данных
    X_train = df_train[feature_cols].fillna(0)
    y_train = df_train['winner_is_A']
    
    X_val = df_val[feature_cols].fillna(0)
    y_val = df_val['winner_is_A']
    
    X_test = df_test[feature_cols].fillna(0)
    y_test = df_test['winner_is_A']
    
    # Объединяем train + val для Grid Search
    X_full = pd.concat([X_train, X_val], ignore_index=True)
    y_full = pd.concat([y_train, y_val], ignore_index=True)
    
    # Сетка параметров для поиска
    param_grid = {
        'depth': [4, 5, 6, 8],
        'learning_rate': [0.01, 0.03, 0.05, 0.1],
        'iterations': [200, 500, 800, 1000],
        'l2_leaf_reg': [1, 3, 5, 7],
    }
    
    total_combinations = 1
    for v in param_grid.values():
        total_combinations *= len(v)
    
    print(f"\n🔍 Grid Search: {total_combinations} комбинаций параметров")
    print(f"   Параметры: {list(param_grid.keys())}")
    
    # Создаём модель
    model = CatBoostClassifier(
        cat_features=cat_features if cat_features else None,
        random_seed=42,
        verbose=0,
        early_stopping_rounds=30,
    )
    
    # Запускаем Grid Search
    print("\n" + "="*60)
    print("🚀 Запуск Grid Search (это может занять несколько минут)...")
    print("="*60)
    
    grid_search_result = model.grid_search(
        param_grid, 
        X=X_full, 
        y=y_full, 
        cv=3,  # 3 фолда для скорости
        stratified=True, 
        plot=False,
        verbose=True
    )
    
    # Лучшие параметры
    best_params = grid_search_result['params']
    
    print("\n" + "="*60)
    print("🏆 ЛУЧШИЕ ПАРАМЕТРЫ")
    print("="*60)
    for name, value in best_params.items():
        print(f"   {name}: {value}")
    
    # Обучаем финальную модель с лучшими параметрами
    print("\n🏁 Обучение финальной модели...")
    final_model = CatBoostClassifier(
        **best_params,
        cat_features=cat_features if cat_features else None,
        random_seed=42,
        verbose=100,
        early_stopping_rounds=50,
    )
    final_model.fit(X_full, y_full, eval_set=(X_test, y_test))
    
    # Оценка на тесте
    y_pred = final_model.predict(X_test)
    y_prob = final_model.predict_proba(X_test)[:, 1]
    
    test_acc = accuracy_score(y_test, y_pred)
    test_auc = roc_auc_score(y_test, y_prob)
    
    print("\n" + "="*60)
    print("📊 РЕЗУЛЬТАТЫ НА ТЕСТЕ")
    print("="*60)
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Test ROC-AUC:  {test_auc:.4f}")
    
    # Feature importance
    print("\n📈 Важность признаков (CatBoost):")
    importances = sorted(
        zip(feature_cols, final_model.feature_importances_),
        key=lambda x: -x[1]
    )
    for name, imp in importances[:15]:
        print(f"   {name:30s}: {imp:.2f}")
    
    # Сохранение
    if not os.path.exists('models'):
        os.makedirs('models')
    
    model_path = 'models/cb_gridsearch.cbm'
    final_model.save_model(model_path)
    print(f"\n💾 Модель сохранена: {model_path}")
    
    # Сохраняем метаданные
    import json
    meta_path = 'models/cb_gridsearch_meta.json'
    with open(meta_path, 'w') as f:
        json.dump({
            'feature_cols': feature_cols,
            'best_params': best_params,
            'test_accuracy': float(test_acc),
            'test_auc': float(test_auc),
        }, f, indent=2)
    print(f"💾 Метаданные сохранены: {meta_path}")
    
    # Код для вставки в train.py
    print("\n" + "="*60)
    print("📝 КОД ДЛЯ ВСТАВКИ В train.py:")
    print("="*60)
    print(f"""
cb_model = CatBoostClassifier(
    iterations={best_params.get('iterations', 500)},
    depth={best_params.get('depth', 6)},
    learning_rate={best_params.get('learning_rate', 0.05)},
    l2_leaf_reg={best_params.get('l2_leaf_reg', 3)},
    random_seed=42,
    early_stopping_rounds=50,
    verbose=100,
    eval_metric='AUC',
    use_best_model=True,
)
""")
    
    return final_model, best_params, test_acc


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--kaggle', action='store_true', help='Использовать Kaggle датасет')
    args = parser.parse_args()
    
    train_cb_gridsearch(use_kaggle=args.kaggle)
