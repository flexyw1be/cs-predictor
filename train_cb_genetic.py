"""
Обучение CatBoost с оптимизацией через самописный генетический алгоритм.
"""
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score

from catboost import CatBoostClassifier
from genetic_algorithm import GeneticOptimizer, accuracy_score_func, roc_auc_score_func
from config import GENETIC_CB_SPACE, get_ga_settings, get_feature_cols


def train_cb_genetic(data_dir='data/processed', use_kaggle=False):
    """
    Тренировка CatBoost с генетической оптимизацией.
    
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
    cat_indices = [feature_cols.index(c) for c in cat_features] if cat_features else None
    
    print(f"   Категориальные: {cat_features}")
    
    # Подготовка данных
    X_train = df_train[feature_cols].fillna(0)
    y_train = df_train['winner_is_A']
    
    X_val = df_val[feature_cols].fillna(0)
    y_val = df_val['winner_is_A']
    
    X_test = df_test[feature_cols].fillna(0)
    y_test = df_test['winner_is_A']
    
    # Объединяем train + val для CV оптимизации
    X_full = pd.concat([X_train, X_val], ignore_index=True)
    y_full = pd.concat([y_train, y_val], ignore_index=True)
    
    # CV splitter (3 фолда для скорости на маленьких данных)
    n_splits = 3 if len(X_full) < 5000 else 5
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    print(f"   CV folds: {n_splits}")
    
    # Получаем настройки GA в зависимости от размера данных
    ga_settings = get_ga_settings(len(X_full))
    print(f"   GA режим: {'FAST' if len(X_full) < 5000 else 'FULL'}")
    
    # Фиксированные параметры модели
    fixed_params = {
        'random_seed': 42,
        'verbose': 0,
        'early_stopping_rounds': 30,  # Меньше для скорости
        'cat_features': cat_features if cat_features else None,
    }
    
    # Создаём обёртку модели с фиксированными параметрами
    class CatBoostWrapper:
        def __init__(self, **kwargs):
            # Преобразуем float параметры
            params = {}
            for k, v in kwargs.items():
                if k == 'iterations':
                    params[k] = int(v)
                elif k == 'depth':
                    params[k] = int(v)
                else:
                    params[k] = v
            
            all_params = {**fixed_params, **params}
            self.model = CatBoostClassifier(**all_params)
            self.cat_features = cat_features
        
        def fit(self, X, y, **kwargs):
            # Используем часть данных как eval_set для early stopping
            from sklearn.model_selection import train_test_split
            X_tr, X_ev, y_tr, y_ev = train_test_split(
                X, y, test_size=0.15, random_state=42, stratify=y
            )
            self.model.fit(X_tr, y_tr, eval_set=(X_ev, y_ev))
            return self
        
        def predict(self, X):
            return self.model.predict(X)
        
        def predict_proba(self, X):
            return self.model.predict_proba(X)
        
        def score(self, X, y):
            y_pred = self.model.predict(X)
            return accuracy_score(y, y_pred)
    
    # Запускаем генетический алгоритм
    print("\n" + "="*60)
    
    optimizer = GeneticOptimizer(
        param_space=GENETIC_CB_SPACE,
        population_size=ga_settings['population_size'],
        generations=ga_settings['generations'],
        mutation_rate=ga_settings['mutation_rate'],
        mutation_strength=ga_settings['mutation_strength'],
        crossover_rate=ga_settings['crossover_rate'],
        elite_size=ga_settings['elite_size'],
        tournament_size=ga_settings['tournament_size'],
        early_stopping=ga_settings['early_stopping'],
        random_state=42,
        verbose=1
    )
    
    best_params, best_score = optimizer.optimize(
        model_class=CatBoostWrapper,
        X=X_full,
        y=y_full,
        cv_splitter=cv,
        scoring_func=accuracy_score_func
    )
    
    # Преобразуем параметры для финальной модели
    final_params = {
        'depth': int(best_params['depth']),
        'iterations': int(best_params['iterations']),
        'learning_rate': best_params['learning_rate'],
        'l2_leaf_reg': best_params['l2_leaf_reg'],
        'bagging_temperature': best_params['bagging_temperature'],
        'random_strength': best_params['random_strength'],
    }
    
    # Обучаем финальную модель с лучшими параметрами
    print("\n🏁 Обучение финальной модели...")
    final_model = CatBoostClassifier(
        **final_params,
        random_seed=42,
        verbose=100,
        early_stopping_rounds=50,
        cat_features=cat_features if cat_features else None,
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
    print(f"CV Score:      {best_score:.4f}")
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
    
    # Сохраняем в формате CatBoost
    model_path = 'models/cb_genetic.cbm'
    final_model.save_model(model_path)
    print(f"\n💾 Модель сохранена: {model_path}")
    
    # Сохраняем метаданные
    import json
    meta_path = 'models/cb_genetic_meta.json'
    with open(meta_path, 'w') as f:
        json.dump({
            'feature_cols': feature_cols,
            'best_params': {k: float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v for k, v in best_params.items()},
            'test_accuracy': float(test_acc),
            'test_auc': float(test_auc),
            'cv_score': float(best_score),
        }, f, indent=2)
    print(f"💾 Метаданные сохранены: {meta_path}")
    
    return final_model, best_params, test_acc


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--kaggle', action='store_true', help='Использовать Kaggle датасет')
    args = parser.parse_args()
    
    train_cb_genetic(use_kaggle=args.kaggle)