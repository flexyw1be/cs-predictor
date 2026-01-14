"""
Обучение RandomForest с оптимизацией через самописный генетический алгоритм.
"""
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, roc_auc_score

from genetic_algorithm import GeneticOptimizer, accuracy_score_func, roc_auc_score_func
from config import GENETIC_RF_SPACE, get_ga_settings, get_feature_cols


def train_rf_genetic(data_dir='data/processed', use_kaggle=False):
    """
    Тренировка Random Forest с генетической оптимизацией.
    
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
    
    # Определяем категориальные признаки
    cat_features = [c for c in feature_cols if c == 'map']
    num_features = [c for c in feature_cols if c != 'map']
    
    print(f"   Числовые: {len(num_features)}")
    print(f"   Категориальные: {len(cat_features)}")
    
    # Подготовка данных
    X_train = df_train[feature_cols].fillna(0)
    y_train = df_train['winner_is_A']
    
    X_val = df_val[feature_cols].fillna(0)
    y_val = df_val['winner_is_A']
    
    X_test = df_test[feature_cols].fillna(0)
    y_test = df_test['winner_is_A']
    
    # Объединяем train + val для CV
    X_full = pd.concat([X_train, X_val], ignore_index=True)
    y_full = pd.concat([y_train, y_val], ignore_index=True)
    
    # Препроцессинг
    print("\n🔧 Препроцессинг данных...")
    if cat_features:
        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_features)
        ])
    else:
        preprocessor = StandardScaler()
    
    X_full_proc = preprocessor.fit_transform(X_full)
    X_test_proc = preprocessor.transform(X_test)
    
    # CV splitter (3 фолда для скорости на маленьких данных)
    n_splits = 3 if len(X_full) < 5000 else 5
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    print(f"   CV folds: {n_splits}")
    
    # Получаем настройки GA в зависимости от размера данных
    ga_settings = get_ga_settings(len(X_full))
    print(f"   GA режим: {'FAST' if len(X_full) < 5000 else 'FULL'}")
    
    # Фиксированные параметры модели
    fixed_params = {
        'random_state': 42,
        'n_jobs': -1,
        'class_weight': 'balanced',
    }
    
    # Создаём обёртку модели с фиксированными параметрами
    class RFWrapper:
        def __init__(self, **kwargs):
            all_params = {**fixed_params, **kwargs}
            self.model = RandomForestClassifier(**all_params)
        
        def fit(self, X, y, **kwargs):
            self.model.fit(X, y)
            return self
        
        def predict(self, X):
            return self.model.predict(X)
        
        def predict_proba(self, X):
            return self.model.predict_proba(X)
        
        def score(self, X, y):
            return self.model.score(X, y)
    
    # Запускаем генетический алгоритм
    print("\n" + "="*60)
    optimizer = GeneticOptimizer(
        param_space=GENETIC_RF_SPACE,
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
        model_class=RFWrapper,
        X=X_full_proc,
        y=y_full,
        cv_splitter=cv,
        scoring_func=accuracy_score_func
    )
    
    # Обучаем финальную модель с лучшими параметрами
    print("\n🏁 Обучение финальной модели...")
    final_model = RandomForestClassifier(
        **best_params,
        **fixed_params
    )
    final_model.fit(X_full_proc, y_full)
    
    # Оценка на тесте
    y_pred = final_model.predict(X_test_proc)
    y_prob = final_model.predict_proba(X_test_proc)[:, 1]
    
    test_acc = accuracy_score(y_test, y_pred)
    test_auc = roc_auc_score(y_test, y_prob)
    
    print("\n" + "="*60)
    print("📊 РЕЗУЛЬТАТЫ НА ТЕСТЕ")
    print("="*60)
    print(f"CV Score:      {best_score:.4f}")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Test ROC-AUC:  {test_auc:.4f}")
    
    # Feature importance
    print("\n📈 Важность признаков:")
    
    # Получаем имена после OneHotEncoder
    if cat_features:
        ohe = preprocessor.named_transformers_['cat']
        cat_feature_names = ohe.get_feature_names_out(cat_features).tolist()
        all_feature_names = num_features + cat_feature_names
    else:
        all_feature_names = num_features
    
    importances = sorted(
        zip(all_feature_names, final_model.feature_importances_),
        key=lambda x: -x[1]
    )
    for name, imp in importances[:15]:
        print(f"   {name:30s}: {imp:.4f}")
    
    # Сохранение
    if not os.path.exists('models'):
        os.makedirs('models')
    
    model_path = 'models/rf_genetic.pkl'
    joblib.dump({
        'model': final_model,
        'preprocessor': preprocessor,
        'feature_cols': feature_cols,
        'best_params': best_params,
        'test_accuracy': test_acc,
        'test_auc': test_auc,
    }, model_path)
    print(f"\n💾 Модель сохранена: {model_path}")
    
    return final_model, best_params, test_acc


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--kaggle', action='store_true', help='Использовать Kaggle датасет')
    args = parser.parse_args()
    
    train_rf_genetic(use_kaggle=args.kaggle)