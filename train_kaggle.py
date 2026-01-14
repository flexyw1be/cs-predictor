"""
Тренировка модели на расширенном Kaggle датасете (~45K матчей).
Использует статистику игроков и продвинутые фичи.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, roc_auc_score, classification_report,
                           precision_score, recall_score, f1_score)
import warnings
warnings.filterwarnings('ignore')

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("CatBoost не установлен, используем GradientBoosting")


def load_data(data_dir='data/processed_kaggle'):
    """Загружает train/val/test данные."""
    train = pd.read_csv(f'{data_dir}/train.csv')
    val = pd.read_csv(f'{data_dir}/val.csv')
    test = pd.read_csv(f'{data_dir}/test.csv')
    return train, val, test


def get_features():
    """Возвращает список признаков для модели."""
    # Базовые фичи из исходных данных
    base_features = [
        'rank_diff',
        'abs_rank_diff',
    ]
    
    # Статистика игроков (из Kaggle датасета)
    player_stats = [
        'team_A_avg_rating',
        'team_A_avg_kd',
        'team_A_avg_adr',
        'team_A_avg_kast',
        'team_B_avg_rating',
        'team_B_avg_kd', 
        'team_B_avg_adr',
        'team_B_avg_kast',
    ]
    
    # Продвинутые фичи (вычисленные)
    advanced_features = [
        'elo_diff',
        'map_elo_diff',
        'h2h_rate',
        'h2h_games',
        'momentum_diff',
        'streak_A',
        'streak_B',
        'days_since_last_A',
        'days_since_last_B',
        'overall_winrate_A',
        'overall_winrate_B',
        'winrate_diff',
        'map_games_A',
        'map_games_B',
        'map_experience_diff',
    ]
    
    return base_features + player_stats + advanced_features


def add_derived_features(df):
    """Добавляет производные признаки."""
    # Разницы в статистике игроков
    if 'team_A_avg_rating' in df.columns:
        df['rating_diff'] = df['team_A_avg_rating'] - df['team_B_avg_rating']
        df['kd_diff'] = df['team_A_avg_kd'] - df['team_B_avg_kd']
        df['adr_diff'] = df['team_A_avg_adr'] - df['team_B_avg_adr']
        df['kast_diff'] = df['team_A_avg_kast'] - df['team_B_avg_kast']
    
    # Streak разница
    if 'streak_A' in df.columns:
        df['streak_diff'] = df['streak_A'] - df['streak_B']
    
    # Усталость (много игр подряд)
    if 'days_since_last_A' in df.columns:
        df['rest_diff'] = df['days_since_last_B'] - df['days_since_last_A']  # больше = лучше отдохнул
    
    return df


def get_extended_features():
    """Расширенный список признаков с производными."""
    base = get_features()
    derived = [
        'rating_diff', 'kd_diff', 'adr_diff', 'kast_diff',
        'streak_diff', 'rest_diff'
    ]
    return base + derived


def prepare_data(train, val, test):
    """Подготавливает данные для обучения."""
    # Добавляем производные фичи
    train = add_derived_features(train.copy())
    val = add_derived_features(val.copy())
    test = add_derived_features(test.copy())
    
    # Получаем список фичей
    all_features = get_extended_features()
    
    # Оставляем только доступные колонки
    available = [f for f in all_features if f in train.columns]
    print(f"Используется {len(available)} признаков из {len(all_features)}")
    
    X_train = train[available].fillna(0)
    y_train = train['winner_is_A']
    
    X_val = val[available].fillna(0)
    y_val = val['winner_is_A']
    
    X_test = test[available].fillna(0)
    y_test = test['winner_is_A']
    
    return X_train, y_train, X_val, y_val, X_test, y_test, available


def train_and_evaluate():
    """Основная функция тренировки и оценки моделей."""
    print("="*70)
    print("ТРЕНИРОВКА МОДЕЛИ НА KAGGLE ДАТАСЕТЕ (45K+ матчей)")
    print("="*70)
    
    # Загрузка данных
    print("\n📊 Загрузка данных...")
    train, val, test = load_data()
    print(f"   Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    
    # Подготовка
    X_train, y_train, X_val, y_val, X_test, y_test, features = prepare_data(train, val, test)
    
    print(f"\n📋 Признаки ({len(features)}):")
    for i, f in enumerate(features):
        print(f"   {i+1:2d}. {f}")
    
    # Словарь для результатов
    results = {}
    
    # --- CatBoost ---
    print("\n" + "-"*70)
    print("🚀 CatBoost")
    print("-"*70)
    
    if HAS_CATBOOST:
        cb = CatBoostClassifier(
            iterations=1000,
            depth=6,
            learning_rate=0.05,
            l2_leaf_reg=3,
            random_seed=42,
            early_stopping_rounds=50,
            verbose=100
        )
        cb.fit(X_train, y_train, eval_set=(X_val, y_val))
        
        y_pred_cb = cb.predict(X_test)
        y_prob_cb = cb.predict_proba(X_test)[:, 1]
        
        acc_cb = accuracy_score(y_test, y_pred_cb)
        auc_cb = roc_auc_score(y_test, y_prob_cb)
        
        print(f"\n✅ Test Accuracy: {acc_cb:.4f} ({acc_cb*100:.2f}%)")
        print(f"✅ Test ROC-AUC:  {auc_cb:.4f}")
        
        results['CatBoost'] = {'accuracy': acc_cb, 'auc': auc_cb, 'model': cb, 'proba': y_prob_cb}
        
        # Feature importance
        print("\n📊 Важность признаков (CatBoost):")
        importance = cb.get_feature_importance()
        feat_imp = sorted(zip(features, importance), key=lambda x: -x[1])
        for name, imp in feat_imp[:15]:
            print(f"   {name:30s}: {imp:.2f}")
    
    # --- Random Forest ---
    print("\n" + "-"*70)
    print("🌲 Random Forest")
    print("-"*70)
    
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=20,
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X_train, y_train)
    
    y_pred_rf = rf.predict(X_test)
    y_prob_rf = rf.predict_proba(X_test)[:, 1]
    
    acc_rf = accuracy_score(y_test, y_pred_rf)
    auc_rf = roc_auc_score(y_test, y_prob_rf)
    
    print(f"✅ Test Accuracy: {acc_rf:.4f} ({acc_rf*100:.2f}%)")
    print(f"✅ Test ROC-AUC:  {auc_rf:.4f}")
    
    results['RandomForest'] = {'accuracy': acc_rf, 'auc': auc_rf, 'model': rf, 'proba': y_prob_rf}
    
    # --- Logistic Regression (baseline) ---
    print("\n" + "-"*70)
    print("📈 Logistic Regression (baseline)")
    print("-"*70)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    
    y_pred_lr = lr.predict(X_test_scaled)
    y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]
    
    acc_lr = accuracy_score(y_test, y_pred_lr)
    auc_lr = roc_auc_score(y_test, y_prob_lr)
    
    print(f"✅ Test Accuracy: {acc_lr:.4f} ({acc_lr*100:.2f}%)")
    print(f"✅ Test ROC-AUC:  {auc_lr:.4f}")
    
    results['LogisticRegression'] = {'accuracy': acc_lr, 'auc': auc_lr}
    
    # --- Ансамбль ---
    print("\n" + "-"*70)
    print("🎯 Ансамбль (усреднение вероятностей)")
    print("-"*70)
    
    if HAS_CATBOOST:
        y_prob_ensemble = (y_prob_cb + y_prob_rf) / 2
    else:
        y_prob_ensemble = y_prob_rf
    
    y_pred_ensemble = (y_prob_ensemble >= 0.5).astype(int)
    acc_ens = accuracy_score(y_test, y_pred_ensemble)
    auc_ens = roc_auc_score(y_test, y_prob_ensemble)
    
    print(f"✅ Test Accuracy: {acc_ens:.4f} ({acc_ens*100:.2f}%)")
    print(f"✅ Test ROC-AUC:  {auc_ens:.4f}")
    
    results['Ensemble'] = {'accuracy': acc_ens, 'auc': auc_ens, 'proba': y_prob_ensemble}
    
    # --- Анализ по порогу уверенности ---
    print("\n" + "="*70)
    print("📊 АНАЛИЗ ПО ПОРОГУ УВЕРЕННОСТИ")
    print("="*70)
    
    best_proba = y_prob_ensemble
    print(f"\n{'Порог':>10} | {'Accuracy':>10} | {'Покрытие':>10} | {'Матчей':>10}")
    print("-"*50)
    
    confidence_results = []
    for threshold in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        mask = (best_proba >= threshold) | (best_proba <= (1 - threshold))
        if mask.sum() > 0:
            acc_at_thresh = accuracy_score(y_test[mask], y_pred_ensemble[mask])
            coverage = mask.sum() / len(y_test)
            print(f"{threshold:>10.2f} | {acc_at_thresh:>10.4f} | {coverage:>10.2%} | {mask.sum():>10d}")
            confidence_results.append({
                'threshold': threshold,
                'accuracy': acc_at_thresh,
                'coverage': coverage,
                'count': mask.sum()
            })
    
    # --- Финальные результаты ---
    print("\n" + "="*70)
    print("📋 СВОДКА РЕЗУЛЬТАТОВ")
    print("="*70)
    
    print(f"\n{'Модель':25s} | {'Accuracy':>10} | {'ROC-AUC':>10}")
    print("-"*50)
    for name, res in sorted(results.items(), key=lambda x: -x[1]['accuracy']):
        print(f"{name:25s} | {res['accuracy']:>10.4f} | {res['auc']:>10.4f}")
    
    # Лучший результат
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\n🏆 Лучшая модель: {best_model[0]}")
    print(f"   Accuracy: {best_model[1]['accuracy']*100:.2f}%")
    print(f"   ROC-AUC: {best_model[1]['auc']:.4f}")
    
    # Вывод по confidence
    if confidence_results:
        best_conf = max(confidence_results, key=lambda x: x['accuracy'])
        print(f"\n🎯 При пороге уверенности {best_conf['threshold']:.2f}:")
        print(f"   Accuracy: {best_conf['accuracy']*100:.2f}%")
        print(f"   Покрытие: {best_conf['coverage']*100:.1f}% матчей ({best_conf['count']} шт)")
    
    return results


if __name__ == "__main__":
    results = train_and_evaluate()
