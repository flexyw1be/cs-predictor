from data_loader import load_data, split_time_series_data
from features import feature_engineering
from ga_optimizer import run_genetic_optimization
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import numpy as np
import matplotlib

matplotlib.use('TkAgg')  # Принудительный запуск в отдельном окне
import matplotlib.pyplot as plt

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import matplotlib.pyplot as plt

from sklearn.metrics import log_loss, mean_squared_error


def calculate_extra_metrics(model, X_test, y_test):
    # 1. Получаем вероятности (нужны для Log Loss и MSE)
    # y_proba[:, 1] — это вероятность победы команды А (класса 1)
    y_proba = model.predict_proba(X_test)[:, 1]

    # 2. Считаем Log Loss (Кросс-энтропия)
    # Чем ниже это число, тем лучше "калибровка" твоих прогнозов
    loss = log_loss(y_test, y_proba)

    # 3. Считаем MSE
    # В классификации это значение показывает средний квадрат отклонения вероятности от факта
    mse = mean_squared_error(y_test, y_proba)

    print(f"\n--- Дополнительные метрики ---")
    print(f"Log Loss (Кросс-энтропия): {loss:.4f}")
    print(f"MSE (Среднеквадратичная ошибка): {mse:.4f}")

    # Интерпретация
    if loss < 0.6:
        print("Результат: Модель хорошо откалибрована (уверенность совпадает с реальностью).")
    else:
        print("Результат: Высокий Loss. Модель часто ошибается в моментах, когда она уверена.")


def calibrate_and_save_model(best_rf_model, X_train, y_train, X_test, y_test):
    print("\n--- Запуск калибровки модели ---")

    # В новых версиях sklearn мы передаем ensemble напрямую,
    # но указываем cv='prefit' правильно.
    # Если ошибка повторяется, попробуй обернуть в конструктор:
    calibrated_model = CalibratedClassifierCV(
        estimator=best_rf_model,
        cv=5,
        method='isotonic'
    )

    # ВАЖНО: При cv='prefit' метод .fit() все равно нужно вызвать,
    # чтобы калибровщик "увидел" распределение вероятностей на тестовых данных
    calibrated_model.fit(X_test, y_test)

    return calibrated_model


def run_model_diagnostic(model, X_test, y_test, features):
    print("\n" + "=" * 30)
    print("ДИАГНОСТИКА ОБУЧЕНИЯ МОДЕЛИ")
    print("=" * 30)

    # 1. Получаем предсказания
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # 2. Метрики точности
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
    print("\nДетальный отчет по классам:")
    print(classification_report(y_test, y_pred))

    # 3. Визуализация Матрицы Ошибок
    # Показывает, сколько раз мы ошиблись с победителем
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=['Win B', 'Win A'],
                yticklabels=['Win B', 'Win A'])
    plt.title('Матрица ошибок (Confusion Matrix)')
    plt.xlabel('Предсказано моделью')
    plt.ylabel('Реальный результат')

    # 4. Визуализация Важности Признаков (Feature Importance)
    # Показывает, на что модель реально смотрела при принятии решения
    plt.subplot(1, 2, 2)
    importances = model.feature_importances_
    indices = np.argsort(importances)[-10:]  # Топ-10 признаков

    plt.barh(range(len(indices)), importances[indices], align='center', color='skyblue')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.title('Топ-10 важных признаков (по версии RF)')
    plt.xlabel('Относительная важность')

    plt.tight_layout()
    plt.show()

    print("--- Диагностика завершена. Графики отображены. ---")


def save_final_dataset(X_train, X_test, y_train, y_test, filename="final_processed_dataset.csv"):
    print(f"--- Сохранение обработанного датасета в {filename} ---")

    # 1. Объединяем тренировочные и тестовые признаки
    X_full = pd.concat([X_train, X_test])

    # 2. Объединяем тренировочные и тестовые целевые метки (кто победил)
    y_full = pd.concat([y_train, y_test])

    # 3. Собираем всё в одну таблицу
    final_df = X_full.copy()
    final_df['target_winner'] = y_full

    # 4. Сохраняем в CSV
    final_df.to_csv(filename, index=False)
    print(f"Успешно! Сохранено {len(final_df)} строк и {len(final_df.columns)} колонок.")


def finalize_model(X_train, y_train, X_test, y_test, best_params):
    print("--- Запуск финального обучения ---")

    # 1. Создаем модель с параметрами, которые нашел наш Генетический Алгоритм
    # Распаковываем словарь параметров через **
    final_model = RandomForestClassifier(**best_params)

    # 2. Обучаем модель
    final_model.fit(X_train, y_train)

    # 3. Проверяем точность на тестовых данных (которые модель не видела)
    predictions = final_model.predict(X_test)
    acc = accuracy_score(y_test, predictions)

    print(f"Точность финальной модели: {acc:.2%}")
    print("\nДетальный отчет:")
    print(classification_report(y_test, predictions))

    # 4. Сохраняем модель и список признаков
    # Важно сохранить и саму модель, и названия колонок (features),
    # чтобы потом подавать данные в правильном порядке.
    model_data = {
        'model': final_model,
        'features': list(X_train.columns),
        'params': best_params
    }

    joblib.dump(model_data, 'cs2_predictor_model.joblib')
    print("--- Модель сохранена в файл cs2_predictor_model.joblib ---")


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

    best_params = best_rf_model.get_params()
    finalize_model(X_train, y_train, X_test, y_test, best_params)
    save_final_dataset(X_train, X_test, y_train, y_test)
    run_model_diagnostic(best_rf_model, X_test, y_test, X_train.columns)
    calibrated_rf = calibrate_and_save_model(best_rf_model, X_train, y_train, X_test, y_test)
    joblib.dump(calibrated_rf, 'cs2_predictor_model_calibrated.joblib')
    calculate_extra_metrics(best_rf_model, X_test, y_test)
    train_acc = best_rf_model.score(X_train, y_train)
    test_acc = best_rf_model.score(X_test, y_test)
    print(f"Точность на тренировке: {train_acc:.2f}")
    print(f"Точность на тесте: {test_acc:.2f}")
    importances = best_rf_model.feature_importances_
    feature_names = X_train.columns
    top_5 = sorted(zip(importances, feature_names), reverse=True)[:5]
    print("Топ-5 признаков, на которых учится модель:", top_5)


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def check_learning_quality(model, X_test, y_test):
    # 1. Считаем предсказания
    y_pred = model.predict(X_test)

    # 2. Строим матрицу ошибок
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Предсказано')
    plt.ylabel('Реально')
    plt.title('Матрица ошибок (Confusion Matrix)')
    plt.show()

    # 3. Выводим важность признаков
    importances = model.feature_importances_
    # (Здесь код для отрисовки столбчатой диаграммы признаков)


if __name__ == "__main__":
    main()
