import pandas as pd
import os
from features import compute_advanced_features


def prepare_datasets(input_path, output_dir='data/processed'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Загрузка данных из {input_path}...")
    df = pd.read_csv(input_path)

    print("Генерация признаков (Elo, H2H, Momentum)...")
    df = compute_advanced_features(df)

    # Сохраняем полный размеченный датасет
    full_path = os.path.join(output_dir, 'full_ml_dataset.csv')
    df.to_csv(full_path, index=False)

    # Разделяем на Train, Val, Test согласно колонке split_row
    # Это гарантирует, что мы используем те же сплиты, что были в задании
    train_df = df[df['split_row'] == 'train']
    val_df = df[df['split_row'] == 'val']
    test_df = df[df['split_row'] == 'test']

    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)

    print(f"Датасеты успешно созданы в папке: {output_dir}")
    print(f"Размер Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")


if __name__ == "__main__":
    # Укажи путь к своему исходному файлу
    RAW_DATA = 'data/map_picks_last6m_top50_ml_ready.csv'
    prepare_datasets(RAW_DATA)