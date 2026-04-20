"""
startup.py — запускается перед main.py на Railway.
Скачивает сырые данные с OWID и пересобирает combined_features.csv.
Если файл уже свежий (меньше 24 часов) — пропускает скачивание.
"""

import pandas as pd
import os
from datetime import datetime, timedelta

os.makedirs("data/raw",       exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
os.makedirs("data/features",  exist_ok=True)

COMBINED_PATH = "data/features/combined_features.csv"
MAX_AGE_HOURS = 24  # пересобираем не чаще раза в сутки


def is_fresh(path, max_age_hours=MAX_AGE_HOURS):
    if not os.path.exists(path):
        return False
    age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(path))
    return age < timedelta(hours=max_age_hours)


# ── Шаг 1: скачиваем сырые данные ────────────────────────────────────────

DATASETS = {
    "covid_cases": {
        "url": "https://catalog.ourworldindata.org/garden/covid/latest/cases_deaths/cases_deaths.csv",
        "columns": [
            "country", "date", "new_cases", "total_cases", "new_deaths",
            "total_deaths", "new_cases_7_day_avg_right", "new_deaths_7_day_avg_right",
            "new_cases_per_million", "new_deaths_per_million",
            "total_cases_per_million", "total_deaths_per_million",
            "weekly_cases", "weekly_deaths",
            "weekly_pct_growth_cases", "weekly_pct_growth_deaths", "cfr",
        ]
    },
    "mpox": {
        "url": "https://catalog.ourworldindata.org/garden/who/latest/monkeypox/monkeypox.csv",
        "columns": [
            "country", "date", "new_cases", "total_cases", "new_deaths",
            "total_deaths", "new_cases_smoothed", "new_deaths_smoothed",
            "new_cases_per_million", "new_cases_smoothed_per_million",
            "new_deaths_per_million", "total_cases_per_million", "total_deaths_per_million",
        ]
    },
    "population": {
        "url": "https://raw.githubusercontent.com/datasets/population/master/data/population.csv",
        "columns": None,  # обрабатываем отдельно
    }
}


def download_dataset(name, url, columns):
    processed_path = f"data/processed/{name}.csv"

    if is_fresh(processed_path):
        print(f"  ⚡ {name}: свежий, пропускаем")
        return pd.read_csv(processed_path)

    print(f"  ⬇️  {name}: скачиваю...")
    try:
        df = pd.read_csv(url)

        if name == "population":
            df = df.sort_values('Year').groupby('Country Name').last().reset_index()
            df = df[['Country Name', 'Value']].rename(columns={
                'Country Name': 'country', 'Value': 'population'
            })
        elif columns:
            available = [c for c in columns if c in df.columns]
            df = df[available]

        df.to_csv(processed_path, index=False)
        print(f"  ✅ {name}: {len(df):,} строк")
        return df

    except Exception as e:
        print(f"  ❌ {name}: {e}")
        if os.path.exists(processed_path):
            print(f"     Использую старый файл")
            return pd.read_csv(processed_path)
        return None


# ── Шаг 2: feature engineering (логика из eda.ipynb) ─────────────────────

def add_features(df, disease_name):
    df = df.copy().sort_values(['country', 'date'])

    df['year']        = df['date'].dt.year
    df['month']       = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['quarter']     = df['date'].dt.quarter
    df['season']      = df['month'].map({
        12: 'winter', 1: 'winter',  2: 'winter',
        3:  'spring', 4: 'spring',  5: 'spring',
        6:  'summer', 7: 'summer',  8: 'summer',
        9:  'autumn', 10: 'autumn', 11: 'autumn',
    })

    if 'population' in df.columns:
        df['new_cases_per_100k']  = df['new_cases']  / (df['population'] / 100_000)
        df['new_deaths_per_100k'] = df['new_deaths'] / (df['population'] / 100_000)
    else:
        df['new_cases_per_100k']  = df.get('new_cases_per_million', 0) / 10
        df['new_deaths_per_100k'] = df.get('new_deaths_per_million', 0) / 10

    smooth_col = 'new_cases_7_day_avg_right' if 'new_cases_7_day_avg_right' in df.columns \
                 else 'new_cases_smoothed'

    for lag in [7, 14, 21]:
        df[f'cases_lag_{lag}d'] = df.groupby('country')[smooth_col].shift(lag)

    df['cases_ma_30d'] = df.groupby('country')[smooth_col].transform(
        lambda x: x.rolling(30, min_periods=1).mean()
    )

    # growth_ratio со сглаживанием (не обнуляется в выходные)
    df['growth_ratio_7d'] = (
        df[smooth_col] / (df['cases_lag_7d'] + 1)
    ).clip(0, 10)

    df['disease'] = disease_name
    return df



def build():
    if is_fresh(COMBINED_PATH):
        print(f"✅ combined_features.csv свежий — пересборка не нужна")
        return

    print("🔧 Собираю данные...\n")

    # Скачиваем
    covid = download_dataset("covid_cases", DATASETS["covid_cases"]["url"], DATASETS["covid_cases"]["columns"])
    mpox  = download_dataset("mpox",        DATASETS["mpox"]["url"],        DATASETS["mpox"]["columns"])
    pop   = download_dataset("population",  DATASETS["population"]["url"],  None)

    if covid is None or mpox is None or pop is None:
        raise RuntimeError("Не удалось загрузить данные — API не запустится")

    # Парсим даты
    for df in [covid, mpox]:
        df['date'] = pd.to_datetime(df['date'])

    # Мёрджим population
    covid = covid.merge(pop, on='country', how='left')
    mpox  = mpox.merge(pop,  on='country', how='left')

    # Feature engineering
    print("\n🔧 Feature engineering...")
    covid_feat = add_features(covid, 'covid')
    mpox_feat  = add_features(mpox,  'mpox')

    # Объединяем
    combined = pd.concat([covid_feat, mpox_feat], ignore_index=True)
    combined = combined.sort_values(['disease', 'country', 'date']).reset_index(drop=True)

    # Сохраняем
    combined.to_csv(COMBINED_PATH, index=False)
    size_mb = os.path.getsize(COMBINED_PATH) / (1024 * 1024)
    print(f"\n✅ Готово: {len(combined):,} строк | {size_mb:.1f} MB")
    print(f"   Болезни: {combined['disease'].value_counts().to_dict()}")
    print(f"   Страны:  {combined['country'].nunique()}")
    print(f"   Период:  {combined['date'].min().date()} → {combined['date'].max().date()}")


if __name__ == "__main__":
    build()