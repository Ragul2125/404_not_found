import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🚀 INTER-CITY SERVICE DEMAND FORECASTING - LIGHTGBM SOLUTION")
print("="*70)

print("\n📊 Loading datasets...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

print(f"✓ Train shape: {train_df.shape}")
print(f"✓ Test shape: {test_df.shape}")
print(f"\nTrain columns: {train_df.columns.tolist()}")
print(f"Test columns: {test_df.columns.tolist()}")

print("\n📅 Processing dates...")
train_df['service_date'] = pd.to_datetime(train_df['service_date'], format='%d-%m-%Y')
test_df['service_date'] = pd.to_datetime(test_df['service_date'], format='%d-%m-%Y')

print(f"✓ Date range in train: {train_df['service_date'].min()} to {train_df['service_date'].max()}")
print(f"✓ Date range in test: {test_df['service_date'].min()} to {test_df['service_date'].max()}")

def create_advanced_features(df):
    df = df.copy()

    print("   🔧 Creating temporal features...")
    df['year'] = df['service_date'].dt.year
    df['month'] = df['service_date'].dt.month
    df['day'] = df['service_date'].dt.day
    df['dayofweek'] = df['service_date'].dt.dayofweek
    df['dayofyear'] = df['service_date'].dt.dayofyear
    df['quarter'] = df['service_date'].dt.quarter
    df['weekofyear'] = df['service_date'].dt.isocalendar().week.astype(int)
    df['days_in_month'] = df['service_date'].dt.days_in_month

    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['is_monday'] = (df['dayofweek'] == 0).astype(int)
    df['is_friday'] = (df['dayofweek'] == 4).astype(int)
    df['is_month_start'] = df['service_date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['service_date'].dt.is_month_end.astype(int)
    df['is_quarter_start'] = df['service_date'].dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = df['service_date'].dt.is_quarter_end.astype(int)
    df['is_year_start'] = (df['month'] == 1).astype(int)
    df['is_year_end'] = (df['month'] == 12).astype(int)

    print("   🔧 Creating cyclical encodings...")
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    df['dayofyear_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
    df['dayofyear_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365)
    df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
    df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)

    print("   🔧 Creating route features...")
    df['route'] = df['origin_hub_id'].astype(str) + '_to_' + df['destination_hub_id'].astype(str)
    df['is_same_hub'] = (df['origin_hub_id'] == df['destination_hub_id']).astype(int)

    df['origin_frequency'] = df.groupby('origin_hub_id')['origin_hub_id'].transform('count')
    df['destination_frequency'] = df.groupby('destination_hub_id')['destination_hub_id'].transform('count')
    df['route_frequency'] = df.groupby('route')['route'].transform('count')

    df['origin_hub_encoded'] = df['origin_hub_id'].astype('category').cat.codes
    df['destination_hub_encoded'] = df['destination_hub_id'].astype('category').cat.codes
    df['route_encoded'] = df['route'].astype('category').cat.codes

    df['hub_sum'] = df['origin_hub_encoded'] + df['destination_hub_encoded']
    df['hub_diff'] = df['origin_hub_encoded'] - df['destination_hub_encoded']
    df['hub_product'] = df['origin_hub_encoded'] * df['destination_hub_encoded']

    print("   🔧 Creating aggregated statistics...")
    df['month_dayofweek'] = df['month'].astype(str) + '_' + df['dayofweek'].astype(str)
    df['quarter_dayofweek'] = df['quarter'].astype(str) + '_' + df['dayofweek'].astype(str)
    df['month_day'] = df['month'].astype(str) + '_' + df['day'].astype(str)

    df['route_month'] = df['route'] + '_' + df['month'].astype(str)
    df['route_dayofweek'] = df['route'] + '_' + df['dayofweek'].astype(str)
    df['route_quarter'] = df['route'] + '_' + df['quarter'].astype(str)

    df['origin_month'] = df['origin_hub_id'].astype(str) + '_' + df['month'].astype(str)
    df['origin_dayofweek'] = df['origin_hub_id'].astype(str) + '_' + df['dayofweek'].astype(str)
    df['dest_month'] = df['destination_hub_id'].astype(str) + '_' + df['month'].astype(str)
    df['dest_dayofweek'] = df['destination_hub_id'].astype(str) + '_' + df['dayofweek'].astype(str)

    categorical_cols = [
        'month_dayofweek', 'quarter_dayofweek', 'month_day',
        'route_month', 'route_dayofweek', 'route_quarter',
        'origin_month', 'origin_dayofweek', 'dest_month', 'dest_dayofweek'
    ]

    for col in categorical_cols:
        df[f'{col}_encoded'] = df[col].astype('category').cat.codes

    return df

print("\n🔧 Creating features for training data...")
train_features = create_advanced_features(train_df)

print("\n🔧 Creating features for test data...")
test_features = create_advanced_features(test_df)

print("\n🎯 Creating target encoding features...")

def add_target_encoding(train_df, test_df, target_col='final_service_units'):
    train_df = train_df.copy()
    test_df = test_df.copy()

    encoding_cols = [
        'route', 'origin_hub_id', 'destination_hub_id',
        'month', 'dayofweek', 'quarter', 'weekofyear',
        'route_month', 'route_dayofweek', 'origin_month', 'dest_month'
    ]

    for col in encoding_cols:
        target_mean = train_df.groupby(col)[target_col].agg(['mean', 'std', 'count']).reset_index()
        target_mean.columns = [col, f'{col}_target_mean', f'{col}_target_std', f'{col}_target_count']

        train_df = train_df.merge(target_mean, on=col, how='left')
        test_df = test_df.merge(target_mean, on=col, how='left')

        train_df[f'{col}_target_mean'].fillna(train_df[target_col].mean(), inplace=True)
        train_df[f'{col}_target_std'].fillna(0, inplace=True)
        train_df[f'{col}_target_count'].fillna(0, inplace=True)

        test_df[f'{col}_target_mean'].fillna(train_df[target_col].mean(), inplace=True)
        test_df[f'{col}_target_std'].fillna(0, inplace=True)
        test_df[f'{col}_target_count'].fillna(0, inplace=True)

    return train_df, test_df

train_features, test_features = add_target_encoding(train_features, test_features)

print("\n📦 Preparing training data...")

exclude_cols = [
    'service_date', 'final_service_units', 'route',
    'origin_hub_id', 'destination_hub_id',
    'month_dayofweek', 'quarter_dayofweek', 'month_day',
    'route_month', 'route_dayofweek', 'route_quarter',
    'origin_month', 'origin_dayofweek', 'dest_month', 'dest_dayofweek'
]

if 'service_key' in train_features.columns:
    exclude_cols.append('service_key')

feature_cols = [col for col in train_features.columns if col not in exclude_cols]

X = train_features[feature_cols]
y = train_features['final_service_units']

print(f"✓ Number of features: {len(feature_cols)}")
print(f"✓ Training samples: {len(X)}")
print(f"✓ Target statistics:")
print(f"   Mean: {y.mean():.2f}")
print(f"   Std: {y.std():.2f}")
print(f"   Min: {y.min():.2f}")
print(f"   Max: {y.max():.2f}")