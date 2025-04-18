import requests
import pandas as pd
from datetime import datetime
import time
from dotenv import load_dotenv
load_dotenv()
import os

API_KEY = os.environ.get('API_KEY')
BASE_URL = os.environ.get('BASE_URL')

def get_crypto_data_paged(fsym="BTC", tsym="USD", interval="histoday", start_date_str='2020-01-01'):
    all_data = []
    toTs = int(time.time())  # начинаем с "сейчас"
    limit = 2000
    start_ts_limit = int(datetime.strptime(start_date_str, "%Y-%m-%d").timestamp())  # целевая нижняя граница

    while True:
        url = f"{BASE_URL}{interval}"
        params = {
            "fsym": fsym,
            "tsym": tsym,
            "limit": limit,
            "toTs": toTs,
            "api_key": API_KEY
        }

        response = requests.get(url, params=params)
        data = response.json()

        if data["Response"] != "Success":
            print("Ошибка:", data.get("Message", "Unknown error"))
            break

        chunk = data["Data"]["Data"]
        if not chunk:
            break

        all_data.extend(chunk)

        # Обновляем toTs на самый ранний момент в текущем чанке
        min_ts = min(item["time"] for item in chunk)
        if min_ts <= start_ts_limit:
            break
        toTs = min_ts - 1  # чтобы избежать дубликата

        print(f"Загружено {len(all_data)} точек, до {datetime.utcfromtimestamp(min_ts).strftime('%Y-%m-%d')}")

        time.sleep(1)  # Чтобы не попасть под лимит API

    # Преобразуем в DataFrame
    btc_price_2020 = pd.DataFrame(all_data).drop_duplicates(subset="time")
    btc_price_2020["time"] = pd.to_datetime(btc_price_2020["time"], unit="s")
    return btc_price_2020.sort_values("time").reset_index(drop=True)    

# Пример: 5 лет дневных данных (365 * 5 = 1825)
btc_price = get_crypto_data_paged(interval="histohour")


# Пример: 2000 часов
btc_price_2020=pd.read_csv('BTC_price.csv')
btc_price_2020["vwap"] = btc_price_2020["volumeto"] / btc_price_2020["volumefrom"]
btc_price_2020=btc_price_2020.drop(columns=['conversionType','conversionSymbol'])
btc_price_2020['EMA_5'] = btc_price_2020['vwap'].ewm(span=5, adjust=False).mean()
btc_price_2020['EMA_30'] = btc_price_2020['vwap'].ewm(span=30, adjust=False).mean()
btc_price_2020['EMA_100'] = btc_price_2020['vwap'].ewm(span=100, adjust=False).mean()
ema12 = btc_price_2020['vwap'].ewm(span=12, adjust=False).mean()
ema26 = btc_price_2020['vwap'].ewm(span=26, adjust=False).mean()
btc_price_2020['macd'] = ema12 - ema26
btc_price_2020['macd_signal'] = btc_price_2020['macd'].ewm(span=9).mean()
btc_price_2020['macd_diff'] = btc_price_2020['macd'] - btc_price_2020['macd_signal']
window = 30  
eps = 1e-9
#macd,macd_z
rolling_mean = btc_price_2020['macd'].rolling(window).mean()
rolling_std = btc_price_2020['macd'].rolling(window).std()
btc_price_2020['macd_z'] = (btc_price_2020['macd'] - rolling_mean) / (rolling_std + eps)

delta = btc_price_2020['vwap'].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

#дивергенция
price_slope = (btc_price_2020['vwap'] - btc_price_2020['vwap'].shift(5)) / 5
macd_slope = (btc_price_2020['macd'] - btc_price_2020['macd'].shift(5)) / 5

# Фича: расхождение направлений
btc_price_2020['macd_price_slope_diff'] = price_slope - macd_slope
btc_price_2020['divergence_strength'] = btc_price_2020['macd_price_slope_diff'].rolling(window=3).sum()

#RSI
vwap_gain = gain.rolling(window=14).mean()
vwap_loss = loss.rolling(window=14).mean()
rs = vwap_gain / vwap_loss
btc_price_2020['rsi'] = 100 - (100 / (1 + rs))

#Bolinger Bands
window = 20
btc_price_2020['bb_ma'] = btc_price_2020['vwap'].rolling(window).mean()
btc_price_2020['bb_std'] = btc_price_2020['vwap'].rolling(window).std()
btc_price_2020['bb_upper'] = btc_price_2020['bb_ma'] + 2 * btc_price_2020['bb_std']
btc_price_2020['bb_lower'] = btc_price_2020['bb_ma'] - 2 * btc_price_2020['bb_std']
#рассчет расстояния до нижней линии; рассчет расстояния до верхней линиии
btc_price_2020['bb_dist_upper'] = (btc_price_2020['bb_upper'] - btc_price_2020['vwap'])/btc_price_2020['vwap']
btc_price_2020['bb_dist_lower'] = (btc_price_2020['vwap'] - btc_price_2020['bb_lower'])/btc_price_2020['vwap']
btc_price_2020['bb_pos'] = (btc_price_2020['vwap'] - btc_price_2020['bb_lower']) / (btc_price_2020['bb_upper'] - btc_price_2020['bb_lower'])
btc_price_2020['bb_break_upper'] = btc_price_2020.apply(lambda row: 1 if row['vwap'] > row['bb_upper'] else 0, axis=1)
btc_price_2020['bb_break_lower'] = btc_price_2020.apply(lambda row: 1 if row['vwap'] < row['bb_lower'] else 0, axis=1)

accumulated = []
acc = 0
#накопление для линий болинджера
for val in btc_price_2020['bb_break_lower']:
    if val == 1:
        acc += 1
        accumulated.append(acc)
    elif val!=1 and acc!=0:
        acc -=1
        accumulated.append(acc)
    else:
        acc = 0
        accumulated.append(acc)

btc_price_2020['bb_break_lower_cumulative'] = accumulated

accumulated1 = []
acc1 = 0

for val in btc_price_2020['bb_break_upper']:
    if val == 1:
        acc1 += 1
        accumulated1.append(acc1)
    elif val!=1 and acc1!=0:
        acc1 -=1
        accumulated1.append(acc1)
    else:
        acc1=0
        accumulated1.append(acc1)

btc_price_2020['bb_break_upper_cumulative'] = accumulated1

# btc_price_2020=btc_price_2020.sort_values(by='time')
# === OBV (On-Balance volumefrom) ===
obv = [0]
for i in range(1, len(btc_price_2020)):
    if btc_price_2020.loc[i, 'close'] > btc_price_2020.loc[i - 1, 'close']:
        obv.append(obv[-1] + btc_price_2020.loc[i, 'volumefrom'])
    elif btc_price_2020.loc[i, 'close'] < btc_price_2020.loc[i - 1, 'close']:
        obv.append(obv[-1] - btc_price_2020.loc[i, 'volumefrom'])
    else:
        obv.append(obv[-1])
btc_price_2020['OBV'] = obv

# === Accumulation/Distribution Line (A/D) ===
mf_multiplier = ((btc_price_2020['close'] - btc_price_2020['low']) - (btc_price_2020['high'] - btc_price_2020['close'])) / (btc_price_2020['high'] - btc_price_2020['low'] + 1e-9)
mf_volumefrom = mf_multiplier * btc_price_2020['volumefrom']
btc_price_2020['AD'] = mf_volumefrom.cumsum()

# === Chaikin Money Flow (CMF) ===
period_cmf = 20
mfv_cmf = mf_multiplier * btc_price_2020['volumefrom']
cmf = mfv_cmf.rolling(period_cmf).sum() / btc_price_2020['volumefrom'].rolling(period_cmf).sum()
btc_price_2020['CMF'] = cmf

# === Money Flow Index (MFI) ===
period_mfi = 14
typical_price = (btc_price_2020['high'] + btc_price_2020['low'] + btc_price_2020['close']) / 3
money_flow = typical_price * btc_price_2020['volumefrom']
pos_flow = []
neg_flow = []
for i in range(1, len(typical_price)):
    if typical_price[i] > typical_price[i - 1]:
        pos_flow.append(money_flow[i])
        neg_flow.append(0)
    elif typical_price[i] < typical_price[i - 1]:
        pos_flow.append(0)
        neg_flow.append(money_flow[i])
    else:
        pos_flow.append(0)
        neg_flow.append(0)

pos_flow = pd.Series(pos_flow).rolling(period_mfi).sum()
neg_flow = pd.Series(neg_flow).rolling(period_mfi).sum()
mfi = 100 - (100 / (1 + (pos_flow / (neg_flow + 1e-9))))
btc_price_2020['MFI'] = mfi.reindex(btc_price_2020.index[1:], method='bfill')  # сдвиг из-за i=1

# === Force Index ===
period_force = 13
btc_price_2020['ForceIndex'] = (btc_price_2020['close'] - btc_price_2020['close'].shift(1)) * btc_price_2020['volumefrom']
btc_price_2020['ForceIndex'] = btc_price_2020['ForceIndex'].rolling(period_force).mean()
btc_price_2020['EMA_5']=btc_price_2020['EMA_5'].pct_change()
btc_price_2020['EMA_30']=btc_price_2020['EMA_30'].pct_change()
btc_price_2020['EMA_100']=btc_price_2020['EMA_100'].pct_change()
btc_price_2020=btc_price_2020.dropna()
btc_price_2020['return']=btc_price_2020['vwap'].pct_change()

btc_price_2020.reset_index(drop=True, inplace=True)
btc_price_2020=btc_price_2020[100:]
btc_price_2020["return"]=btc_price_2020["return"].shift(-1)
btc_price_2022=btc_price_2020[10000:]
btc_price_2020_c=btc_price_2020.drop(columns=['time', 'high', 'low', 'open', 'volumefrom', 'volumeto', 'close','vwap'])
from scipy import stats
import numpy as np

def IQR(df,column):
    Q1 = np.percentile(df[column], 25)
    Q3 = np.percentile(df[column], 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return(f"Выбросов по IQR в {column} : {len(outliers)}")
def normal_test(df,column):
   
    ks_test = stats.kstest(df[column], 'norm', args=(np.mean(df[column]), np.std(df[column])))
    skewness = df[column].skew()
    result = stats.anderson(df[column], dist='norm')
    # print("Statistic:", result.statistic)
    # print("Critical Values:", result.critical_values)
    # print("Significance Levels:", result.significance_level)
    print(f"Kolmogorov-Smirnov {column} test: D={ks_test.statistic}, p-value={ks_test.pvalue}")
    stat = result.statistic
    crit = result.critical_values[2]  # для уровня 5%
    if stat < crit:
        print("Распределение похоже на нормальное (на уровне 5%)")
    else:
        print('Распределение не нормальное (на уровне 5%), по тесту Андерсона')
    print(f"Skewness {column} :", skewness)
    print (IQR(df,column))
    
for col in btc_price_2020_c.columns:
    normal_test(btc_price_2020_c,col)
    import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from scipy.stats import skew

def count_outliers_iqr(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return ((series < lower) | (series > upper)).sum()

def normalize_features(df, feature_cols=None,exclude='return'):
    df_transformed = pd.DataFrame(index=df.index)
    summary = []

    if feature_cols is None:
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in feature_cols if col != exclude]
    for col in feature_cols:
        series = df[col].dropna()
        if series.nunique() < 2:
            continue  # skip constant or near-constant columns

        s = skew(series)
        outliers = count_outliers_iqr(series)
        method = ""

        # Группа 1: нормальное распределение
        if abs(s) < 0.3 and outliers < 100:
            scaler = StandardScaler()
            df_transformed[col] = scaler.fit_transform(df[[col]])
            method = "StandardScaler"

        # Группа 2: умеренная скошенность или выбросы
        elif abs(s) < 10 and outliers < 1000:
            scaler = RobustScaler()
            df_transformed[col] = scaler.fit_transform(df[[col]])
            method = "RobustScaler"

        # Группа 3: сильная скошенность / выбросы
        else:
            qt = QuantileTransformer(output_distribution='normal', random_state=42)
            df_transformed[col] = qt.fit_transform(df[[col]])
            method = "QuantileTransformer"
        df_transformed[exclude]=df[exclude]
        summary.append({
            "feature": col,
            "skewness": round(s, 3),
            "outliers": outliers,
            "method": method
        })

    summary_df = pd.DataFrame(summary)
    return df_transformed

btc_price_2020_n=normalize_features(btc_price_2020_c)
btc_price_2020_n=btc_price_2020_n.dropna()
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
def VIF(df):
    df1 = df.copy()  # Копируем, чтобы не изменять оригинал

# Удаляем нечисловые колонки (например, дату)
    df1 = df1.select_dtypes(include=[np.number])

    # Добавляем константу
    X = sm.add_constant(df1)

    # Рассчитываем VIF
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]


    print(vif_data)
VIF(btc_price_2020_n)
btc_price_2020_v=btc_price_2020_n.drop(columns=['EMA_30','bb_std','bb_ma','bb_upper','bb_lower','bb_dist_upper','bb_dist_lower','macd'])
VIF(btc_price_2020_v)
btc_price_2020_5=btc_price_2020_v.copy()
btc_price_2020_5["return"] = pd.qcut(btc_price_2020_5["return"], q=5, labels=False)
import seaborn as sns
from matplotlib import pyplot as plt

correlation_matrix = btc_price_2020_5.corr()

# Настройка графика  
plt.figure(figsize=(12, 10))  
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, square=True)  

# Показать график  
plt.title('Корреляционная матрица')  
plt.show()
from xgboost import XGBClassifier
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, KFold
)
def xgb_best_per(df,column):
    X=df.drop([column],axis=1)
    y=df[column]
    unique_values = list(set(y))
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, 
                                                            test_size=0.2, 
                                                            random_state=42,
                                                            stratify=y)
    model = XGBClassifier(eval_metric='logloss')
    model.fit(X_train, y_train)

    # Получаем важности
    importances = model.feature_importances_

    # Оборачиваем в DataFrame
    importance_df = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    # Берем топ-N (например, 10)
    top_features = importance_df["Feature"].head(10).tolist()

    # Обучаем модель на этих признаках
    X_train_top = X_train[top_features]
    X_test_top = X_test[top_features]

    model_top = XGBClassifier(eval_metric='logloss')
    model_top.fit(X_train_top, y_train)
    y_pred = model_top.predict(X_test_top)
    return importance_df
xgb_best_per(btc_price_2020_5,'return')
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.pipeline import Pipeline
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import MinMaxScaler

def classification_ml1(df, column, n_features_to_select=10):
    scaler = MinMaxScaler()
    
    X = df.drop([column], axis=1)
    
    y = df[column]
    unique_values = list(set(y))

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train_full, X_test_full, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y
    )

    models_and_params = {
        "LogisticRegression": {
            "model": LogisticRegression(max_iter=4000),
            "params": {
                "C": [0.1, 1, 10],
                "solver": ["liblinear"]
            }
        },
        "RandomForest": {
            "model": RandomForestClassifier(),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [None, 5, 10]
            }
        },
        "XGBoost": {
            "model": XGBClassifier(eval_metric='logloss'),
            "params": {
                "n_estimators": [50, 100],
                "max_depth": [3, 6],
                "learning_rate": [0.1, 0.2]
            }
        },
        "KNeighbors": {
            "model": KNeighborsClassifier(),
            "params": {
                "n_neighbors": [3, 5, 7],
                "weights": ["uniform", "distance"]
            }
        },
        "SVM": {
            "model": SVC(),
            "params": {
                "C": [0.1, 1, 10],
                "kernel": ["linear", "rbf"]
            }
        },
        "DecisionTree": {
            "model": DecisionTreeClassifier(),
            "params": {
                "max_depth": [None, 5, 10],
                "min_samples_split": [2, 5]
            }
        },
        "CatBoost": {
            "model": CatBoostClassifier(verbose=0, train_dir="catboost_info"),
            "params": {
                "iterations": [100, 200],
                "learning_rate": [0.05, 0.1],
                "depth": [4, 6]
            }
        }
    }

    results = []

    best_f1 = 0
    best_model_name = ""
    best_y_pred = None
    best_y_test = None

    for name, mp in models_and_params.items():
        print(f"\n=== Обработка модели: {name} ===")

        # Копии X
        X_train = X_train_full.copy()
        X_test = X_test_full.copy()

        # Применяем RFE
        
        try:
            print(f">> Выполняется RFE (выбор {n_features_to_select} признаков)...")
            rfe_estimator = mp["model"]
            rfe_selector = RFE(estimator=rfe_estimator, n_features_to_select=n_features_to_select, step=1)
            rfe_selector = rfe_selector.fit(X_train, y_train)

            selected_features = X_train.columns[rfe_selector.support_]
            print(f">> Отобрано признаков: {list(selected_features)}")

            # Обновляем данные
            X_train = X_train[selected_features]
            X_test = X_test[selected_features]
        except Exception as e:
            print(f">> Ошибка при RFE: {e}")
            print(">> Используем все признаки.")

        # Grid Search
        grid = GridSearchCV(mp["model"], mp["params"], cv=5, n_jobs=-1, scoring="f1_macro")
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)

        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        f1 = report["macro avg"]["f1-score"]

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_y_pred = y_pred
            best_y_test = y_test

        results.append({
            "Set": unique_values,
            "Model": name,
            "Accuracy": report["accuracy"],
            "Precision": report["macro avg"]["precision"],
            "Recall": report["macro avg"]["recall"],
            "F1-score": f1
        })
        
        model_result=({
            'Model': name,
            "Set": unique_values,
            "Model": name,
            "Accuracy": report["accuracy"],
            "Precision": report["macro avg"]["precision"],
            "Recall": report["macro avg"]["recall"],
            "F1-score": f1   
        })
        
        model_result_df=pd.DataFrame(model_result)
        print(model_result)
        
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix: {name}")
        plt.show()

    results_df = pd.DataFrame(results)

    # Confusion matrix для лучшей модели
    print(f"\n=== Лучшая модель: {best_model_name} (F1-score = {best_f1:.4f}) ===")
    cm = confusion_matrix(best_y_test, best_y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix: {best_model_name}")
    plt.show()

    return results_df

classification_ml1(btc_price_2020_5, 'return', n_features_to_select=12)