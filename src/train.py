from src import config
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd

def train(data):
    df = pd.read_csv(data)
    MODELS = {
        'linearregression': LinearRegression(),
        'randomforest': RandomForestRegressor(n_estimators=200)
    }

    X = df.iloc[:, :-1]
    y = df.price

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2, random_state=1)

    model = MODELS[config.MODEL]
    model.fit(X_train, y_train)

    y_predict = model.predict(X_test)
    print(config.MODEL, 'Accuracy: ', r2_score(y_test, y_predict))

train(config.PROCESSED_TRAIN_DATA)