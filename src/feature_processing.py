import pandas as pd
from src import config
from sklearn.preprocessing import LabelEncoder
import joblib


def feature_processing(data, mode):
    # Read the input data from a CSV file into a DataFrame
    df = pd.read_csv(data)

    # Feature Imputation - Fill missing values with appropriate defaults
    df.society.fillna('No society', inplace=True)
    df.bhk.fillna(0, inplace=True)
    df.bath.fillna(1, inplace=True)
    df.balcony.fillna(0, inplace=True)

    # Label encoding the categorical variables
    labelencoders = {}

    # Iterate through categorical columns and perform label encoding
    for c in ['area_type', 'availability', 'location', 'society']:
        labelencoders[c] = LabelEncoder()
        if mode == 'train':
            df[c] = labelencoders[c].fit_transform(df[c])
        else:
            labelencoders = joblib.load(config.MODELS_PATH+ 'feature_encoders.pkl')
            df[c] = labelencoders[c].transform(df[c])
    # Save the processed DataFrame to a CSV file
    df.to_csv(config.PROCESSED_TRAIN_DATA, index=False)

    # Save the Label Encoders for later use
    if mode == 'train':
        joblib.dump(labelencoders, config.MODELS_PATH + 'feature_encoders.pkl')

feature_processing(config.TRAIN_DATA)