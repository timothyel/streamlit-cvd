import pandas as pd
import numpy as np

to_scale = ['diastolic', 'systolic', 'height', 'weight', 'age']


def preprocess(df, scaler):
    # impute data with 0 value with median from training set
    df[to_scale] = scaler.transform(df[to_scale])

    #convert_cholesterol
    def categorize_cholesterol(value):
        value = float(value)
        if value < 200:
            return 1
        elif 200 <= value <= 239:
            return 2
        else:
            return 3
    def categorize_glucose(value):
        value = float(value)
        if value < 140:
            return 1
        elif 140 <= value <= 199:
            return 2
        else:
            return 3
        
    df['cholesterol'] = df['cholesterol'].apply(categorize_cholesterol)
    df['gluc'] = df['gluc'].apply(categorize_glucose)
    df['gender'] = df['gender'].apply(lambda x: 1 if x == 'Male' else 0)
    df['active'] = df['active'].apply(lambda x: 1 if x == 'Active' else 0)
    df['smoke'] = df['smoke'].apply(lambda x: 1 if x == 'Yes' else 0)
    df['alco'] = df['alco'].apply(lambda x: 1 if x == 'Yes' else 0)

    
    return df

