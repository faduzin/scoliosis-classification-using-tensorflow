import pandas as pd
from sklearn.preprocessing import StandardScaler


def class_remapping(data, column_name):
    try:
        data['class'] = (data['Scolio'] > 10).astype(int)
        return data
    except Exception as e:
        print(f"Error: {e}")
        return None


def scale_data(data):
    try:    
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        return data
    except Exception as e:
        print(f"Error: {e}")
        return None
    

def extract_individual_n(data):
    try:
        temp_list = []
        for name in data['Name']:
            temp_list.append(name.split('_')[1])
        data['individual_n'] = temp_list
        print('Individual number extracted successfully.')
        return data
    except Exception as e:
        print(f"Error: {e}")
        return None