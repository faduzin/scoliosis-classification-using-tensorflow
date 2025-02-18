import pandas as pd


def load_data(file_path, sep=','):
    try:
        data = pd.read_csv(file_path, sep=sep)
        print('Data loaded successfully.')
        return data
    except FileNotFoundError:
        print('File not found.')
        return None
    

def save_data(df, file_path):
    try: # Tenta salvar o arquivo
        df.to_csv(file_path, index=False) # Salva o dataframe
        print(f"Arquivo salvo em: {file_path}.") # Exibe mensagem de sucesso
    except: # Se houver erro
        raise("Falha ao salvar arquivo.") # Exibe mensagem de erro