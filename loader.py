import os
import requests
import numpy as np
import pandas as pd

DATA_PATH = 'data/MMM_MMM_DAE.csv'

def download_data(url = DATA_PATH, force_download=False, ):
    # Utility function to donwload data if it is not in disk
    data_path = os.path.join('data', os.path.basename(url.split('?')[0]))
    if not os.path.exists(data_path) or force_download:
        # ensure data dir is created
        os.makedirs('data', exist_ok=True)
        # request data from url
        response = requests.get(url, allow_redirects=True)
        # save file
        with open(data_path, "w") as f:
            # Note the content of the response is in binary form: 
            # it needs to be decoded.
            # The response object also contains info about the encoding format
            # which we use as argument for the decoding
            f.write(response.content.decode(response.apparent_encoding))

    return data_path


def load_formatted_data(data_fname:str) -> pd.DataFrame:
    """ One function to read csv into a dataframe with appropriate types/formats.
        Note: read only pertinent columns, ignore the others.
    """
    # Precision on columns to read. 
    column_names = ['nom','acc','acc_etg','acc_lib','adr_num','adr_voie','appartenan','date_insta','dermnt','disp_compl','disp_h','disp_j','dtpr_bat','dtpr_lcad','dtpr_lcped','lat_coor1','lc_ped','long_coor1','num_serie','ref','tel1']
    # Precisions on column's dtypes. 
    column_types = {'nom':'object',
                    'acc':'object',
                    'acc_etg':'int64',
                    'acc_lib':'bool',
                    'appartenan':'object',
                    'date_insta':'object',
                    'dermnt':'object',
                    'disp_compl':'object',
                    'disp_h':'object',
                    'disp_j':'object',
                    'dtpr_bat':'object',
                    'dtpr_lcad':'object',
                    'dtpr_lcped':'object',
                    'lat_coor1':'float64',
                    'lc_ped':'bool',
                    'long_coor1':'float64',
                    'num_serie':'object',
                    'ref':'object',
                    'tel1':'object',
                    } 
    # We define dates as object because it must be readable by human, we don't use it as a datetime64.  
    #df.replace(" ", pd.NA, inplace=True)
    df = pd.read_csv(
        data_fname,
        usecols=column_names,
        #dtype=column_types,
        encoding='utf-8'
        )
    df['acc_etg'].replace(0, "RDC", inplace=True)
    df['acc_etg'].rename('Etage')
    df['acc'].rename('Interieur')
    df['acc_lib'].rename('Acces_libre')
    df['appartenan'].rename('Propriétaire')
    df['date_insta'].rename('Date_instal')
    df['dermnt'].rename('Derniere_maintenance')
    df['disp_compl'].rename('Fermeture_occasionel') #cleaner les datas
    df['disp_h'].rename('Heure_disp')
    df['disp_j'].rename('Jour_disp')
    df['dtpr_bat'].rename('Date_péremption_batterie')
    df['dtpr_lcad'].rename('Date_péremption_elec_adulte')
    df['dtpr_lcped'].rename('Date_péremption_elec_pédiatrique')
    df['lc_ped'].rename('Electrode_enfant')
    df['lat_coor1'].rename('Latitude')
    df['long_coor1'].rename('Longitude')
    df['num_serie'].rename('Num_serie')
    df['ref'].rename('Référent')
    df['tel1'].rename('Tel')
    return df



# once they are all done, call them in the general sanitizing function
def sanitize_data(df:pd.DataFrame) -> pd.DataFrame:
    """ One function to do all sanitizing"""
        # Remove duplicate rows
    df.drop_duplicates(inplace=True)
        # Replace missing values based on data type
    for col in df.columns:
        if df[col].dtype == 'bool':
            df[col].fillna(False, inplace=True)
        else:
            df[col].fillna(pd.NA, inplace=True)
    
        # Replace "-" with "NA"
    df.replace("-", pd.NA, inplace=True)
        # Convert string columns to lower case
    df = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)

    specific_column = 'Tel'
    if specific_column in df.columns:
        df[specific_column] = df[specific_column].str.replace('+', '')
    return df

# Define a framing function
def frame_data(df:pd.DataFrame) -> pd.DataFrame:
    """ One function all framing (column renaming, column merge)"""
    df.rename(columns={'old': 'new'}, inplace=True)

    # Merge columns adr_num and adr_voie
    df['adr_num'].replace(pd.NA,'', inplace=True)
    df['address'] = df['adr_num'].astype(str) + ' ' + df['adr_voie']
    df.drop(['adr_num', 'adr_voie'], axis=1, inplace=True)
    return df


# once they are all done, call them in the general clean loading function
def load_clean_data(data_path:str=DATA_PATH)-> pd.DataFrame:
    """one function to run it all and return a clean dataframe"""
    df = (load_formatted_data(data_path)
          .pipe(sanitize_data)
          .pipe(frame_data)
    )
    #print(df.to_string(index=False))
    return df


# if the module is called, run the main loading function
if __name__ == '__main__':
    load_clean_data(download_data())
    #print(load_clean_data(download_data()))
