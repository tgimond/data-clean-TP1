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
                    'acc_lib':'object',
                    'appartenan':'object',
                    'date_insta':'object',
                    'dermnt':'object',
                    'disp_compl':'object',
                    'disp_h':'object',
                    'disp_j':'object',
                    'dtpr_bat':'object',
                    'dtpr_lcad':'object',
                    'dtpr_lcped':'object',
                    'lat_coor1':'object',
                    'lc_ped':'object',
                    'long_coor1':'object',
                    'num_serie':'object',
                    'ref':'object',
                    'tel1':'object',
                    } 
    # We define dates as object because it must be readable by human, we don't use it as a datetime64.  
    #df.replace(" ", pd.NA, inplace=True)
    df = pd.read_csv(
        data_fname,
        usecols=column_names,
        dtype=column_types,
        encoding='utf-8'
        )
    df.rename(columns={
        'nom': 'Nom',
        'acc_etg': 'Etage',
        'acc': 'Interieur',
        'acc_lib': 'Acces_libre',
        'appartenan': 'Propriétaire',
        'date_insta': 'Date_instal',
        'dermnt': 'Derniere_maintenance',
        'disp_compl': 'Fermeture_occasionelle',
        'disp_h': 'Heure_disp',
        'disp_j': 'Jour_disp',
        'dtpr_bat': 'Date_péremption_batterie',
        'dtpr_lcad': 'Date_péremption_elec_adulte',
        'dtpr_lcped': 'Date_péremption_elec_pédiatrique',
        'lc_ped': 'Electrode_enfant',
        'lat_coor1': 'Latitude',
        'long_coor1': 'Longitude',
        'num_serie': 'Num_serie',
        'ref': 'Référent',
        'tel1': 'Tel'
    }, inplace=True)
    #print(df.columns)
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
    df.replace(" ", pd.NA, inplace=True)
        # Convert string columns to lower case
    df = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)
    if 'Fermeture_occasionelle' in df.columns:
        df.replace(r'^ma.*', pd.NA, regex=True, inplace=True)
    
    # Convertir la colonne 'dates' en objets datetime
    df['Date_péremption_batterie'] = pd.to_datetime(df['Date_péremption_batterie'])
    # Formater la colonne 'dates' dans le format souhaité
    df['Date_péremption_batterie'] = df['Date_péremption_batterie'].dt.strftime('%Y-%m-%d')

    df['Date_instal'] = pd.to_datetime(df['Date_instal'])
    df['Date_instal'] = df['Date_instal'].dt.strftime('%Y-%m-%d')

    df['Date_péremption_elec_adulte'] = pd.to_datetime(df['Date_péremption_elec_adulte'])
    df['Date_péremption_elec_adulte'] = df['Date_péremption_elec_adulte'].dt.strftime('%Y-%m-%d')

    df['Date_péremption_elec_pédiatrique'] = pd.to_datetime(df['Date_péremption_elec_pédiatrique'])
    df['Date_péremption_elec_pédiatrique'] = df['Date_péremption_elec_pédiatrique'].dt.strftime('%Y-%m-%d')
    
    df['Etage'].replace(0, "RDC", inplace=True)

    # Supprimer la ligne "tous les ans" de la colonne "Derniere_maintenance"
    df = df[df['Derniere_maintenance'] != 'tous les ans']
    df['Derniere_maintenance'] = pd.to_datetime(df['Derniere_maintenance'])
    df['Derniere_maintenance'] = df['Derniere_maintenance'].dt.strftime('%Y-%m-%d')

    df['Nom'].replace("- "," - ")
    df['Nom'].replace(" -"," - ")

    df['Acces_libre']=df['Acces_libre'].replace({'oui':True,'non':False})
    df['Acces_libre']=df['Acces_libre'].astype('boolean')

    df['Electrode_enfant'] = df['Electrode_enfant'].replace('2021-08-17', np.nan)
    df['Electrode_enfant']=df['Electrode_enfant'].replace({'oui':True,'non':False})
    df['Electrode_enfant']=df['Electrode_enfant'].astype('boolean')

    # Convertir la colonne en numérique en remplaçant les valeurs non numériques par NaN
    df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    df['Longitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    # Convertir la colonne en type float64
    df['Latitude'] = df['Latitude'].astype('float64')
    df['Longitude'] = df['Longitude'].astype('float64')

    if 'Tel' in df.columns:
        df['Tel'].fillna('', inplace=True)
        df['Tel'] = df['Tel'].str.replace(r'\D', '').str.replace('+', '').str.replace('33', '0').str.replace('  ', ' ').str.replace('\n', ' ')
        df['Tel'] = df['Tel'].str.replace(r'(?<=\d)\s(?=\d)', '')    
        df['Tel'] = df['Tel'].str.rstrip()
        df['Tel'] = np.where(df['Tel'].str.len() != 14, pd.NA, df['Tel'])

    df['Jour_disp']=df['Jour_disp'].replace('lundi, mardi, mercredi, jeudi, vendredi, samedi, dimanche','7j/7')

    df['Num_serie'] = df['Num_serie'].str.upper() #We keep the capitalized letter in the Num_serie

    return df

# Define a framing function
def frame_data(df:pd.DataFrame) -> pd.DataFrame:
    """ One function all framing (column renaming, column merge)"""
    # Merge columns adr_num and adr_voie
    df['adr_num'].replace(pd.NA,'', inplace=True)
    df['address'] = df['adr_num'].astype(str) + ' ' + df['adr_voie']
    df['address'] = df['address'].str.split(',').str[0]
    df.drop(['adr_num', 'adr_voie'], axis=1, inplace=True)
    return df


# once they are all done, call them in the general clean loading function
def load_clean_data(data_path:str=DATA_PATH)-> pd.DataFrame:
    """one function to run it all and return a clean dataframe"""
    df = (load_formatted_data(data_path)
          .pipe(sanitize_data)
          .pipe(frame_data)
    )
    for value in df['Etage']:
        print(value)
    #print(df.to_string(index=False))
    return df


# if the module is called, run the main loading function
if __name__ == '__main__':
    load_clean_data(download_data())
    print(load_clean_data(download_data()))
