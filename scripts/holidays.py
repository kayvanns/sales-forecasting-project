import pandas as pd
import numpy as np


def preprocess_holidays(holidays):
    # Transferred holidays: use the transfer date instead of the original date
    tr1 = holidays[(holidays.type == "Holiday") & (holidays.transferred == True)].drop("transferred", axis=1).reset_index(drop=True)
    tr2 = holidays[holidays.type == "Transfer"].drop("transferred", axis=1).reset_index(drop=True)
    tr = pd.concat([tr1, tr2], axis=1).iloc[:, [5, 1, 2, 3, 4]]

    holidays = holidays[(holidays.transferred == False) & (holidays.type != "Transfer")].drop("transferred", axis=1)
    holidays = pd.concat([holidays, tr], ignore_index=True).reset_index(drop=True)

    # Normalize descriptions: strip numeric suffixes and separators
    holidays["description"] = (
        holidays["description"]
        .str.replace("-", "", regex=False)
        .str.replace("+", "", regex=False)
        .str.replace(r'\d+', '', regex=True)
    )

    # Collapse Additional, Bridge types into Holiday; strip "Puente " prefix
    holidays["type"] = holidays["type"].replace({"Additional": "Holiday", "Bridge": "Holiday"})
    holidays["description"] = holidays["description"].str.replace("Puente ", "", regex=False)

    # Work days (bridge payback days) — excluded from holiday features
    work_day = holidays[holidays.type == "Work Day"]
    holidays = holidays[holidays.type != "Work Day"]

    # Events are always national scope
    events = (
        holidays[holidays.type == "Event"]
        .drop(["type", "locale", "locale_name"], axis=1)
        .rename(columns={"description": "events"})
    )
    holidays = holidays[holidays.type != "Event"].drop("type", axis=1)

    regional = (
        holidays[holidays.locale == "Regional"]
        .rename(columns={"locale_name": "state", "description": "holiday_regional"})
        .drop("locale", axis=1)
        .drop_duplicates()
    )
    national = (
        holidays[holidays.locale == "National"]
        .rename(columns={"description": "holiday_national"})
        .drop(["locale", "locale_name"], axis=1)
        .drop_duplicates()
    )
    local = (
        holidays[holidays.locale == "Local"]
        .rename(columns={"description": "holiday_local", "locale_name": "city"})
        .drop("locale", axis=1)
        .drop_duplicates()
    )

    events["events"] = np.where(events.events.str.contains("futbol"), "Futbol", events.events)

    return holidays, regional, national, local, events, work_day

national_holiday_cols = [
    'holiday_national_Batalla_de_Pichincha',
    'holiday_national_Carnaval',
    'holiday_national_Dia_de_Difuntos',
    'holiday_national_Dia_de_la_Madre',
    'holiday_national_Dia_del_Trabajo',
    'holiday_national_Independencia_de_Cuenca',
    'holiday_national_Independencia_de_Guayaquil',
    'holiday_national_Navidad',
    'holiday_national_Primer_Grito_de_Independencia',
    'holiday_national_Primer_dia_del_ano',
    'holiday_national_Viernes_Santo',
]

regional_holiday_cols = [
    'holiday_regional_Provincializacion_Santa_Elena',
    'holiday_regional_Provincializacion_de_Cotopaxi',
    'holiday_regional_Provincializacion_de_Imbabura',
    'holiday_regional_Provincializacion_de_Santo_Domingo',
]

local_holiday_cols = [
    'holiday_local_Cantonizacion_de_Cayambe',
    'holiday_local_Cantonizacion_de_El_Carmen',
    'holiday_local_Cantonizacion_de_Guaranda',
    'holiday_local_Cantonizacion_de_Latacunga',
    'holiday_local_Cantonizacion_de_Libertad',
    'holiday_local_Cantonizacion_de_Quevedo',
    'holiday_local_Cantonizacion_de_Riobamba',
    'holiday_local_Cantonizacion_de_Salinas',
    'holiday_local_Cantonizacion_del_Puyo',
    'holiday_local_Fundacion_de_Ambato',
    'holiday_local_Fundacion_de_Cuenca',
    'holiday_local_Fundacion_de_Esmeraldas',
    'holiday_local_Fundacion_de_Guayaquil',
    'holiday_local_Fundacion_de_Ibarra',
    'holiday_local_Fundacion_de_Loja',
    'holiday_local_Fundacion_de_Machala',
    'holiday_local_Fundacion_de_Manta',
    'holiday_local_Fundacion_de_Quito',
    'holiday_local_Fundacion_de_Riobamba',
    'holiday_local_Fundacion_de_Santo_Domingo',
    'holiday_local_Independencia_de_Ambato',
    'holiday_local_Independencia_de_Guaranda',
    'holiday_local_Independencia_de_Latacunga',
]

event_cols = [
    'events_Black_Friday',
    'events_Cyber_Monday',
    'events_Dia_de_la_Madre',
    'events_Futbol',
    'events_Terremoto_Manabi',
]

all_holiday_cols = national_holiday_cols + regional_holiday_cols + local_holiday_cols + event_cols

def consolidate_holidays(df):

    df['n_national_holidays'] = df[national_holiday_cols].sum(axis=1)
    df['n_regional_holidays'] = df[regional_holiday_cols].sum(axis=1)
    df['n_local_holidays']    = df[local_holiday_cols].sum(axis=1)
    df['n_events']            = df[event_cols].sum(axis=1)

    df['is_black_friday'] = df['events_Black_Friday']
    df['is_cyber_monday'] = df['events_Cyber_Monday']
    df['is_earthquake']   = df['events_Terremoto_Manabi']
    df['is_christmas']    = df['holiday_national_Navidad']
    df['is_new_year']     = df['holiday_national_Primer_dia_del_ano']

    drop_cols = [
        *all_holiday_cols,
        'national_independence',
        'local_cantonizacio',
        'local_fundacion',
        'local_independencia',
    ]
    geo_cols = [c for c in df.columns if c.startswith('city_') or c.startswith('state_')]
    df = df.drop(columns=[c for c in drop_cols + geo_cols if c in df.columns])
    return df



