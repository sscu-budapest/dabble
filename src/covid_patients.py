import datetime
import datetime as dt
import json
import re
from itertools import count

import datazimmer as dz
import numpy as np
import pandas as pd
from aswan import get_soup
from tqdm import tqdm


class Condition(dz.CompositeTypeBase):
    lungs = bool
    heart = bool
    blood_pressure = bool
    diabetes = bool
    obesity = bool


class CovidVictim(dz.AbstractEntity):

    serial = dz. Index & int

    age = int
    estimated_date = dt.datetime
    is_male = bool
    raw_conditions = str

    condition = Condition

    # TODO: this should be inherited from elsewhere
    positive_rate = float
    total_vaccinations = int
    people_vaccinated = int
    people_fully_vaccinated = int
    total_boosters = int


hun_url = dz.SourceUrl("https://koronavirus.gov.hu/elhunytak")
wd_url = dz.SourceUrl("https://www.worldometers.info/coronavirus/country/hungary/")
owid_url = dz.SourceUrl("https://covid.ourworldindata.org")
OWID_SRC = f"{owid_url}/data/owid-covid-data.csv"


def get_hun_victim_df():
    dfs = []
    for p in tqdm(count()):
        soup = get_soup(f"{hun_url}?page={p}")
        elem = soup.find(class_="views-table")
        try:
            dfs.append(pd.read_html(str(elem))[0])
        except ValueError:
            break

    return (
        pd.concat(dfs, ignore_index=True)
        .astype({"Kor": int})
        .assign(is_male=lambda df: (df["Nem"].str.lower().str[0] == "f"))
        .drop("Nem", axis=1)
        .rename(columns={"Kor": CovidVictim.age, "Sorszám": CovidVictim.serial})
    )


def get_count_df(patient_df):

    soup = get_soup(wd_url)
    js_str = soup.find("script", text=re.compile("'graph-deaths-daily', .*")).contents[
        0
    ]

    daily_df = (
        pd.DataFrame(
            {
                k: json.loads(re.compile(rf"{k}: (\[.*\])").findall(js_str)[0])
                for k in ["data", "categories"]
            }
        )
        .assign(date=lambda df: pd.to_datetime(df["categories"]))
        .fillna(0)
        .sort_values("date")
        .loc[lambda df: df["data"].cumsum() < patient_df.shape[0], :]
    )

    mismatch = patient_df.shape[0] - daily_df["data"].sum()
    pad_dic = {"data": [mismatch], "date": pd.to_datetime(datetime.date.today())}
    return pd.concat([daily_df, pd.DataFrame(pad_dic)], ignore_index=True)


cond_map = [
    (CovidVictim.condition.heart, "szív"),
    (CovidVictim.condition.lungs, "tüdő"),
    (CovidVictim.condition.obesity, "elhízás"),
    (CovidVictim.condition.blood_pressure, "vérnyomás"),
    (CovidVictim.condition.diabetes, "cukorbetegség"),
]

victim_table = dz.ScruTable(CovidVictim)


@dz.register_data_loader(cron="0 16 * * *")
def create():

    victim_df = get_hun_victim_df()
    daily_df = get_count_df(victim_df)
    owid_df = pd.read_csv(OWID_SRC).loc[lambda df: df["location"] == "Hungary", :]

    (
        victim_df.sort_values(CovidVictim.serial)
        .assign(
            estimated_date=np.repeat(daily_df["date"], daily_df["data"])
            .astype(str)
            .values,
            raw_conditions=lambda df: df["Alapbetegségek"]
            .str.lower()
            .str.replace("õ", "ő"),
            **{k: _getcond(v) for k, v in cond_map},
        )
        .merge(
            owid_df.rename(columns={"date": CovidVictim.estimated_date}),
            how="left",
        )
        .sort_values(CovidVictim.estimated_date)
        .fillna(method="ffill")
        .fillna(0)
        .pipe(victim_table.replace_all)
    )


def _getcond(s):
    def f(df):
        return df[CovidVictim.raw_conditions].str.contains(s).astype(bool)

    return f
