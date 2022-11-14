import datetime as dt
import json
import re
from dataclasses import dataclass
from urllib.parse import parse_qsl, urlparse

import aswan
import datazimmer as dz
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

hun_url = dz.SourceUrl("https://koronavirus.gov.hu/elhunytak")
wd_url = dz.SourceUrl("https://www.worldometers.info/coronavirus/country/hungary/")
owid_url = dz.SourceUrl("https://covid.ourworldindata.org")
OWID_SRC = f"{owid_url}/data/owid-covid-data.csv"


class Condition(dz.CompositeTypeBase):
    lungs = bool
    heart = bool
    blood_pressure = bool
    diabetes = bool
    obesity = bool


class CovidVictim(dz.AbstractEntity):

    serial = dz.Index & int

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


@dataclass
class CovidState(dz.PersistentState):
    top_serial: int = 0


class VictimCollector(aswan.RequestSoupHandler):
    def parse(self, soup: "BeautifulSoup"):

        state = CovidState.load()

        page_no = int(dict(parse_qsl(urlparse(self._url).query))["page"])
        elem = soup.find(class_="views-table")
        df = pd.read_html(str(elem))[0]
        if df["Sorszám"].astype(int).min() > state.top_serial:
            self.register_links_to_handler([f"{hun_url}?page={page_no + 1}"])
        return df


class HunCovidProject(dz.DzAswan):
    name: str = "hun-covid"
    cron: str = "0 16 * * *"
    starters = {
        VictimCollector: [f"{hun_url}?page=0"],
        aswan.RequestHandler: [wd_url],
    }


def get_hun_victim_df(project: dz.DzAswan) -> pd.DataFrame:
    dfs = []
    for pcev in project.get_unprocessed_events(VictimCollector):
        dfs.append(pcev.content)
    return (
        pd.concat(dfs, ignore_index=True)
        .astype({"Kor": int})
        .assign(
            is_male=lambda df: (df["Nem"].str.lower().str[0] == "f"),
            raw_conditions=lambda df: df["Alapbetegségek"]
            .str.lower()
            .str.replace("õ", "ő"),
        )
        .drop(["Nem", "Alapbetegségek"], axis=1)
        .rename(columns={"Kor": CovidVictim.age, "Sorszám": CovidVictim.serial})
    )


def get_count_df(dza: dz.DzAswan, total_count):

    js_rex = re.compile("'graph-deaths-daily', .*")
    soup = BeautifulSoup(
        next(dza.get_unprocessed_events(aswan.RequestHandler)).content, "html5lib"
    )
    js_str = soup.find("script", text=js_rex).contents[0]
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
        .loc[lambda df: df["data"].cumsum() < total_count, :]
    )

    mismatch = total_count - daily_df["data"].sum()
    pad_dic = {"data": [mismatch], "date": pd.to_datetime(dt.date.today())}
    return pd.concat([daily_df, pd.DataFrame(pad_dic)], ignore_index=True)


def extend_with_old(df: pd.DataFrame):
    old_victims: pd.DataFrame = victim_table.get_full_df()
    if not old_victims.empty:
        return pd.concat([df, old_victims.loc[:, df.columns]])
    return df


cond_map = [
    (CovidVictim.condition.heart, "szív"),
    (CovidVictim.condition.lungs, "tüdő"),
    (CovidVictim.condition.obesity, "elhízás"),
    (CovidVictim.condition.blood_pressure, "vérnyomás"),
    (CovidVictim.condition.diabetes, "cukorbetegség"),
]

victim_table = dz.ScruTable(CovidVictim)


@dz.register_data_loader(extra_deps=[HunCovidProject])
def create():

    dza = HunCovidProject()

    victim_base_df = (
        get_hun_victim_df(dza)
        .pipe(extend_with_old)
        .drop_duplicates(subset=[CovidVictim.serial])
    )
    daily_df = get_count_df(dza, victim_base_df.shape[0])
    owid_df = pd.read_csv(OWID_SRC).loc[lambda df: df["location"] == "Hungary", :]

    (
        victim_base_df.sort_values(CovidVictim.serial)
        .assign(
            estimated_date=np.repeat(daily_df["date"], daily_df["data"])
            .astype(str)
            .values,
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
