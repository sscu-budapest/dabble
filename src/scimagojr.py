import datazimmer as dz
import pandas as pd

class Journal(dz.AbstractEntity):

    sourceid = dz. Index & int


    rank = int
    journal_rating = float
    title = str
    type = str
    issn = str
    h_index = int
    total_docs_2021 = int
    ref_per_doc = float
    sjr_best_quartile = str
    total_docs_3years = int
    total_refs = int
    total_cites_3years = int
    citable_docs_3years = int
    country = str
    region = str
    publisher = str
    coverage = str
    categories = str


journal_table = dz.ScruTable(Journal)


@dz.register_data_loader
def proc():
    df = pd.read_csv("https://www.scimagojr.com/journalrank.php?out=xls", sep=";")
    (
        df.rename(
            columns=lambda s: s.lower()
            .replace(" ", "_")
            .replace(".", "")
            .replace("(", "")
            .replace(")", "")
        )
        .assign(
            journal_rating=lambda df: df["sjr"].pipe(_f2str),
            ref_per_doc=lambda df: df["ref_/_doc"].pipe(_f2str),
        )
        .pipe(journal_table.replace_records)
    )


def _f2str(s):
    return s.str.replace(",", ".").astype(float)
