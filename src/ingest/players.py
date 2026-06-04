"""Load player data."""
import logging

import pandas as pd

from .common import get_data_dir, load_csv_file, select_dataset_csv

logger = logging.getLogger(__name__)


def load_transfermarkt_player_scores() -> pd.DataFrame:
    """Load Transfermarkt player scores dataset."""
    raw_dir = get_data_dir("raw/kaggle") / "transfermarkt_player_scores"
    # This dataset ships many relational tables; select the players table.
    players_file = select_dataset_csv(raw_dir, preferred_names=["players.csv"])
    df = load_csv_file(players_file)
    df.columns = df.columns.str.lower().str.strip()
    logger.info(f"Loaded Transfermarkt data: {len(df)} rows, {len(df.columns)} columns")

    return df


def load_fifa23_players() -> pd.DataFrame:
    """Load FIFA 23 player dataset."""
    raw_dir = get_data_dir("raw/kaggle") / "fifa23_players_clean"
    # Directory contains multiple FIFA editions (17-23); select FIFA 23.
    fifa_file = select_dataset_csv(
        raw_dir, preferred_names=["CLEAN_FIFA23_official_data.csv"]
    )
    df = load_csv_file(fifa_file)
    df.columns = df.columns.str.lower().str.strip()
    logger.info(f"Loaded FIFA 23 data: {len(df)} rows, {len(df.columns)} columns")

    return df


def load_world_cup_2022_players() -> pd.DataFrame:
    """Load World Cup 2022 player data."""
    raw_dir = get_data_dir("raw/kaggle") / "world_cup_2022_player_data"
    # Directory contains many per-aspect tables (player_defense, player_passing,
    # ...); select the core player_stats table.
    stats_file = select_dataset_csv(raw_dir, preferred_names=["player_stats.csv"])
    df = load_csv_file(stats_file)
    df.columns = df.columns.str.lower().str.strip()
    logger.info(f"Loaded World Cup 2022 player data: {len(df)} rows, {len(df.columns)} columns")

    return df
