import pandas as pd
from skforecast.datasets import fetch_dataset


def load_data(verbose: bool = True) -> pd.DataFrame:
    """Load air_quality_valencia dataset from skforecast public datasets."""
    data = fetch_dataset(name='air_quality_valencia', verbose=verbose)
    data.index = pd.DatetimeIndex(data.index, freq='h')
    return data
