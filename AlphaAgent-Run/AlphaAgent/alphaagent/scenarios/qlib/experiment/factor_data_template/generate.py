import qlib
import pandas as pd

qlib.init(provider_uri="~/.qlib/qlib_data/cn_data_2024h1", region="cn")
from qlib.data import D

# All instruments in the dataset (no market filter)
instruments = D.instruments("all")
fields = ["$open", "$close", "$high", "$low", "$volume", "$vwap"]

raw = D.features(instruments, fields, freq="day")
raw = raw.sort_index()

# Unify index to MultiIndex(datetime, instrument)
if raw.index.names == ["instrument", "datetime"]:
    data = raw.swaplevel().sort_index()
else:
    data = raw.sort_index()

data = data.loc["2012-07-01":"2025-06-30"].sort_index()

# If qlib doesn't provide $vwap, approximate with (O+C+H+L)/4
if "$vwap" not in data.columns or data["$vwap"].isna().all():
    data["$vwap"] = (data["$open"] + data["$close"] + data["$high"] + data["$low"]) / 4.0

print("full data preview:")
print(data.head())
print(data.index.names)
print(data.columns.tolist())

# Main output
data.to_hdf("./daily_pv.h5", key="data")
data.to_hdf("./daily_pv_all.h5", key="data")

# Debug subset: max 100 instruments
ins_list = data.index.get_level_values("instrument").unique().tolist()[:100]
data_debug = data.loc[(slice(None), ins_list), :].sort_index()
data_debug.to_hdf("./daily_pv_debug.h5", key="data")

print("saved:")
print("  ./daily_pv.h5")
print("  ./daily_pv_all.h5")
print("  ./daily_pv_debug.h5")
