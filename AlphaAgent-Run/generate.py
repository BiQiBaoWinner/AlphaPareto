import qlib
import pandas as pd

qlib.init(provider_uri="~/.qlib/qlib_data/cn_data_2024h1", region="cn")
from qlib.data import D

instruments = D.instruments("csi300")
fields = ["$open", "$close", "$high", "$low", "$volume"]

raw = D.features(instruments, fields, freq="day")
raw = raw.sort_index()

# Unify index to MultiIndex(datetime, instrument)
if raw.index.names == ["instrument", "datetime"]:
    data = raw.swaplevel().sort_index()
else:
    data = raw.sort_index()

data = data.loc["2012-07-01":"2025-06-30"].sort_index()

# Calculate daily return per instrument
close_s = data["$close"]
ret_s = (
    close_s.groupby(level="instrument", group_keys=False)
    .pct_change()
    .fillna(0.0)
)
data["$return"] = ret_s.astype(float)

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
