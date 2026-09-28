import qlib

qlib.init(provider_uri="~/.qlib/qlib_data/cn_data_2024h1")

from qlib.data import D

instruments = D.instruments()
fields = ["$open", "$close", "$high", "$low", "$volume", "$factor"]
data = D.features(instruments, fields, freq="day").swaplevel().sort_index().loc["2012-07-01":].sort_index()

data.to_hdf("./daily_pv_all.h5", key="data")


fields = ["$open", "$close", "$high", "$low", "$volume", "$factor"]
debug_data = D.features(instruments, fields, start_time="2022-01-01", end_time="2023-12-31", freq="day").swaplevel().sort_index()
# Pick first 100 instruments that actually exist in this date range
available_instruments = debug_data.index.get_level_values("instrument").unique()
selected = available_instruments[:100]
debug_data = debug_data.swaplevel().loc[selected].swaplevel().sort_index()

debug_data.to_hdf("./daily_pv_debug.h5", key="data")
