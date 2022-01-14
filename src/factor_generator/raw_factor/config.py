"""
config for factor generator 
"""

# load packages
import getpass
USER = getpass.getuser()

# init dataserver 
from src.data_ingestion.PqiDataSdk_Offline import PqiDataSdkOffline
ds = PqiDataSdkOffline()

# set up factor i/o path 
my_factor_path = 'data/features/factor'
my_support_factor_path = 'data/features/support_factor'

# set time frame
start_date = '20150101'
end_date = '20211231'
tickers = ds.get_ticker_list()

# set data types (only 'eod' is available for offline data)
required_data_types = [
    'eod'
] 

# specify features to read 
required_field_types_dict = {
    'eod': [],
    'fund': []
}

# pick factor type 
is_support_factor = False  # true for support factor，false for alpha factors

# others 
# for ff3 
mkt_is_equal_weight = True  # True for equal weight，False for fmv weight
smb_is_fmv = True  # True for fmv，false for tmv 
hml_is_bm = True  # True for bm，False for ep (ep not available)

# factor list (hidden for github)
from src.factor_generator.raw_factor.factor_list import * 
