# Data Folder Explained

DO NOT PUT EXTRA STUFF UNDER ANY OF THE SUBFOLDERS!

For optimal I/O and memory economy, **feather** is extensively used. Each EOD dataframe contains a column called 'index' containing the stock codes.

There are three folders:

- raw: raw data sent from a confidential source.
- parsed: parsed data for readability and optimal I/O
  - dates: an npy storing all trade dates in this offline dataset;
  - fund_eod_data: fundamental features. Currently only has `TotalEquity` for Book-to-Market ratio calculation;
  - index_eod_data: EOD dataframes for major indices.
  - index_stock_weight: EOD stock weights for a given index;
  - industry_class: SW level 1 industry classification;
  - stock_basics: stock basic information. Currently only has `ListDate`;
  - stock_eod_data: all stock EOD features
- features: put different types of factors inside
  - factor: factor for investing;
  - risk_factor: for estimating covariance matrix;
  - support_factor: factors that support computing factors to invest;
  - ml_factor: predicted stock returns, the machine learning outputs;
  - dynamic_ind: the dynamic industry classification eod features.
