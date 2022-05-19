import traceback

import numpy as np
import pandas as pd
import os
import datetime

def drop_duplicate_rows(df):
    """
    Drops the duplicate rows in the dataframe based on Date column.

    Parameters
    ----------

    df : dataframe

    Returns
    -------

    df: dataframe
        updated dataframe after droping duplicates.

    """
    df = df.drop_duplicates(subset=["Date"], keep="first")
    return df


def fill_with_previous_values(df):
    """
    Fills the null values in the dataframe with the values from the previous row.

    Parameters
    ----------

    df : dataframe

    Returns
    -------

    df : dataframe
        updated dataframe after filling with previous values.

    """
    df.fillna(method="ffill", inplace=True)
    return df


def add_missing_rows(df, ind):
    """

    Adds rows to the stock dataframe.

    If the date is present in index dataframe and not present in stock dataframe,
    then a new row (as date and NAN values) is added to stock dataframe.

    Parameters
    ----------

    df : dataframe
        stock dataframe

    ind : dataframe
        index dataframe

    Returns
    -------

    df : dataframe
        updated dataframe after adding new rows.

    """

    df.Date = pd.to_datetime(df.Date)
    ind.Date = pd.to_datetime(ind.Date)
    s = df.Date.head(1).values[0]
    e = df.Date.tail(1).values[0]
    ind = ind[ind.Date.between(e, s)]
    df = df.set_index("Date")
    ind = ind.set_index("Date")
    missing = set(ind.index)-set(df.index)
    for i in missing:
        df.loc[i] = np.nan
    df = df.sort_index(ascending=False)
    df = df.reset_index()

    return df


def cleaning(df, ind):
    """
    Removes duplicate rows, Adds missing rows, fills null values from pervious row to the stock dataframe.

    Parameters
    ----------

    df : dataframe
        stock dataframe

    ind : dataframe
        index dataframe

    Returns
    -------

    df : dataframe
        updated dataframe after performing all the operations.

    """

    df = drop_duplicate_rows(df)
    ind = drop_duplicate_rows(ind)
    df = add_missing_rows(df, ind)
    df = fill_with_previous_values(df)
    df.reset_index(drop=True, inplace=True)
    df = drop_duplicate_rows(df)
    df = df.sort_values(by=["Date"], ascending=[False])
    return df, ind

def stock_split(stock_df, start_date, end_date, r1, r2):
    """
    For an r1:r2 stock split, if y is the stock value before the split,
    then the value of the stock will be y*(r2/r1),
    for the data between the given dates.

    Parameters
    ----------

    stock : dataframe

    start_date : datetime

    end_date : datetime

    r1 : integer

    r2 : integer

    Returns
    -------

    stock : dataframe
        updated dataframe after splitting
    """
    specific_dates = stock_df[stock_df.Date.between(end_date, start_date)]
    for index, row in specific_dates.iterrows():
        specific_dates.loc[index,
                           "Open"] = specific_dates.loc[index, "Open"] * (r1/r2)
        specific_dates.loc[index,
                           "Low"] = specific_dates.loc[index, "Low"] * (r1/r2)
        specific_dates.loc[index,
                           "High"] = specific_dates.loc[index, "High"] * (r1/r2)
        specific_dates.loc[index, "Close"] = specific_dates.loc[index,
                                                                      "Close"] * (r1/r2)

        try:
            stock_df.loc[index] = specific_dates.loc[index]
        except:
            traceback.print_exc()

    return stock_df

def create_dividend(stock_df, dividend_df):
    """
    Creates new Dividend Value column in the stock dataframe.

    Parameters
    ----------

    dividend_df : dataframe

    stock_df : dataframe

    Returns
    -------

    stock : dataframe
        updated dataframe with dividend column

    """
    dividend_df['Date'] = pd.to_datetime(dividend_df['Date'], errors='coerce').dt.strftime('%Y-%m-%d')
    stock_df['Date'] = pd.to_datetime(stock_df['Date'], errors='coerce').dt.strftime('%Y-%m-%d')
    stock_df['Dividend Value'] = pd.DataFrame([[0]], index=stock_df.index)
    last_date = stock_df["Date"].values[-1]
    rows = dividend_df.shape[0]
    if rows == 1:
        from_date = dividend_df.iloc[0]["Date"]
        if from_date > last_date:
            amount = dividend_df["Dividends"][0]
            mask = (stock_df['Date'] <= from_date) & (stock_df['Date'] >= last_date)
            stock_df["Dividend Value"].loc[mask] = stock_df["Dividend Value"].loc[mask].replace(0, amount)
        return stock_df
    for row in range(rows):
        to_date = dividend_df["Date"][row]
        amount = dividend_df["Dividends"][row]
        # print(amount)
        from_date = dividend_df["Date"][row + 1]

        if from_date < last_date:
            from_date = last_date
            mask = (stock_df['Date'] >= from_date) & (stock_df['Date'] <= to_date)
            stock_df["Dividend Value"].loc[mask] = stock_df["Dividend Value"].loc[mask].replace(0, amount)
            break
        mask = (stock_df['Date'] > from_date) & (stock_df['Date'] <= to_date)
        # mask_length = np.count_nonzero(mask.values)
        stock_df["Dividend Value"].loc[mask] = stock_df["Dividend Value"].loc[mask].replace(0, amount)
    return stock_df


def apply_corporate_actions(stock_df,dividend_df,stock_split_df):
    for index, row in stock_split_df.iterrows():
        try:
            start_date = stock_split_df.loc[index, "Date"]
            ratio = stock_split_df.loc[index, "Stock Splits"]
            r1, r2 = ratio*10,10
            r1, r2 = int(r1), int(r2)
            end_date = stock_df.head(1)["Date"].values[0]
            stock_df = stock_split(stock_df, start_date, end_date, r1, r2)
        except:
            pass

    stock_df = create_dividend(stock_df, dividend_df)

    return stock_df

def calculate_beta(stock_df, ind_df, full_stock):
    """
    Creates a new Beta column in the stock dataframe
    beta = covariance(X, Y)/var(Y)
    X = %returns of company
    Y = %returns of NDX
    %returns of company = ((Close Price of today / Close Price of previous trading day) - 1) * 100
    %returns of NDX = from new Index dataframe. (% Return)
    Parameters
    ----------
    stock_df : dataframe
    Returns
    -------
    stock : dataframe
        updated dataframe with new Beta column
    """
    # path = os.path.join(os.getcwd(), "Data")

    stock_df["% Return of Company"] = (
        (full_stock["Close"] / full_stock['Close'].shift(-1))-1)*100

    full_stock["% Return of Company"] = (
        (full_stock["Close"] / full_stock['Close'].shift(-1))-1)*100

    ind_df["Date"] = pd.to_datetime(ind_df["Date"])
    stock_df["Date"] = pd.to_datetime(stock_df["Date"])

    s = full_stock.Date.head(1).values[0]
    e = full_stock.Date.tail(1).values[0]
    ind_df = ind_df[ind_df.Date.between(e, s)]
    ind_df = ind_df.iloc[::-1]
    ind_df.rename(columns={'Close': 'Close Price of NDX',
               '% Return': '% Return of NDX'}, inplace=True)
    ind_df = ind_df.loc[:,["Date",'Close Price of NDX','% Return of NDX']]
    ind_df["Date"] = pd.to_datetime(ind_df["Date"])
    ind_df = ind_df.copy()
    stock = stock_df.set_index("Date")
    ind_df = ind_df.set_index("Date")
    full_stock = full_stock.set_index("Date")
    for date, row in stock.iterrows():
        try:
            stock.loc[date, 'Close Price of NDX'] = ind_df.loc[date,
                                                                'Close Price of NDX']
            stock.loc[date, '% Return of NDX'] = ind_df.loc[date,
                                                             '% Return of NDX']
        except:
            pass
    stock = stock.reset_index()
    full_stock = full_stock.reset_index()
    ind_df = ind_df.reset_index()
    NDX = ind_df["% Return of NDX"]
    # print('NDX')
    # print(NDX)
    company = full_stock["% Return of Company"]
    results = list()
    for i in range(stock.shape[0]):
        cov = np.ma.cov(np.ma.masked_invalid(np.array(company[i:], NDX[i:-1])), rowvar=False)
        var = np.nanvar(NDX[i:-1])
        res = var/cov
        results.append(res)
    stock["Beta"] = results
    return stock

def add_risk_free_column(stock, riskrates, full_stock):
    """
    Creates a new Rate column in the stock dataframe using riskfreerate file.
    Parameters
    ----------
    stock : dataframe
    Returns
    -------
    res : dataframe
        updated dataframe with Rate column
    """

    riskrates["Date"] = pd.to_datetime(riskrates["Date"])
    stock["Date"] = pd.to_datetime(stock["Date"])

    # riskrates["Rate"] = pd.to_numeric(riskrates["Rate"])
    riskrates["Rate"] = (riskrates["Rate"].astype(str)).apply(pd.to_numeric, errors='coerce')
    # stock[direct_columns] = (stock[direct_columns].astype(
    #     str)).apply(pd.to_numeric, errors='coerce')
    resdf = riskrates.copy()
    stock = stock.set_index("Date")
    resdf = resdf.set_index("Date")
    for date, row in stock.iterrows():
        try:
            stock.loc[date, 'Rate'] = resdf.loc[date, 'Rate']
        except:
            stock.loc[date, 'Rate'] = np.nan

    stock = stock.reset_index()
    resdf = resdf.reset_index()

    return stock

def calculate_alpha(stock, ind, full_stock):
    """
    Creates a new Alpha column in the stock dataframe
    alpha = %YTDCompany - (riskfreerate + (Beta * (%YTDNDX - riskfreerate)))
    %YTD Company = percentage of year to date of the company
    %YTD NDX = percentage of year to date of the index file.(%YTD)
    Beta = beta value from calculate_beta method.
    %YTDCompany = ((Close Price of last available day / Close Price of today) - 1) * 100
    riskfreerate :
    Parameters
    ----------
    stock : dataframe
    Returns
    -------
    stock : dataframe
        updated dataframe with new Alpha column
    """
    # path = os.path.join(os.getcwd(), "Data")

    stock["% YTD of Company"] = ((full_stock.tail(1)['Close'].values[0]/full_stock["Close"])-1)*100
    # ind = pd.read_csv(os.path.join(path, "Index.csv"))
    ind["Date"] = pd.to_datetime(ind["Date"])
    s = stock.Date.head(1).values[0]
    e = stock.Date.tail(1).values[0]
    ind = ind[ind.Date.between(e, s)]
    ind = ind.loc[:,["Date","% YTD"]]
    ind.columns = ["Date","% YTD of NDX"]
    ind["Date"] = pd.to_datetime(ind["Date"])
    stock["Date"] = pd.to_datetime(stock["Date"])

    # inddf = ind[ind.Date.between(
    #     stock.iloc[-1]['Date'], stock.iloc[0]['Date'])]

    inddf = ind.copy()

    stock = stock.set_index("Date")
    inddf = inddf.set_index("Date")

    for date, row in stock.iterrows():
        try:
            stock.loc[date, '% YTD of NDX'] = inddf.loc[date, '% YTD of NDX']
        except:
            pass
    stock = stock.reset_index()
    inddf = inddf.reset_index()

    # stock = pd.merge(stock, ind, on="Date", how="left")
    # stock["Beta"] = pd.to_numeric(stock["Beta"], errors='coerce')
    stock["Beta"] = (stock["Beta"].astype(str)).apply(
        pd.to_numeric, errors='coerce')
    stock["Alpha"] = stock["% YTD of Company"] - \
                     (stock["Rate"] + (stock["Beta"] * (stock["% YTD of NDX"] - stock["Rate"])))
    return stock

def create_lower_upper_bands(stock, full_stock):
    """

    Creates lower band, upper band, band area columns in the stock dataframe.

    Parameters
    ----------

    stock : dataframe

    Returns
    -------

    stock : dataframe
        updated dataframe with lower, upper, band area columns

    """
    for i, row in stock.iterrows():
        maxv = full_stock.loc[i:]["Close"].max()
        minv = full_stock.loc[i:]["Close"].min()
        # print(i,maxv,minv)
        stock.loc[i, "Upper Band"] = maxv
        stock.loc[i, "Lower Band"] = minv
        stock.loc[i, "Band Area"] = maxv - minv
    return stock

def create_new_LB_UB(stock, full_stock):
    """
    Creates new growth rate columns in the stock dataframe.
    Previous and Next ,Lower Band, Upper Band for
    for 30,90,180,360,720,1080 days

    Lower Lower = Close Price of that day/min close price in the band
    Upper Band = Close Price of that day/max close price in the band

    Parameters
    ----------

    stock : dataframe

    Returns
    -------

    stock : dataframe
        updated dataframe with newly created columns.
    """

    bands = [30, 60, 90, 180, 360, 720, 1080]
    for b in bands:
        pcols = ["Previous " + str(b) + " days LB",
                 "Previous " + str(b) + " days UB"]
        stock[pcols] = pd.DataFrame([[0]*len(pcols)], index=stock.index)
        for index, row in stock.iterrows():
            start = row['Date']
#             start = row['Date'] - datetime.timedelta(days=1)
            end = start - datetime.timedelta(days=b)
            specific_dates = full_stock[full_stock.Date.between(end, start)]
            low = specific_dates["Close"].min()
            high = specific_dates["Close"].max()
            today = row["Close"]
            stock.loc[index, pcols] = [low/today, high/today]

    bands = [30, 60, 90, 180, 360, 720, 1080]
    for b in bands:
        ncols = ["Next " + str(b) + " days LB", "Next " + str(b) + " days UB"]
        stock[ncols] = pd.DataFrame([[0]*len(ncols)], index=stock.index)
        for index, row in stock.iterrows():
            start = row['Date']
#             start = row['Date'] + datetime.timedelta(days=1)
            end = start + datetime.timedelta(days=b)
            specific_dates = full_stock[full_stock.Date.between(start, end)]
            low = specific_dates["Close"].min()
            high = specific_dates["Close"].max()
            today = row["Close"]
            stock.loc[index, ncols] = [low/today, high/today]
    return stock

def create_eps_pe_ratio_revenue_income_expenditure_net_profit(rev, stk):
    """
    Creates eps, pe, revenue, income, expenditure, profit columns.

    Creates 2,4,8 bands for eps, pe, revenue, income, expenditure, profit columns.

    Parameters
    ----------

    rev : dataframe
        revenue dataframe

    stk : dataframe
        stock dataframe

    Returns
    -------

    stk : dataframe
        updated dataframe after creating the columns.
    """

    stk["Date"] = pd.to_datetime(stk["Date"])
    s = min(rev.Year)
    e = max(rev.Year)
    cols = ['Revenue', 'Net Profit', 'Income', 'Expenditure', 'EPS']
    stk[cols] = pd.DataFrame([[0]*len(cols)], index=stk.index)

    rep = ['Total Revenue', 'Gross Profit', 'Net Income', 'Total Operating Expenses', 'EPS']

    for index, row in stk.iterrows():
        q = (row.Date.month-1)//3 + 1
        samp = rev[(rev['Year'] == row.Date.year) & (rev['Quarter'] == q)]
        if samp.shape[0] != 0:
            stk.loc[index, cols] = samp.iloc[0][rep].values
        else:
            stk.loc[index, cols] = [np.nan]*5

    stk['year'] = pd.DatetimeIndex(stk['Date']).year
    # stk = stk[(stk.year >= s)&(stk.year <= e) & stk["Revenue"] !=0 ]
    # stk = stk.drop(["year"],axis=1)

    bands = [2, 4, 8]

    for band in bands:
        bcols = ['Revenue last '+str(band)+' quarters', 'Income last '+str(band)+' quarters', 'Expenditure  last '+str(
            band)+' quarters', 'Net Profit  last '+str(band)+' quarters', 'EPS last '+str(band)+' quarters']
        stk[bcols] = pd.DataFrame([[0]*len(bcols)], index=stk.index)
        for index, row in stk.iterrows():
            q = (row.Date.month-1)//3 + 1
            samp = rev[(rev['Year'] == row.Date.year) & (rev['Quarter'] == q)]
            if samp.shape[0] == 0:
                r = 1
            else:
                r = samp.index.values[0]
            if r+band+1 <= rev.shape[0]:
                v = range(r+1, r+band+1)
                stk.loc[index, bcols] = rev.loc[v, rep].sum().values
    stk["p/e"] = stk["Close"]/stk["EPS"]
    return stk

def add_next_day_columns(stock, full_stock):
    """
    Creates new Next Day columns in the stock dataframe.

    Parameters
    ----------

    stock : dataframe

    Returns
    -------

    stock : dataframe
        updated dataframe with Next Day columns.
    """

    new_columns = ["Next Day Open", "Next Day High",
                   "Next Day Low", "Next Day Close"]
    columns = ["Open", "High", "Low", "Close"]
    stock[new_columns] = pd.DataFrame([[np.nan]*4], index=stock.index)
    stock[new_columns] = full_stock[columns].shift(1)
    return stock


direct_columns = ['Open', 'High', 'Low', 'Close', 'Next Day Open',
                  'Next Day High', 'Next Day Low', 'Next Day Close','Spread High-Low', 'Spread Close-Open', 'Alpha', 'Beta']
growth_direct_rate_columns = [col + " GR" for col in direct_columns]


def find_gain_loss(stock, full_stock):
    """
    Creates new growth rate columns in the stock dataframe.

    Growth rate = (X-Y)/Y

    X = value of today
    Y = value of the previous trading day

    Parameters
    ----------

    stock : dataframe

    Returns
    -------

    stock : dataframe
        updated dataframe with newly created columns.

    """
    direct_columns = ['Open', 'High', 'Low', 'Close',
                      'Next Day Open', 'Next Day High', 'Next Day Low',
                      'Next Day Close', 'Spread High-Low', 'Spread Close-Open', 'Alpha', 'Beta']
    growth_direct_rate_columns = [col + " GR" for col in direct_columns]
    stock[direct_columns] = stock[direct_columns].apply(pd.to_numeric, errors="coerce")

    stock[growth_direct_rate_columns] = pd.DataFrame([[np.nan]*len(growth_direct_rate_columns)], index=stock.index)
    # print("After Growth Rate")
    # print(full_stock.head(2))
    result = stock.append(full_stock.head(2))
    # print(result.head(5))
    result = result.drop_duplicates(subset=["Date"], keep="first")
    result = result.reset_index(drop=True)
    result[direct_columns] = result[direct_columns].apply(
        pd.to_numeric, errors="coerce")

    for i in range(stock.shape[0]):
      try:
        today = result.iloc[i][direct_columns]
        previous = result.iloc[i+1][direct_columns]
        vals = (today - previous)/previous
        vals = vals.values
        stock.loc[i, growth_direct_rate_columns] = vals
      except:
        pass
    return stock

def sequential_increase(stock, full_stock):
    """
    Creates new Sequential Increase column in the stock dataframe.

    Parameters
    ----------

    stock : dataframe

    Returns
    -------

    stock : dataframe
        updated dataframe with newly created column.
    """
    stock["Sequential Increase"] = np.nan
    c = 0
    # stock.at[stock.shape[0]-2, "Sequential Increase"] = 0
    # stock.at[stock.shape[0]-1, "Sequential Increase"] = 0
    for i in range(stock.shape[0], 0, -1):
        try:
            if full_stock.at[i, "Close"] > full_stock.at[i+1, "Close"]:
                c += 1
                stock.at[i-1, "Sequential Increase"] = c
            else:
                stock.at[i-1, "Sequential Increase"] = 0
                c = 0
        except:
            pass
    return stock


def sequential_decrease(stock, full_stock):
    """
    Creates new Sequential Decrease column in the stock dataframe.

    Parameters
    ----------

    stock : dataframe

    Returns
    -------

    stock : dataframe
        updated dataframe with newly created column.
    """

    stock["Sequential Decrease"] = np.nan
    c = 1
    # stock.at[stock.shape[0]-2, "Sequential Decrease"] = 0
    # stock.at[stock.shape[0]-1, "Sequential Decrease"] = 0
    for i in range(stock.shape[0], 0, -1):
        try:
            if full_stock.at[i, "Close"] < full_stock.at[i+1, "Close"]:
                stock.at[i-1, "Sequential Decrease"] = c
                c += 1
            else:
                stock.at[i-1, "Sequential Decrease"] = 0
                c = 1
        except:
            pass
    return stock


def sequential_increase_percentage(stock, full_stock):
    """
    Creates new Sequential Increase % column in the stock dataframe.

    Parameters
    ----------

    stock : dataframe

    Returns
    -------

    stock : dataframe
        updated dataframe with newly created column.
    """
    stock["Sequential Increase %"] = np.nan
    for i in range(stock.shape[0]):
        try:
            if stock.at[i, "Sequential Increase"] != 0:
                inc = stock.at[i, "Sequential Increase"]
            else:
                inc = 1
            fr = full_stock.at[i+1, "Close"]
            to = full_stock.at[i+1+inc, "Close"]
            stock.at[i, "Sequential Increase %"] = (fr - to) / to
        except:
            pass
    return stock


def sequential_decrease_percentage(stock, full_stock):
    """
    Creates new Sequential Decrease % column in the stock dataframe.

    Parameters
    ----------

    stock : dataframe

    Returns
    -------

    stock : dataframe
        updated dataframe with newly created column.
    """

    stock["Sequential Decrease %"] = ""
    for i in range(stock.shape[0]):
        try:
            if stock.at[i, "Sequential Decrease"] != 0:
                inc = stock.at[i, "Sequential Decrease"]
            else:
                inc = 1
            fr = full_stock.at[i+1, "Close"]
            to = full_stock.at[i+1+inc, "Close"]
            stock.at[i, "Sequential Decrease %"] = (to - fr) / fr
        except:
            pass
    return stock


def sequential_increase_decrease(stock, full_stock):
  bands = [90, 180, 365]
  for b in bands:
      bcols = ["Max Inc % in "+str(b)+" days", "Max Dec % in "+str(b)+" days", "Min Inc % in "+str(
          b)+" days", "Min Dec % in "+str(b)+" days", "Avg Inc % in "+str(b)+" days", "Avg Dec % in "+str(b)+" days"]
      stock[bcols] = pd.DataFrame([[0]*len(bcols)], index=stock.index)
      for i in range(stock.shape[0]):
          s = i+1
          # print("S and S+b", s,s+b)
          specific_bands = stock.loc[s:s+b]
          specific_bands.sort_index(inplace=True)
          try:
          # print((specific_bands["Sequential Decrease %"].apply(pd.to_numeric, errors="coerce").dropna().values))
            inc_max_val = max(specific_bands["Sequential Increase %"].apply(pd.to_numeric, errors="coerce").dropna().values)
            inc_min_val = min(specific_bands["Sequential Increase %"].apply(pd.to_numeric, errors="coerce").dropna().values)
            inc_avg_val = np.mean(specific_bands["Sequential Increase %"].apply(pd.to_numeric, errors="coerce").dropna().values)
            dec_max_val = max(specific_bands["Sequential Decrease %"].apply(pd.to_numeric, errors="coerce").dropna().values)
            dec_min_val = min(specific_bands["Sequential Decrease %"].apply(pd.to_numeric, errors="coerce").dropna().values)
            dec_avg_val = np.mean(specific_bands["Sequential Decrease %"].apply(pd.to_numeric, errors="coerce").dropna().values)
            stock.loc[i, bcols] = [inc_max_val,dec_max_val,inc_min_val,dec_min_val,inc_avg_val,dec_avg_val]
          except:
            pass
  return stock

def dividend_growthrate(dividend_df):
    dividend_df["Date"] = pd.to_datetime(dividend_df["Date"])
    result = {}
    for index, row in dividend_df.iterrows():
      month = row["Date"].month
      q = 1 if 1 <= month <= 3 else 2 if 4 <= month <= 6 else 3 if 6 <= month <= 9 else 4
      year = row["Date"].year
      amount = row["Dividends"]
      res_year = result.get(year, {1: [0], 2: [0], 3: [0], 4: [0]})
      q_res_year = res_year[q]
      if len(q_res_year) == 1:
          if q_res_year[0] == 0:
              q_res_year[0] = amount
          else:
              q_res_year.append(amount)
      else:
          q_res_year.append(amount)
      res_year[q] = q_res_year
      result[year] = res_year
    for year, quaters in result.items():
      for q, a in quaters.items():
        quaters[q] = sum(a) / len(a)
      result[year] = quaters
    # print(result)
    years = []
    quarters = []
    dividend = []
    for year in result:
      each_year_quarters = result[year]
      quarters_list = list(each_year_quarters.keys())
      for q in quarters_list:
        years.append(year)
        quarters.append(q)
        dividend.append(each_year_quarters[q])

    div_gr_df = pd.DataFrame()
    div_gr_df["year"] = years
    div_gr_df["quarter"] = quarters
    div_gr_df["Dividend"] = dividend
    # print(div_gr_df)
    div_gr_dict = {}
    div_gr_df = div_gr_df.sort_values(by=["year", "quarter"], ascending=False, ignore_index=True)
    rows = div_gr_df.shape[0]
    for row in range(rows):
      year = div_gr_df["year"][row]
      quarter = div_gr_df["quarter"][row]
      year_dict = div_gr_dict.get(year,{})
      year_dict[quarter] = 0
      if row < rows - 1:
        today_amount = div_gr_df["Dividend"][row]
        previous_day_amount = div_gr_df["Dividend"][row + 1]
        # print(today_amount,previous_day_amount)
        gr_val = 0
        if previous_day_amount != 0:
            gr_val = ((today_amount - previous_day_amount) / previous_day_amount)
        div_gr_df.loc[row, ["Dividend Gr"]] = gr_val
        year_dict[quarter] = gr_val
      div_gr_dict[year] = year_dict
    return div_gr_df,div_gr_dict

def update_quarterwise_growth_rate(stock_df,revenue_df,div_gr_df):

    stock_df['Date'] = pd.to_datetime(stock_df['Date'])
    gr_cols = ["Total Revenue Gr","Gross Profit Gr","Net Income Gr","Total Operating Expenses Gr","EPS Gr"]
    stock_df[gr_cols] = pd.DataFrame([[0]*len(gr_cols)], index=stock_df.index)
    stock_df["Dividend Gr"] = pd.DataFrame([[0]],index=stock_df.index)
    for index, row in stock_df.iterrows():
        q = (row.Date.month-1)//3 + 1
        rev_samp = revenue_df[(revenue_df['Year'] == row.Date.year) & (revenue_df['Quarter'] == q)]
        div_samp = div_gr_df[(div_gr_df['year'] == row.Date.year) & (div_gr_df['quarter'] == q)]
        if rev_samp.shape[0] > 0:
          li = (rev_samp[gr_cols].values).tolist()[0]
          stock_df.loc[index, gr_cols] = li
          # print(stock_df.loc[index, gr_cols])
        if div_samp.shape[0] > 0:
          gr_val = div_samp["Dividend Gr"].values
          # print(gr_val)
          stock_df.loc[index, "Dividend Gr"] = gr_val
      # print(stock_df.loc[index, "Dividend Gr"])
    return stock_df

def close_price_as_percent_of_LV_HV_BA(stock, full_stock):
    """
    Creates new growth rate columns in the stock dataframe.
    For Close Price as% Lowest Value, close price as% Highest Value, close price as% Band Area
    for 7, 30, 90, 180, 365 bands

    Close Price as % of Lowest Value = Close Price of that day/min close price in the band
    Close Price as % of Highest Value = Close Price of that day/max close price in the band
    Close Price as % of Band Area = Close Price of that day / (max-min close price in the band)

    Parameters
    ----------

    stock : dataframe

    Returns
    -------

    stock : dataframe
        updated dataframe with newly created columns.
    """
    bands = [7, 30, 90, 180, 365]
    for b in bands:
        bcols = ["CP % LV "+str(b)+" days", "CP % HV " +
                 str(b)+" days", "CP % BA "+str(b)+" days"]
        stock[bcols] = pd.DataFrame([[0]*len(bcols)], index=stock.index)
        for i, row in stock.iterrows():
            start = row['Date']
            end = start - datetime.timedelta(days=b)
            specific_dates = full_stock[full_stock.Date.between(end, start)]
            low = specific_dates["Close"].min()
            high = specific_dates["Close"].max()
            today = row["Close"]
            try:
                if (high == low):
                    stock.loc[i, bcols] = [today/low, today/high, ""]
                else:
                    stock.loc[i, bcols] = [today/low,
                                           today/high, today/(high-low)]
            except:
                pass
    return stock

def perform_operation(symbol):
    # try:
        print("*******************************")
        print(symbol)
        print("*******************************")
        index_df = pd.read_csv(os.path.join(path, "Index.csv"))
        dividend_df = pd.read_csv(os.path.join(path, "Corporate_actions/Dividends/"+symbol+".csv"))
        stock_split_df = pd.read_csv(os.path.join(path, "Corporate_actions/Stock_split/"+symbol + ".csv"))
        revenue_df = pd.read_csv(os.path.join(path, "Revenue/"+symbol+".csv"))
        stock_df = pd.read_csv(os.path.join(path, "Stock_data/"+symbol+".csv"))
        # gr_stock_df = pd.read_csv(os.path.join(path, "GRStock/"+"gr"+symbol+".csv"))
        riskfreerate_df = pd.read_csv(os.path.join(path, "RiskFreeRateFull.csv"))
        # gr_stock_df['Date'] = pd.to_datetime(gr_stock_df['Date'])
        stock_df['Date'] = pd.to_datetime(stock_df['Date'])
        stock_df = stock_df.sort_values(by=["Date"], ascending=False, ignore_index=True)
        # start = gr_stock_df.iloc[0]['Date']
        end = stock_df.iloc[0]['Date']
        start = (end - datetime.timedelta(days=3 * 365))
        date_flag = False
        if str(start) < str(stock_df.iloc[0]['Date']):
            date_flag = True
            start = (start + datetime.timedelta(days=2))
        dates_index = stock_df.index[stock_df['Date'] == str(start)].tolist()
        if len(dates_index) == 0:
            while (True):
                print(start)
                # break
                if not date_flag:
                    start = (start - datetime.timedelta(days=1))
                else:
                    start = (start + datetime.timedelta(days=1))
                dates_index = stock_df.index[stock_df['Date'] == str(start)].tolist()
                if len(dates_index) == 1:
                    break
                continue
        print("start", start)
        print("end", end)
        full_stock = stock_df.copy()
        # print("Full_stock")
        # print(full_stock)
        if start == end:
            return
        stock_df = stock_df[stock_df.Date.between(start, end)]
        # stock_df["Date"] = pd.to_datetime(stock_df["Date"])
        # mask = (stock_df['Date'] > '2021-01-04') & (stock_df['Date'] <= '2021-12-31')
        # stock_df = stock_df.loc[mask]
        if stock_df.shape[0] == 0:
            return
        stock_df = apply_corporate_actions(stock_df, dividend_df,stock_split_df)
        # print("After corporate actions")
        # print(stock_df)
        stock_df = calculate_beta(stock_df, index_df, full_stock)
        # print("BETA Calculated : ")
        # print(stock_df["Beta"])
        stock_df = add_risk_free_column(stock_df, riskfreerate_df, full_stock)
        # print("Added Risk Free Rates")
        # print(stock_df)
        stock_df = calculate_alpha(stock_df, index_df, full_stock)
        # print("Alpha Value")
        # print(stock_df)
        stock_df = create_lower_upper_bands(stock_df, full_stock)
        # print("Lower and Upper bands")
        # print(stock_df)
        stock_df = create_new_LB_UB(stock_df, full_stock)
        stock_df = create_eps_pe_ratio_revenue_income_expenditure_net_profit(revenue_df, stock_df)
        stock_df = add_next_day_columns(stock_df, full_stock)
        stock_df[direct_columns] = stock_df[direct_columns].apply(pd.to_numeric, errors="coerce")
        stock_df = find_gain_loss(stock_df, full_stock)
        stock_df = sequential_increase(stock_df, full_stock)
        stock_df = sequential_decrease(stock_df, full_stock)
        stock_df = sequential_increase_percentage(stock_df, full_stock)
        stock_df = sequential_decrease_percentage(stock_df, full_stock)
        stock_df = sequential_increase_decrease(stock_df, full_stock)
        stock_df = stock_df.drop(columns=["Unnamed: 0"], axis=1, errors='ignore')
        dividend_gr_df, div_gr_dict = dividend_growthrate(dividend_df)
        stock_df = update_quarterwise_growth_rate(stock_df, revenue_df, dividend_gr_df)
        stock_df = close_price_as_percent_of_LV_HV_BA(stock_df, full_stock)
        # result = stock_df.append(gr_stock_df)
        stock_df.drop_duplicates()
        if not os.path.exists(os.path.join(path,"GRStock")):
            os.makedirs("NASDAQ_Data/GRStock")
        stock_df.to_csv(os.path.join(path, "GRStock", "gr_" + symbol + ".csv"), index=None)
    # except:
    #     pass



if __name__ == "__main__":
    path = os.path.join(os.getcwd(), "NASDAQ_Data")
    df = pd.read_csv(os.path.join(path, "equity_data.csv"))
    sorted_df = df.sort_values(by=["Market Cap"], ascending=False)
    symbols = sorted_df["Symbol"].values.tolist()
    no_data = ["HDB","RIO","SNY","UL","DEO","CVX","BHP","T"]
    count = 1
    for symbol in symbols[:101]:
        print(count)
        if symbol in no_data:
            continue
    #     # try:
        perform_operation(symbol)
        count += 1
        # except:
        #     pass
    # perform_operation("TSLA")