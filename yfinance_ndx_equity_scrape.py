import pandas as pd
import os
import datetime
import time
import yfinance as yf
from yahoofinancials import YahooFinancials
import numpy as np


def convert_date_to_unix_timestamp(stock_df):
    """
    Adds a new Unix Date column to the given dataframe

    Parameters
    ----------
    stock_df : dataframe

    Returns
    -------
    stock_df : dataframe
        updated dataframe with a new Unix Date column.
    """
    stock_df["Unix Date"] = pd.to_datetime(stock_df["Date"]).apply(
        lambda x: time.mktime(x.timetuple()))
    return stock_df

def download_stock_data(symbol):
    path = os.path.join(os.getcwd(), "NASDAQ_Data","Stock_data")

    if not os.path.exists(path):
        os.makedirs("NASDAQ_Data/Stock_data")

    end_date = datetime.datetime.now()
    if os.path.exists(os.path.join(path, symbol+".csv")):
    # print(end_date)
        old_df = pd.read_csv(os.path.join(path, symbol + ".csv"))
        # old_df["Date"] = pd.to_datetime(old_df["Date"])
        dates = old_df["Date"].tolist()
        print("CSV_date ",dates[0])
        last = (datetime.datetime.strptime(dates[0],"%Y-%m-%d").date())
        print("After Conversion ",type(last),last)
        # current_date = (end_date - datetime.timedelta(days = 1)).strftime("%Y-%m-%d")
        # print("Current Date",type(current_date),current_date)
        if end_date.strftime("%Y-%m-%d") <= str(last):
            print("{} Stock Data Fetched already".format(symbol))
            return
        else:
            stock_df = yf.download(symbol,
                                   start=last,
                                   end=end_date,
                                   progress=False,
                                   )
            stock_df.to_csv(os.path.join(path, symbol + "(1).csv"))
            new_df = pd.read_csv(os.path.join(path, symbol + "(1).csv"))
            # print(new_df)
            new_df["Date"] = pd.to_datetime(new_df["Date"]).dt.strftime('%Y-%m-%d')
            # new_df = new_df.drop(columns=["Unnamed: 13"], axis=1, errors='ignore')
            new_df = new_df.dropna(how='all')
            new_df = convert_date_to_unix_timestamp(new_df)
            res = pd.concat([old_df,new_df], ignore_index=True)
            res.drop_duplicates(subset="Date",keep='first', inplace=True)
            # print(res)
            res = res.sort_values(by=["Date"], ascending=False,ignore_index=True)
            res["Spread High-Low"] = res["High"] - res["Low"]
            res["Spread Close-Open"] = res["Close"] - res["Open"]
            os.remove(os.path.join(path, symbol + ".csv"))
            res.to_csv(os.path.join(path, symbol + ".csv"), index=False)
            os.remove(os.path.join(path, symbol + "(1).csv"))
            return res
    # print(start_date.year, start_date.month, start_date.day)
    else:
        start_date = end_date - datetime.timedelta(days=10 * 365)
        # current_date = (end_date - datetime.timedelta(days=2)).strftime("%Y-%m-%d")
        stock_df = yf.download(symbol,
                          start=start_date,
                          end=end_date,
                          progress=False,
                          )
        stock_df.to_csv(os.path.join(path, symbol + ".csv"))
        stock = pd.read_csv(os.path.join(path, symbol + ".csv"))
        stock.Date = pd.to_datetime(stock.Date, errors="coerce")
        # stock = stock.drop(columns=["Unnamed: 13"], axis=1, errors='ignore')
        stock = stock.dropna(how='all')
        stock["Spread High-Low"] = stock["High"] - stock["Low"]
        stock["Spread Close-Open"] = stock["Close"] - stock["Open"]
        stock = convert_date_to_unix_timestamp(stock)
        stock = stock.sort_values(by=["Date"], ascending=False,ignore_index=True)
        stock.to_csv(os.path.join(path, symbol + ".csv"), index=None)
        return stock
    # print(aapl_df.tail())

def download_actions(symbol):
    path = os.path.join(os.getcwd(), "NASDAQ_Data", "Corporate_actions")

    if not os.path.exists(path):
        os.makedirs("NASDAQ_Data/Corporate_actions")

    # end_date = datetime.date.today()
    # last_date = end_date - datetime.timedelta(days=2 * 365)
    # last_date = datetime.datetime.strptime(str(last_date),"%Y-%m-%d").date()
    ticker = yf.Ticker(symbol)
    stock_actions_dividends = ticker.actions["Dividends"]
    dividends_df = stock_actions_dividends.to_frame()
    # print(dividends_df)
    # dividends_df = dividends_df[dividends_df["Dividends"] != 0]
    dividends_path = os.path.join(path, "Dividends")
    if not os.path.exists(dividends_path):
        os.makedirs("NASDAQ_Data/Corporate_actions/Dividends")
    if os.path.exists(os.path.join(dividends_path, symbol + ".csv")):
        os.remove(os.path.join(dividends_path, symbol + ".csv"))
    dividends_df = dividends_df.sort_values(by=["Date"], ascending=False)
    dividends_df.to_csv(os.path.join(dividends_path, symbol + ".csv"))

    stock_actions_split = ticker.actions["Stock Splits"]
    stock_splits_df = stock_actions_split.to_frame()
    stock_splits_df = stock_splits_df[stock_splits_df["Stock Splits"] != 0]
    stock_split_path = os.path.join(path, "Stock_split")
    if not os.path.exists(stock_split_path):
        os.makedirs("NASDAQ_Data/Corporate_actions/Stock_split")
    if os.path.exists(os.path.join(stock_split_path, symbol + ".csv")):
        os.remove(os.path.join(stock_split_path, symbol + ".csv"))
    stock_splits_df.to_csv(os.path.join(stock_split_path, symbol + ".csv"))
    # print(stock_actions_split.tail())
    return

def download_eps_revenue_expenditure_profit_income(symbol):
    path = os.path.join(os.getcwd(), "NASDAQ_Data", "Revenue")

    if not os.path.exists(path):
        os.makedirs("NASDAQ_Data/Revenue")
    ticker = YahooFinancials(symbol)
    earnings_data = ticker.get_stock_earnings_data()
    # print(eps)
    try:
        eps_data = earnings_data[symbol]['earningsData']['quarterly']
        # print(eps_data)
        eps_list = []
        for each in eps_data:
            eps_list.append(each['actual'])
        eps_list.reverse()
    except:
        pass
    quarter_ticker = yf.Ticker(symbol)
    financials_df = quarter_ticker.quarterly_financials
    financials_df = financials_df.loc[["Total Revenue","Gross Profit","Net Income","Total Operating Expenses"]]
    # financials_df.loc["EPS"] = eps_list
    financials_df = financials_df.transpose()
    financials_df.index.name = 'Date'
    financials_df = financials_df.reset_index(level=0)
    financials_df["Date"] = financials_df["Date"].dt.strftime('%Y-%m-%d')
    cols = ["Total Revenue Gr","Gross Profit Gr","Net Income Gr","Total Operating Expenses Gr", "EPS Gr"]
    financials_df[cols] = pd.DataFrame([[0] * len(cols)], index=financials_df.index)
    quarter = []
    years = []

    rows = financials_df.shape[0]
    for row in range(rows):
        date = str(financials_df["Date"][row])
        month = datetime.datetime.strptime(date,"%Y-%m-%d").month
        year = datetime.datetime.strptime(date,"%Y-%m-%d").year
        years.append(year)
        q = ""
        if 1 <= month <= 3:
            q = "1"
        elif 4 <= month <= 6:
            q = "2"
        elif 6 <= month <= 9:
            q = "3"
        else:
            q = "4"
        quarter.append(q)
        if row < rows-1:
            today_values = np.array(financials_df.loc[row,["Total Revenue","Gross Profit","Net Income","Total Operating Expenses", "EPS"]].values)
            previous_day_values = np.array(financials_df.loc[row+1,["Total Revenue","Gross Profit","Net Income","Total Operating Expenses", "EPS"]].values)
            sub_val = np.subtract(today_values,previous_day_values)
            gr_val = (np.divide(sub_val,previous_day_values)).tolist()
            financials_df.loc[row,cols] = gr_val
    financials_df["Quarter"] = quarter
    financials_df["Year"] = years
    financials_df["Date"] = pd.to_datetime(financials_df["Date"])
    financials_df = financials_df.sort_values(by=["Year","Quarter"], ascending=False,ignore_index=True)
    if os.path.exists(os.path.join(path,symbol+".csv")):
        old_df = pd.read_csv(os.path.join(path,symbol+".csv"))
        old_df["Date"] = pd.to_datetime(old_df["Date"])
        # os.remove(os.path.join(path,symbol+".csv"))
        res = pd.concat([old_df, financials_df], ignore_index=True)
        res.drop_duplicates(subset="Date", keep='first', inplace=True)
        res = res.sort_values(by=["Year","Quarter"], ascending=False,ignore_index=True)
        os.remove(os.path.join(path, symbol + ".csv"))
        res.to_csv(os.path.join(path, symbol + ".csv"), index=False)
        return res
    else:
        financials_df.to_csv(os.path.join(path,symbol+".csv"),index=False)
        return financials_df

def caluculate_beta(symbol):
    ticker = yf.Ticker(symbol)
    print("Y-finance beta : ",ticker.info["beta"])
    return

def download_index_data():
    symbol = "^NDX"
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=10*365)
    if os.path.exists(os.path.join(path,"Index.csv")):
        old_df = pd.read_csv(os.path.join(path,"Index.csv"))
        old_df["Date"] = pd.to_datetime(old_df["Date"])
        last_date = old_df["Date"].dt.strftime('%Y-%m-%d').tolist()[0]
        # print(last_date)
        if str(last_date) == str(end_date.date):
            print("Index Data Fetched")
            return
        end_date = datetime.datetime.now()
        os.remove(os.path.join(path,"Index.csv"))
        new_df = yf.download(symbol,
                               start=last_date,
                               end=end_date,
                               progress=False,
                               )
        new_df.to_csv(os.path.join(path, "index.csv"))
        new_df = pd.read_csv(os.path.join(path, "index.csv"))
        new_df["Date"] = pd.to_datetime(new_df["Date"])
        res = pd.concat([old_df,new_df],ignore_index=True)
        res = res.sort_values(by=["Date"], ascending=False,ignore_index=True)
        res.drop_duplicates(subset=["Date"], keep="first", inplace=True)
        res[["Open", "High", "Low", "Close"]] = res[["Open", "High", "Low", "Close"]].apply(pd.to_numeric)
        res["% Return"] = ((res["Close"] / res['Close'].shift(1)) - 1) * 100
        res["% YTD"] = ((res.tail(1)['Close'].values[0] / res["Close"]) - 1) * 100
        os.remove(os.path.join(path, "index.csv"))
        res.to_csv(os.path.join(path, "Index.csv"), index=False)
        # print(res)
    else:
        ticker = yf.Ticker(symbol)
        index_df = yf.download(symbol,
                               start=start_date,
                               end=end_date,
                               progress=False,
                               )
        index_df.to_csv(os.path.join(path,"index.csv"))
        index_df = pd.read_csv(os.path.join(path,"index.csv"))
        index_df["Date"] = pd.to_datetime(index_df["Date"])
        index_df[["Open", "High", "Low", "Close"]] = index_df[["Open", "High", "Low", "Close"]].apply(pd.to_numeric)
        index_df.drop_duplicates(subset=["Date"], keep="first", inplace=True)
        index_df = index_df.sort_values(by=["Date"], ascending=False,ignore_index=True)
        index_df["% Return"] = ((index_df["Close"] / index_df['Close'].shift(1)) - 1) * 100
        index_df["% YTD"] = ((index_df.tail(1)['Close'].values[0] / index_df["Close"]) - 1) * 100
        os.remove(os.path.join(path, "index.csv"))
        index_df.to_csv(os.path.join(path,"Index.csv"), index=False)
        # print(index_df)
    return

if __name__ == "__main__":
    path = os.path.join(os.getcwd(), "NASDAQ_Data")
    df = pd.read_csv(os.path.join(path, "equity_data.csv"))
    download_index_data()
    sorted_df = df.sort_values(by=["Market Cap"], ascending=False)
    symbols = sorted_df["Symbol"].values.tolist()
    no_data = ['BABA','HD','HDB','HSBC','PBR','RIO','SNY','UL','DXCM','LLY','KO','ORCL','BHP','T','UNP','DEO','BP','BTI','CM']
    for symbol in symbols[:101]:
        print(symbol)
        # try:
        if symbol in no_data:
            continue
        download_stock_data(symbol)
        download_actions(symbol)
        # download_eps_revenue_expenditure_profit_income(symbol)
        # except:
        #     pass
    # print(download_stock_data("AAPL"))
    # download_actions("AAPL")
    # print(download_eps_revenue_expenditure_profit_income("AAPL"))
    # caluculate_beta("AAPL")
