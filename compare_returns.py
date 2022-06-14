import pandas as pd
# import numpy as np
import os
# import datetime

def calculate_idol_returns(file,symbol,returns_dict):
    # print(file)
    stock_df = pd.read_csv(os.path.join(simpath,file))
    stock_df = stock_df.tail(180)
    # stock_df["date"] = pd.to_datetime(stock_df["date"])
    invest_date = stock_df['date'].iloc[0]
    close_invest = stock_df['close'].iloc[0]
    exit_date = stock_df['date'].iloc[-1]
    close_exit = stock_df["close"].iloc[-1]
    investment = 100000
    shares_bought = int(investment/close_invest)
    invested = shares_bought*close_invest
    # print(exit_date,invested)
    returns = shares_bought*close_exit
    idol_returns = float(returns - invested)
    # returns_dict = {'Symbol':symbol,'Invested On':invest_date,'Close Price on Invest':close_invest,'Exit On':exit_date,'Close Price on Exit': close_exit,'Investment':investment,'No of Shares':shares_bought,'Idol Returns':idol_returns}
    symbol_list = returns_dict.get('Symbol',[])
    symbol_list.append(symbol)
    returns_dict["Symbol"] = symbol_list
    invest_date_list = returns_dict.get('Idol Invested On', [])
    invest_date_list.append(invest_date)
    returns_dict['Idol Invested On'] = invest_date_list
    invest_cp_list = returns_dict.get('Idol Close Price on Invest', [])
    invest_cp_list.append(close_invest)
    returns_dict['Idol Close Price on Invest'] = invest_cp_list
    exit_date_list = returns_dict.get('Idol Exit On', [])
    exit_date_list.append(exit_date)
    returns_dict["Idol Exit On"] = exit_date_list
    exit_cp_list = returns_dict.get('Idol Close Price on Exit', [])
    exit_cp_list.append(close_exit)
    returns_dict["Idol Close Price on Exit"] = exit_cp_list
    investment_list = returns_dict.get('Idol Investment', [])
    investment_list.append(invested)
    returns_dict["Idol Investment"] = investment_list
    shares_list = returns_dict.get('Idol No of Shares', [])
    shares_list.append(shares_bought)
    returns_dict["Idol No of Shares"] = shares_list
    idol_list = returns_dict.get('Idol Returns', [])
    idol_list.append(idol_returns)
    returns_dict["Idol Returns"] = idol_list
    # print(returns_dict)
    return returns_dict

def calculate_actual_returns(file,symbol,returns_dict):
    stock_df = pd.read_csv(os.path.join(simpath,file))
    stock_df = stock_df.tail(180)
    single_investment = 100000
    total_investment = 0
    shares_bought = []
    total_shares_bought = 0
    remained_amount = 0
    remained_shares = 0
    invested_dates_list_per_stock = []
    exit_dates_list_per_stock = []
    inv_close_list_per_stock = []
    ex_close_list_per_stock = []
    exit_flag = False
    actual_returns = 0
    invested = 0
    invested_list = []
    for index,row in stock_df.iterrows():
        if row['invest']:
            exit_flag = False
            inv_date = row['date']
            invested_dates_list_per_stock.append(inv_date)
            inv_close = row['close']
            inv_close_list_per_stock.append(inv_close)
            shares = int(single_investment/inv_close)
            total_shares_bought += shares
            remained_shares += shares
            shares_bought.append(shares)
            invested = shares*inv_close
            invested_list.append(invested)
            total_investment += invested
            remained_amount += invested
        if row['exit']:
            if len(invested_dates_list_per_stock) == 0:
                continue
            exit_flag = True
            ex_date = row['date']
            exit_dates_list_per_stock.append(ex_date)
            ex_close = row['close']
            ex_close_list_per_stock.append(ex_close)
            remained_amount = 0
            remained_shares = 0
            returns = total_shares_bought*ex_close
            actual_returns = returns - total_investment
    # symbol_list = returns_dict.get('Symbol', [])
    # symbol_list.append(symbol)
    # returns_dict["Symbol"] = symbol_list
    invest_date_list = returns_dict.get('Actual Invested On', [])
    invest_date_list.append(invested_dates_list_per_stock)
    returns_dict['Actual Invested On'] = invest_date_list
    invest_cp_list = returns_dict.get('Actual Close Price on Invest', [])
    invest_cp_list.append(inv_close_list_per_stock)
    returns_dict['Actual Close Price on Invest'] = invest_cp_list
    exit_date_list = returns_dict.get('Actual Exit On', [])
    exit_date_list.append(exit_dates_list_per_stock)
    returns_dict["Actual Exit On"] = exit_date_list
    exit_cp_list = returns_dict.get('Actual Close Price on Exit', [])
    exit_cp_list.append(ex_close_list_per_stock)
    returns_dict["Actual Close Price on Exit"] = exit_cp_list
    investment_list = returns_dict.get('Actual Investment', [])
    investment_list.append(invested_list)
    returns_dict["Actual Investment"] = investment_list
    shares_list = returns_dict.get('Actual No of Shares', [])
    shares_list.append(shares_bought)
    returns_dict["Actual No of Shares"] = shares_list
    actual_list = returns_dict.get('Actual Returns', [])
    actual_list.append(actual_returns)
    returns_dict["Actual Returns"] = actual_list
    total_investment_list = returns_dict.get('Actual Total Investment', [])
    total_investment_list.append(total_investment)
    returns_dict["Actual Total Investment"] = total_investment_list
    if not exit_flag:
        unsold_shares_list = returns_dict.get('Unsold Shares',[])
        unsold_shares_list.append(remained_shares)
        returns_dict['Unsold Shares'] = unsold_shares_list
        remained_amount_list = returns_dict.get('Remained Amount',[])
        remained_amount_list.append(remained_amount)
        returns_dict["Remained Amount"] = remained_amount_list
    else:
        unsold_shares_list = returns_dict.get('Unsold Shares', [])
        unsold_shares_list.append([])
        returns_dict['Unsold Shares'] = unsold_shares_list
        remained_amount_list = returns_dict.get('Remained Amount', [])
        remained_amount_list.append([])
        returns_dict["Remained Amount"] = remained_amount_list
    return returns_dict

npath = os.path.join(os.getcwd(),"NASDAQ_Data")
simpath = os.path.join(npath,"Simulation")

equity_data = pd.read_csv(os.path.join(npath,"equity_data.csv"))
sorted_df = equity_data.sort_values(by=["Market Cap"], ascending=False)
symbols = sorted_df["Symbol"].values.tolist()
symbols = symbols[:101]
no_data = ['BABA','HD','HDB','HSBC','PBR','RIO','SNY','UL','DXCM','LLY','KO','ORCL','BHP','T','UNP','DEO','BP','BTI','CM']
for i in no_data:
    if i in symbols:
        symbols.remove(i)
print(len(symbols))
files = os.listdir(simpath)
print("Files",len(files))
returns_dict = {}
for file in files:
    symbol = file.split("_")[0]
    if symbol not in no_data:
        if "_30" not in file:
            continue
        else:
            # print(symbol)
            returns_dict = calculate_idol_returns(file,symbol,returns_dict)
            returns_dict = calculate_actual_returns(file,symbol,returns_dict)
# for key in returns_dict:
#     print(key)
#     print(len(returns_dict[key]))
return_df = pd.DataFrame.from_dict(returns_dict)
return_df.to_csv(os.path.join(npath,'Compare_Returns.csv'),index=None)