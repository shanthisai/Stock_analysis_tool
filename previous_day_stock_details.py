import pandas as pd
import os

def previous_day_stock_details(symbol,name,each_stock_df,prev_day_dict):
    # each_stock_df["Date"] = pd.to_datetime(each_stock_df["Date"])
    symbols_list = prev_day_dict.get("Symbol",[])
    symbols_list.append(symbol)
    names_list = prev_day_dict.get("Company",[])
    names_list.append(name)
    prev_day_dict["Symbol"] = symbols_list
    prev_day_dict["Company"] = names_list
    last_day_list = (each_stock_df.iloc[0].to_list())
    for i in range(len(last_day_list[:-1])):
        each_col_list = prev_day_dict.get(columns[i],[])
        each_col_list.append(last_day_list[i])
        prev_day_dict[columns[i]] = each_col_list
    # print(prev_day_dict)
    return prev_day_dict

path = os.path.join(os.getcwd(), "NASDAQ_Data")
df = pd.read_csv(os.path.join(path, "equity_data.csv"))
stock_data_path = os.path.join(path, "Stock_data")
sorted_df = df.sort_values(by=["Market Cap"], ascending=False)
symbols = sorted_df["Symbol"].values.tolist()
names = df['Name'].values.tolist()[:101]
flag = True
prev_day_dict = {}
no_data = ['BABA','HD','HDB','HSBC','PBR','RIO','SNY','UL','DXCM','LLY','KO','ORCL','BHP','T','UNP','DEO','BP','BTI','CM']
for i in range(len(symbols[:101])):
    print(symbols[i])
    if symbols[i] in no_data:
        continue
    else:
        each_stock_df = pd.read_csv(os.path.join(stock_data_path, symbols[i]+".csv"))
        columns = each_stock_df.columns
        columns = columns[:-1]
        if flag:
            prev_day_dict = previous_day_stock_details(symbols[i],names[i],each_stock_df,prev_day_dict)
            flag = False
        else:
            prev_day_dict = previous_day_stock_details(symbols[i],names[i],each_stock_df,prev_day_dict)
prev_df = pd.DataFrame.from_dict(prev_day_dict)
if os.path.exists(os.path.join(path,"previous_day_stock_details.csv")):
    os.remove(os.path.join(path,"previous_day_stock_details.csv"))
prev_df.to_csv(os.path.join(path,"previous_day_stock_details.csv"))
