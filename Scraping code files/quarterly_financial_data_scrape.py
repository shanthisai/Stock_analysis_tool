import os
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time
from sklearn.impute import KNNImputer

download_path = os.path.join(os.getcwd(),"NASDAQ_Data","Revenue")
if not os.path.exists(download_path):
    os.makedirs("NASDAQ_Data/Revenue")
basic_url = 'https://www.macrotrends.net'

def create_driver():

    chrome_options = Options()
    chrome_options.add_argument("start-maximized")
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--log-level=3")
    # chrome_options.add_argument('--no-sandbox')
    # chrome_options.add_argument('--disable-dev-shm-usage')
    user_agent = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.50 Safari/537.36'
    chrome_options.add_argument(f'user-agent={user_agent}')
    driver = webdriver.Chrome(executable_path= r'./chromedriver.exe',options=chrome_options)
    return driver

def download_financial_data(name,symbol):
    driver = create_driver()
    driver.get(basic_url)
    input_bar = driver.find_element(By.CLASS_NAME,"js-typeahead")
    if len(symbol) < 3:
        input_bar.send_keys(name)
    else:
        input_bar.send_keys(symbol)
    time.sleep(5)
    drop_down_div = driver.find_element(By.CLASS_NAME,'typeahead__container.cancel.backdrop.result')
    drop_down_ul = drop_down_div.find_element(By.CLASS_NAME,"typeahead__list")
    li_list = drop_down_ul.find_elements(By.TAG_NAME,"li")
    income_statement_href = li_list[6].find_element(By.TAG_NAME,"a").get_attribute("href")
    income_statement_href = income_statement_href+"?freq=Q"
    driver.get(income_statement_href)
    table_content_elem = driver.find_element(By.ID,"contenttablejqxgrid")
    data_elems = table_content_elem.find_elements(By.TAG_NAME,'div')
    col_names = ["Total Revenue","Gross Profit","Net Income","Total Operating Expenses", "EPS"]
    required_info = ["revenue","gross-profit","net-income","operating-expenses","eps-earnings-per-share-diluted"]
    basic_href = data_elems[0].find_element(By.LINK_TEXT,"Revenue").get_attribute("href")
    # print("Basic Href : ", basic_href)
    flag = True
    date_flag = True
    financial_data = {}
    for j in range(len(required_info)):
        data_type = financial_data.get(col_names[j],[])
        dates = []
        quarters = []
        if flag:
            current_href = basic_href
            flag = False
        else:
            current_href = basic_href.replace("revenue",required_info[j])
        # print("Current Href : ",current_href)
        data_driver = create_driver()
        data_driver.get(current_href)
        timeout = 30
        WebDriverWait(data_driver, timeout).until(EC.visibility_of_element_located((By.ID, 'style-1')))
        table_elem = data_driver.find_element(By.ID,"style-1")
        div_elems = table_elem.find_elements(By.TAG_NAME,"div")
        text_data = (div_elems[1].text).split("\n")
        if required_info[j] == 'eps-earnings-per-share-diluted':
            text_data = text_data[1:]
        else:
            text_data = text_data[2:]
        for row in text_data:
            row = row.split(" ")
            if date_flag:
                dates.append(row[0])
                quarter = int(row[0][5:7])
                q = 1 if 1 <= quarter <= 3 else 2 if 4 <= quarter <= 6 else 3 if 6 <= quarter <= 9 else 4
                quarters.append(q)
            try:
                if "$0.00" == row[1]:
                    data_type.append(0)
                else:
                    data_type.append(row[1].replace("$","").replace(",",""))
            except IndexError:
                data_type.append(0)
        if date_flag:
            financial_data["Date"] = dates
            financial_data["Quarter"] = quarters
        date_flag = False
        financial_data[col_names[j]] = data_type
        data_driver.close()
    # print("Result : ",financial_data)
    financial_df = pd.DataFrame(data=financial_data)
    # print("@@@@@@@@@@@@@@@@@@@@ Initial @@@@@@@@@@@@@@@@@@@@@@@@")
    # print(financial_df)
    financial_df["Date"] = pd.to_datetime(financial_df["Date"])
    financial_df[col_names] = financial_df[col_names].apply(pd.to_numeric, errors="coerce")
    # print("@@@@@@@@@@@@@@@@@@@@ After changing data type @@@@@@@@@@@@@@@@@@@@@@@@")
    # print(financial_df.head(10))
    rows = financial_df.shape[0]
    for row in range(rows):
        if row == 0 or row == rows-1:
            for col in col_names:
                required_values = 0
                if financial_df[col][row] == 0:
                    if row == 0:
                        required_values = financial_df[col][row+1]+financial_df[col][row+2] ##### Lower rows #####
                    if row == row-1:
                        required_values = financial_df[col][row-1]+financial_df[col][row-2] ##### Upper rows #####
                    required_val = required_values/2
                    financial_df[col][row] = required_val
        if row == 1 or row == rows-2:
            for col in col_names:
                if financial_df[col][row] == 0:
                    upper_values = 0
                    lower_values = 0
                    if row == 1:
                        upper_values = financial_df[col][row-1]
                        lower_values = financial_df[col][row+1]+financial_df[col][row+2]
                    if row == rows-2:
                        upper_values = financial_df[col][row - 1] + financial_df[col][row - 2]
                        lower_values = financial_df[col][row + 1]
                    required_val = (upper_values+lower_values)/3
                    financial_df[col][row] = required_val
        if 1 < row <= rows-3:
            for col in col_names:
                if financial_df[col][row] == 0:
                    upper_values = financial_df[col][row-1]+financial_df[col][row-2]
                    lower_values = financial_df[col][row+1]+financial_df[col][row+2]
                    required_val = (upper_values+lower_values)/4
                    financial_df[col][row] = required_val
    # print("@@@@@@@@@@@@@@@@@@@@ After changing data types and null values @@@@@@@@@@@@@@@@@@@@@@@@")
    # print(financial_df)
    cols = ["Total Revenue Gr", "Gross Profit Gr", "Net Income Gr", "Total Operating Expenses Gr", "EPS Gr"]
    financial_df[cols] = pd.DataFrame([[0] * len(cols)], index=financial_df.index)
    rows = financial_df.shape[0]
    years = []
    for row in range(rows):
        years.append(financial_df.iloc[row]["Date"].year)
        if row < rows-1:
            today_values = np.array(financial_df.loc[row,col_names].values)
            previous_day_values = np.array(financial_df.loc[row+1,col_names].values)
            sub_val = np.subtract(today_values,previous_day_values)
            gr_val = (np.divide(sub_val,previous_day_values)).tolist()
            financial_df.loc[row,cols] = gr_val
    # print("@@@@@@@@@@@@@@@@@@@@ Final @@@@@@@@@@@@@@@@@@@@@@@@")
    # print(financial_df.head(10))
    financial_df["Year"] = years
    financial_df.to_csv(os.path.join(download_path,symbol+".csv"),index=False)
    return financial_df
    # return

if __name__ == "__main__":
    path = os.path.join(os.getcwd(),"NASDAQ_Data")
    df = pd.read_csv(os.path.join(path, "equity_data.csv"))
    # download_index_data()
    sorted_df = df.sort_values(by=["Market Cap"], ascending=False)
    symbols = sorted_df["Symbol"].values.tolist()
    names = sorted_df["Name"].values.tolist()
    symbols = symbols[:101]
    names = names[:101]
    undone_stocks = {}
    undone_names = []
    undone_symbols = []
    for i in range(len(names)):
        print("###############################################")
        print("Index : ",i," ----- Name : {}".format(names[i]),symbols[i])
        print("###############################################")
        name = names[i].split(" ")[0]
        symbol = symbols[i]
        try:
            download_financial_data(name,symbol)
        except Exception as e:
            print("Exception as : ",e)
            undone_symbols.append(symbol)
            undone_names.append(name)
        time.sleep(5)
    undone_stocks["Symbol"] = undone_symbols
    undone_stocks["Name"] = undone_names
    pd.DataFrame(data=undone_stocks).to_csv(os.path.join(download_path,"Undone.csv"))
    # li = ['TGT','BUD']
    # for i in li:
    #     download_financial_data("Estee",i)