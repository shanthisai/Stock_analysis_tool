import os
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import pandas as pd
import time
from selenium.webdriver.support.wait import WebDriverWait


def download_equity():
    path = os.path.join(os.getcwd(), "NASDAQ_Data")

    security_url = "https://www.nasdaq.com/market-activity/stocks/screener"

    # if os.path.exists(os.path.join(path, "Equity.csv")):
    #     print("Equity.csv exists")
    #     return

    chrome_options = Options()
    # chrome_options.add_argument("start-maximized")
    chrome_options.add_argument("--headless")
    chrome_options.add_argument('window-size=1920x1080')
    user_agent = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.50 Safari/537.36'
    chrome_options.add_argument(f'user-agent={user_agent}')
    chrome_options.add_experimental_option("prefs", {"download.default_directory": path})
    driver = webdriver.Chrome(
        executable_path=r'./chromedriver.exe', options=chrome_options)
    driver.maximize_window()
    driver.get(security_url)
    driver.find_element(By.XPATH,".//*[contains(text(),'Mega (>$200B)')]").click()
    driver.find_element(By.XPATH,".//*[contains(text(),'Large ($10B-$200B)')]").click()
    # mega_checkbox = driver.find_element(By.XPATH,"//label/span*[contains(text(),'Mega (>$200B)')]").click()
    # driver.execute_script("arguments[0].click();", mega_checkbox)
    # large_checkbox = driver.find_element(By.XPATH,"//label/span*[contains(text(),'Large ($10B-$200B))')]").click()
    # driver.execute_script("arguments[0].click();", large_checkbox)
    # WebDriverWait(driver, 5).until(EC.element_to_be_clickable((By.CLASS_NAME, 'nasdaq-screener__form-button--apply')))
    element = driver.find_element(By.CLASS_NAME,'nasdaq-screener__form-button--apply')
    driver.execute_script("arguments[0].click();", element)
    time.sleep(3)
    download_element = driver.find_element(By.CLASS_NAME,"nasdaq-screener__download").click()
    # driver.execute_script("arguments[0].click();",download_element)
    time.sleep(3)
    driver.close()

if __name__ == "__main__":
    path = os.path.join(os.getcwd(), "NASDAQ_Data")
    if not os.path.exists(path):
        os.makedirs("NASDAQ_Data")
    dir_list = os.listdir(path)
    for file in dir_list:
        if file.startswith("equity"):
            os.remove(os.path.join(path,"equity_data.csv"))
   
    download_equity()
    dir_list = os.listdir(path)
    for file in dir_list:
        if file.startswith("nasdaq"):
            old_file = os.path.join(path, file)
            new_file = os.path.join(path, "equity_data.csv")
            os.rename(old_file, new_file)
    df = pd.read_csv(os.path.join(path, "equity_data.csv"))
    sorted_df = df.sort_values(by=["Market Cap"], ascending=False)
    os.remove(os.path.join(path,"equity_data.csv"))
    sorted_df.to_csv(os.path.join(path,"equity_data.csv"),index=False)
