import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import pandas as pd
import time
import datetime
from bs4 import BeautifulSoup
from selenium.webdriver.support.select import Select


def download_risk_free_rate():
    """
    Downloads the Risk Free Rate file.

    risk_free_rate_url = "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve&field_tdr_date_value=2022"

    creates the driver.
    opens the risk_free_rate_url.
    downloads the file.

    Methods:
    --------

    create_driver : creates the chrome driver.

    download : extracts the data from the page and saves to a csv file.

    """
    path = os.path.join(os.getcwd(), "NASDAQ_Data")
    risk_free_rate_url = "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve&field_tdr_date_value=2022"

    if not os.path.exists(path):
        os.makedirs("NASDAQ_Data")

    def create_driver():
        """
        Creates a Chrome Driver.

        Returns
        --------
        driver : driver
            chrome web driver.
        """

        chrome_options = Options()
        chrome_options.add_argument("start-maximized")
        chrome_options.add_argument("--headless")
        # chrome_options.add_argument('window-size=1920x1080')
        # user_agent = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.50 Safari/537.36'
        # chrome_options.add_argument(f'user-agent={user_agent}')
        chrome_options.add_experimental_option("prefs", {"download.default_directory": path})
        driver = webdriver.Chrome(
            executable_path=r'./chromedriver.exe', options=chrome_options)
        return driver

    def download():
        """
        downloads the risk free rate file.

        """

        href_class = driver.find_element(By.CLASS_NAME,'csv-feed.views-data-export-feed')
        href_elem = href_class.find_element(By.TAG_NAME,'a')
        href_elem.click()
        time.sleep(3)
        driver.quit()
        # print("New")
        risk_free_rate = pd.read_csv(os.path.join(path,'daily-treasury-rates.csv'))
        risk_free = risk_free_rate.loc[:,["Date", "3 Mo"]]
        risk_free["Date"] = pd.to_datetime(risk_free["Date"]).dt.strftime('%Y-%m-%d')
        risk_free.columns = ["Date", "Rate"]
        risk_free.dropna(inplace=True)
        # print(risk_free)
        if os.path.exists(os.path.join(path,"RiskFreeRateFull.csv")):
            # print("old")
            old_risk_free_rate = pd.read_csv(os.path.join(path, "RiskFreeRateFull.csv"))
            # print(old_risk_free_rate)
            new_df = pd.concat([old_risk_free_rate,risk_free],ignore_index=True)
            new_df.drop_duplicates(subset="Date", keep='first', inplace=True)
            new_df = new_df.sort_values(by=["Date"], ascending=False,ignore_index=True)
            # print("After merging")
            # print(new_df)
            os.remove(os.path.join(path,"RiskFreeRateFull.csv"))
            os.remove(os.path.join(path,'daily-treasury-rates.csv'))
            new_df = new_df.drop(columns=["Unnamed: 0.1"], axis=1, errors='ignore')
            new_df = new_df.drop(columns=["Unnamed: 0"], axis=1, errors='ignore')
            new_df.to_csv(os.path.join(path,"RiskFreeRateFull.csv"), index=None)
        else:
            risk_free = risk_free.drop(columns=["Unnamed: 0.1"], axis=1, errors='ignore')
            risk_free = risk_free.drop(columns=["Unnamed: 0"], axis=1, errors='ignore')
            risk_free.to_csv(os.path.join(path, "RiskFreeRateFull.csv"), index=None)
            os.remove(os.path.join(path,'daily-treasury-rates.csv'))
    year = 2022
    while (year>=2019):
        driver = create_driver()
        risk_free_rate_url = "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve&field_tdr_date_value="+str(year)
        driver.get(risk_free_rate_url)
        download()
        year -= 1


if __name__ == "__main__":
    download_risk_free_rate()