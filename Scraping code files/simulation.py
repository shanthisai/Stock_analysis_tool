import os
import pandas as pd
import datetime
import re
import traceback
import time
pd.options.mode.chained_assignment = None

def simulation(df, investment, days):
    invest = False
    shares = 0
    df['date'] = pd.to_datetime(df['date'])
    start = df.iloc[-1]['date']
    end = start - datetime.timedelta(days=days)
    refdf = df[df['date'].between(end, start)]
    refdf = refdf.sort_values(by=["date"],ascending=[True])
    simulation_result = []
    for _, row in refdf.iterrows():
        if row["invest"]:
            if invest is False:
                if investment < row['close']:
                    break
                shares = int(investment / row['close'])
                invested = shares * row['close']
                investment = investment - invested
                invest = True
                res = {"investment": invested, "shares": shares,
                       "entry": True, "exit": False, "date": row["date"].strftime("%d-%m-%Y"), "close": row["close"]}
                simulation_result.append(res)
        if row['exit']:
            if invest:
                investment = investment + shares * row['close']
                res = {"investment": investment, "shares": shares,
                       "entry": False, "exit": True, "date": row["date"].strftime("%d-%m-%Y"), "close": row["close"]}
                simulation_result.append(res)
                invest = False

    else:
        if invest and not row['invest']:
            investment = investment + shares * row['close']
            invest = False
            res = {"investment": investment, "shares": shares,
                   "entry": False, "exit": True, "date": row["date"].strftime("%d-%m-%Y"), "close": row["close"]}
            simulation_result.append(res)
    returns = []
    for i in range(0, len(simulation_result), 2):
        try:
            a = simulation_result[i]['investment']
            b = simulation_result[i+1]['investment']
            r = ((b-a)/a)
            returns.append(r)
        except:
            pass
    try:
        average_return_percent = sum(returns)/len(returns)
        return {"average_return_percent": average_return_percent, "simulation_result": simulation_result}
    except:
        return None

def simulate(investment, days):
    topreturns = []
    for i in range(len(symbols)):
        try:
            symbol = (symbols[i])
            company = names[i]
            print(symbol,company)
            spath = symbol + "_" + str(days) + ".csv"
            df = pd.read_csv(os.path.join(simpath, spath))
            result = simulation(df, investment, days)
            if result == None:
                continue
            result.update({"company": company})
            result.update({"symbol": symbol})
            topreturns.append(result)
            simdf = pd.DataFrame(result["simulation_result"])
            simdf.to_csv(os.path.join(simrespath, spath), index=None)
        except :
            pass
    if topreturns == []:
        return
    cols = ["company","symbol","average_return_percent","simulation_result"]
    topreturnscompanies = pd.DataFrame(topreturns)
    topreturnscompanies = topreturnscompanies[cols]
    topreturnscompanies = topreturnscompanies.sort_values(by=["average_return_percent"], ascending=[False])
    if os.path.exists(os.path.join(toppath, "sim_" + str(days)+".csv")):
        os.remove(os.path.join(toppath, "sim_" + str(days)+".csv"))
        topreturnscompanies.to_csv(os.path.join(toppath, "sim_" + str(days)+".csv"), index=None)
    else:
        topreturnscompanies.to_csv(os.path.join(toppath, "sim_" + str(days)+".csv"), index=None)

# sp500 = pd.read_csv(os.path.join(os.getcwd(), "Data",
                    # "SP500companies.csv")).set_index("Security Code")

npath = os.path.join(os.getcwd(),"NASDAQ_Data")
simpath = os.path.join(os.getcwd(), "NASDAQ_Data", "Simulation")
simrespath = os.path.join(os.getcwd(), "NASDAQ_Data", "SimulationResult")
toppath = os.path.join(os.getcwd(), "NASDAQ_Data", "Top")

if not os.path.exists(simrespath):
    os.makedirs(simrespath)
if not os.path.exists(toppath):
    os.makedirs(toppath)

investment = 100000
df = pd.read_csv(os.path.join(npath, "equity_data.csv"))
# download_index_data()
sorted_df = df.sort_values(by=["Market Cap"], ascending=False)
symbols = sorted_df["Symbol"].values.tolist()
names = sorted_df["Name"].values.tolist()
names = names[:101]
symbols = symbols[:101]
no_data = ["HDB","RIO","SNY","UL","DEO","CVX","BHP","T"]
for i in no_data:
    if i in symbols:
        sym_index = symbols.index(i)
        symbols.remove(i)
        names.remove(names[sym_index])
for days in [30, 60, 90, 180, 360, 720, 1080]:
    try:
        print(days)
        simulate(investment, days)
    except:
        traceback.print_exc()
    time.sleep(15)