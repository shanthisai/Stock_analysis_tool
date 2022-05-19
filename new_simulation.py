import os
import pandas as pd
import traceback
import datetime
import re
import ray
import numpy as np


def simulation(df, investment, days, i):
    invest = False
    shares = 0
    df['date'] = pd.to_datetime(df['date'])
    start = df.iloc[0]['date'] - datetime.timedelta(days=i)
    end = start - datetime.timedelta(days=days)
    refdf = df[df['date'].between(end, start)]
    refdf = refdf.sort_values(by=["date"], ascending=[True])
    simulation_result = []
    for _, row in refdf.iterrows():
        if row["invest"]:
            if not invest:
                if investment < row['close']:
                    break
                shares = int(investment / row['close'])
                invested = shares * row['close']
                investment = investment - invested
                invest = True
                res = {"investment": invested, "shares": shares,
                       "entry": True, "exit": False, "date": row["date"].strftime("%d-%m-%Y"), "close": row["close"],
                       "predictedub": row["predicted ub"]}
                simulation_result.append(res)
        if row['exit']:
            if invest:
                investment = investment + shares * row['close']
                res = {"investment": investment, "shares": shares,
                       "entry": False, "exit": True, "date": row["date"].strftime("%d-%m-%Y"), "close": row["close"],
                       "predictedub": row["predicted ub"]}
                simulation_result.append(res)
                invest = False

    else:
        if invest and not row['invest']:
            investment = investment + shares * row['close']
            invest = False
            res = {"investment": investment, "shares": shares,
                   "entry": False, "exit": True, "date": row["date"].strftime("%d-%m-%Y"), "close": row["close"],
                   "predictedub": row["predicted ub"]}
            simulation_result.append(res)
    if len(simulation_result) < 2:
        return None
    returns = []
    for i in range(0, len(simulation_result), 2):
        try:
            a = simulation_result[i]['investment']
            b = simulation_result[i + 1]['investment']
            r = ((b - a) / a)
            returns.append(r)
        except:
            pass
    try:
        average_return_percent = sum(returns) / len(returns)
        return {"average_return_percent": average_return_percent, "simulation_result": simulation_result}
    except:
        pass


# @ray.remote
def simulate(symbol, days, company):
    try:
        print(symbol, company)
        df = pd.read_csv(os.path.join(simpath, symbol + "_" + str(days) + ".csv"))
        df['date'] = pd.to_datetime(df['date'])
        df = df.iloc[::-1]
        df = df.reset_index(drop=True)
        result = []
        investment = 100000
        actstart = df.iloc[0]['date']
        actend = df.iloc[-1]['date']
        for i in range((actstart - actend).days):
            start = df.iloc[0]['date'] - datetime.timedelta(days=i)
            end = start - datetime.timedelta(days=days)
            if (end - actend).days <= 0:
                break
            res = simulation(df, investment, days, i)
            if res != None:
                result.append(res)

        if result == []:
            return None

        rows = []
        for res in result:
            curdf = pd.DataFrame(res["simulation_result"])
            curdf["actual_returns"] = curdf["investment"]
            curdf["predicted_returns"] = curdf["predictedub"] * \
                                         curdf["shares"] * curdf["close"]
            curdf["actual_returns"] = curdf.apply(
                lambda row: None if row["entry"] else row["actual_returns"], axis=1)
            curdf["predicted_returns"] = curdf.apply(
                lambda row: None if row["exit"] else row["predicted_returns"], axis=1)
            curdf["actual_returns_percent"] = (
                                                      curdf["actual_returns"].shift(-1) - curdf["investment"]) / curdf[
                                                  "investment"]
            curdf["predicted_returns_percent"] = (
                                                         curdf["predicted_returns"] - curdf["investment"]) / curdf[
                                                     "investment"]
            curdf["returns_percent_diff"] = curdf["predicted_returns_percent"] - \
                                            curdf["actual_returns_percent"]
            rows.append([curdf["actual_returns_percent"].mean(),
                         curdf["predicted_returns_percent"].mean()])

        soldf = pd.DataFrame(
            rows, columns=["actual_returns_percent", "predicted_returns_percent"])
        soldf["returns_percent_diff"] = soldf["predicted_returns_percent"] - \
                                        soldf["actual_returns_percent"]
        posmean = soldf[soldf["returns_percent_diff"]
                        > 0]["returns_percent_diff"].mean()
        negmean = soldf[soldf["returns_percent_diff"]
                        < 0]["returns_percent_diff"].mean()
        actmean = soldf["actual_returns_percent"].mean()
        predmean = soldf["predicted_returns_percent"].mean()
        left, right = (1 + negmean) * actmean, (1 + posmean) * actmean
        return [symbol, company, actmean, predmean, left, right]
    except:
        return None


def create_top_file(days):
    spath = "simres_{}.csv".format(days)
    df = pd.read_csv(os.path.join(toppath, spath))
    df["minimum"] = df.apply(lambda row: row["actual"] if np.isnan(row["minimum"]) else row["minimum"], axis=1)
    df["maximum"] = df.apply(lambda row: row["actual"] if np.isnan(row["maximum"]) else row["maximum"], axis=1)
    df[["minimum", "maximum"]] = df.apply(lambda row: pd.Series(
        [np.nanmin([row["minimum"], row["maximum"]]), np.nanmax([row["minimum"], row["maximum"]])]), axis=1)
    df["min"] = df["actual"] - df["minimum"]
    df["max"] = df["maximum"] - df["actual"]
    df["suggest"] = df.apply(
        lambda row: "buy" if row["minimum"] <= row["actual"] <= row["minimum"] * 1.15 else "sell" if row[
                                                                                                         "maximum"] * 0.85 <=
                                                                                                     row["actual"] <=
                                                                                                     row[
                                                                                                         "maximum"] else None,
        axis=1)
    sell = df.sort_values(by=["max"], ascending=[True])
    buy = df.sort_values(by=["min"], ascending=[True])
    df.to_csv(os.path.join(toppath, spath), index=None)
    buy.to_csv(os.path.join(toppath, "buy_" + str(days) + ".csv"), index=None)
    sell.to_csv(os.path.join(toppath, "sell_" + str(days) + ".csv"), index=None)


npath = os.path.join(os.getcwd(),"NASDAQ_Data")
simpath = os.path.join(os.getcwd(), "NASDAQ_Data", "Simulation")
simrespath = os.path.join(os.getcwd(), "NASDAQ_Data", "SimulationResult")
toppath = os.path.join(os.getcwd(), "NASDAQ_Data", "Top")

if not os.path.exists(toppath):
    os.makedirs(toppath)

# ray.init(ignore_reinit_error=True)
result = []


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

for d in [30, 60, 90, 180, 360, 720, 1080]:
    days = d
    for i in range(len(symbols)):
        symbol = symbols[i]
        name = names[i]
        try:
            result.append(simulate(symbol, days, name))
        except:
            traceback.print_exc()
    try:
        simres = result
        simres = [res for res in simres if res is not None]
        if simres != []:
            columns = ["symbol", "company", "actual",
                       "predicted", "minimum", "maximum"]
            simdf = pd.DataFrame(simres, columns=columns)
            simdf = simdf.sort_values(by=["actual"], ascending=[False])
            simdf.to_csv(os.path.join(toppath, "simres" + "_" + str(days) + ".csv"), index=None)
            create_top_file(days)
    except Exception as e:
        print(e)