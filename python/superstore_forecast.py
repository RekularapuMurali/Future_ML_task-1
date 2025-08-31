import warnings
warnings.filterwarnings("ignore")
print("Script started...")

import pandas as pd, numpy as np, matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

SRC="Sample - Superstore.csv"
FREQ="MS"
HORIZON=12
TEST_MONTHS=6

df=pd.read_csv(SRC,encoding="ISO-8859-1",parse_dates=["Order Date"])
s=df.groupby(pd.Grouper(key="Order Date",freq=FREQ))["Sales"].sum().asfreq(FREQ).fillna(0)
p=s.reset_index().rename(columns={"Order Date":"ds","Sales":"y"})

split=len(p)-TEST_MONTHS
train=p.iloc[:split]
test=p.iloc[split:]

m=Prophet(yearly_seasonality=True,weekly_seasonality=False,daily_seasonality=False)
m.add_country_holidays(country_name="US")
m.fit(train)

future=m.make_future_dataframe(periods=HORIZON,freq=FREQ)
fc=m.predict(future)[["ds","yhat","yhat_lower","yhat_upper"]]

hist=p.merge(fc,how="left",on="ds")
hist.rename(columns={"ds":"date","y":"actual","yhat":"forecast"},inplace=True)

test_merge=test.merge(fc,how="left",on="ds")
mae=float(mean_absolute_error(test_merge["y"],test_merge["yhat"]))
from sklearn.metrics import mean_squared_error
import numpy as np

rmse = np.sqrt(mean_squared_error(test_merge["y"], test_merge["yhat"]))

pd.DataFrame([{"mae":mae,"rmse":rmse,"horizon_months":HORIZON,"freq":FREQ}]).to_csv("metrics.csv",index=False)

out=hist.copy()
out.loc[out["forecast"].isna(),"forecast"]=out["actual"]
out.to_csv("forecast_output.csv",index=False)

plt.figure()
plt.plot(out["date"],out["actual"])
plt.plot(out["date"],out["forecast"])
plt.title("Superstore Sales: Actual vs Forecast (Monthly)")
plt.xlabel("Date"); plt.ylabel("Sales")
plt.tight_layout(); plt.savefig("actual_vs_forecast.png"); plt.close()

for comp in ["trend","weekly","yearly"]:
    if comp in fc:
        plt.figure(); plt.plot(fc["ds"],fc[comp]); plt.title(comp.title())
        plt.tight_layout(); plt.savefig(f"component_{comp}.png"); plt.close()

print("Script finished successfully.")

