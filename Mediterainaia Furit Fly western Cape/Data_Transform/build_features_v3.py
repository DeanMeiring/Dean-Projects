"""
Build the v3 biological and wind features, one row per region per day.

  Degree-days   : base 9.6 C, the lower developmental threshold for South African
                  C. capitata (Grout & Stoltz, 2007). Rolling 30/60/90-day sums plus a
                  season-to-date sum from 1 July (start of the southern spring).
  Chill portions: Dynamic Model (Fishman et al., 1987; Erez et al., 1990) run on hourly
                  temperature. Season-to-date from 1 March (start of the chill season)
                  and a 30-day rolling sum. Implementation follows chillR's Dynamic_Model.
  Cape Doctor   : strong south-easterly wind. A day counts when the dominant direction is
                  90-180 degrees and the max wind speed is in the top 25% for that region.

Inputs : wc_v3_hourly_temp.csv, wc_v3_daily_wind.csv, wc_v2_weather_history.csv
Output : wc_v3_daily_features.csv
"""
import os

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HOURLY = os.path.join(SCRIPT_DIR, "wc_v3_hourly_temp.csv")
WIND = os.path.join(SCRIPT_DIR, "wc_v3_daily_wind.csv")
DAILY_WX = os.path.join(SCRIPT_DIR, "wc_v2_weather_history.csv")
OUT = os.path.join(SCRIPT_DIR, "wc_v3_daily_features.csv")

GDD_BASE = 9.6
SE_MIN_DEG, SE_MAX_DEG, SE_SPEED_PCTL = 90, 180, 0.75


def dynamic_model_portions(temp_c):
    """Chill portions gained in each hour (Dynamic Model; same constants as chillR)."""
    e0, e1, a0, a1 = 4153.5, 12888.8, 139500.0, 2.567e18
    slp, tetmlt = 1.6, 277.0
    aa, ee = a0 / a1, e1 - e0

    tk = np.asarray(temp_c, dtype=float) + 273.0
    sr = np.exp(slp * tetmlt * (tk - tetmlt) / tk)
    xi = sr / (1 + sr)
    xs = aa * np.exp(ee / tk)
    ak1 = a1 * np.exp(-e1 / tk)

    n = len(tk)
    inter_e = np.zeros(n)
    for i in range(1, n):
        prev = inter_e[i - 1]
        s = prev if prev < 1 else prev - prev * xi[i - 1]
        inter_e[i] = xs[i] - (xs[i] - s) * np.exp(-ak1[i])
    return np.where(inter_e >= 1, inter_e * xi, 0.0)


def season_cumsum(df, value, start_month):
    """Running total that resets each year on the 1st of start_month."""
    season = df["date"].dt.year - (df["date"].dt.month < start_month).astype(int)
    return df.groupby(["region", season])[value].cumsum()


def main():
    # --- chill portions from hourly temperature
    h = pd.read_csv(HOURLY, parse_dates=["time"]).sort_values(["region", "time"])
    h["temperature_2m"] = h.groupby("region")["temperature_2m"].transform(lambda s: s.interpolate(limit=6))
    h["cp"] = h.groupby("region", group_keys=False)["temperature_2m"].apply(
        lambda s: pd.Series(dynamic_model_portions(s.fillna(s.mean()).values), index=s.index))
    h["date"] = h["time"].dt.normalize()
    daily = h.groupby(["region", "date"], as_index=False)["cp"].sum()

    # --- degree-days (base 9.6 C) from the daily mean temperature
    wx = pd.read_csv(DAILY_WX, parse_dates=["time"]).rename(columns={"time": "date"})
    wx["gdd96"] = np.maximum(0, wx["temperature_2m_mean"] - GDD_BASE)
    daily = daily.merge(wx[["region", "date", "gdd96"]], on=["region", "date"], how="left")

    # --- Cape Doctor days
    w = pd.read_csv(WIND, parse_dates=["time"]).rename(columns={"time": "date"})
    # Gridded 10 m wind at these inland valleys rarely reaches coastal Cape Doctor speeds,
    # so "strong" is relative: above that region's 75th percentile of daily max wind.
    strong = w.groupby("region")["wind_speed_10m_max"].transform(lambda s: s.quantile(SE_SPEED_PCTL))
    w["cape_doctor"] = (w["wind_direction_10m_dominant"].between(SE_MIN_DEG, SE_MAX_DEG)
                        & (w["wind_speed_10m_max"] >= strong)).astype(int)
    daily = daily.merge(w[["region", "date", "wind_speed_10m_max", "wind_gusts_10m_max", "cape_doctor"]],
                        on=["region", "date"], how="left")

    daily = daily.sort_values(["region", "date"]).reset_index(drop=True)
    daily["cp_season"] = season_cumsum(daily, "cp", start_month=3)
    daily["gdd96_season"] = season_cumsum(daily.assign(gdd96=daily["gdd96"].fillna(0)), "gdd96", start_month=7)

    out = []
    for region, g in daily.groupby("region"):
        g = g.set_index("date")
        out.append(pd.DataFrame({
            "region": region,
            "cp_30d_sum": g["cp"].rolling("30D").sum(),
            "cp_season": g["cp_season"],
            "gdd96_30d_sum": g["gdd96"].rolling("30D").sum(),
            "gdd96_60d_sum": g["gdd96"].rolling("60D").sum(),
            "gdd96_90d_sum": g["gdd96"].rolling("90D").sum(),
            "gdd96_season": g["gdd96_season"],
            "wind_max_14d_mean": g["wind_speed_10m_max"].rolling("14D").mean(),
            "gust_max_14d_max": g["wind_gusts_10m_max"].rolling("14D").max(),
            "cape_doctor_days_14d": g["cape_doctor"].rolling("14D").sum(),
        }).reset_index())
    feats = pd.concat(out)
    feats.to_csv(OUT, index=False)

    # Sanity check: total chill per winter (1 Mar - 31 Aug) by region.
    chk = daily[daily["date"].dt.month.between(3, 8)]
    print("Chill portions per winter (Mar-Aug), median and range by region:")
    print(chk.groupby(["region", chk["date"].dt.year])["cp"].sum().groupby("region")
          .agg(["median", "min", "max"]).round(1))
    print("\nCape Doctor days per year (median):")
    print(daily.groupby(["region", daily["date"].dt.year])["cape_doctor"].sum().groupby("region").median())
    print(f"\nSaved {len(feats)} rows to {OUT}")


if __name__ == "__main__":
    main()
