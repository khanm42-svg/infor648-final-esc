# INFO 648 Final Project: Neighborhood Trajectory Prediction

**Census Division:** East South Central
**States:** Alabama, Kentucky, Mississippi, Tennessee

## How to Run

1. Open `INFO648_Final_EastSouthCentral.ipynb` in [Google Colab](https://colab.research.google.com)
2. Run the first cell — a file picker will appear
3. Upload `student_tracts_raw.csv` and `forecast_tracts_2020.csv`
4. Go to **Runtime → Run all**
   
## Data Sources

Provided by the course instructor, drawn from IPUMS NHGIS time-series tables standardized to 2010 census-tract boundaries. Not included in this repository due to file size.

- `student_tracts_raw.csv` — 2010 demographic features + 2020 population (~73,000 tracts nationwide, used for training and testing)
- `forecast_tracts_2020.csv` — 2020 demographic features for the same tracts (used for 2020→2030 forecast only, no label)

## Streamlit Prediction App (Extra Credit)

A live web app lets you enter a tract's characteristics and get a growth prediction instantly.

**Live URL:** https://infor648-final-esc-cngotpytzuzxdnfqifoz7h.streamlit.app/

**To run locally:**
```bash
pip install -r requirements.txt
streamlit run app.py
```

**To deploy:**
1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
2. Click **New app** → select this repo
3. Set **Main file path** to `app.py`
4. Click **Deploy**
5. Copy the live URL and paste it above

> **Note:** .joblib files were downloaded from colab and uploaded ontoGithub
