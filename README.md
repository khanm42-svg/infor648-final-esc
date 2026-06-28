# INFO 648 Final Project: Neighborhood Trajectory Prediction
**Census Division:** East South Central
**States:** Alabama, Kentucky, Mississippi, Tennessee
**Term:** Summer 2026
## How to Run
1. Open `INFO648_Final_ESC.ipynb` in [Google Colab](https://colab.research.google.com)
2. Run the first cell — a file picker will appear
3. Upload `student_tracts_raw.csv` and `forecast_tracts_2020.csv`
4. Go to **Runtime → Run all**
## Data Sources

Provided by the professor, drawn from IPUMS NHGIS time-series tables standardized to 2010 census-tract boundaries. Not included in this repository due to file size.
- `student_tracts_raw.csv` — 2010 demographic features + 2020 population (~73,000 tracts nationwide, used for training and testing)
- `forecast_tracts_2020.csv` — 2020 demographic features for the same tracts (used for 2020→2030 forecast only, no label)
