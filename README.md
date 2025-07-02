# Sports-Equipment Delivery App – Feasibility Analytics

This repo houses:
* `synthetic_sports_app_survey.csv` – 500-row synthetic consumer survey.
* `analysis.py` – one-shot script that:
  1. Classifies willingness to adopt the app  
  2. Segments customers with K-means  
  3. Predicts monthly spend with linear regression  
  4. Surfaces association rules among preferences
* `outputs/` – auto-generated metrics, plots & CSVs.

## Quick start
```bash
pip install -r requirements.txt
python analysis.py --csv synthetic_sports_app_survey.csv
