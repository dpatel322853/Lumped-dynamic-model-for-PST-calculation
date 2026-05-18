# DPSIM - Merged PST Calculator

Merged Streamlit app for Low Temperature Protection of CS piping downstream of LP Ethylene Superheater.

## What is merged

1. Existing lumped dynamic PST model supplied by Dhawal Patel.
2. Screening / plug-flow PST calculation for quick validation.
3. Sensitivity analysis for HX cooling time constant and XV stroke time.
4. Report export to Excel, CSV, PNG and PDF.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

Upload:

- `app.py`
- `requirements.txt`
- `README.md`

Set the main file path as:

```text
app.py
```

## Engineering note

This tool is for screening and engineering basis development. Validate with approved project data, vendor TT/XV response data and rigorous dynamic simulation before formal SIS/PST documentation.
