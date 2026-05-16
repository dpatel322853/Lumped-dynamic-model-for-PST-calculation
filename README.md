
# Streamlit PST Calculator App

## Purpose
This Streamlit app converts the Python PST calculator into a deployable web app. It is intended for first-pass Process Safety Time (PST) estimation for low-temperature protection of CS piping downstream of an LP Ethylene Superheater after methanol heating-medium failure.

## Features
- Input all key process, exchanger, pipe, TT, logic, solenoid, and XV parameters from the UI
- Run dynamic cooldown simulation
- Visualize:
  - Superheater outlet actual temperature
  - TT measured temperature
  - CS pipe fluid temperature
  - CS pipe wall temperature
  - XV opening profile
- Download:
  - Excel report
  - CSV time series
  - PNG plot
  - PDF report

## Local Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Create a GitHub repository, e.g. `pst-streamlit-app`.
2. Upload these files:
   - `app.py`
   - `requirements.txt`
   - `README.md`
3. Go to Streamlit Community Cloud.
4. Select the repository.
5. Main file path: `app.py`
6. Click Deploy.

## Engineering Note
This is a simplified lumped dynamic model. For formal SIS/PST validation, benchmark against Aspen HYSYS Dynamics or approved dynamic simulation and use project-approved datasheets.
