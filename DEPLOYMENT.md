# Run & Deploy

## Windows — easiest
Double-click `run.bat`.

It creates `.venv`, installs the packages, and starts Streamlit.

## Manual
Open PowerShell in this folder:

```powershell
py -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Then open:

`http://localhost:8501`

## Streamlit Community Cloud
1. Push this folder to GitHub.
2. Go to Streamlit Community Cloud.
3. Create a new app from the repository.
4. Select `app.py` as the main file.
5. Deploy.

The CSV is already inside `data/`, so no database is required.

## Vercel
This exact Streamlit application should not be deployed directly to Vercel. Vercel is better suited to a React/Next.js frontend or a separately designed web frontend. For this project, Streamlit Community Cloud is the simplest deployment target while preserving all Plotly interactions.
