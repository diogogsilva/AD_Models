from fastapi import FastAPI, HTTPException

app = FastAPI()

@app.get('/')
def get_options():
    return {"Indicators": ["pH", "amonia"]}

@app.get('/{indicator}')
def get_options(indicator):
    if indicator in ['ph', 'amonia']:
        return get_features_list(indicator)
    else:
        raise HTTPException(status_code=404, detail="Indicator not found")

@app.get('/{indicator}/{method}')
def get_method(indicator, method: int):
    if indicator in ['ph', 'amonia'] and method in [0, 1, 2]:
        return get_feature(indicator, method)
    else:
        raise HTTPException(status_code=404, detail="Indicator and/or Method not found")


def get_features_list(indicator):
    if indicator == 'ph':
        return {'methods': ['Anomaly Detection', 'Forecast', 'Anomaly Detection with Forecasted Data']}
    elif indicator == 'amonia':
        return {'methods': ['Anomaly Detection']}

def get_feature(indicator, method):
    return 'Valid option'