from pathlib import Path
import re, html
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

bundle = joblib.load(Path(__file__).with_name('spam_model.joblib'))
app = FastAPI(title='Spam classifier')

def clean_text(x):
    x = html.unescape('' if x is None else str(x))
    x = re.sub(r'<style.*?</style>|<script.*?</script>', ' ', x, flags=re.I|re.S)
    x = re.sub(r'<[^>]+>', ' ', x)
    x = re.sub(r'https?://\S+|www\.\S+', ' URL ', x, flags=re.I)
    x = re.sub(r'\S+@\S+', ' EMAIL ', x)
    x = re.sub(r'[^\w\s!?$%+\-]', ' ', x, flags=re.UNICODE)
    return re.sub(r'\s+', ' ', x).strip().lower()

class EmailInput(BaseModel):
    text: str

@app.get('/health')
def health():
    return {'status': 'ok'}

@app.post('/predict')
def predict(item: EmailInput):
    if len(item.text) > 100_000:
        raise HTTPException(status_code=413, detail='Текст слишком длинный (максимум 100000 символов)')
    score = float(bundle['pipeline'].predict_proba([clean_text(item.text)])[0, 1])
    return {'label': 'spam' if score >= bundle['threshold'] else 'ham',
            'spam_probability': score, 'threshold': bundle['threshold']}


@app.get('/', response_class=HTMLResponse, include_in_schema=False)
def demo():
    return Path(__file__).with_name('index.html').read_text(encoding='utf-8')
