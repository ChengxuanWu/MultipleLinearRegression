import json
import os
from http.server import BaseHTTPRequestHandler

# Load model weights
weights_path = os.path.join(os.path.dirname(__file__), '..', 'model_weights.json')
try:
    with open(weights_path, 'r', encoding='utf-8') as f:
        MODELS = json.load(f)
except Exception:
    MODELS = {}

def predict_boston_linear(inputs):
    lin = MODELS.get('boston', {}).get('linear', {})
    features = MODELS.get('boston', {}).get('features', [])
    intercept = lin.get('intercept', 40.36)
    coefs = lin.get('coefficients', [])
    
    val = intercept
    for feat, coef in zip(features, coefs):
        val += coef * float(inputs.get(feat, 0))
    return round(float(val), 2)

def predict_boston_ridge(inputs):
    ridge = MODELS.get('boston', {}).get('ridge_poly', {})
    features = MODELS.get('boston', {}).get('features', [])
    mean = ridge.get('scaler_mean', [])
    scale = ridge.get('scaler_scale', [])
    powers = ridge.get('powers', [])
    coefs = ridge.get('coefficients', [])
    intercept = ridge.get('intercept', 21.55)
    
    # Standardize 7 features
    scaled = []
    for i, feat in enumerate(features):
        x = float(inputs.get(feat, 0))
        m = mean[i] if i < len(mean) else 0.0
        s = scale[i] if i < len(scale) and scale[i] != 0 else 1.0
        scaled.append((x - m) / s)
        
    # Evaluate polynomial features
    poly_terms = []
    for p in powers:
        term = 1.0
        for i, exp in enumerate(p):
            if exp > 0:
                term *= (scaled[i] ** exp)
        poly_terms.append(term)
        
    val = intercept
    for term, c in zip(poly_terms, coefs):
        val += c * term
    return round(float(val), 2)

def predict_startups(inputs):
    s = MODELS.get('startups', {})
    intercept = s.get('intercept', 46975.86)
    coefs = s.get('coefficients', [0.79658, 0.02990])
    
    rd = float(inputs.get('rd', 0))
    mkt = float(inputs.get('mkt', 0))
    
    val = intercept + coefs[0] * rd + coefs[1] * mkt
    return round(float(val), 2)

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_POST(self):
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            data = json.loads(body) if body else {}

            project = data.get('project', 'boston')
            inputs = data.get('inputs', {})
            mode = data.get('mode', 'high-r2')

            if project == 'boston':
                if mode == 'high-r2':
                    pred = predict_boston_ridge(inputs)
                    r2 = 0.8029
                    rmse = 3.8017
                else:
                    pred = predict_boston_linear(inputs)
                    r2 = 0.6685
                    rmse = 4.9302
                resp = {
                    'project': 'boston',
                    'mode': mode,
                    'prediction_k': pred,
                    'prediction_dollar': pred * 1000.0,
                    'r2': r2,
                    'rmse': rmse
                }
            elif project == 'startups':
                pred = predict_startups(inputs)
                resp = {
                    'project': 'startups',
                    'prediction_dollar': pred,
                    'r2': 0.9474,
                    'rmse': 8750.2
                }
            else:
                resp = {'error': f'Unknown project: {project}'}

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode('utf-8'))
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({'error': str(e)}).encode('utf-8'))

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps({'status': 'online', 'service': 'CRISP-DM Vercel ML API'}).encode('utf-8'))
