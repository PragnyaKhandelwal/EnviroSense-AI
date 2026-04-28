# VPS deployment (clean isolation)

## 1) Python deps in project venv only

```bash
cd /home/pragnya/EnviroSenseAI
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Backend env

```bash
cp backend/.env.example backend/.env
# edit backend/.env values for DB credentials and CORS
```

## 3) Frontend build

```bash
cd /home/pragnya/EnviroSenseAI/frontend-pwa
cp .env.production.example .env.production
npm ci
npm run build
```

## 4) Systemd API service

```bash
sudo cp /home/pragnya/EnviroSenseAI/deploy/systemd/envirosense-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now envirosense-api.service
sudo systemctl status envirosense-api.service --no-pager
```

## 5) Nginx site

```bash
sudo cp /home/pragnya/EnviroSenseAI/deploy/nginx/envirosense.conf /etc/nginx/sites-available/envirosense
sudo ln -sf /etc/nginx/sites-available/envirosense /etc/nginx/sites-enabled/envirosense
sudo nginx -t
sudo systemctl restart nginx
sudo systemctl status nginx --no-pager
```

## 6) Smoke tests

```bash
curl -sS http://127.0.0.1:8000/api/health
curl -sS "http://127.0.0.1:8000/api/pipeline?device=ESP32_Node_1" | head
curl -I http://127.0.0.1/
```
