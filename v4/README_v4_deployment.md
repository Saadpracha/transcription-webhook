# Server Deployment Steps for New Version

## 1. Go to project folder
```bash
cd ~/webhook/v4/Audio-Transcription
```

## 2. Create virtual environment
```bash
python3 -m venv venv
```

## 3. Activate virtual environment
```bash
source venv/bin/activate
```

## 4. Upgrade pip
```bash
pip install --upgrade pip
```

## 5. Install requirements
```bash
pip install -r requirements.txt
```

## 6. Run project manually on port 8004
```bash
uvicorn fastapi_whisper_service:app --host 0.0.0.0 --port 8004 --log-level info
```

## 7. Nginx file to update
File:
```bash
/etc/nginx/sites-available/webhook
```

Add/update this block:
```nginx
location /v4/ {
    proxy_pass http://localhost:8004;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
}
```

## 8. Systemd file to create
File:
```bash
/etc/systemd/system/fastapi-v4.service
```

Content:
```ini
[Unit]
Description=FastAPI v4 Whisper Service
After=network.target

[Service]
User=saad
WorkingDirectory=/home/saad/webhook/v4/Audio-Transcription
EnvironmentFile=/home/saad/webhook/v4/Audio-Transcription/.env
ExecStart=/home/saad/webhook/v4/Audio-Transcription/venv/bin/uvicorn \
    fastapi_whisper_service:app \
    --host 0.0.0.0 \
    --port 8004 \
    --log-level info

Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

## 9. Systemd commands
```bash
sudo systemctl daemon-reload
sudo systemctl enable fastapi-v4
sudo systemctl start fastapi-v4
systemctl status fastapi-v4
```

## 10. Check logs
```bash
journalctl -u fastapi-v4.service -f
```

## 11. Test port
```bash
curl -i http://127.0.0.1:8004/
curl -i http://127.0.0.1:8004/docs
```

## 12. Test public URL
```bash
curl -i -X POST "https://webhook.lead2424.com/v4/" -H "Content-Type: application/json" -d '{"audio":"FILE_URL","transcribe":true}'
```
