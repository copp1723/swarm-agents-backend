# 🚀 Swarm Multi-Agent Backend - Render Deployment

## Build Command
```bash
pip install -r requirements.txt
```

## Start Command
```bash
gunicorn --bind 0.0.0.0:$PORT --workers 4 --timeout 30 --keep-alive 2 src.main:app
```

## Environment Variables (Set in Render Dashboard)

### Required Environment Variables:
- `FLASK_ENV=production`
- `FLASK_DEBUG=False`
- `SECRET_KEY=your-super-secret-production-key`
- `OPENROUTER_API_KEY=sk-or-v1-your-openrouter-api-key`
- `SUPERMEMORY_API_KEY=your-supermemory-api-key`
- `MAILGUN_API_KEY=your-mailgun-api-key`
- `MAILGUN_DOMAIN=your-domain.com`
- `CORS_ORIGINS=https://your-frontend-domain.com`
- `JWT_SECRET_KEY=your-jwt-secret-key`

### Database (Render PostgreSQL):
- `DATABASE_URL` (automatically provided by Render PostgreSQL add-on)

### Optional Configuration:
- `LOG_LEVEL=INFO`
- `RATE_LIMIT_ENABLED=True`
- `WORKERS=4`
- `TIMEOUT=30`

## Render Service Configuration

### Service Type: Web Service
### Runtime: Python 3.11
### Build Command: `pip install -r requirements.txt`
### Start Command: `gunicorn --bind 0.0.0.0:$PORT --workers 4 --timeout 30 --keep-alive 2 src.main:app`
### Auto-Deploy: Yes (from main branch)

## Health Check Endpoint
- URL: `/api/health`
- Expected Response: 200 OK with JSON status

## Custom Domain Setup
1. Add your custom domain in Render dashboard
2. Update DNS records with your domain provider
3. SSL certificate will be automatically provisioned

## Database Setup
1. Add PostgreSQL add-on in Render
2. Database URL will be automatically set as environment variable
3. Tables will be created automatically on first run

## Monitoring
- Health checks: `/api/health`
- Logs: Available in Render dashboard
- Metrics: CPU, Memory, Response time tracking

## Scaling
- Horizontal scaling: Increase number of instances
- Vertical scaling: Upgrade service plan
- Auto-scaling: Available on higher plans

