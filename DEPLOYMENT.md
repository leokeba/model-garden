# Deployment Guide

## Overview
Model Garden now includes an integrated web UI that is automatically served by the FastAPI backend. No need to run separate frontend development servers!

## Quick Start

### Development
```bash
# Start the server (serves both API and UI)
uv run model-garden serve

# Access the application
# - Web UI: http://localhost:8000
# - API Docs: http://localhost:8000/docs
# - API: http://localhost:8000/api/v1/*
```

### Development with Auto-Reload
```bash
# Enable auto-reload for code changes
uv run model-garden serve --reload
```

### Production
```bash
# Run on all interfaces (for remote access)
uv run model-garden serve --host 0.0.0.0 --port 8000

# Or with a process manager like systemd, supervisor, or PM2
```

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  FastAPI Backend                     │
│                 (Port 8000)                          │
├─────────────────────────────────────────────────────┤
│                                                      │
│  API Routes:                                         │
│  - /health                                           │
│  - /api/v1/models                                    │
│  - /api/v1/training/jobs                             │
│  - /api/v1/system/status                             │
│                                                      │
│  Static Files:                                       │
│  - /_app/* → frontend/build/_app/*                   │
│  - /* → frontend/build/index.html (SPA fallback)     │
│                                                      │
└─────────────────────────────────────────────────────┘
```

## Frontend Build Process

The frontend is built as a static site using SvelteKit's static adapter:

```bash
cd frontend
npm run build
```

**Output**: `frontend/build/`
- Contains all static HTML, JS, CSS files
- Prerendered pages for faster initial load
- Client-side routing for SPA behavior

**Build Artifacts** (committed to git):
- `frontend/build/` - Production build
- Served automatically by FastAPI backend

## Configuration

### SvelteKit Configuration
File: `frontend/svelte.config.js`

```javascript
adapter: adapter({
  pages: 'build',
  assets: 'build',
  fallback: 'index.html',  // Enables client-side routing
  precompress: false,
  strict: true
})
```

### FastAPI Static File Serving
File: `model_garden/api.py`

```python
# Mount static assets
app.mount("/_app", StaticFiles(directory="frontend/build/_app"), name="static-assets")

# Catch-all route for SPA
@app.get("/{full_path:path}")
async def serve_spa(full_path: str):
    # Serves frontend/build/index.html for all non-API routes
    ...
```

## Routing

### API Routes (Priority)
FastAPI routes are defined first and take priority:
- `/health` → Health check endpoint
- `/api/v1/*` → All API endpoints
- `/docs` → Swagger UI
- `/redoc` → ReDoc documentation

### Frontend Routes (Fallback)
All other routes serve the SvelteKit SPA:
- `/` → Dashboard
- `/models` → Models listing
- `/models/{name}` → Model detail page
- `/training` → Training jobs listing
- `/training/{id}` → Job detail page
- `/training/new` → Create new training job

## Rebuilding Frontend

When you make changes to the frontend code:

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies (first time only)
npm install

# Build for production
npm run build

# The backend will automatically serve the new build
# (restart the backend if it's running)
```

## Environment Variables

Currently no environment variables are required. Future versions may add:
- `MODEL_GARDEN_HOST` - Server host (default: 0.0.0.0)
- `MODEL_GARDEN_PORT` - Server port (default: 8000)
- `MODEL_GARDEN_ENV` - Environment (development/production)

## Process Management

### Systemd Service (Linux)

Create `/etc/systemd/system/model-garden.service`:

```ini
[Unit]
Description=Model Garden API Server
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/model-garden
Environment="PATH=/path/to/.venv/bin"
ExecStart=/path/to/.venv/bin/uv run model-garden serve --host 0.0.0.0
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable model-garden
sudo systemctl start model-garden
sudo systemctl status model-garden
```

### PM2 (Node.js Process Manager)

```bash
# Install PM2 globally
npm install -g pm2

# Start with PM2
pm2 start "uv run model-garden serve" --name model-garden

# Save configuration
pm2 save

# Enable startup script
pm2 startup
```

## Nginx Reverse Proxy (Optional)

For production deployments behind nginx:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support (for future real-time features)
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

## Performance

### Build Sizes
Frontend production build:
- **Total**: ~180 KB (gzipped)
- **Initial load**: ~60 KB
- **Code splitting**: Yes
- **Lazy loading**: Yes

### Load Times
- **First contentful paint**: < 1s
- **Time to interactive**: < 2s
- **Total page size**: < 200 KB

### Caching
Static assets are served with proper cache headers:
- `/_app/immutable/*` → Immutable, cached forever
- Other files → Standard browser caching

## Troubleshooting

### Frontend not loading
1. Check if build directory exists: `ls frontend/build/`
2. Rebuild frontend: `cd frontend && npm run build`
3. Check server logs for errors
4. Verify API is running: `curl http://localhost:8000/health`

### API routes not working
1. Ensure API routes are defined before static file mounting
2. Check route priority in `model_garden/api.py`
3. Test API directly: `curl http://localhost:8000/api/v1/models`

### 404 on client-side routes
1. Verify fallback is configured in `svelte.config.js`
2. Check catch-all route in FastAPI
3. Ensure `index.html` exists in build directory

### Changes not reflected
1. Rebuild frontend: `cd frontend && npm run build`
2. Restart backend: `pkill -f "model-garden serve" && uv run model-garden serve`
3. Clear browser cache (Ctrl+Shift+R)

## Development Tips

### Frontend Development
For rapid frontend development, you can still use the Vite dev server:

```bash
# Terminal 1: Backend
uv run model-garden serve

# Terminal 2: Frontend dev server (optional)
cd frontend && npm run dev
```

Access:
- Frontend dev: http://localhost:5173 (hot reload)
- Backend API: http://localhost:8000/api/v1/*

### Backend Development
Use `--reload` flag for auto-reload on code changes:

```bash
uv run model-garden serve --reload
```

Note: This will restart the server on any Python file change.

## Security Considerations

### Production Checklist
- [ ] Update CORS origins in `model_garden/api.py`
- [ ] Add authentication/authorization (Phase 2)
- [ ] Enable HTTPS (use nginx/traefik)
- [ ] Set up rate limiting
- [ ] Configure firewall rules
- [ ] Use environment variables for secrets
- [ ] Enable security headers
- [ ] Regular dependency updates

### Current Security Status
⚠️ **Development Mode**: 
- CORS allows all origins (`allow_origins=["*"]`)
- No authentication required
- Suitable for local development only

For production, update CORS settings:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-domain.com"],  # Restrict to your domain
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)
```

## Updates and Maintenance

### Updating Dependencies

Frontend:
```bash
cd frontend
npm update
npm audit fix
npm run build
```

Backend:
```bash
uv sync --upgrade
```

### Version Management
Current setup uses:
- **Frontend**: SvelteKit 2.x, Svelte 5.x, TailwindCSS 3.x
- **Backend**: FastAPI 0.109+, Python 3.11+

## Monitoring

### Health Check
```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "message": "Model Garden API is running"
}
```

### System Status
```bash
curl http://localhost:8000/api/v1/system/status
```

Returns:
- CPU and memory usage
- GPU availability and memory
- Disk usage
- Active training jobs count
- Model count

## Support

For issues or questions:
- Check [API_COMPARISON.md](./API_COMPARISON.md) for API coverage
- See [docs/07-troubleshooting.md](./docs/07-troubleshooting.md)
- Open an issue on GitHub
