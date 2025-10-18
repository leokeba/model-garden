# Running Model Garden as a Systemd Service

This guide explains how to run the Model Garden web UI as a persistent systemd service on Debian/Ubuntu systems, ensuring it continues running even when SSH sessions drop and automatically restarts on system reboot.

## Prerequisites

- Debian/Ubuntu-based Linux system
- Model Garden installed with `uv`
- Root or sudo access
- Project located at a permanent path (e.g., `/home/leo/Dev/model-garden`)

## Step 1: Create a Systemd Service File

Create a new service file for Model Garden:

```bash
sudo nano /etc/systemd/system/model-garden.service
```

Add the following configuration (adjust paths as needed):

```ini
[Unit]
Description=Model Garden API Server
After=network.target
Documentation=https://github.com/leokeba/model-garden

[Service]
Type=simple
User=leo
Group=leo
WorkingDirectory=/home/leo/Dev/model-garden

# Environment variables
Environment="PATH=/home/leo/.local/bin:/usr/local/bin:/usr/bin:/bin"
Environment="HF_HOME=/home/leo/Dev/model-garden/storage/cache"
Environment="TRANSFORMERS_CACHE=/home/leo/Dev/model-garden/storage/cache"
Environment="HF_DATASETS_CACHE=/home/leo/Dev/model-garden/storage/datasets"

# Command to execute
ExecStart=/home/leo/.local/bin/uv run model-garden serve

# Restart policy
Restart=on-failure
RestartSec=10s

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=model-garden

# Resource limits (adjust based on your needs)
LimitNOFILE=65536
# Optional: Limit memory if needed
# MemoryLimit=32G

[Install]
WantedBy=multi-user.target
```

### Configuration Explanation

- **User/Group**: Replace `leo` with your actual username
- **WorkingDirectory**: Full path to your Model Garden installation
- **Environment PATH**: Include the path to `uv` (check with `which uv`)
- **HF_HOME**: HuggingFace cache directory (critical for Model Garden)
- **ExecStart**: Full path to `uv` command (verify with `which uv`)
- **Restart**: Automatically restart on failure
- **StandardOutput/StandardError**: Logs sent to systemd journal

## Step 2: Find the Correct uv Path

Before enabling the service, verify the `uv` installation path:

```bash
which uv
```

Common paths:
- `/home/username/.local/bin/uv` (user installation)
- `/usr/local/bin/uv` (system installation)

Update the `ExecStart` path in the service file accordingly.

## Step 3: Set Proper Permissions

Ensure the service file has correct permissions:

```bash
sudo chmod 644 /etc/systemd/system/model-garden.service
```

## Step 4: Reload Systemd and Enable Service

```bash
# Reload systemd to recognize the new service
sudo systemctl daemon-reload

# Enable the service to start on boot
sudo systemctl enable model-garden.service

# Start the service
sudo systemctl start model-garden.service
```

## Step 5: Verify Service Status

Check that the service is running:

```bash
sudo systemctl status model-garden.service
```

Expected output:
```
● model-garden.service - Model Garden API Server
     Loaded: loaded (/etc/systemd/system/model-garden.service; enabled; vendor preset: enabled)
     Active: active (running) since ...
```

## Managing the Service

### Start/Stop/Restart

```bash
# Start the service
sudo systemctl start model-garden.service

# Stop the service
sudo systemctl stop model-garden.service

# Restart the service
sudo systemctl restart model-garden.service

# Reload configuration after editing service file
sudo systemctl daemon-reload
sudo systemctl restart model-garden.service
```

### View Logs

```bash
# View recent logs
sudo journalctl -u model-garden.service -n 50

# Follow logs in real-time
sudo journalctl -u model-garden.service -f

# View logs since boot
sudo journalctl -u model-garden.service -b

# View logs with timestamps
sudo journalctl -u model-garden.service --since "2024-01-01" --until "2024-01-02"
```

### Check Service Status

```bash
# Detailed status
sudo systemctl status model-garden.service

# Check if enabled on boot
sudo systemctl is-enabled model-garden.service

# Check if currently active
sudo systemctl is-active model-garden.service
```

## Accessing the Web UI

Once the service is running, access the web UI at:
- `http://localhost:8000` (local)
- `http://your-server-ip:8000` (remote)

## Advanced Configuration

### Custom Port

To run on a different port, modify the service file:

```ini
ExecStart=/home/leo/.local/bin/uv run model-garden serve --port 8080
```

### GPU Memory Management

If you experience GPU memory issues, you can add a pre-start cleanup:

```ini
[Service]
# ... other settings ...
ExecStartPre=/home/leo/Dev/model-garden/cleanup_gpu.sh
ExecStart=/home/leo/.local/bin/uv run model-garden serve
```

Make sure `cleanup_gpu.sh` is executable:
```bash
chmod +x /home/leo/Dev/model-garden/cleanup_gpu.sh
```

### Environment File

For complex environment configurations, create a separate file:

```bash
sudo nano /etc/model-garden/environment
```

Add environment variables:
```
HF_HOME=/home/leo/Dev/model-garden/storage/cache
TRANSFORMERS_CACHE=/home/leo/Dev/model-garden/storage/cache
HF_DATASETS_CACHE=/home/leo/Dev/model-garden/storage/datasets
CUDA_VISIBLE_DEVICES=0
```

Update the service file:
```ini
[Service]
EnvironmentFile=/etc/model-garden/environment
```

### Running with Screen/Tmux (Alternative)

If you prefer not to use systemd, you can use `screen` or `tmux`:

**Using Screen:**
```bash
# Install screen
sudo apt-get install screen

# Start a detached session
screen -dmS model-garden bash -c 'cd /home/leo/Dev/model-garden && uv run model-garden serve'

# Reattach to session
screen -r model-garden

# Detach: Ctrl+A, then D
```

**Using Tmux:**
```bash
# Install tmux
sudo apt-get install tmux

# Start a detached session
tmux new-session -d -s model-garden 'cd /home/leo/Dev/model-garden && uv run model-garden serve'

# Attach to session
tmux attach -t model-garden

# Detach: Ctrl+B, then D
```

## Troubleshooting

### Service Fails to Start

1. Check logs for errors:
   ```bash
   sudo journalctl -u model-garden.service -n 100 --no-pager
   ```

2. Verify paths in service file are correct
3. Check permissions on working directory
4. Ensure `uv` is installed and accessible

### Permission Denied Errors

Ensure the user specified in the service file has:
- Read/write access to the working directory
- Access to GPU devices (add user to `video` group if needed):
  ```bash
  sudo usermod -aG video leo
  ```

### Port Already in Use

Check if another service is using port 8000:
```bash
sudo lsof -i :8000
```

Kill the process or change the port in the service file.

### GPU Not Accessible

If running as a service user, ensure they have GPU access:
```bash
# Add user to render and video groups
sudo usermod -aG render,video leo
```

## Security Considerations

### Firewall Configuration

If accessing remotely, configure the firewall:

```bash
# Allow port 8000 (UFW)
sudo ufw allow 8000/tcp

# Or use iptables
sudo iptables -A INPUT -p tcp --dport 8000 -j ACCEPT
```

### Reverse Proxy with Nginx

For production, use Nginx as a reverse proxy:

```bash
sudo apt-get install nginx
```

Create Nginx configuration:
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
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

### SSL with Let's Encrypt

```bash
sudo apt-get install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

## Monitoring

### Check Service Health

Create a simple health check script:

```bash
#!/bin/bash
# /home/leo/bin/check-model-garden.sh

if systemctl is-active --quiet model-garden.service; then
    echo "Model Garden is running"
    exit 0
else
    echo "Model Garden is not running"
    systemctl status model-garden.service
    exit 1
fi
```

### Add to Cron for Monitoring

```bash
crontab -e
```

Add:
```
*/5 * * * * /home/leo/bin/check-model-garden.sh >> /var/log/model-garden-health.log 2>&1
```

## Summary

Running Model Garden as a systemd service provides:
- ✅ Automatic startup on system boot
- ✅ Automatic restart on failure
- ✅ Persistence across SSH disconnections
- ✅ Centralized logging via journald
- ✅ Standard system service management

For development work, you may still want to run manually with `uv run model-garden serve`, but for production deployments, systemd is the recommended approach.
