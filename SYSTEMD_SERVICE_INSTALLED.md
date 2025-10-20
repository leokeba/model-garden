# Model Garden Systemd Service - Installation Complete

✅ **Model Garden has been successfully installed as a systemd service!**

## Service Details

- **Service Name:** `model-garden.service`
- **Status:** Active and running
- **Auto-start:** Enabled (will start automatically on system boot)
- **User:** root
- **Working Directory:** `/root/model-garden`
- **Server URL:** http://0.0.0.0:8000

## Service Configuration

The service file is located at: `/etc/systemd/system/model-garden.service`

**Environment Variables:**
- `HF_HOME=/root/model-garden/storage/cache`
- `TRANSFORMERS_CACHE=/root/model-garden/storage/cache`
- `HF_DATASETS_CACHE=/root/model-garden/storage/datasets`
- `PYTORCH_ALLOC_CONF=expandable_segments:True`

## Managing the Service

### Start/Stop/Restart Commands

```bash
# Check status
sudo systemctl status model-garden.service

# Start the service
sudo systemctl start model-garden.service

# Stop the service
sudo systemctl stop model-garden.service

# Restart the service
sudo systemctl restart model-garden.service

# Reload after editing service file
sudo systemctl daemon-reload
sudo systemctl restart model-garden.service
```

### View Logs

```bash
# View recent logs (last 50 lines)
sudo journalctl -u model-garden.service -n 50

# Follow logs in real-time
sudo journalctl -u model-garden.service -f

# View logs since boot
sudo journalctl -u model-garden.service -b

# View logs with time range
sudo journalctl -u model-garden.service --since "1 hour ago"
```

### Disable/Enable Auto-start

```bash
# Disable auto-start on boot
sudo systemctl disable model-garden.service

# Re-enable auto-start on boot
sudo systemctl enable model-garden.service

# Check if enabled
sudo systemctl is-enabled model-garden.service
```

## Accessing the Web UI

The Model Garden web interface is now accessible at:

- **Local:** http://localhost:8000
- **Network:** http://YOUR_SERVER_IP:8000

## Features

✅ Automatic startup on system boot
✅ Automatic restart on failure (10 second delay)
✅ Persistence across SSH disconnections
✅ Centralized logging via systemd journal
✅ Standard system service management

## Troubleshooting

### Service won't start

Check the logs:
```bash
sudo journalctl -u model-garden.service -n 100 --no-pager
```

### Check if port 8000 is in use

```bash
sudo lsof -i :8000
```

### Restart the service

```bash
sudo systemctl restart model-garden.service
```

## Next Steps

1. Access the web UI at http://localhost:8000 (or your server IP)
2. The service will automatically start on system reboot
3. Use `journalctl` to monitor logs
4. For production, consider setting up a reverse proxy with Nginx (see docs/16-systemd-service-setup.md)

## Additional Documentation

For more advanced configuration options, including:
- Custom port configuration
- GPU memory management
- Reverse proxy setup with Nginx
- SSL configuration
- Health monitoring

See the full documentation: `docs/16-systemd-service-setup.md`

---

**Installation Date:** October 20, 2025
**Installation User:** root
**Installation Path:** /root/model-garden
