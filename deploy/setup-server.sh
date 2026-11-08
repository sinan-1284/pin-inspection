#!/bin/bash
# Run on socket.ddlc.ai as sinan:  bash pin-setup.sh
# Publishes the pin station (reverse SSH tunnel from the Pi on localhost:8090)
# at https://pin.ddlc.ai through its own Cloudflare tunnel. health.ddlc.ai is
# not touched.
set -e
CF=~/bin/cloudflared
NAME=pin-inspection
HOST=pin.ddlc.ai

tunnel_id() { $CF tunnel list -o json 2>/dev/null | python3 -c "import json,sys;print(''.join(t['id'] for t in json.load(sys.stdin) if t['name']=='$NAME'))"; }
ID=$(tunnel_id)
[ -n "$ID" ] || { $CF tunnel create $NAME; ID=$(tunnel_id); }
echo "Tunnel $NAME = $ID"

cat > ~/.cloudflared/pin.yml <<CONF
tunnel: $NAME
credentials-file: $HOME/.cloudflared/$ID.json
ingress:
  - hostname: $HOST
    service: http://localhost:8090
  - service: http_status:404
CONF

# --config: without it cloudflared routes to the tunnel named in config.yml (ddlc-health)
$CF --config ~/.cloudflared/pin.yml tunnel route dns --overwrite-dns $NAME $HOST

mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/pin-tunnel.service <<UNIT
# Publishes the pin station (Pi -> SSH tunnel -> localhost:8090) at $HOST
[Unit]
Description=Cloudflare Tunnel for $HOST
After=network-online.target

[Service]
ExecStart=%h/bin/cloudflared --no-autoupdate --config %h/.cloudflared/pin.yml tunnel run $NAME
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
UNIT
systemctl --user daemon-reload
systemctl --user enable --now pin-tunnel
systemctl --user restart pin-tunnel
sleep 8
echo
echo "Check: $(curl -s -o /dev/null -w '%{http_code}' https://$HOST/) (401 = working, asks for the login)"
