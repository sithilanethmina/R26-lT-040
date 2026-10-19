#!/bin/bash
# ==============================================================================
# FairPriceLK - Oracle Cloud VM One-Time Setup & Bootstrap Script
# Compatible with: Ubuntu 22.04/24.04 LTS & Oracle Linux 8/9 on OCI VM.Standard.E2.1.Micro
# ==============================================================================

set -euo pipefail

echo "========================================================================"
echo " Starting FairPriceLK Oracle Cloud VM Setup"
echo " Target: VM.Standard.E2.1.Micro (1 OCPU, 1 GB RAM, AMD x86_64)"
echo "========================================================================"

# 1. Configure 2GB Swap Space (Crucial for 1GB RAM instances to prevent OOM)
if [ ! -f /swapfile ]; then
    echo "[1/5] Creating 2GB swapfile to prevent OOM errors..."
    sudo fallocate -l 2G /swapfile || sudo dd if=/dev/zero of=/swapfile bs=1M count=2048
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
    # Set swappiness to 20 so RAM is preferred, but swap prevents crashes
    sudo sysctl vm.swappiness=20
    echo 'vm.swappiness=20' | sudo tee -a /etc/sysctl.conf
    echo "  [+] 2GB swapfile created and enabled."
else
    echo "[1/5] Swapfile already exists. Skipping."
fi

# 2. Update packages and install prerequisites
echo "[2/5] Updating packages and installing prerequisites..."
if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update -y
    sudo apt-get install -y curl git ufw iptables ca-certificates gnupg lsb-release
elif command -v dnf >/dev/null 2>&1; then
    sudo dnf update -y
    sudo dnf install -y curl git iptables ca-certificates gnupg2
fi

# 3. Install Docker Engine and Docker Compose Plugin
if ! command -v docker >/dev/null 2>&1; then
    echo "[3/5] Installing Docker Engine..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    rm -f get-docker.sh
    echo "  [+] Docker installed successfully."
else
    echo "[3/5] Docker is already installed."
fi

# Unconditionally configure user permissions and ensure docker service + socket are running
echo "  [*] Configuring Docker service and socket..."
sudo usermod -aG docker "$USER" || true
sudo systemctl daemon-reload || true
sudo systemctl enable docker || true
sudo systemctl restart docker || sudo systemctl start docker || sudo service docker start || true
sudo systemctl enable docker.socket || true
sudo systemctl start docker.socket || true

# Verify docker socket is available
if sudo docker info >/dev/null 2>&1; then
    echo "  [+] Docker daemon is active and running."
else
    echo "  [!] Warning: Docker daemon is not responding yet."
fi

# 4. Open Ports 80 and 8000 in OS Firewall
echo "[4/5] Opening Ports 80 and 8000 in OS Firewall..."
# Ubuntu UFW
if command -v ufw >/dev/null 2>&1; then
    sudo ufw allow 22/tcp || true
    sudo ufw allow 80/tcp || true
    sudo ufw allow 8000/tcp || true
    sudo ufw --force enable || true
fi

# Oracle Cloud iptables rules (OCI Ubuntu/Oracle Linux images have strict iptables by default)
if sudo iptables -L INPUT -n --line-numbers 2>/dev/null | grep -q "REJECT"; then
    echo "  [*] Inserting iptables ACCEPT rules for ports 80 and 8000 before REJECT rule..."
    sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 80 -j ACCEPT || \
    sudo iptables -I INPUT 1 -p tcp --dport 80 -j ACCEPT || true
    sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 8000 -j ACCEPT || \
    sudo iptables -I INPUT 1 -p tcp --dport 8000 -j ACCEPT || true
    
    # Persist iptables
    if command -v netfilter-persistent >/dev/null 2>&1; then
        sudo netfilter-persistent save || true
    elif [ -d /etc/iptables ]; then
        sudo iptables-save | sudo tee /etc/iptables/rules.v4 > /dev/null || true
    fi
fi

# Oracle Linux firewalld
if command -v firewall-cmd >/dev/null 2>&1 && systemctl is-active --quiet firewalld; then
    sudo firewall-cmd --permanent --add-port=80/tcp || true
    sudo firewall-cmd --permanent --add-port=8000/tcp || true
    sudo firewall-cmd --reload || true
fi

echo "  [+] Ports 80 and 8000 opened in OS firewall."

# 5. Prepare Repository Workspace
APP_DIR="$HOME/R26-lT-040"
echo "[5/5] Checking repository deployment directory: $APP_DIR"
if [ ! -d "$APP_DIR" ]; then
    echo "  Cloning repository into $APP_DIR..."
    git clone https://github.com/sithilanethmina/R26-lT-040.git "$APP_DIR" || {
        echo "Note: If repository is private, set up an SSH deploy key or Personal Access Token."
    }
fi

if [ -d "$APP_DIR" ] && [ ! -f "$APP_DIR/.env" ]; then
    echo "  Creating default $APP_DIR/.env file..."
    cat << 'EOF' > "$APP_DIR/.env"
PORT=8000
CONSOLIDATED_MODE=true
SAVE_SCREENSHOTS=false
# Optional: Set GEMINI_API_KEY if using unstructured ad text/screenshot extraction
GEMINI_API_KEY=
GEMINI_MODEL=gemini-2.5-flash-lite
EOF
fi

echo "========================================================================"
echo " Setup Complete!"
echo " Next Steps:"
echo " 1. Make sure OCI Security List allows Ingress TCP on port 8000."
echo " 2. In GitHub, add Repository Secrets:"
echo "    - SSH_HOST:        Your Oracle VM Public IP"
echo "    - SSH_USER:        ubuntu (for Ubuntu) or opc (for Oracle Linux)"
echo "    - SSH_PRIVATE_KEY: The private SSH key matching ~/.ssh/authorized_keys"
echo " 3. Push or merge changes to 'main' branch to trigger auto-deployment."
echo "========================================================================"
