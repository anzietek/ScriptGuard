#!/bin/bash

# Server Settings
SERVER_IP="62.171.130.236"
USER="deployer"
# CHANGE THIS to the path of your private key
KEY_PATH="$HOME/.ssh/id_ed25519"

# Clear screen and display info
clear
echo -e "\033[36m=== ScriptGuard Secure Tunnel ===\033[0m" # Cyan
echo "Target: $USER@$SERVER_IP"
echo "Auth:   Private Key ($KEY_PATH)"
echo ""
echo "Opening local ports:"
echo -e " \033[32m[Postgres] 127.0.0.1:5432 -> Server:5432\033[0m" # Green
echo -e " \033[32m[Qdrant]   127.0.0.1:6333 -> Server:6333\033[0m" # Green
echo -e " \033[32m[PgAdmin]  127.0.0.1:5050 -> Server:5050\033[0m" # Green
echo ""
echo -e "\033[33mWARNING: Do not close this window. Minimize it.\033[0m" # Yellow
echo ""

# Check if key exists
if [ ! -f "$KEY_PATH" ]; then
    echo -e "\033[31mError: Private key file not found at: $KEY_PATH\033[0m"
    echo "Please update the KEY_PATH variable in the script."
    exit 1
fi

# Start SSH with Key
ssh -i "$KEY_PATH" -N -L 5432:127.0.0.1:5432 -L 6333:127.0.0.1:6333 -L 5050:127.0.0.1:5050 $USER@$SERVER_IP

# Exit message
echo ""
echo -e "\033[31mConnection closed.\033[0m" # Red
read -p "Press Enter to exit..."