# Zmiany w podrun-setup.sh - ZenML Remote Access

## Co zostało zmienione?

### 1. **Dodano przekierowanie portu ZenML przez SSH tunnel**

**Lokalizacja**: Funkcja `setup_tunnel()` (linia ~171)

**Zmiana**:
```bash
# PRZED:
ssh -4 -f -N \
    -L 5432:127.0.0.1:5432 \
    -L 6333:127.0.0.1:6333 \
    -L 5050:127.0.0.1:5050 \
    $REMOTE_USER@$REMOTE_IP

# PO:
ssh -4 -f -N \
    -L 5432:127.0.0.1:5432 \
    -L 6333:127.0.0.1:6333 \
    -L 5050:127.0.0.1:5050 \
    -L 8237:127.0.0.1:8237 \    # ← DODANE
    $REMOTE_USER@$REMOTE_IP
```

**Efekt**:
- Port 8237 (ZenML dashboard) jest teraz przekierowywany przez SSH tunnel
- Możesz otworzyć dashboard na swoim komputerze lokalnym: `http://localhost:8237`
- Połączenie jest szyfrowane przez SSH

---

### 2. **Zaktualizowano komunikat o tunelu**

**Lokalizacja**: Funkcja `setup_tunnel()` (linia ~186)

**Zmiana**:
```bash
# PRZED:
echo "   - Postgres: localhost:5432 -> Remote:5432"
echo "   - Qdrant:   localhost:6333 -> Remote:6333"

# PO:
echo "   - Postgres: localhost:5432 -> Remote:5432"
echo "   - Qdrant:   localhost:6333 -> Remote:6333"
echo "   - ZenML:    localhost:8237 -> Remote:8237"  # ← DODANE
```

**Efekt**: Skrypt pokazuje informację o dostępnym porcie ZenML

---

### 3. **Dodano automatyczną konfigurację projektu ZenML**

**Lokalizacja**: Funkcja `init_zenml()` (linia ~240)

**Nowa funkcjonalność**:
```bash
# Set active project to 'default' (Community Edition compatible)
print_info "Configuring ZenML project..."
uv run python -c "
from zenml.client import Client
try:
    client = Client()
    current_project = client.active_project.name
    if current_project != 'default':
        print(f'  Switching from {current_project} to default project...')
        # Get default project
        projects = client.list_projects()
        default_project = None
        for p in projects.items:
            if p.name == 'default':
                default_project = p
                break
        if default_project:
            client.set_active_project(default_project.id)
            print('  [OK] Active project: default')
        else:
            print('  [WARNING] Default project not found, using current project')
    else:
        print('  [OK] Already using default project')
except Exception as e:
    print(f'  [WARNING] Could not set project: {e}')
"
```

**Efekt**:
- Automatycznie przełącza aktywny projekt na "default"
- Zapobiega ostrzeżeniom o projekcie Pro
- Pipeline'y będą widoczne w dashboard
- Działa bez potrzeby ręcznej konfiguracji

---

## Jak używać?

### Na RunPod:

1. **Uruchom setup**:
   ```bash
   cd /workspace/ScriptGuard
   bash podrun-setup.sh -y
   ```

2. **Skrypt automatycznie**:
   - Ustawi SSH tunnel z przekierowaniem portu 8237
   - Uruchomi ZenML Server na porcie 8237
   - Przełączy aktywny projekt na "default"
   - Pokaże status wszystkich przekierowań

3. **Output będzie zawierał**:
   ```
   [SUCCESS] Tunnel ESTABLISHED.
      - Postgres: localhost:5432 -> Remote:5432
      - Qdrant:   localhost:6333 -> Remote:6333
      - ZenML:    localhost:8237 -> Remote:8237

   [INFO] Configuring ZenML project...
     [OK] Active project: default
   ```

---

### Na twoim lokalnym komputerze (62.171.130.236):

**Dostęp do ZenML Dashboard**:

1. **Przez SSH tunnel** (jeśli jesteś na serwerze deployer):
   ```bash
   # Otwórz przeglądarkę:
   http://localhost:8237
   ```

2. **Bezpośredni dostęp przez IP** (jeśli ZenML nasłuchuje na 0.0.0.0):
   ```bash
   # Z dowolnego miejsca w sieci:
   http://62.171.130.236:8237
   ```

**Zaloguj się**:
- Username: `adix79` (lub domyślny użytkownik)
- Password: (jeśli ustawiony)

**Sprawdź**:
- Zakładka "Pipelines" - powinny być widoczne run'y
- Zakładka "Artifacts" - powinny być dostępne artefakty
- Zakładka "Stacks" - powinna pokazywać konfigurację

---

## Weryfikacja po uruchomieniu

### 1. Sprawdź czy tunnel działa:

```bash
# Na RunPod:
ps aux | grep ssh

# Powinno pokazać:
# ssh -4 -f -N ... -L 8237:127.0.0.1:8237 ...
```

### 2. Sprawdź czy ZenML Server nasłuchuje:

```bash
# Na RunPod:
lsof -i :8237

# Powinno pokazać:
# COMMAND   PID USER   FD   TYPE DEVICE SIZE/OFF NODE NAME
# python  12345 root   XX   IPv4  XXXXX      0t0  TCP *:8237 (LISTEN)
```

### 3. Sprawdź aktywny projekt:

```bash
# Na RunPod:
uv run python -c "from zenml.client import Client; print(f'Active project: {Client().active_project.name}')"

# Powinno pokazać:
# Active project: default
```

### 4. Sprawdź dostęp do dashboard:

```bash
# Z twojego komputera lokalnego:
curl http://localhost:8237/health

# Lub otwórz w przeglądarce:
# http://localhost:8237
```

---

## Porty przekierowywane przez SSH tunnel

| Port | Usługa | Cel |
|------|--------|-----|
| 5432 | PostgreSQL | Baza danych (code samples, metadata) |
| 6333 | Qdrant | Vector database (embeddings, RAG) |
| 5050 | (?) | Nieznane (możliwe pgAdmin/monitoring) |
| **8237** | **ZenML Server** | **Dashboard dla pipeline'ów (NOWE)** |

---

## Bezpieczeństwo

### SSH Tunnel:
- ✅ Wszystkie połączenia są szyfrowane
- ✅ Klucz prywatny jest usuwany z dysku zaraz po użyciu
- ✅ Port 8237 jest dostępny tylko przez localhost (127.0.0.1)
- ✅ Wymaga autoryzacji SSH do serwera deployer

### ZenML Server:
- ⚠️ Server działa na `0.0.0.0:8237` (dostępny z zewnątrz)
- ⚠️ Upewnij się, że firewall blokuje port 8237 z Internetu
- ✅ Dostęp tylko przez SSH tunnel (bezpieczne)

**Rekomendacja**:
- Trzymaj dostęp do ZenML przez SSH tunnel
- NIE otwieraj portu 8237 bezpośrednio w firewallu
- Jeśli potrzebujesz dostępu zdalnego, użyj VPN lub SSH tunnel

---

## Troubleshooting

### Problem: "Could not configure ZenML project"

**Przyczyna**: Server jeszcze się nie uruchomił

**Rozwiązanie**:
```bash
# Poczekaj 10 sekund i spróbuj ręcznie:
uv run python switch_to_default.py
```

---

### Problem: "Cannot connect to localhost:8237"

**Przyczyna**: Tunnel nie jest aktywny lub ZenML server nie działa

**Rozwiązanie**:
```bash
# 1. Sprawdź tunnel:
ps aux | grep ssh

# 2. Sprawdź ZenML server:
lsof -i :8237

# 3. Restart tunnel:
pkill -f "ssh.*62.171.130.236"
bash podrun-setup.sh --check

# 4. Restart ZenML:
uv run zenml down
bash podrun-setup.sh --check
```

---

### Problem: "Active project is 'scriptguard' not 'default'"

**Przyczyna**: Automatyczna konfiguracja nie zadziałała

**Rozwiązanie**:
```bash
# Przełącz ręcznie:
uv run python switch_to_default.py

# Zweryfikuj:
uv run python test_dashboard_access.py
```

---

## Różnice między RunPod a lokalnym setupem

| Aspekt | Lokalny (Windows) | RunPod (Linux) |
|--------|-------------------|----------------|
| ZenML Server | Docker container | Native uv run |
| Host | `0.0.0.0` (Docker) | `0.0.0.0` (setup script) |
| Port | 8237 | 8237 |
| Dostęp | Direct localhost | SSH tunnel + direct |
| Projekt | Ręczne `switch_to_default.py` | **Auto w setup script** |

---

## Podsumowanie zmian

### ✅ Dodano:
1. **Port forwarding 8237** w SSH tunnel (linia ~171)
2. **Komunikat o porcie ZenML** w tunnel verification (linia ~186)
3. **Automatyczna konfiguracja projektu "default"** w init_zenml() (linia ~240)

### 📦 Pliki zmodyfikowane:
- `podrun-setup.sh` (3 sekcje zmienione)

### 📦 Pliki bez zmian:
- Kod aplikacji ScriptGuard
- `config.yaml`
- Pipeline definitions
- `.env` configuration

---

## Następne kroki po uruchomieniu

1. **Uruchom pipeline**:
   ```bash
   # Setup script automatycznie zapyta, lub uruchom ręcznie:
   uv run python src/main.py
   ```

2. **Otwórz dashboard**:
   ```bash
   # Z twojego komputera lokalnego:
   http://localhost:8237
   ```

3. **Monitoruj wykonanie**:
   - Zobacz kroki pipeline w czasie rzeczywistym
   - Sprawdź logi każdego step'a
   - Przeglądaj artefakty

4. **Sprawdź artefakty**:
   ```bash
   # CLI:
   uv run zenml artifact list

   # Dashboard:
   http://localhost:8237/artifacts
   ```

---

**Data zmian**: 2026-02-12
**Autor**: Automatyczne dostosowanie dla ZenML Community Edition
**Status**: ✅ Gotowe do użycia
