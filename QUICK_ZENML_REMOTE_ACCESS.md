# 🚀 Quick Guide: ZenML Remote Access na RunPod

## ✅ Co zostało dodane?

### 1. SSH Tunnel z portem 8237
```bash
# W podrun-setup.sh dodano:
-L 8237:127.0.0.1:8237    # ZenML Dashboard
```

### 2. Auto-konfiguracja projektu "default"
```bash
# Po uruchomieniu ZenML Server automatycznie:
- Przełącza projekt na "default"
- Eliminuje ostrzeżenia o Pro
- Zapewnia widoczność w dashboard
```

---

## 🎯 Jak używać?

### Na RunPod:
```bash
cd /workspace/ScriptGuard
bash podrun-setup.sh -y
```

### Na swoim komputerze (62.171.130.236):
```bash
# Otwórz przeglądarkę:
http://localhost:8237

# Login jako:
Username: adix79
```

---

## 🔍 Weryfikacja

### ✅ Sprawdź czy działa:

**1. Tunnel aktywny?**
```bash
ps aux | grep "8237"
# Powinno pokazać: ssh ... -L 8237:127.0.0.1:8237
```

**2. Projekt ustawiony?**
```bash
uv run python -c "from zenml.client import Client; print(Client().active_project.name)"
# Output: default
```

**3. Dashboard dostępny?**
```bash
curl http://localhost:8237/health
# Output: {"status":"ok"}
```

---

## 📊 Architektura połączenia

```
[RunPod Container]                [SSH Tunnel]                [Twój Komputer]
                                                              (62.171.130.236)
ZenML Server ━━━━━━━━━━━━━━━━━━━> Port 8237 ━━━━━━━━━━━━━> localhost:8237
0.0.0.0:8237                      (encrypted)                 (browser)

PostgreSQL   ━━━━━━━━━━━━━━━━━━━> Port 5432 ━━━━━━━━━━━━━> localhost:5432
Qdrant       ━━━━━━━━━━━━━━━━━━━> Port 6333 ━━━━━━━━━━━━━> localhost:6333
```

---

## 🎉 Rezultat

### Przed:
- ❌ Projekt: "scriptguard" (Pro only)
- ❌ Dashboard: Brak dostępu zdalnego
- ❌ Ostrzeżenie: "not accessible through dashboard"

### Po:
- ✅ Projekt: "default" (Community Edition)
- ✅ Dashboard: Dostępny przez http://localhost:8237
- ✅ Pipeline'y: Widoczne w czasie rzeczywistym
- ✅ Artefakty: Dostępne w UI

---

## 📝 Pliki zmienione

**Tylko 1 plik**:
- `podrun-setup.sh` (3 linie dodane)

**0 zmian w kodzie aplikacji!**

---

## 🔥 One-liner do testowania

Po uruchomieniu setup:

```bash
# Sprawdź wszystko na raz:
echo "=== Tunnel ===" && ps aux | grep "8237.*ssh" && \
echo "=== Project ===" && uv run python -c "from zenml.client import Client; print(f'Project: {Client().active_project.name}')" && \
echo "=== Dashboard ===" && curl -s http://localhost:8237/health

# Oczekiwany output:
# === Tunnel ===
# ssh ... -L 8237:127.0.0.1:8237 ...
# === Project ===
# Project: default
# === Dashboard ===
# {"status":"ok"}
```

---

**Status**: ✅ Ready to deploy
**Testowane**: 2026-02-12
**Czas setup**: ~2 minuty
