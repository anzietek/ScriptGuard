# Chunking Configuration Fix - Summary

## ❌ Problem Znaleziony

System używał **sliding window chunking** zamiast **hierarchical chunking**, mimo że w konfiguracji było:
```yaml
hierarchical:
  enabled: true
```

---

## 🔍 Analiza Logów (scriptguard_2026-02-11.log)

### Chunking Issue
```
Line 8738: Chunking Strategy: sliding_window        ← ŹLE!
Line 8741: Default strategy: sliding_window

Line (middle): Chunk Type Distribution:
  sliding_window: 98505 (100.0%)                    ← 100% sliding window
  hierarchical:   0 (0%)                            ← 0% hierarchical!
```

### Root Cause
```python
# code_similarity_store.py line 136
chunking_strategy = code_emb_config.get("chunking_strategy", "sliding_window")
                                                             ^^^^^^^^^^^^^^^^
                                                             DEFAULT VALUE
```

**Kod szukał parametru `chunking_strategy` w config.yaml, ale go NIE BYŁO!**

---

## ✅ Fix Zastosowany

### Dodano do config.yaml (linia ~212):
```yaml
code_embedding:
  model: "microsoft/unixcoder-base"

  # Chunking strategy selection
  chunking_strategy: "hierarchical"  # ← NOWY PARAMETR!
                                      # Options: "hierarchical", "sliding_window"

  fewshot:
    enabled: true
    k: 3
    balance_labels: true

  hierarchical:
    enabled: true
    max_function_tokens: 1024
    fallback_to_sliding: true
    languages: ["python"]
```

---

## 📊 Rozkład Danych (z logów)

### Pipeline Flow:
```
1. Raw data collected
   ↓
2. DEDUPLICATION (MinHash LSH)
   → 10,177 samples
   ↓
3. VALIDATION + QUALITY FILTER
   → 9,767 samples
   ├── Malicious: 1,846 (19%)
   └── Benign:    7,921 (81%)
   ↓
4. TRAIN/TEST SPLIT (80/20)
   ├── Train: ~7,814 samples
   └── Test:  ~1,953 samples
   ↓
5. AUGMENTATION (3 variants per malicious)
   "Found 1846 malicious samples to augment"
   → 13,351 training samples
   ├── Malicious: ~5,908 (augmented)
   └── Benign:    ~7,443
   ↓
6. VECTORIZATION (chunking)
   → 11,418 samples → 98,505 chunks
   Average: 8.63 chunks/sample
```

### Finalne Proporcje Train Set:

| Etap | Malicious | Benign | Razem | Ratio |
|------|-----------|--------|-------|-------|
| Po deduplikacji | ~2,000 | ~8,177 | 10,177 | 1:4 |
| Po walidacji | 1,846 | 7,921 | 9,767 | 1:4.3 |
| Train (przed aug) | ~1,477 | ~6,337 | ~7,814 | 1:4.3 |
| **Train (po aug)** | **~5,908** | **~7,443** | **13,351** | **~1:1.3** ✅ |

**Obserwacje:**
- ✅ Augmentacja zbalansowała dataset (malicious:benign ≈ 1:1.3)
- ✅ MinHash LSH: 0 exact duplicates (10177 → 10177) - samples są unikalne
- ⚠️  Sliding window utworzył 98,505 chunks (8.63/sample) - za dużo!
- ⚠️  Hierarchical powinien tworzyć mniej chunks (funkcje zamiast fixed-size)

---

## 🎯 Oczekiwane Rezultaty po Fix

### Przed (Sliding Window):
```
Samples: 11,418
Chunks: 98,505
Avg chunks/sample: 8.63
Strategy: 100% sliding_window
```

### Po (Hierarchical):
```
Samples: 11,418
Chunks: ~30,000-50,000 (szacowane)  ← MNIEJ chunks
Avg chunks/sample: ~3-5             ← WIĘCEJ kontekstu per chunk
Strategy: ~80-90% hierarchical, ~10-20% sliding (fallback)
```

**Korzyści:**
- ✅ Chunki = funkcje/klasy (semantycznie sensowne)
- ✅ Mniej chunks = szybszy retrieval w RAG
- ✅ Więcej kontekstu per chunk = lepsze embeddings
- ✅ Lepsza jakość Few-Shot RAG

---

## 🧪 Weryfikacja

### Po następnym uruchomieniu pipeline, sprawdź logi:

**Powinno być:**
```
✓ Chunking Strategy: hierarchical
✓ Chunk Type Distribution:
    hierarchical: ~8000-10000 (80-90%)
    sliding_window: ~1000-2000 (10-20%)  ← fallback dla długich funkcji
```

**Nie powinno być:**
```
✗ Chunking Strategy: sliding_window
✗ Chunk Type Distribution:
    sliding_window: 98505 (100.0%)
```

### Komenda do sprawdzenia:
```bash
grep "Chunking Strategy\|Chunk Type Distribution" logs/scriptguard_*.log | tail -20
```

---

## 📝 Dodatkowe Uwagi

### Dlaczego hierarchical jest lepsze?

**Sliding Window:**
- Dzieli kod na fixed-size chunks (512 tokens + 64 overlap)
- Ignoruje strukturę kodu (może podzielić funkcję w połowie)
- Tworzy dużo małych chunks (8.63/sample)
- Duplikuje kod przez overlap

**Hierarchical:**
- Ekstraktuje funkcje/klasy z AST
- Zachowuje semantyczną strukturę
- Tworzy mniej, większych chunks (cała funkcja)
- Fallback do sliding window dla długich funkcji (>1024 tokens)

### MinHash LSH a Chunking

**Pytanie użytkownika:** Czy MinHash LSH ma wpływ na hierarchical chunking?

**Odpowiedź:** **NIE**, to niezależne systemy:
1. **MinHash LSH** = Deduplikacja PRZED chunkingiem (krok 2 pipeline)
2. **Hierarchical Chunking** = Vectorization PO deduplikacji (krok 6 pipeline)

Są oddzielone 4 krokami pipeline i działają na różnych etapach.

---

## ✅ Status

- ✅ Problem zidentyfikowany (brakujący `chunking_strategy` parameter)
- ✅ Fix zastosowany (dodano `chunking_strategy: "hierarchical"` do config.yaml)
- ✅ Pamięć zaktualizowana (MEMORY.md)
- ⏳ Wymaga weryfikacji przy następnym uruchomieniu pipeline

**Next Steps:**
1. Uruchom pipeline ponownie: `python src/main.py --config config.yaml`
2. Sprawdź logi czy chunking strategy = hierarchical
3. Porównaj ilość chunks (powinno być ~50k zamiast 98k)
4. Sprawdź metryki treningu (powinny być lepsze z hierarchical)

---

## 🔧 Rollback (jeśli potrzeba)

Jeśli hierarchical nie działa dobrze, zmień w config.yaml:
```yaml
code_embedding:
  chunking_strategy: "sliding_window"  # Wróć do sliding window
```
