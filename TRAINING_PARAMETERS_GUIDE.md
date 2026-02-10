# ScriptGuard Training Parameters Guide

**Wersja:** 1.0
**Ostatnia aktualizacja:** 2026-02-10
**Model:** bigcode/starcoder2-3b + QLoRA

---

## 📋 Spis treści

1. [Model & Hardware](#1-model--hardware)
2. [Batching & Memory](#2-batching--memory)
3. [QLoRA Configuration](#3-qlora-configuration)
4. [Optimization & Regularization](#4-optimization--regularization)
5. [Learning Rate Scheduling](#5-learning-rate-scheduling)
6. [Evaluation & Monitoring](#6-evaluation--monitoring)
7. [Early Stopping](#7-early-stopping)
8. [Precision & Performance](#8-precision--performance)
9. [Experiment Tracking](#9-experiment-tracking)

---

## 1. Model & Hardware

### `model_id`
**Typ:** `string`
**Domyślnie:** `"bigcode/starcoder2-3b"`

**Opis:**
Identyfikator modelu bazowego z Hugging Face Hub. StarCoder2-3B to model specjalizowany w kodzie, zoptymalizowany pod QLoRA fine-tuning.

**Wpływ:**
- Większy model (7B, 15B) → lepsza accuracy, ale wymaga więcej VRAM
- Mniejszy model (1B) → szybszy training, mniejsza accuracy

**Sugerowane wartości:**

| GPU VRAM | Model | Use Case |
|----------|-------|----------|
| 8-12 GB | `bigcode/starcoder2-1b` | Prototyping, szybki development |
| 16-24 GB | `bigcode/starcoder2-3b` | **Produkcja (24GB optimal)** |
| 40-80 GB | `bigcode/starcoder2-7b` | Maximum accuracy, research |

**Aktualna wartość:** `bigcode/starcoder2-3b` ✅

---

### `device`
**Typ:** `string`
**Domyślnie:** `"cuda"`

**Opis:**
Urządzenie do trenowania modelu.

**Wpływ:**
- `cuda` → GPU training (100x szybsze)
- `cpu` → Bardzo wolne (tylko do testów)

**Sugerowane wartości:**
- **Zawsze `cuda`** jeśli masz GPU
- `cpu` tylko do debugowania (bez treningu)

**Aktualna wartość:** `cuda` ✅

---

### `gradient_checkpointing`
**Typ:** `boolean`
**Domyślnie:** `true`

**Opis:**
Zapisuje tylko wybrane aktywacje zamiast wszystkich, ponownie je obliczając podczas backward pass.

**Wpływ:**
- ✅ `true` → Zmniejsza VRAM usage o **~40%**, training **~20% wolniejszy**
- ❌ `false` → Więcej VRAM, szybszy training (może OOM na 3B modelu)

**Sugerowane wartości:**

| VRAM | Rekomendacja |
|------|--------------|
| < 16 GB | `true` (required) |
| 16-24 GB | `true` (optimal dla 3B) |
| > 40 GB | `false` (szybszy training) |

**Aktualna wartość:** `true` ✅

---

### `use_flash_attention_2`
**Typ:** `boolean`
**Domyślnie:** `true`

**Opis:**
Używa Flash Attention 2 (zoptymalizowana implementacja attention mechanism).

**Wpływ:**
- ✅ `true` → **2-3x szybszy training**, mniej VRAM
- ⚠️ Wymaga Ampere+ GPU (RTX 3000+, A100)
- ❌ Nie działa na Windows (używa eager attention fallback)

**Sugerowane wartości:**

| Platform | GPU | Rekomendacja |
|----------|-----|--------------|
| Linux | RTX 3090/4090, A5000+ | `true` |
| Windows | Any | `true` (auto fallback do eager) |
| Colab/RunPod | T4, A100 | `true` |

**Aktualna wartość:** `true` ✅

---

### `group_by_length`
**Typ:** `boolean`
**Domyślnie:** `true`

**Opis:**
Grupuje próbki o podobnej długości w tym samym batchu, zmniejszając padding.

**Wpływ:**
- ✅ `true` → **~15% szybszy training**, mniej WASTED compute
- ❌ `false` → Losowe długości → dużo paddingu → wolniejszy

**Sugerowane wartości:**
- **Zawsze `true`** dla kodu (różne długości plików)

**Aktualna wartość:** `true` ✅

---

## 2. Batching & Memory

### `per_device_train_batch_size`
**Typ:** `int`
**Domyślnie:** `4`

**Opis:**
Liczba próbek przetwarzanych jednocześnie na GPU podczas treningu.

**Wpływ:**
- Większy batch → stabilniejszy gradient, szybszy training, **więcej VRAM**
- Mniejszy batch → mniej VRAM, bardziej "noisy" gradient

**Sugerowane wartości:**

| VRAM | Batch Size | Gradient Accumulation | Efektywny Batch |
|------|------------|----------------------|-----------------|
| 8-12 GB | 1-2 | 16 | 16-32 |
| 16 GB | 2 | 8-16 | 16-32 |
| 24 GB | **4** | **8** | **32** ✅ |
| 40 GB | 8 | 4 | 32 |
| 80 GB | 16 | 2 | 32 |

**Reguła:** Efektywny batch size = `per_device_train_batch_size × gradient_accumulation_steps`
**Optimal:** 32-64 dla fine-tuningu małych modeli

**Aktualna wartość:** `4` (efektywny: 32) ✅

---

### `per_device_eval_batch_size`
**Typ:** `int`
**Domyślnie:** `4`

**Opis:**
Batch size podczas ewaluacji (może być większy niż train, bo nie trzeba gradientów).

**Wpływ:**
- Większy → szybsza ewaluacja
- Brak limitu VRAM (forward pass only)

**Sugerowane wartości:**

| Train Batch | Eval Batch |
|-------------|------------|
| 1-2 | 4-8 |
| 4 | 8-16 |
| 8 | 16-32 |

**Aktualna wartość:** `4` (można zwiększyć do 8-16) ⚠️

---

### `gradient_accumulation_steps`
**Typ:** `int`
**Domyślnie:** `8`

**Opis:**
Liczba kroków forward/backward przed aktualizacją wag. Symuluje większy batch size.

**Wpływ:**
- Większy → efektywnie większy batch, stabilniejszy gradient
- `effective_batch = per_device_train_batch_size × gradient_accumulation_steps`

**Sugerowane wartości:**

| VRAM | Config | Efektywny Batch |
|------|--------|-----------------|
| 8-12 GB | batch=1, accum=32 | 32 |
| 16 GB | batch=2, accum=16 | 32 |
| 24 GB | **batch=4, accum=8** | **32** ✅ |
| 40+ GB | batch=8, accum=4 | 32 |

**Target:** Efektywny batch 32-64

**Aktualna wartość:** `8` ✅

---

## 3. QLoRA Configuration

### `use_qlora`
**Typ:** `boolean`
**Domyślnie:** `true`

**Opis:**
Włącza QLoRA (Quantized Low-Rank Adaptation) - efektywna metoda fine-tuningu.

**Wpływ:**
- ✅ `true` → Model w 4-bit, tylko adaptery w FP16 → **4x mniej VRAM**
- ❌ `false` → Full fine-tuning → wymaga 80+ GB VRAM dla 3B modelu

**Sugerowane wartości:**
- **Zawsze `true`** dla GPU <80GB

**Aktualna wartość:** `true` ✅

---

### `lora_r`
**Typ:** `int`
**Domyślnie:** `16`
**Zakres:** `4-64`

**Opis:**
Ranga macierzy LoRA. Wyższy rank → więcej parametrów do trenowania.

**Wpływ:**
- Niższy (4-8) → mniej parametrów, szybszy training, **może underfitować**
- Wyższy (32-64) → więcej parametrów, wolniejszy, **może overfitować**

**Sugerowane wartości:**

| Rozmiar Datasetu | Task Complexity | lora_r |
|------------------|-----------------|--------|
| < 1k samples | Simple | 8 |
| 1k - 10k | Moderate | **16** ✅ |
| 10k - 100k | Complex | 32 |
| > 100k | Very Complex | 64 |

**Reguła:** `lora_alpha = 2 × lora_r` (typowo)

**Aktualna wartość:** `16` ✅

---

### `lora_alpha`
**Typ:** `int`
**Domyślnie:** `32`

**Opis:**
Scaling factor dla LoRA adaptacji. Kontroluje siłę updatów.

**Wpływ:**
- Wyższy → silniejsze updaty (szybsza konwergencja, ryzyko overfittingu)
- Niższy → delikatniejsze updaty (stabilniejszy training)

**Sugerowane wartości:**
- `lora_alpha = 2 × lora_r` (standard)
- Dla overfittingu: `lora_alpha = lora_r` (słabsze updaty)

**Aktualna wartość:** `32` (2×16) ✅

---

### `lora_dropout`
**Typ:** `float`
**Domyślnie:** `0.15`
**Zakres:** `0.0-0.5`

**Opis:**
Dropout w warstwach LoRA (regularizacja).

**Wpływ:**
- Wyższy (0.15-0.3) → silniejsza regularizacja, **zapobiega overfittingowi**
- Niższy (0.0-0.05) → słabsza regularizacja, ryzyko overfittingu

**Sugerowane wartości:**

| Symptom | lora_dropout |
|---------|--------------|
| Overfitting (train>>test) | 0.2-0.3 |
| Balanced | **0.15** ✅ |
| Underfitting (train=test, obie niskie) | 0.05-0.1 |

**Aktualna wartość:** `0.15` ✅

---

### `target_modules`
**Typ:** `list[string]`
**Domyślnie:** `["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`

**Opis:**
Które warstwy modelu dostają LoRA adaptery.

**Wpływ:**
- Więcej modułów → więcej parametrów, lepsza adaptacja, **więcej VRAM**
- Mniej modułów → mniej parametrów, szybszy training

**Sugerowane wartości:**

| Preset | Modules | Use Case |
|--------|---------|----------|
| Minimal | `q_proj, v_proj` | Quick experiments |
| Standard | `q_proj, v_proj, k_proj, o_proj` | Balanced |
| **Full Attention** | **q, v, k, o, gate, up, down** | **Production** ✅ |

**Aktualna wartość:** Full (7 modułów) ✅

---

## 4. Optimization & Regularization

### `learning_rate`
**Typ:** `float`
**Domyślnie:** `4e-5` (0.00004)
**Zakres:** `1e-6` - `5e-4`

**Opis:**
Szybkość uczenia się modelu (rozmiar kroku w kierunku gradientu).

**Wpływ:**
- Wyższy (1e-4+) → szybsza konwergencja, **niestabilny training**
- Niższy (1e-5) → stabilniejszy, **wolniejsza konwergencja**

**Sugerowane wartości:**

| Model Size | Task | Learning Rate |
|------------|------|---------------|
| 1B | Fine-tuning | 1e-4 |
| 3B | Fine-tuning | **4e-5** ✅ |
| 7B+ | Fine-tuning | 2e-5 |
| Any | From scratch | 1e-3 - 5e-3 |

**Symptomy:**
- Loss explodes → **zmniejsz** (2e-5)
- Loss nie spada → zwiększ (1e-4)

**Aktualna wartość:** `4e-5` ✅

---

### `weight_decay`
**Typ:** `float`
**Domyślnie:** `0.15`
**Zakres:** `0.0-0.3`

**Opis:**
L2 regularizacja - karze duże wagi (zapobiega overfittingowi).

**Wpływ:**
- Wyższy (0.15-0.3) → silniejsza regularizacja, **zapobiega overfittingowi**
- Niższy (0.01) → słabsza regularizacja
- `0.0` → brak regularizacji (tylko dla dużych datasetów)

**Sugerowane wartości:**

| Dataset Size | Overfitting Risk | weight_decay |
|--------------|------------------|--------------|
| < 1k samples | High | 0.2-0.3 |
| 1k-10k | Moderate | **0.15** ✅ |
| 10k-100k | Low | 0.05-0.1 |
| > 100k | Very Low | 0.01 |

**Aktualna wartość:** `0.15` (wysoka - walczy z overfittingiem) ✅

---

### `label_smoothing_factor`
**Typ:** `float`
**Domyślnie:** `0.1`
**Zakres:** `0.0-0.3`

**Opis:**
Zmiękcza labele (zamiast [0,1] używa [0.05, 0.95]). Model jest mniej "pewny".

**Wpływ:**
- ✅ `0.1-0.15` → Model nie overfituje, **lepiej generalizuje**
- ✅ Pomaga z **Specificity: 0.00** (model przestaje klasyfikować wszystko jako malicious)
- ❌ `0.0` → Model pewny siebie (overfitting risk)

**Sugerowane wartości:**

| Problem | label_smoothing_factor |
|---------|------------------------|
| Model klasyfikuje wszystko jako jedna klasa | **0.1-0.15** ✅ |
| Overfitting (train>>test) | 0.15-0.2 |
| Underfitting | 0.0-0.05 |

**WAŻNE:** To jest **kluczowy parametr** dla Twojego problemu Specificity: 0.00!

**Aktualna wartość:** `0.1` ✅

---

### `optim`
**Typ:** `string`
**Domyślnie:** `"paged_adamw_8bit"`

**Opis:**
Optimizer używany do aktualizacji wag.

**Wpływ:**
- `paged_adamw_8bit` → **Najmniej VRAM**, niemal identyczny do AdamW
- `adamw_8bit` → Nieco więcej VRAM
- `adamw_torch` → Full precision, dużo VRAM

**Sugerowane wartości:**

| VRAM | Optimizer |
|------|-----------|
| < 16 GB | `paged_adamw_8bit` |
| 16-24 GB | **`paged_adamw_8bit`** ✅ |
| > 40 GB | `adamw_torch` (nieznacznie lepsze wyniki) |

**Aktualna wartość:** `paged_adamw_8bit` ✅

---

## 5. Learning Rate Scheduling

### `lr_scheduler_type`
**Typ:** `string`
**Domyślnie:** `"cosine"`

**Opis:**
Jak learning rate zmienia się podczas treningu.

**Wpływ:**
- `linear` → LR liniowo spada (prosty, stabilny)
- `cosine` → **Smooth spadek, warm restarts** (lepsze dla overfittingu)
- `constant` → Stały LR (tylko z warmup)

**Sugerowane wartości:**

| Training Length | Scheduler |
|----------------|-----------|
| 1-3 epochs | **`cosine`** ✅ |
| 5+ epochs | `cosine_with_restarts` |
| Quick experiments | `linear` |

**Aktualna wartość:** `cosine` ✅

---

### `warmup_steps`
**Typ:** `int`
**Domyślnie:** `100`
**Zakres:** `0-500`

**Opis:**
Liczba kroków z małym LR na początku (stopniowo rośnie do `learning_rate`).

**Wpływ:**
- Zapobiega dużym gradientom na początku (stabilizuje training)
- Za dużo → training za wolny
- Za mało → niestabilny start

**Sugerowane wartości:**

| Total Steps | warmup_steps |
|-------------|--------------|
| < 500 | 50 |
| 500-2000 | **100** ✅ |
| 2000-5000 | 200-300 |
| > 5000 | 500 |

**Reguła:** ~5-10% total steps

**Aktualna wartość:** `100` ✅

---

## 6. Evaluation & Monitoring

### `evaluation_strategy`
**Typ:** `string`
**Domyślnie:** `"steps"`

**Opis:**
Jak często uruchamiać ewaluację na test secie.

**Wpływ:**
- `steps` → Co `eval_steps` kroków (najlepsze)
- `epoch` → Po każdym epoce (OK dla wielu epochs)
- `no` → Brak ewaluacji (tylko training loss)

**Sugerowane wartości:**
- **Zawsze `steps`** (pozwala na early stopping)

**Aktualna wartość:** `steps` ✅

---

### `eval_steps`
**Typ:** `int`
**Domyślnie:** `50`

**Opis:**
Co ile kroków uruchomić ewaluację.

**Wpływ:**
- Mniejszy (25-50) → częsta ewaluacja, **łatwiej wykryć overfitting**
- Większy (200+) → rzadka ewaluacja, szybszy training

**Sugerowane wartości:**

| Total Steps | eval_steps |
|-------------|------------|
| < 500 | 25 |
| 500-2000 | **50** ✅ |
| 2000-5000 | 100 |
| > 5000 | 200 |

**Reguła:** ~2-5% total steps

**Aktualna wartość:** `50` ✅

---

### `save_steps`
**Typ:** `int`
**Domyślnie:** `50`

**Opis:**
Co ile kroków zapisać checkpoint.

**Wpływ:**
- Mniejszy → częste zapisy, **więcej dysku**
- Większy → rzadkie zapisy, ryzyko utraty progressu

**Sugerowane wartości:**
- Powinien być **równy `eval_steps`** (zapisuj po każdej ewaluacji)

**Aktualna wartość:** `50` (= eval_steps) ✅

---

### `load_best_model_at_end`
**Typ:** `boolean`
**Domyślnie:** `true`

**Opis:**
Po zakończeniu trainingu wczytaj najlepszy checkpoint (wg `metric_for_best_model`).

**Wpływ:**
- ✅ `true` → Używasz najlepszego modelu (nie ostatniego)
- Wymaga ewaluacji (`evaluation_strategy != "no"`)

**Sugerowane wartości:**
- **Zawsze `true`** (ostatni checkpoint może być overfitted)

**Aktualna wartość:** `true` ✅

---

### `metric_for_best_model`
**Typ:** `string`
**Domyślnie:** `"eval_loss"`

**Opis:**
Metryka używana do wyboru najlepszego checkpointa.

**Wpływ:**
- `eval_loss` → Wybiera checkpoint z najniższym eval loss (typowy wybór)
- `eval_accuracy` → Wybiera checkpoint z najwyższą accuracy

**Sugerowane wartości:**
- **`eval_loss`** dla większości przypadków ✅

**Aktualna wartość:** `eval_loss` ✅

---

### `test_split_size`
**Typ:** `float`
**Domyślnie:** `0.2`
**Zakres:** `0.1-0.3`

**Opis:**
Frakcja danych używana jako test set.

**Wpływ:**
- Większy (0.2-0.3) → bardziej reliable eval, **mniej danych do treningu**
- Mniejszy (0.1) → więcej danych do treningu, mniej reliable eval

**Sugerowane wartości:**

| Dataset Size | test_split_size |
|--------------|-----------------|
| < 1k | 0.3 |
| 1k-10k | **0.2** ✅ |
| > 10k | 0.1 |

**Aktualna wartość:** `0.2` ✅

---

## 7. Early Stopping

### `early_stopping`
**Typ:** `boolean`
**Domyślnie:** `true`

**Opis:**
Zatrzymuje training gdy eval metric przestaje się poprawiać.

**Wpływ:**
- ✅ `true` → Oszczędza czas, **zapobiega overfittingowi**
- ❌ `false` → Training do końca epochs (może overfitować)

**Sugerowane wartości:**
- **Zawsze `true`** (zwłaszcza dla małych datasetów)

**Aktualna wartość:** `true` ✅

---

### `early_stopping_patience`
**Typ:** `int`
**Domyślnie:** `2`
**Zakres:** `1-5`

**Opis:**
Ile ewaluacji bez poprawy przed zatrzymaniem trainingu.

**Wpływ:**
- Niższy (1-2) → szybkie zatrzymanie (może przedwcześnie)
- Wyższy (5+) → dłuższe czekanie (może overfitować)

**Sugerowane wartości:**

| eval_steps | Patience | Stops After |
|------------|----------|-------------|
| 25 | 3 | 75 steps bez poprawy |
| 50 | **2** | **100 steps bez poprawy** ✅ |
| 100 | 2 | 200 steps bez poprawy |

**Reguła:** Patience × eval_steps = 100-200 steps

**Aktualna wartość:** `2` ✅

---

### `early_stopping_threshold`
**Typ:** `float`
**Domyślnie:** `0.001`

**Opis:**
Minimalna poprawa metryki uznawana za "improvement" (0.001 = 0.1%).

**Wpływ:**
- Większy (0.01) → wymaga wyraźnej poprawy (wcześniejsze stopping)
- Mniejszy (0.0001) → akceptuje małe poprawy (dłuższy training)

**Sugerowane wartości:**
- `0.001` (0.1% improvement) - standard ✅
- `0.0` - każda poprawa się liczy

**Aktualna wartość:** `0.001` ✅

---

## 8. Precision & Performance

### `bf16`
**Typ:** `boolean`
**Domyślnie:** `true`

**Opis:**
Brain Float 16 precision (FP16 z większym zakresem).

**Wpływ:**
- ✅ `true` → **2x szybszy training**, połowa VRAM, identyczne wyniki
- Wymaga Ampere+ GPU (RTX 3000+, A100)

**Sugerowane wartości:**

| GPU | bf16 |
|-----|------|
| RTX 3090/4090, A5000+ | **`true`** ✅ |
| RTX 2080, V100 | `false` (użyj fp16) |
| CPU | `false` |

**Aktualna wartość:** `true` ✅

---

### `tf32`
**Typ:** `boolean`
**Domyślnie:** `true`

**Opis:**
TensorFloat32 - używa FP32 range z FP16 precision w operacjach tensorowych.

**Wpływ:**
- ✅ `true` → **20% szybszy training** na Ampere GPU
- Tylko Ampere+ (RTX 3000+, A100)
- Nie wpływa na VRAM

**Sugerowane wartości:**

| GPU | tf32 |
|-----|------|
| RTX 3090/4090, A5000, A100 | **`true`** ✅ |
| Starsze GPU | `false` (brak wsparcia) |

**Aktualna wartość:** `true` ✅

---

## 9. Experiment Tracking

### `report_to`
**Typ:** `list[string]`
**Domyślnie:** `["wandb"]`

**Opis:**
Gdzie logować metryki treningowe.

**Wpływ:**
- `wandb` → Weights & Biases (cloud tracking)
- `tensorboard` → Lokalne tensorboard
- `none` → Brak trackingu

**Sugerowane wartości:**
- Development: `["wandb"]` ✅
- Production: `["wandb", "tensorboard"]`
- Offline: `["tensorboard"]`

**Aktualna wartość:** `["wandb"]` ✅

---

### `run_name`
**Typ:** `string`
**Domyślnie:** `"scriptguard-balanced-v2"`

**Opis:**
Nazwa eksperymentu w Wandb.

**Wpływ:**
- Sensowna nazwa → łatwiej porównać eksperymenty

**Sugerowane wartości:**
- Format: `{projekt}-{dataset}-{version}`
- Przykład: `scriptguard-balanced-v3`, `scriptguard-dedup-85`

**Aktualna wartość:** `scriptguard-balanced-v2` ✅

---

## 🎯 Quick Reference: Problemy i Rozwiązania

### Problem: Overfitting (train 98%, test 85%)

**Rozwiązanie:**
```yaml
weight_decay: 0.2              # było 0.15
lora_dropout: 0.2              # było 0.15
label_smoothing_factor: 0.15   # było 0.1
early_stopping_patience: 2     # już OK
```

---

### Problem: Underfitting (train i test oba ~70%)

**Rozwiązanie:**
```yaml
learning_rate: 1e-4            # było 4e-5
lora_r: 32                     # było 16
weight_decay: 0.05             # było 0.15
num_epochs: 3                  # było 1
```

---

### Problem: Specificity: 0.00 (klasyfikuje wszystko jako malicious)

**Rozwiązanie:** ✅ **Już zaimplementowane:**
```yaml
label_smoothing_factor: 0.1    # Model mniej pewny
augment_after_split: true      # Zapobiega data leakage
balance_method: "hybrid"       # Lepszy balans klas
dedup_threshold: 0.85          # Więcej diversity
```

---

### Problem: OOM (Out of Memory)

**Rozwiązanie:**
```yaml
per_device_train_batch_size: 2  # było 4
gradient_accumulation_steps: 16 # było 8 (ten sam efektywny batch)
gradient_checkpointing: true    # już OK
```

---

### Problem: Training za wolny

**Rozwiązanie:**
```yaml
group_by_length: true          # już OK
use_flash_attention_2: true    # już OK
tf32: true                     # już OK
per_device_train_batch_size: 8 # było 4 (jeśli masz VRAM)
eval_steps: 100                # było 50 (rzadsza eval)
```

---

## 📊 Optimized Configurations

### Small Dataset (< 2k samples)
```yaml
learning_rate: 2e-5
weight_decay: 0.2
label_smoothing_factor: 0.15
lora_dropout: 0.2
early_stopping_patience: 2
num_epochs: 3
```

### Medium Dataset (2k-10k samples) - **YOUR CASE**
```yaml
learning_rate: 4e-5            # ✅ Current
weight_decay: 0.15             # ✅ Current
label_smoothing_factor: 0.1    # ✅ Current
lora_dropout: 0.15             # ✅ Current
early_stopping_patience: 2     # ✅ Current
num_epochs: 1                  # ✅ Current
```

### Large Dataset (> 10k samples)
```yaml
learning_rate: 4e-5
weight_decay: 0.05
label_smoothing_factor: 0.05
lora_dropout: 0.1
early_stopping_patience: 3
num_epochs: 1
```

---

## 🚀 Performance Optimization Checklist

- ✅ `gradient_checkpointing: true` - oszczędza VRAM
- ✅ `use_flash_attention_2: true` - 2-3x szybszy
- ✅ `tf32: true` - 20% szybszy (Ampere GPU)
- ✅ `bf16: true` - 2x szybszy
- ✅ `group_by_length: true` - 15% szybszy
- ✅ `per_device_train_batch_size: 4` - optimal dla 24GB
- ✅ `gradient_accumulation_steps: 8` - efektywny batch 32

**Wszystkie optymalizacje już włączone!** 🎉

---

## 📖 Dalsze Materiały

- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [Hugging Face Training Guide](https://huggingface.co/docs/transformers/training)
- [Label Smoothing Explained](https://arxiv.org/abs/1906.02629)

---

**Ostatnia aktualizacja:** 2026-02-10
**Autor:** Claude (Anthropic) + ScriptGuard Team
