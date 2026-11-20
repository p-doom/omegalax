# 🚀 Omegalax Test-Infrastruktur & Verifikation

Dieses Dokument bietet einen strukturierten Überblick über die Test-Suite der JAX-basierten Trainingsplattform **Omegalax**. Alle Tests basieren auf dem `absltest`-Framework von Google.

---

## 🔍 A. Modell-Paritätsprüfungen (HuggingFace Alignment / Smoke-Tests)

**Ziel:** Absolute mathematische Korrektheit der JAX/Flax-Implementierungen gegenüber ihren PyTorch/HuggingFace-Referenzen garantieren.

### Funktionsweise
1. Ein minimales HuggingFace-Modell (PyTorch) mit sehr kleinen "Smoke-Test"-Dimensionen wird im RAM erstellt.
2. Das Modell wird als `safetensors` abgespeichert und über die eigene JAX-Parameter-Ladelogik eingelesen.
3. Identische Inputs durchlaufen beide Vorwärtspfade.
4. Über `assert_logits_close` wird sichergestellt, dass die Ausgaben (Logits) bis auf minimale numerische Toleranzgrenzen exakt übereinstimmen.

### Wichtige Testdateien
* **Synthetische Smoke-Tests:**
  * `test_qwen3_5_smoke.py` (MoE-Modelle)
  * `test_qwen3_dense_smoke.py` (Dense-Modelle)
  * `test_qwen3_vl_smoke.py` (Vision-Language-Modelle)
* **Real-Weights Smoke-Tests:**
  * `test_qwen3_5_0_8b.py`:
    Verwendet echte, vortrainierte Gewichte (über `snapshot_download("Qwen/Qwen3.5-0.8B")`). 
    Dieser Test tokenisiert den Beispielsatz *"Why is the sky blue..."* unter Verwendung von Chat-Templates und Left-Padding. Die PyTorch-Token-IDs von der GPU/CPU werden als int32-NumPy-Array (über `np.array(..., dtype=np.int32)`) für das JAX-Modell vorbereitet. Der Test filtert Padding-Tokens über die `attention_mask` aus und fordert mindestens eine **80%ige Übereinstimmung** (`top1_min_match=0.8`) der Top-1-Vorhersagen zwischen JAX und PyTorch.

---

## ⚙️ B. Komponenten- & Modul-Tests

Hier werden isolierte Layer, mathematische Operationen und verteilte Berechnungen getestet:

* **`test_lora.py`**: Überprüft, ob LoRA-Adapter Gewichte korrekt anpassen und die Isolation der Parameter intakt bleibt.
* **`test_rope_dtype.py`**: Stellt sicher, dass Rotary Position Embeddings mit verschiedenen Datentypen stabil und fehlerfrei laufen.
* **`test_tp_attention.py`**: Validiert die Korrektheit des Attention-Mechanismus unter Tensor Parallelism (TP).
* **Gated Delta Net GPU-Kernels (Pallas/Triton):**
  * `test_gated_delta_rule_pallas.py` *(Vorwärtspfad)*: Vergleicht die extrem schnelle, Triton-basierte Pallas-Implementierung (`chunk_gated_delta_rule_pallas`) mit der reinen JAX-Referenz (`chunk_gated_delta_rule_xla` aus `xla_reference.py`) auf Parität bei verschiedenen Tensor-Dimensionen (Toleranz: max. `5e-2` Absolutfehler unter `bf16`).
  * `test_gated_delta_rule_pallas_bwd.py` *(Rückwärtspfad)*: Verwendet `jax.grad` über einen quadratischen Loss (`L = sum(out**2)`), um sicherzustellen, dass die berechneten Gradienten (`dq`, `dk`, `dv`, `dg`, `dbeta`) der Pallas-Implementierung mathematisch identisch zu den perfekten Referenz-Gradienten der JAX-Version sind.

---

## 📦 C. Daten-Pipeline & Collators

Validiert, dass Trainingsdaten korrekt aufbereitet, gemischt und verteilt werden:

* **`test_grain_pipeline.py`** & **`test_data_mixing.py`**: Überprüfen das Einlesen von Grain-Datasets sowie das Mischen mehrerer Datenquellen nach vordefinierten Gewichtungen.
* **`test_sft_collators.py`**: Stellt sicher, dass das Padding, die Truncation und die Maskierung bei multimodalen (VLM) und reinen Text-Datensätzen fehlerfrei funktionieren.

---

## 🏃 D. Ende-zu-Ende Trainings- & Export-Tests

Integrationstests für den gesamten Lebenszyklus eines Modells:

* **`test_sft_training.py`**: Führt echte, extrem kurze 1-Schritt-SFT-Trainingsläufe auf synthetischen Batches aus. Dadurch wird sichergestellt, dass Loss-Berechnung, Gradienten-Akkumulation, Gewichts-Updates und Optimierung einwandfrei zusammenspielen.
* **`test_export_roundtrip_smoke.py`**: Garantiert, dass das Serialisieren (Aufspalten, Speichern, Laden, Zusammenführen) von Flax/NNX-Zuständen ohne Informationsverlust gelingt.
* **`test_perf.py`**: Verifiziert die korrekte Berechnung von Performance-Metriken (TFLOPS-Tracking, Peak-Memory, Schrittgeschwindigkeiten).