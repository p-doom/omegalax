# Required Changes for State Passing

## train_text_sft.py

### Warum die aktuelle train_text_sft.py nicht kompatibel ist

Im aktuellen Code von `train_text_sft.py` gibt es zwei fundamentale Blöcke, die State Passing verhindern:

#### A. Die Daten-Pipeline (`_grain_iter`)
* **Aktuell:** Die Daten werden komplett zufällig gemischt (`shuffle=True`) und als unabhängige Sequenzen geladen.
* **Problem:** Für State Passing müssen Segment 1, Segment 2 und Segment 3 desselben Dokuments nacheinander verarbeitet werden. Wenn die Batches gemischt sind, reichst du den Zustand von Dokument A an ein Segment von Dokument B weiter.
* **Anpassung:** Du müsstest den Iterator so umbauen, dass er Dokumente sequenziell in Chunks zerlegt und sicherstellt, dass aufeinanderfolgende Batches im selben Batch-Index zum selben Dokument gehören.

#### B. Neue CLI-Flags
Du müsstest Steuerungsparameter hinzufügen wie:
* `--state_passing` (zum Ein- und Ausschalten des Features).
* `--segment_length` (wie lang ein Segment für einen einzelnen Forward Pass sein soll).




Wenn du eine Custom Architektur mit einer eigenen Parameteranzahl (z. B. andere Anzahl an Schichten, Dimensionen oder Köpfen) trainieren möchtest, gibt es dafür zwei sehr saubere Wege.

Methode A: Völlig codefrei (Über eine lokale Hugging-Face config.json)
Da die Ladelogik voll HF-kompatibel ist, musst du keine einzige Zeile Code ändern, um die Dimensionen anzupassen:

Ordner erstellen: Erstelle einen lokalen Ordner (z. B. my_custom_model/).
JSON anlegen: Erstelle in diesem Ordner eine Datei namens config.json.
Parameter eintragen: Definiere darin deine Wunsch-Parameter im Standard-Format (z. B.):
json
{
  "model_type": "qwen3_5",
  "text_config": {
    "vocab_size": 248320,
    "hidden_size": 1024,
    "num_hidden_layers": 12,
    "num_attention_heads": 16,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "intermediate_size": 4096,
    "layer_types": ["linear_attention", "full_attention", ...]
  },
  "vision_config": { ... }
}
Skript starten: Rufe das Skript einfach mit dem Pfad zu diesem Ordner auf:
bash
python scripts/train_text_sft.py --model_id="my_custom_model" ...
Die Schnittstelle in 

api.py
 und die Lade-Funktion in 

config.py
 erkennen automatisch, dass es ein lokaler Pfad ist, und bauen das Modell exakt nach diesen Maßen.

Methode B: Über ein lokales Python-Preset (Smoke Spec)
Falls du deine Custom-Parameter lieber als feste Option im Code verankern willst:

Datei öffnen: Gehe in 

omegalax/models/qwen3_5/config.py
.
Preset hinzufügen: Trage eine neue Konfiguration in das Dictionary _QWEN3_5_SMOKE_SPECS ein (z. B. "my-custom-dense"):
python
"my-custom-dense": {
    "vision_config": _SMOKE_VISION,
    "text_config": {
        "vocab_size": 50000,
        "hidden_size": 512,
        "num_hidden_layers": 8,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "intermediate_size": 2048,
        "layer_types": ("linear_attention", "full_attention", ...),
    }
}
Skript starten: Rufe das Skript mit diesem Preset auf:
bash
python scripts/train_text_sft.py --model_id="my-custom-dense" ...