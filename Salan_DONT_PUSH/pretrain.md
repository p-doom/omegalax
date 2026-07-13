# Pretraining-Hyperparameter in omegalax einstellen

Diese Notiz mappt die NVIDIA-GatedDeltaNet-Hyperparameter auf diese Codebase.
Wichtig: `omegalax` hat aktuell einen SFT-Trainer (`scripts/train_text_sft.py`,
`scripts/train_vlm_sft.py`), keinen sauber getrennten Pretraining-Trainer. Viele
Optimizer-, Scheduler-, Batch- und Architektur-Hyperparameter sind trotzdem dieselben
Stellschrauben. Fuer echtes Pretraining muss aber die Daten-/Loss-Seite kritisch
geprueft werden.

## Executive Summary

Wenn du wirklich ein eigenes Qwen3.5-artiges Hybrid-Modell pretrainen willst,
reicht es nicht, nur andere Zahlen in `scripts/train_text_sft.py` zu setzen. Das
aktuelle Script ist fuer SFT gebaut: Es erwartet Chat/SFT-Daten, nutzt einen
Assistant-only `loss_mask`, und stoppt nach `num_steps` statt nach einem
expliziten Tokenbudget. Fuer eine saubere Pretraining-Baseline brauchst du daher
eine kleine, aber wichtige Erweiterung der Codebase.

### Was du konkret bauen solltest

```yaml
required_work:
  1_pretrain_data_pipeline:
    goal: plain next-token pretraining statt Chat/SFT
    why: aktuelle SFT-Pipeline trainiert nur auf Assistant-Tokens
    likely_files:
      - scripts/compile_pretrain_dataset.py
      - scripts/build_pretrain_chunk_index.py
      - omegalax/data/pretrain_collator.py

  2_pretrain_script:
    goal: eigenes scripts/train_text_pretrain.py
    why: SFT-Script hat falsche Semantik fuer Pretraining
    base_on:
      - scripts/train_text_sft.py
      - omegalax/trainers/text.py
    key_difference:
      - loss_mask=1 fuer alle echten Tokens
      - optional --max_tokens oder automatische num_steps-Berechnung
      - pretraining-Defaults statt SFT-Defaults

  3_optimizer_flags:
    goal: NVIDIA-Recipe wirklich reproduzierbar machen
    why: adam_beta2=0.95 ist aktuell nicht einstellbar
    likely_files:
      - scripts/train_text_pretrain.py
      - omegalax/trainers/text.py
    add:
      - adam_beta1
      - adam_beta2
      - adam_eps

  4_model_config:
    goal: eigene 0.1B-1B Qwen3.5-Hybrid-Config erstellen
    why: Architekturparameter kommen aus config.json, nicht aus Trainer-Flags
    must_decide:
      - hidden_size
      - num_hidden_layers
      - layer_types
      - linear_key_head_dim
      - linear_num_key_heads
      - linear_num_value_heads
      - linear_value_head_dim
      - intermediate_size oder MoE-Parameter

  5_run_recipe:
    goal: Tokenbudget, Batchtokens, Warmup und LR-Sweep festlegen
    why: max_tokens gibt es aktuell nicht direkt
    compute:
      - tokens_per_step = batch_size * max_length * grad_accum_steps
      - num_steps = target_tokens / tokens_per_step
      - warmup_steps = 0.01 * num_steps
```

### Minimaler sinnvoller Projektplan

1. Erstelle zuerst einen echten Pretraining-Collator.
   Er soll `token_ids_BT`, `attention_mask_BT` und `loss_mask_BT` liefern, wobei
   `loss_mask_BT=1` fuer alle echten Tokens und `0` fuer Padding ist. Ohne diesen
   Schritt trainierst du kein klassisches Language-Model-Pretraining.

2. Kopiere danach `scripts/train_text_sft.py` zu `scripts/train_text_pretrain.py`
   und entferne die SFT-spezifischen Annahmen. Das neue Script kann denselben
   Trainer verwenden, solange die Batch-Dicts gleich aussehen. Der kritische
   Unterschied liegt in der Datenquelle und Loss-Mask.

3. Erweitere `TrainConfig` und `build_optimizer()` um `adam_beta1`,
   `adam_beta2` und `adam_eps`. Sonst kannst du die NVIDIA-Recipe nicht sauber
   nachstellen, weil `optax.adamw` aktuell implizit `beta2=0.999` statt `0.95`
   nutzt.

4. Fuege optional `--max_tokens` zum Pretrain-Script hinzu. Dann kann das Script
   `num_steps` und `warmup_steps` automatisch aus `batch_size`, `max_length` und
   `grad_accum_steps` berechnen. Das reduziert Fehler bei Runs mit verschiedenen
   Batch-Konfigurationen.

5. Erstelle eine lokale HF-style `config.json` fuer dein Zielmodell. Dort
   definierst du `layer_types` und die GDN-Dimensionen. `gated_delta_per_layer`
   aus NVIDIA hat in dieser Codebase keine Wirkung.

6. Starte mit einem kleinen Smoke-Pretraining-Run auf kurzer Sequenzlaenge, dann
   einem 4096er Sanity-Run, dann erst dem eigentlichen LR-Sweep.

### To-do-Liste

```yaml
todo_now:
  - [ ] Zielarchitektur als lokale HF-style config.json definieren
  - [ ] Pretraining-Datenformat festlegen: plain token stream oder Dokumente
  - [ ] Pretraining-Collator bauen: loss_mask fuer alle non-pad Tokens
  - [ ] scripts/train_text_pretrain.py aus train_text_sft.py ableiten
  - [ ] Pretraining-Defaults setzen: cosine, lr_end_factor=0.1, wd=0.1, grad_clip=1.0
  - [ ] adam_beta1/beta2/eps als Flags implementieren
  - [ ] max_tokens oder klare num_steps-Berechnung implementieren
  - [ ] Test: tokens_per_step, warmup_steps und loss_mask verifizieren
  - [ ] Smoke-Run mit winzigem Modell und max_length 128/512
  - [ ] Sanity-Run mit Zielarchitektur und max_length 4096
  - [ ] LR-Sweep: 1e-4, 2e-4/3e-4, 4e-4

todo_later:
  - [ ] local/sliding-window attention fuer full_attention layers evaluieren
  - [ ] weight-decay mask fuer Norm/Bias/GDN-Spezialparameter pruefen
  - [ ] Parameterzaehler fuer total/block-only/tied-embedding einbauen
  - [ ] Validierungsset und Perplexity-Logging fuer Pretraining standardisieren
  - [ ] Checkpoint-Export auf Pretraining-Runs testen
```

### Meine Einschaetzung

Ja: Du solltest wahrscheinlich ein eigenes `train_text_pretrain.py` machen.
Nicht weil der Trainer komplett anders sein muss, sondern weil die Semantik des
Run-Entrypoints anders ist. SFT und Pretraining unterscheiden sich hier vor allem
in Datenformat, Loss-Mask, Defaults und Tokenbudget-Handling. Wenn du das alles
in `train_text_sft.py` hineinbaust, wird das Script schwerer zu verstehen und du
riskierst versehentlich SFT- und Pretraining-Runs zu vermischen.

Der vorhandene `omegalax/trainers/text.py` kann wahrscheinlich weiterverwendet
werden, weil der eigentliche Loss bereits next-token prediction macht. Der
entscheidende Punkt ist, dass `loss_mask_BT` fuer Pretraining anders erzeugt
werden muss.

## Kurzfazit

Die wichtigsten Hyperparameter, die du wirklich aktiv setzen solltest:

```yaml
must_set:
  learning_rate: CLI --learning_rate
  lr_schedule: CLI --lr_schedule=cosine
  lr_end_factor: CLI --lr_end_factor=0.1
  warmup_steps: CLI --warmup_steps=<aus Tokenbudget berechnen>
  weight_decay: CLI --weight_decay=0.1
  max_grad_norm: CLI --max_grad_norm=1.0
  batch_tokens: CLI --batch_size * --max_length * --grad_accum_steps
  sequence_length: CLI --max_length und Chunk-Index --max_length
  architecture: HF-style config.json, nicht ueber Trainer-Flags

currently_missing_or_mismatched:
  adam_beta2_0_95: nicht exposed; optax.adamw nutzt Default beta2=0.999
  adam_eps: nicht exposed; optax Default ist effektiv 1e-8
  max_tokens: nicht exposed; du setzt stattdessen --num_steps
  local_window_2048: in Qwen3.5 Attention aktuell nicht implementiert
  pure_pretraining_objective: aktuelle Pipeline ist SFT/chat-loss-mask-orientiert
```

Wenn du nur eine Sache aus dieser Datei mitnimmst: Nicht nur die Zahlen kopieren.
In dieser Codebase sind `adam_beta2`, `max_tokens`, `local_window` und die
Pretraining-Datenpipeline die Stellen, an denen die NVIDIA-Recipe nicht 1:1
abgebildet ist.

## Wo Training-Hyperparameter gesetzt werden

Der text-only Einstieg ist:

```text
scripts/train_text_sft.py
```

Die Flags stehen dort in den Zeilen 41-52:

```python
flags.DEFINE_integer("max_length", 512, "Maximum sequence length.")
flags.DEFINE_integer("num_steps", 100, "Number of training steps.")
flags.DEFINE_integer("batch_size", 8, "Global batch size across all JAX processes.")
flags.DEFINE_float("learning_rate", 2e-5, "Learning rate.")
flags.DEFINE_float("weight_decay", 0.01, "Weight decay.")
flags.DEFINE_integer("warmup_steps", 0, "Linear LR warmup steps.")
flags.DEFINE_enum("lr_schedule", "linear", ["linear", "cosine", "wsd"], ...)
flags.DEFINE_float("lr_end_factor", 0.0, ...)
flags.DEFINE_float("max_grad_norm", 1.0, ...)
flags.DEFINE_integer("grad_accum_steps", 1, ...)
```

Diese Werte werden in `omegalax/trainers/text.py` in `TrainConfig` uebernommen
und in `build_optimizer()` benutzt.

## Mapping: NVIDIA-HParams zu omegalax

| NVIDIA-Feld                         | In omegalax setzen                                       |                      Status | Kommentar                                                                                                        |
| ----------------------------------- | -------------------------------------------------------- | --------------------------: | ---------------------------------------------------------------------------------------------------------------- |
| `learning_rate`                     | `--learning_rate`                                        |                      direkt | Sinnvoll zu sweepen. Der Default `2e-5` im SFT-Script ist fuer Pretraining viel zu niedrig.                      |
| `scheduler: cosine`                 | `--lr_schedule=cosine`                                   |                      direkt | Der Code-Default ist `linear`, also nicht NVIDIA-like. Explizit setzen.                                          |
| `lr_end_factor: 0.1`                | `--lr_end_factor=0.1`                                    |                      direkt | Entspricht `min_lr = lr / 10`.                                                                                   |
| `warmup: 1% tokens`                 | `--warmup_steps`                                         |                    indirekt | Code nimmt Steps, nicht Tokens. Du musst umrechnen.                                                              |
| `weight_decay: 0.1`                 | `--weight_decay=0.1`                                     |                      direkt | Aber aktuell ohne Masking: Norms und Spezialparameter koennen mit decay bekommen.                                |
| `adam_beta1: 0.9`                   | nicht exposed                                            |                teilweise ok | Optax Default ist 0.9, passt.                                                                                    |
| `adam_beta2: 0.95`                  | nicht exposed                                            |                     Problem | Optax Default ist 0.999. Das ist ein echter Unterschied zur NVIDIA-Recipe.                                       |
| `adam_eps: 1e-8`                    | nicht exposed                                            |               vermutlich ok | Optax AdamW Default ist 1e-8.                                                                                    |
| `max_grad_norm: 1.0`                | `--max_grad_norm=1.0`                                    |                      direkt | Default ist schon 1.0. Explizit setzen fuer Reproduzierbarkeit.                                                  |
| `batch_tokens: ~2M`                 | `--batch_size * --max_length * --grad_accum_steps`       | direkt, aber anders benannt | `batch_size` ist globale Sequenzanzahl pro Microstep; `grad_accum_steps` multipliziert Token pro Optimizer-Step. |
| `sequence_length: 4096`             | `--max_length=4096` plus Chunk-Index `--max_length=4096` |                      direkt | Muss in Dataset-Build und Training gleich sein.                                                                  |
| `max_tokens: 15B/100B`              | `--num_steps`                                            |                    indirekt | Es gibt kein `--max_tokens`. Du berechnest Steps aus Tokenbudget.                                                |
| `sliding_window/local_window: 2048` | aktuell nicht vorhanden                                  |                       fehlt | Qwen3.5 full attention ist im Code volle kausale Attention, kein lokales Fenster.                                |


## Gemini Comments:  (lesen und berücksichtigen)

1. Skalierung der Gewichtsinitialisierung (Depth-Scaling)
Das Problem: In modernen LLMs (wie LLaMA, Qwen) werden die Gewichte der residualen Projektionslayer (o_proj in 

attention.py
 und down_proj in 

model.py
) mit einer modifizierten Standardabweichung initialisiert, die mit der Tiefe des Netzwerks herunterskaliert wird. Typischerweise wird die Standardabweichung mit 1 / sqrt(2 * num_hidden_layers) multipliziert.
Aktueller Zustand im Code: Der Code verwendet für fast alle linearen Layer pauschal nnx.initializers.lecun_normal(). Bei tieferen Modellen (z. B. ab 24 Layern) führt dies beim Pretraining oft zu einer Signalexplosion im Vorwärtspfad und zu instabilen Gradienten in den ersten Schritten.
Empfehlung: Überschreibe die kernel_init-Funktionen für Projektions-Layer so, dass sie die Anzahl der Layer berücksichtigen.

Depth-Scaling der Init

  Für Qwen3.5 dense ist dieser Punkt noch sauberer als bei MoE, weil du keine Expert-Down-Projections berücksichtigen
  musst. Relevant sind vor allem die residualen Output-Projektionen:

  - Attention.o_proj in Projects/omegalax/omegalax/models/qwen3_5/attention.py:58
  - MLP.down_proj in Projects/omegalax/omegalax/models/qwen3_5/model.py:53
  - falls du weiterhin linear_attention-Layer hast: GatedDeltaNet.out_proj in Projects/omegalax/omegalax/models/
    qwen3_5/deltanet.py:110

  Gemini hat hier im Kern recht: Wenn du from scratch pretrainierst und nicht HF-Gewichte lädst, ist pauschales
  lecun_normal() für alle linearen Layer riskanter als eine LLM-typische residual-scaled Init. Ich würde nicht Q/K/V,
  Gate/Up usw. pauschal skalieren, sondern gezielt die Residual-Branches, die wieder in den Residual Stream addiert
  werden.

  Urteil: sinnvoller Todo. Für 24+ Layer würde ich das ernst nehmen.


Infrastruktur- / Parallelisierungs-Hyperparameter
Diese bestimmen die Verteilung auf deinen GPUs/TPUs, um Out-of-Memory (OOM) Fehler zu vermeiden.

Relevante Parameter:
- tp_size (Tensor Parallelism: Aufsplittung einzelner Layer auf N GPUs)
- fsdp_size (Fully Sharded Data Parallelism: Sharding von Optimizer/Gewichten)
- dp_size (klassisches Data Parallelism)


Trainings- und Optimierungs-Hyperparameter
- grad_accum_steps (Anzahl der Akkumulationsschritte zur Simulation größerer Batches)



  1. Depth-Scaling der Init

  Für Qwen3.5 dense ist dieser Punkt noch sauberer als bei MoE, weil du keine Expert-Down-Projections
  berücksichtigen musst. Relevant sind vor allem die residualen Output-Projektionen:

  - Attention.o_proj in Projects/omegalax/omegalax/models/qwen3_5/attention.py:58
  - MLP.down_proj in Projects/omegalax/omegalax/models/qwen3_5/model.py:53
  - falls du weiterhin linear_attention-Layer hast: GatedDeltaNet.out_proj in Projects/omegalax/omegalax/
    models/qwen3_5/deltanet.py:110

  Gemini hat hier im Kern recht: Wenn du from scratch pretrainierst und nicht HF-Gewichte lädst, ist pauschales
  lecun_normal() für alle linearen Layer riskanter als eine LLM-typische residual-scaled Init. Ich würde nicht
  Q/K/V, Gate/Up usw. pauschal skalieren, sondern gezielt die Residual-Branches, die wieder in den Residual
  Stream addiert werden.

  Urteil: sinnvoller Todo. Für 24+ Layer würde ich das ernst nehmen.

  2. AdamW Epsilon

  Der Code setzt eps aktuell nicht explizit: Projects/omegalax/omegalax/trainers/text.py:85. Gemini hat also
  recht, dass es implizit läuft.

  Aber: Die Begründung “BF16-Unterlauf wegen eps=1e-8” ist in deiner Codebase nicht besonders stark, weil dein
  Optimizer Gradients/Updates in fp32 verarbeitet: Projects/omegalax/omegalax/trainers/optim.py:27. eps=1e-8
  ist außerdem auch in vielen AdamW-Rezepten normal.

  Wichtiger als eps ist bei dir weiterhin: adam_beta2=0.95 ist aktuell nicht exposed. Ohne Änderung nutzt du
  wahrscheinlich Optax-Default 0.999, was deutlich weiter von NVIDIA/LLM-Pretraining-Recipes weg ist als
  eps=1e-8.

  Urteil: eps als Flag exposen ja. Default nicht blind auf 1e-6 ändern. Erst beta2=0.95 fixen.

  3. Gradienten-Clipping

  Geminis Kommentar ist widersprüchlich: Er sagt 1.0 sei zu hoch, empfiehlt dann aber 1.0 oder niedriger. Für
  Qwen3.5 dense ist max_grad_norm=1.0 eine völlig plausible Pretraining-Baseline.

  In deinem CLI ist 1.0 bereits Default: Projects/omegalax/scripts/train_text_sft.py:51. In TrainConfig direkt
  ist der Default aber 0.0: Projects/omegalax/omegalax/trainers/text.py:57. Also: im Pretrain-Script explizit
  setzen.

  Urteil: 1.0 ist nicht “zu hoch” als Baseline. 0.5 ist ein Fallback bei Instabilität, nicht mein erster
  Default.






## Rechenregeln fuer Tokenbudget

In dieser Codebase ist:

```text
tokens_per_optimizer_step = batch_size * max_length * grad_accum_steps
num_steps = target_train_tokens / tokens_per_optimizer_step
warmup_steps = warmup_fraction * num_steps
```

Beispiel fuer NVIDIA-aehnliche `~2.1M` Tokens pro Optimizer-Step:

```yaml
max_length: 4096
batch_size: 512
grad_accum_steps: 1
tokens_per_optimizer_step: 512 * 4096 * 1 = 2_097_152
```

Wenn dein Hardware-Batch kleiner ist, kannst du denselben effektiven Batch ueber
Gradient Accumulation bauen:

```yaml
max_length: 4096
batch_size: 64
grad_accum_steps: 8
tokens_per_optimizer_step: 64 * 4096 * 8 = 2_097_152
```

Fuer 15B Tokens:

```yaml
target_train_tokens: 15_000_000_000
tokens_per_optimizer_step: 2_097_152
num_steps: 7153
warmup_steps_1_percent: 72
```

Fuer 100B Tokens:

```yaml
target_train_tokens: 100_000_000_000
tokens_per_optimizer_step: 2_097_152
num_steps: 47684
warmup_steps_1_percent: 477
```

Das unterscheidet sich vom NVIDIA-Repo: Dort wird `max_tokens` direkt im
Trainingsscript verarbeitet. Hier ist `num_steps` die harte Stopp-Bedingung.

## Beispiel-Run fuer text-only Training

Das ist ein SFT-Command im aktuellen Repo-Stil, aber mit pretraining-aehnlichen
Optimizer-HParams:

```bash
uv run scripts/train_text_sft.py \
  --model_id /path/to/my_qwen3_5_config_dir \
  --tokenizer Qwen/Qwen3.5-0.8B \
  --data_path /path/to/train_chunks_4096 \
  --max_length 4096 \
  --batch_size 64 \
  --grad_accum_steps 8 \
  --num_steps 7153 \
  --learning_rate 3e-4 \
  --lr_schedule cosine \
  --lr_end_factor 0.1 \
  --warmup_steps 72 \
  --weight_decay 0.1 \
  --max_grad_norm 1.0 \
  --tp_size 1 \
  --fsdp_size 1 \
  --save_dir runs/pretrain_like/my_qwen3_5_0_5b \
  --save_every 500 \
  --log_every 10
```

Wichtig:

- `--model_id` kann ein lokaler Pfad zu einem HF-style `config.json` sein.
- Wenn der Config-Pfad keine Tokenizer-Dateien enthaelt, setze `--tokenizer`
  explizit.
- `--max_length` muss zum Chunk-Index passen.
- `--batch_size` ist globale Batch-Groesse ueber alle JAX-Prozesse, nicht
  per-device batch.
- `--grad_accum_steps` zaehlt in `tokens_per_optimizer_step` hinein.

## Learning Rate

In `scripts/train_text_sft.py` ist der Default:

```yaml
learning_rate: 2e-5
```

Das ist ein SFT-Default, keine Pretraining-Baseline. Fuer Pretraining eines
0.1B-1B Hybrid-Modells ist das wahrscheinlich zu klein.

Empfehlung fuer deine Baseline-Suite:

```yaml
lr_sweep:
  safe: 1e-4
  main: 2e-4 oder 3e-4
  aggressive: 4e-4
fixed:
  lr_schedule: cosine
  lr_end_factor: 0.1
  warmup_steps: 1% bis 2% von num_steps
  weight_decay: 0.1
  max_grad_norm: 1.0
```

Warum nicht einfach nur `1e-4`? Weil `1e-4` aus dem NVIDIA 0.4B-Repo-Launchscript
konservativ wirkt. Das Paper-Main-Setup nennt `4e-4` fuer 1.3B/100B. Fuer deine
Architektur ist die richtige LR nicht garantiert, aber LR ist der erste Sweep,
den ich machen wuerde.

## Scheduler und `lr_end_factor`

Der Scheduler wird in `omegalax/trainers/lr_schedule.py` gebaut.

Fuer NVIDIA-like Cosine:

```bash
--lr_schedule cosine --lr_end_factor 0.1
```

Das erzeugt:

```text
linear warmup von 0 auf peak_lr
danach cosine decay bis peak_lr * lr_end_factor
```

Das entspricht konzeptionell:

```yaml
min_lr: learning_rate / 10
```

Der Code-Default ist aber:

```yaml
lr_schedule: linear
lr_end_factor: 0.0
```

`linear` bedeutet hier nicht linear decay. In `build_lr_schedule()` ist `linear`
nach Warmup effektiv konstant auf Peak-LR. Fuer Pretraining also explizit
`cosine` setzen.

## Warmup

NVIDIA gibt Warmup oft in Tokens an. `omegalax` gibt Warmup in Steps an:

```bash
--warmup_steps <int>
```

Umrechnung:

```text
warmup_steps = warmup_tokens / (batch_size * max_length * grad_accum_steps)
```

Oder bei 1 Prozent Warmup:

```text
warmup_steps = 0.01 * num_steps
```

Kritischer Punkt: Wenn du `grad_accum_steps` aenderst, aendert sich das
Tokenbudget pro Optimizer-Step. Dann musst du `num_steps` und `warmup_steps`
neu berechnen, sonst vergleichst du nicht denselben Trainingslauf.

## AdamW, Betas und Weight Decay

Der Optimizer wird hier gebaut:

```text
omegalax/trainers/text.py:73-90
```

Aktuell:

```python
chain = []
if train_cfg.max_grad_norm > 0:
    chain.append(optax.clip_by_global_norm(train_cfg.max_grad_norm))
chain.append(optax.adamw(lr, weight_decay=train_cfg.weight_decay))
tx = optax.chain(*chain)
```

Das bedeutet:

```yaml
exposed:
  learning_rate: yes
  weight_decay: yes
  max_grad_norm: yes
not_exposed:
  adam_beta1: no
  adam_beta2: no
  adam_eps: no
```

Optax Defaults sind praktisch:

```yaml
adam_beta1: 0.9
adam_beta2: 0.999
adam_eps: 1e-8
```

Damit passt `beta1=0.9`, aber `beta2=0.95` aus der NVIDIA-Recipe wird aktuell
nicht reproduziert. Das ist nicht nur Kosmetik. `beta2=0.95` reagiert schneller
auf Aenderungen in der Gradientenvarianz; `0.999` ist deutlich traeger und ist
eher ein klassischer Finetuning/Transformer-Default.

Meine Empfehlung:

1. Fuer eine echte NVIDIA-nahe Baseline solltest du `beta1`, `beta2`, `eps` als
   Flags in `TrainConfig` und `build_optimizer()` aufnehmen.
2. Wenn du keinen Code anfassen willst, dokumentiere jeden Run klar als
   `adam_beta2=0.999`, nicht als NVIDIA-Replikation.
3. Fuer kleine erste Stabilitaets-Smokes ist `0.999` okay, aber fuer den
   eigentlichen Pretraining-Vergleich wuerde ich `0.95` implementieren.

### Weight-decay-Masking

`optax.adamw(..., weight_decay=...)` wird aktuell ohne Mask benutzt. Das heisst:
Weight decay wird auf alle trainierbaren Parameter angewendet, sofern Optax hier
nicht intern anders behandelt.

In dieser Architektur betrifft das nicht nur grosse Matrices, sondern potentiell
auch:

- RMSNorm-Gewichte
- `dt_bias`
- `A_log`
- GatedDeltaNet-Norm-Gewichte

Viele LLM-Recipes decayern Norm/Bias-Parameter nicht. Das NVIDIA-Repo nutzt
PyTorch `AdamW(model.parameters(), weight_decay=0.1)` ohne offensichtliche
Parametergruppen in der betrachteten Stelle, also ist "alles decayern" nicht
absurd. Trotzdem: Wenn du Instabilitaet in GDN-Layern siehst, ist WD-Masking
eine plausible Stellschraube.

## Gradient Clipping

Setzen:

```bash
--max_grad_norm 1.0
```

Code:

```text
omegalax/trainers/text.py:82-86
```

Wenn `max_grad_norm > 0`, wird `optax.clip_by_global_norm(max_grad_norm)` vor
AdamW in die Optax-Chain gesetzt.

Bewertung:

- Fuer Hybrid/GDN-Pretraining sinnvoll.
- Ich wuerde `1.0` erstmal lassen.
- Wenn Loss-Spikes/NaNs auftreten, zuerst LR/Warmup pruefen, dann Clip ggf.
  strenger testen.

## Batch Tokens

In `omegalax/trainers/text.py` wird geloggt:

```python
global_tokens_per_step = train_cfg.seq_len * train_cfg.batch_size * accum_steps
```

Das ist die korrekte Definition fuer deinen effektiven Optimizer-Step.

Entsprechend:

```yaml
batch_tokens:
  formula: max_length * batch_size * grad_accum_steps
```

NVIDIA `512 x 4k` entspricht:

```yaml
max_length: 4096
global_sequences_per_optimizer_step: 512
tokens: 2_097_152
```

In `omegalax` kannst du das auf verschiedene Arten erreichen:

```yaml
option_a:
  batch_size: 512
  grad_accum_steps: 1

option_b:
  batch_size: 256
  grad_accum_steps: 2

option_c:
  batch_size: 64
  grad_accum_steps: 8
```

Alle drei haben bei `max_length=4096` denselben effektiven Tokenbatch. Sie haben
aber nicht zwingend dieselbe Performance, weil Grad-Accumulation mehr
Microsteps bedeutet.

## Sequence Length

Es gibt keinen `block_size` wie im NVIDIA-LitGPT-Repo. In dieser Codebase ist
die relevante Sequenzlaenge:

```bash
--max_length
```

Sie wird an zwei Stellen benutzt:

1. Beim Chunk-Index-Build:

```bash
uv run scripts/build_sft_chunk_index.py \
  --data-path /path/to/payload \
  --out-dir /path/to/chunks_4096 \
  --model-id /path/to/my_qwen3_5_config_dir \
  --tokenizer Qwen/Qwen3.5-0.8B \
  --max-length 4096
```

2. Beim Training:

```bash
uv run scripts/train_text_sft.py \
  --data-path /path/to/chunks_4096 \
  --max_length 4096
```

Der Collator erzeugt feste `(B, max_length)` Arrays. Wenn ein Beispiel laenger
ist, wirft er einen Fehler. Mixed datasets mit unterschiedlichen `max_length`
werden ebenfalls abgelehnt.

Bewertung:

- Fuer dein Hybrid-Modell wuerde ich `4096` als bessere Baseline nehmen als
  `2048`, wenn Compute reicht.
- `2048` ist ok fuer billige Smoke-Runs.
- Wenn du `4096` nutzt, baue den Dataset-Index direkt mit `4096`. Spaeter nur
  das Trainer-Flag zu aendern reicht nicht.

## `max_tokens` gibt es nicht

NVIDIA:

```yaml
max_tokens: 15B
warmup_tokens: 150M
```

omegalax:

```yaml
num_steps: int
warmup_steps: int
```

Du musst also vor dem Run rechnen:

```text
num_steps = ceil(max_tokens / (batch_size * max_length * grad_accum_steps))
warmup_steps = ceil(warmup_fraction * num_steps)
```

Das ist wichtig fuer Vergleichbarkeit. Wenn du nur `num_steps` kopierst, aber
Batch oder Sequenzlaenge aenderst, trainierst du auf einem anderen Tokenbudget.

## Architektur-Hyperparameter

Die Modellarchitektur kommt nicht aus Trainer-Flags, sondern aus der
Qwen3.5-Config:

```text
omegalax/models/qwen3_5/config.py
```

Die relevanten Felder in `Qwen3_5TextConfig` sind:

```yaml
text_config:
  vocab_size
  hidden_size
  num_hidden_layers
  num_attention_heads
  num_key_value_heads
  head_dim
  layer_types
  rope_theta
  partial_rotary_factor
  mrope_section
  attention_bias
  tie_word_embeddings
  linear_conv_kernel_dim
  linear_key_head_dim
  linear_num_key_heads
  linear_num_value_heads
  linear_value_head_dim
  intermediate_size          # dense FFN
  moe_intermediate_size      # MoE
  shared_expert_intermediate_size
  num_experts
  num_experts_per_tok
```

Fuer eigene Modelle ist der sauberste Weg: Erstelle ein lokales HF-style
`config.json` und uebergib den Ordner als `--model_id`.

Minimaler dense Qwen3.5-text-only Config-Sketch:

```json
{
  "model_type": "qwen3_5",
  "tie_word_embeddings": false,
  "image_token_id": 248056,
  "video_token_id": 248057,
  "vision_start_token_id": 248053,
  "vision_end_token_id": 248054,
  "vision_config": {
    "depth": 2,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_heads": 4,
    "patch_size": 16,
    "temporal_patch_size": 2,
    "spatial_merge_size": 2,
    "in_channels": 3,
    "out_hidden_size": 128,
    "num_position_embeddings": 100,
    "dtype": "bfloat16"
  },
  "text_config": {
    "dtype": "bfloat16",
    "vocab_size": 1024,
    "hidden_size": 128,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 32,
    "rms_norm_eps": 1e-6,
    "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    "rope_parameters": {
      "rope_theta": 10000,
      "partial_rotary_factor": 0.25,
      "mrope_section": [2, 1, 1],
      "mrope_interleaved": true,
      "rope_type": "default"
    },
    "attention_bias": false,
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 16,
    "linear_num_key_heads": 2,
    "linear_num_value_heads": 4,
    "linear_value_head_dim": 32,
    "intermediate_size": 256
  }
}
```

Fuer einen echten 0.1B-1B Run musst du die Dimensionen natuerlich skalieren.
Der Punkt ist: Die Struktur muss so aussehen, weil `make_config_from_hf()` diese
Keys verlangt.

## `layer_types` ersetzt `gated_delta_per_layer`

NVIDIA GatedDeltaNet-H1 nutzt im betrachteten Repo:

```yaml
gated_delta_per_layer: 2
```

Das bedeutet dort ungefaehr: jede zweite Schicht GDN, jede andere Attention.

In `omegalax` gibt es dieses Feld nicht. Hier entscheidet:

```yaml
layer_types:
  - linear_attention
  - full_attention
  - ...
```

Code:

```text
omegalax/models/qwen3_5/model.py:183-190
```

Wenn `layer_type == "full_attention"`, wird `Attention` gebaut. Sonst wird
`GatedDeltaNet` gebaut.

Fuer NVIDIA-H1-like 1:1 bei 12 Layern:

```yaml
layer_types:
  - linear_attention
  - full_attention
  - linear_attention
  - full_attention
  - linear_attention
  - full_attention
  - linear_attention
  - full_attention
  - linear_attention
  - full_attention
  - linear_attention
  - full_attention
```

Fuer Qwen3.5-like ist eher typisch:

```yaml
layer_types:
  - linear_attention
  - linear_attention
  - linear_attention
  - full_attention
  - linear_attention
  - linear_attention
  - linear_attention
  - full_attention
```

In `config.py` gibt es zwar `_generate_layer_types()` mit Full Attention jede
vierte Schicht, aber beim Laden aus HF-Config wird `layer_types` explizit aus
der Config gelesen. Fuer dein eigenes Modell solltest du es also direkt im
`config.json` setzen.

## GatedDeltaNet-Dimensionen in dieser Codebase

In NVIDIA wurde fuer das 0.4B-H1-Modell gerechnet:

```yaml
hidden_size: 1536
expand_k: 0.75
expand_v: 1.5
num_heads: 9
head_qk_dim: 128
head_v_dim: 256
```

In `omegalax` ist das anders modelliert. Die GDN-Dimensionen stehen direkt in
der Qwen3.5-Config:

```yaml
linear_key_head_dim
linear_num_key_heads
linear_num_value_heads
linear_value_head_dim
linear_conv_kernel_dim
```

Code:

```text
omegalax/models/qwen3_5/deltanet.py:52-63
```

Konkret:

```python
self.num_v_heads = cfg.linear_num_value_heads
self.num_k_heads = cfg.linear_num_key_heads
self.head_k_dim = cfg.linear_key_head_dim
self.head_v_dim = cfg.linear_value_head_dim
self.key_dim = self.head_k_dim * self.num_k_heads
self.value_dim = self.head_v_dim * self.num_v_heads
self.gqa_factor = self.num_v_heads // self.num_k_heads
```

Der recurrent state der Delta-Rule ist grob:

```text
num_value_heads * linear_key_head_dim * linear_value_head_dim
```

Nicht:

```text
num_key_heads * linear_key_head_dim * linear_value_head_dim
```

Warum? Q/K werden in `deltanet.py` bei GQA auf Value-Heads broadcasted:

```python
local_v_heads == local_k_heads * gqa_factor
```

Darum musst du sicherstellen:

```text
linear_num_value_heads % linear_num_key_heads == 0
```

Sonst kann der Assert im GDN-Pfad brechen.

Beispiel mit den Qwen3.5-Defaults in dieser Codebase:

```yaml
linear_num_key_heads: 16
linear_key_head_dim: 128
linear_num_value_heads: 64
linear_value_head_dim: 128

key_dim_total: 16 * 128 = 2048
value_dim_total: 64 * 128 = 8192
state_values_per_batch_element_per_linear_layer: 64 * 128 * 128 = 1_048_576
bf16_state_memory_per_batch_element_per_linear_layer: ~2 MiB
```

Das ist deutlich groesser als der NVIDIA-0.4B-H1-State von ca. 295k Werten pro
GDN-Layer. Wenn du GDN-Memory/Speed kontrollieren willst, sind diese Felder
wichtiger als `num_attention_heads`.

## Dense Attention und `local_window`

NVIDIA nennt fuer Hybrid-Modelle oft Sliding-Window-Attention mit Fenster 2048.
In dieser Codebase sehe ich fuer Qwen3.5-Text-Attention aktuell:

```text
omegalax/models/qwen3_5/attention.py
```

Dort wird aufgerufen:

```python
dot_product_attention(..., is_causal=True, ...)
```

Es gibt kein `local_window`, `sliding_window` oder Window-Mask-Argument in der
Qwen3.5-Text-Config. Das heisst:

```yaml
current_behavior:
  full_attention_layers: full causal attention
  local_window_2048: not implemented
```

Das ist ein wichtiger Unterschied zur NVIDIA-Recipe. Bei `max_length=4096` ist
full attention noch oft machbar, aber:

- Kosten der Full-Attention-Layer steigen quadratisch mit Sequenzlaenge.
- Je hoeher der Anteil `full_attention` in `layer_types`, desto teurer wird es.
- Wenn du wirklich Qwen3.5/NVIDIA-like Sliding Window willst, brauchst du eine
  lokale Attention-Maske oder Backend-Unterstuetzung fuer Window Attention.

## MLP `intermediate_size`

In `omegalax/models/qwen3_5/model.py` ist der dense MLP explizit:

```python
self.gate_proj = Linear(hidden_size, intermediate_size)
self.up_proj = Linear(hidden_size, intermediate_size)
self.down_proj = Linear(intermediate_size, hidden_size)
```

Das entspricht SwiGLU:

```text
hidden_size -> gate/up intermediate_size -> down hidden_size
```

Bewertung:

- `intermediate_size` ist ein Architekturparameter, kein Trainings-HParam.
- Fuer Dense-FFN-Modelle musst du ihn in `config.json` setzen.
- Fuer MoE-Modelle wird stattdessen `moe_intermediate_size` und
  `shared_expert_intermediate_size` verwendet.
- Wenn du Parameterbudget 0.1B-1B targetest, ist `intermediate_size` einer der
  groessten Hebel fuer Modellgroesse.




  

## Pretraining-Problem: aktuelle Datenpipeline ist SFT

Die aktuellen Scripts und Collator sind fuer Chat/SFT gebaut:

```text
scripts/compile_sft_dataset.py
scripts/build_sft_chunk_index.py
scripts/train_text_sft.py
omegalax/data/collator_qwen3.py
```

Der `TextSFTCollator` baut eine `loss_mask`, die nur Assistant-Content trainiert.
Der Loss selbst ist next-token prediction, aber maskiert:

```text
omegalax/trainers/loss.py:65-69
```

```python
hidden_BTD = hidden_BTD[:, :-1, :]
targets_BT = targets_BT[:, 1:]
mask_BT = mask_BT[:, 1:]
```

Das ist fuer SFT korrekt. Fuer klassisches Pretraining willst du normalerweise:

```yaml
loss_mask:
  all_real_tokens: 1
  pads: 0
```

Statt nur Assistant-Tokens. Wenn du mit der aktuellen SFT-Pipeline "pretrainst",
trainierst du nicht auf allen Tokens. Das waere eine falsche Baseline.

Konsequenz:

- Fuer echte Pretraining-Runs brauchst du entweder einen Pretraining-Collator
  oder eine Datenpipeline, die `loss_mask_BT` fuer alle nicht-pad Tokens setzt.
- Die Optimizer-HParams aus NVIDIA sind erst dann sinnvoll vergleichbar.
- Ohne diesen Schritt vergleichst du SFT-Dynamik, nicht Pretraining-Dynamik.

## Welche HParams wuerde ich wirklich adjustieren?

### Sofort adjustieren

```yaml
learning_rate:
  why: groesster Stabilitaets/Speed-Hebel
  recommendation: sweep 1e-4, 2e-4/3e-4, 4e-4

lr_schedule:
  why: default ist nicht NVIDIA-like
  recommendation: cosine

lr_end_factor:
  why: bildet min_lr=lr/10 ab
  recommendation: 0.1

warmup_steps:
  why: muss zum Tokenbudget passen
  recommendation: 1% num_steps, bei Instabilitaet 2%

batch_tokens:
  why: beeinflusst Optimizer-Dynamik stark
  recommendation: ~2M fuer 0.4B-1B, kleiner testen fuer 0.1B falls wenig Budget

max_length:
  why: bestimmt Kontextlaenge, Compute, GDN-Nutzung
  recommendation: 4096 fuer Main-Baseline, 2048 fuer billige Smoke-Runs
```

### Implementieren oder bewusst dokumentieren

```yaml
adam_beta2:
  current: 0.999
  target_nvidia: 0.95
  recommendation: als Flag hinzufuegen, wenn du NVIDIA-like sein willst

adam_eps:
  current: vermutlich 1e-8
  target_nvidia: 1e-8
  recommendation: optional als Flag hinzufuegen fuer explizite Reproduzierbarkeit

weight_decay_mask:
  current: kein explizites Masking
  recommendation: erstmal lassen, aber bei GDN-Instabilitaet pruefen

local_window:
  current: fehlt
  recommendation: nur relevant, wenn du wirklich Sliding-Window-Dense-Attention willst
```

### Nicht als erstes anfassen

```yaml
max_grad_norm:
  default: 1.0
  recommendation: erstmal lassen

adam_beta1:
  default: 0.9
  recommendation: passt

adam_eps:
  default: 1e-8
  recommendation: passt, solange Optax Default nicht geaendert wird

linear_conv_kernel_dim:
  default_qwen3_5: 4
  recommendation: nicht initial sweepen

rms_norm_eps:
  default_qwen3_5: 1e-6
  recommendation: nicht initial sweepen
```

## Konkrete Probleme, die deine aktuelle Auswahl nicht beachtet

1. `adam_beta2=0.95` wird aktuell nicht gesetzt.
   Wenn du nichts aenderst, trainierst du mit Optax Default `0.999`.

2. `max_tokens` wird nicht direkt gesetzt.
   Du musst `num_steps` aus Batch, Sequenzlaenge und Grad-Accumulation berechnen.

3. `warmup: 1% tokens` wird nicht direkt gesetzt.
   Du musst `warmup_steps` aus `num_steps` berechnen.

4. `local_window=2048` existiert aktuell nicht in Qwen3.5 Attention.
   Full-attention Layers laufen als volle kausale Attention.

5. Die aktuelle Pipeline ist SFT-maskiert.
   Fuer Pretraining brauchst du eine Loss-Maske auf allen echten Tokens.

6. Qwen3.5-GDN-Dims sind nicht dieselben wie NVIDIA-GDN-Dims.
   In dieser Codebase sind `linear_num_value_heads`, `linear_key_head_dim` und
   `linear_value_head_dim` die State-Memory-Hebel.

7. `layer_types` muss explizit stimmen.
   `gated_delta_per_layer=2` aus NVIDIA hat hier keine Wirkung.

8. `--max_length` muss auch beim Chunk-Index-Build verwendet werden.
   Nur den Training-Run auf 4096 zu stellen reicht nicht.

## Empfohlene Baseline fuer diese Codebase

Fuer einen ersten serioesen 0.1B-1B Hybrid-Pretraining-Run:

```yaml
optimizer:
  learning_rate: 2e-4 oder 3e-4
  lr_schedule: cosine
  lr_end_factor: 0.1
  warmup: 1% num_steps
  weight_decay: 0.1
  adam_beta1: 0.9
  adam_beta2: 0.95  # erst nach Code-Anpassung wirklich gesetzt
  adam_eps: 1e-8
  max_grad_norm: 1.0

batching:
  max_length: 4096
  batch_tokens: ~2M
  num_steps: target_tokens / batch_tokens

architecture:
  layer_types: explizit in config.json
  full_attention_ratio: bewusst waehlen
  local_window: nicht annehmen, solange nicht implementiert
  gdn_state_dims: linear_num_value_heads * linear_key_head_dim * linear_value_head_dim

data:
  objective: next-token prediction
  loss_mask: all non-pad tokens, nicht nur Assistant-Tokens
```

Wenn du noch keine echte Pretraining-Pipeline hast, ist die erste technische
Aufgabe nicht LR-Sweeping, sondern:

```text
Pretraining-Collator/Dataset bauen, der token_ids_BT, attention_mask_BT und
loss_mask_BT mit loss_mask=1 fuer alle echten Tokens liefert.
```

Erst danach sind die NVIDIA-HParams als Pretraining-Baseline sauber interpretierbar.
