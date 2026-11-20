# OmegaLAX: Review von State-Weitergabe, Experimenten und Tests

Stand: 24. September 2026. Hauptcheckout: `feature/statepassing-pretrain-experiments`, Commit `17ffbb7`. Ergänzend geprüft wurde der Delta des aktuellen Coding-Trace-Worktrees `/fast/home/salan.isaqzoi/.codex/worktrees/realistic-trace-sft`, Commit `d1e1cbe`, gegenüber seinem gemeinsamen Ausgangsstand. Dieser Branch enthält bereits einige Korrekturen, die im Hauptcheckout fehlen. `omegalax_optimization` wurde nicht als zweiter Produktionsstand geprüft.

Das Review umfasst Datenfenster und Masken, GDN-/Conv-Weitergabe, BPTT, Loss und Akkumulation, Initialisierung, Checkpoints, natürliche State-Usage-/Full-Context-Evals sowie die synthetischen D1-/D7-/Bank-Aufgaben. Die Implementierung wurde mit Projektbrief, Experimentbeschreibung und aktuellen Run-Einträgen abgeglichen. Es wurden keine Produktionsdateien geändert und keine wissenschaftlichen Trainingsruns gestartet.

## Ergebnis

Die zentralen Statepassing-Pfade haben substanzielle Tests mit echten Modellen, Gradienten und Checkpoints. Trotzdem ist die Codebase nicht durchgehend gegen stille Änderungen der Trainings- und Eval-Semantik abgesichert. Besonders kritisch sind die unterschiedlichen Initialisierungspfade, das verlorene `last_to_first`-Routing und die Gewichtung bei variablen Loss-Masken. Die Testsuite deckt **nicht alle relevanten Produktionspfade und Kombinationen** ab.

P1 bedeutet hier: vor Verwendung des betroffenen Experimentpfads beheben. P2 bezeichnet einen reproduzierbaren Fehler oder eine konkrete Zuverlässigkeitslücke mit engerem Auslöser. Ein Finding bedeutet nicht, dass sämtliche bisherigen Ergebnisse betroffen sind; die Grenzen stehen jeweils dabei.

| ID | Priorität | Befund | Betroffener Stand |
|---|---|---|---|
| F1 | P1 | Curriculum ignoriert HF-Gewichte und initialisiert zufällig. | Hauptcheckout; im Trace-Branch bereits korrigiert |
| F2 | P1 | Externer Eval und Chat verlieren `last_to_first`. | Hauptcheckout; bereits historisch dokumentiert |
| F3 | P2 | Äquivalente No-BPTT-Flags verändern die Ziel-Tokens. | Hauptcheckout und gemeinsamer Trainerpfad |
| F4 | P2 | Gradient Accumulation gewichtet Microbatches anders als die geloggte Token-Loss. | Standard-, Tail- und Curriculum-Training |
| F5 | P2 | Natürlicher Resume schützt BPTT und Trainingsmodus nicht. | Natürlicher Trainer und Training-Contract |
| F6 | P2 | Curriculum-Signalpfad speichert denselben Schritt zweimal. | Hauptcheckout und Trace-Curriculum |
| F7 | P2 | Full-Context-Vergleich akzeptiert unterschiedliche Eval-Kohorten. | Natürlicher Full-Context-Vergleich |
| F8 | P2 | Trace-Loss-Masken fehlen in der Dataset-/Manifest-Identität. | Trace-Branch |
| F9 | P2 | Gültige Evalgruppen ohne Ziel-Tokens verursachen Division durch null. | Aggregation; durch Trace-Masken erreichbar |

## Fehler und stille Abweichungen

### F1 — HF-Initialisierung erreicht im Hauptcheckout den Curriculum-Trainer nicht

Bei einem frischen Curriculum-Run mit `--init_hf_model_path` wird der Pfad zwar als `model_source` zur Konfigurationsauflösung verwendet. Der Aufruf von `run_statepassing_curriculum` reicht ihn aber nicht als Gewichtsquelle weiter, und der Curriculum-Trainer ruft immer die zufällige Modellinitialisierung auf. Der Run kann somit mit korrekter Qwen-Architektur und plausiblen Logs starten, obwohl die gewünschten vortrainierten Gewichte fehlen.

Belege: [CLI-Aufruf:535](scripts/train_text_pretrain.py), [zufällige Initialisierung:1251](omegalax/trainers/pretrain.py). Der separate normale Pfad lädt HF-Gewichte ausdrücklich ab Zeile 914. Im Trace-Branch behebt `d1e1cbe` genau diese Abweichung, indem er das Argument weiterreicht und die Gewichte lädt. Der dokumentierte Coding-Trace-Run auf diesem Branch ist deshalb **nicht** allein wegen F1 als zufällig initialisiert einzustufen.

Die bestehenden HF-Tests in `test_memory_tasks_hf_initialization.py` prüfen den Tail-/Memory-Pfad, nicht dieses Curriculum. Nötig ist ein kleiner echter Safetensors-Export mit erkennbaren Gewichten, der durch den Curriculum-Einstieg geladen wird; ein erfolgreicher Loss-Schritt allein würde den Fehler nicht erkennen. Die vorhandene Branch-Korrektur sollte zusammen mit diesem Test integriert werden.

### F2 — `last_to_first` wird beim externen Eval still zu `same_layer`

Der Trainer leitet bei `last_to_first` den letzten GDN-Layer-State in den ersten Layer des nächsten Segments. Der Contract speichert diese Information jedoch außerhalb von `eval_statepassing_config`. `resolve_eval_config` verliert sie, und der Eval selektiert anschließend die States nach ihrer unveränderten Layerposition. Das Flag ist auch in `StatePassingConfig` des Chatpfads nicht repräsentiert.

Belege: [Training-Contract:151](omegalax/training_contract.py), [Contract-Auflösung:219](omegalax/training_contract.py), [Eval-State-Selektion:45](omegalax/evals/experiments.py), [Chatkonfiguration:32](omegalax/text/chat.py). Die Reproduktion baut einen echten `last_to_first`-Contract und löst ihn wieder auf: Das Ergebnis enthält kein Routing-Feld. Die anschließende State-Selektion kann daher nur `same_layer` ausführen.

Das ist ein **bekannter, weiterhin vorhandener Fehler**. Der Forschungs-Worklog und der historische Run-Eintrag erklären die externen State-Usage-Zahlen dieses Spezialarms bereits für nicht interpretierbar. Die heutigen `same_layer`-Hauptarme sind davon nicht betroffen. `test_last_to_first_routing_is_resume_protected_without_changing_eval_config` prüft lediglich den Contract-Konflikt, nicht die gleiche Zustandsübergabe im Eval. Routing muss entweder vollständig transportiert oder dieser Evalpfad ausdrücklich abgewiesen werden.

### F3 — `bptt_chunks=None`, `0` und `1` trainieren im No-BPTT-Modus unterschiedliche Ziele

Die State-Gradienten verwenden `resolved_bptt_chunks`; die Loss-Maske erhält dagegen das ursprüngliche Argument. Im No-BPTT-Modus werden deshalb bei allen drei Schreibweisen Gradienten an jeder Segmentgrenze getrennt, aber nur bei explizitem `0/1` verschwinden die Boundary-Targets. Damit variiert neben der Gradiententechnik auch die Menge der vorherzusagenden Tokens.

Belege: [BPTT-Auflösung:416](omegalax/trainers/pretrain.py), [Maskenaufruf:481](omegalax/trainers/pretrain.py), [Boundary-Maske:274](omegalax/trainers/pretrain.py). Auf demselben echten Tiny-Modell mit zwei Segmenten zu vier Tokens ergibt `None` sieben Ziel-Tokens und NLL `4.248393`; `0/1` ergeben sechs Ziel-Tokens und NLL `4.308475`. Bei langen Segmenten ist der relative Tokenunterschied kleiner; er bleibt eine systematische Änderung des Objectives.

Auch dieses Problem ist bereits dokumentiert und noch offen. Der vorhandene Maskentest prüft nur die Boundary-Untermaske; er prüft nicht die Gleichheit der tatsächlichen Trainerziele bei äquivalenten Flags. Die gewünschte Boundary-Semantik muss explizit festgelegt und einmal kanonisch verwendet werden. Das Review ersetzt diese wissenschaftliche Entscheidung nicht durch eine neue Maske.

### F4 — Akkumulierte Gradienten entsprechen bei variablen Masken nicht der angezeigten Gesamt-Loss

Jeder Microbatch berechnet zuerst `nll_sum / supervised_tokens`. `optax.MultiSteps` mittelt diese Gradienten gleichgewichtet. Die Trainingsmetrik addiert dagegen NLL-Summen und Ziel-Tokenzahlen und bildet daraus eine nach Tokens gewichtete Loss. Bei einem Microbatch mit einem und einem mit sieben Ziel-Tokens erhält im Training jedes Token des ersten Microbatches das siebenfache Gewicht eines Tokens des zweiten.

Belege: [Loss-Normalisierung:498](omegalax/trainers/pretrain.py), [MultiSteps:209](omegalax/trainers/text.py), [Tail-MultiSteps:193](omegalax/trainers/tail_pretrain.py), [Metrikaggregation:645](omegalax/trainers/pretrain.py). Die Reproduktion verwendet dieselben produktiven Forward-/Loss-/Gradientenfunktionen: Die maximale Abweichung zwischen dem Microbatch-Mittel und dem Gradienten des zusammengefassten Batches beträgt `24.20703125`; tokengewichtete Kombination und zusammengefasster Batch unterscheiden sich nur um etwa `9.54e-6`.

Bei `grad_accum_steps=1` oder identischen Ziel-Tokenzahlen tritt dieser Unterschied nicht auf, etwa in den festen D1-/D7-Synthetic-Batches. Bei Akkumulation betrifft er gepaddete natürliche Daten und besonders unterschiedlich dichte Assistant-Masken im Coding-Trace-Training. Die Größe einer historischen Ergebnisverschiebung wurde nicht gemessen. Falls ausdrücklich gleiche Microbatch-Gewichte gewünscht sind, muss dies als Objective dokumentiert werden; andernfalls müssen Gradienten und Metrik dieselbe Normalisierung verwenden. Es fehlt ein Test „zwei unterschiedlich maskierte Microbatches gegen denselben zusammengefassten Batch“.

Für das aktuelle Trace-Experiment ist dies konkret relevant: Der Produktionswrapper setzt ausdrücklich `--grad_accum_steps=8` ([run.sbatch:63](/fast/home/salan.isaqzoi/.codex/worktrees/realistic-trace-sft/Salan_DONT_PUSH/trace_sft_20260921/production/run.sbatch:63)), und der Run-Eintrag dokumentiert Assistant-only-Loss. Die tatsächlichen Zielzahlen jedes historischen Microbatches wurden hier nicht rekonstruiert; damit ist die ausgeführte Gewichtungsregel bekannt, ihre quantitative Wirkung auf diesen Run jedoch nicht.

### F5 — Natürlicher Resume akzeptiert Änderungen an BPTT und Trainingsmodus

`build_training_contract` enthält weder `pretrain_mode` noch `bptt_chunks`. Der zusätzliche LR-Contract schützt ausschließlich die Lernrate und ihren Zeitplan. Die rekonstruierte Train-Step-Funktion verwendet beim Resume die neu übergebenen Flags. Somit kann derselbe natürliche Checkpoint mit BPTT4 fortgesetzt oder auf BPTT1/No-BPTT umgestellt werden, ohne dass die vorhandenen Contract-Vergleiche diese Objective-Änderung erkennen.

Belege: [Contract-Signatur und Payload:98](omegalax/training_contract.py), [CLI-Contract:494](scripts/train_text_pretrain.py), [neu aufgebauter Train-Step:944](omegalax/trainers/pretrain.py), [LR-Contract:90](omegalax/trainers/text.py). Das ist eine statisch nachgewiesene Schutzlücke; es wurde kein produktiver Checkpoint mit geänderten Flags fortgesetzt. Der strengere Synthetic-Memory-Contract enthält seine State-Einstellungen bereits und ist nicht der hier beanstandete Pfad.

Ein Resume-Test sollte eine absichtliche Änderung von BPTT/Modus ablehnen, bevor Training oder Iteratorkonsum beginnt. Die kanonische BPTT-Einstellung und Boundary-Loss-Semantik gehören in den natürlichen Trainingsvertrag. Es geht nicht darum, legitime neue Experimente zu verbieten, sondern sie von einer Fortsetzung desselben Experiments unterscheidbar zu machen.

### F6 — Signal und regulärer Save kollidieren im Curriculum-Trainer

Der Curriculum-Trainer speichert zuerst auf einem regulären `save_every`-Schritt. Wenn anschließend ein Signal erkannt wird, speichert er denselben Schritt erneut mit `force=True`. Orbax wirft dann `StepAlreadyExistsError`, bevor der vorgesehene Requeue ausgeführt wird. Der normale Trainer und der Tail-Trainer vermeiden diese Doppelung bereits; die Curriculum-Schleife enthält die Reparatur nicht.

Belege: [regulärer Save:1420](omegalax/trainers/pretrain.py), [Signal-Save:1444](omegalax/trainers/pretrain.py). Mit echtem Tiny-Modell, Grain-Iterator, `SIGUSR1` und Orbax wurde exakt `StepAlreadyExistsError: Checkpoint for step 1 already exists.` reproduziert. Es wurde kein produktiver Job requeued. Der bereits geschriebene Checkpoint wird dadurch nicht automatisch unbrauchbar; der automatische Fortsetzungspfad scheitert jedoch.

Im selben Signalblock wird außerdem nach dem letzten Curriculum-Schritt noch ein Requeue angefordert. Ein anschließender Resume wird vom Stop-Check abgewiesen, weil der Zielschritt schon erreicht ist. Für diese zweite Kante liegt statische Evidenz vor. Tests für reale Curriculum-Saves bei Signal, Phasenwechsel und finalem Schritt fehlen; der vorhandene Phasenwechseltest ersetzt Checkpointmanager und Savefunktion durch Mocks.

### F7 — Full-Context-Vergleich verwechselt gleiche Tokenzahl mit gleicher Kohorte

`compare_relative_nll` gleicht lediglich Summary-Schlüssel und `token_count` ab. Es prüft nicht, ob die beiden Resultate dieselben Dokumente, Masken oder dasselbe Manifest verwenden. Zwei gleich große Samples gleicher Dokumentlängen können damit als gültiger Vergleich mit einer scheinbaren NLL-Verbesserung ausgegeben werden.

Beleg: [Vergleich:581](omegalax/evals/full_context.py). Die echte Vergleichsfunktion akzeptiert in der Reproduktion verschiedene `manifest_hash`-/`dataset_hash`-Werte (`cohort-A` und `cohort-B`) bei gleicher Tokenzahl und schreibt `complete: true`. Die Eingaben waren kleine Resultdateien, keine produktiven Evals. Das beweist die fehlende Prüfung, nicht eine Verfälschung bestimmter bestehender Grafiken.

Nötig sind ein Abgleich der tatsächlichen Eval-Kohorte und Token-Selektion sowie ein negativer Test mit gleich vielen Tokens aus verschiedenen Dokumenten. Die robusteren Identitätsprüfungen anderer Evalpfade sollten hier konsistent angewandt werden.

### F8 — Trace-Masken können sich bei unveränderter Manifest-Identität ändern

Im Trace-Branch verwendet `FullDocumentLoader` jetzt `doc.metadata['loss_spans']`. Der Manifest-Builder hasht weiterhin nur Dokumentidentität und Tokenfolge. Eine Änderung der Assistant-Zielbereiche verändert damit die evaluierte Aufgabe, aber weder `dataset_hash` noch `manifest_hash`. Bereits vorhandene Eval-Shards können weiterhin als zur gleichen Aufgabe gehörend erscheinen.

Belege im Trace-Worktree: [Manifest-Hash:207](/fast/home/salan.isaqzoi/.codex/worktrees/realistic-trace-sft/omegalax/evals/manifest.py:207), [Loss-Spans im Loader:505](/fast/home/salan.isaqzoi/.codex/worktrees/realistic-trace-sft/omegalax/evals/manifest.py:505). Die Reproduktion schreibt zwei kleine echte ArrayRecord-Dokumente, baut das Manifest, ändert ausschließlich die Spans und baut es erneut. Die Zielanzahl je Chunk ändert sich von `[3, 3]` zu `[0, 3]`, beide Hashes bleiben gleich.

Die kanonischen Zielmasken beziehungsweise deren Semantik müssen Bestandteil der Dataset-Identität sein. Der vorhandene Trace-Iterator-Test beweist das Durchreichen der Maske, nicht ihre Bindung an Eval-Identität und Resume. Eine nachträgliche Maskenänderung am produktiven Datensatz wurde nicht festgestellt.

### F9 — Sparse Trace-Masken erreichen eine ungeschützte Division durch null

Ein Segment kann ausschließlich System-, User- oder Tool-Tokens enthalten und deshalb null Assistant-Ziel-Tokens haben. Wenn eine aggregierte Gruppe aus Dokumentlänge und Segmentposition insgesamt null Ziel-Tokens enthält, berechnet `aggregate_metrics` trotzdem `nll_sum / token_count` und bricht ab. Das ist bei vollständigen NTP-Masken wesentlich seltener, aber ein gültiger Fall der neuen Trace-Masken.

Beleg: [Aggregation:108](omegalax/evals/aggregation.py). Die kleine Trace-Reproduktion erzeugt einen tatsächlich maskierten Null-Target-Chunk; die Aggregation des entsprechenden Resultats wirft `ZeroDivisionError: float division by zero`. Der Trace-Filter garantiert späte Targets in vollständigen C2-Fenstern, aber nicht Targets in jeder einzelnen Evalposition. Ein einzelnes leeres Dokumentsegment innerhalb einer sonst nichtleeren Gruppe reicht nicht für den Fehler; maßgeblich ist die gesamte aggregierte Gruppe.

Solche Gruppen sollten ausdrücklich als nicht auswertbar behandelt werden. Eine künstliche Loss von null wäre wissenschaftlich irreführend. Benötigt wird ein echter Loader→Eval→Aggregation-Test mit Segmenten ohne Assistant-Targets. Es wurde nicht behauptet, dass der aktuelle produktive Eval bereits an dieser Stelle gescheitert ist.

## Testsuite: relevante Schwächen

### T1 — Der dokumentierte Gesamtlauf garantiert keine GPU-Abdeckung

[README:149](README.md) empfiehlt `unittest discover` über sämtliche Tests in einem Prozess. Mehrere früh importierte Chat-Testmodule setzen `JAX_PLATFORMS=cpu` und `OMEGALAX_DELTANET_KERNEL=xla` global. Die Pallas-Tests entscheiden ihren GPU-Skip anschließend bereits beim Import. Damit kann ein Lauf auf einem GPU-Knoten gerade die wichtigen GPU-Tests überspringen. Zusätzlich führt [test_tp_attention.py:7](tests/test_tp_attention.py) schon beim Import Distributed-Initialisierung und Berechnungen aus und fängt mehrere Fehler nur mit `print` ab; es ist kein regulärer assertion-basierter Testfall.

Für dieses Review wurden CPU- und GPU-Module deshalb in getrennten frischen Prozessen ausgeführt. Dabei trat ein weiterer Aufrufkonflikt auf: Tokamax versucht ohne vorherige Abseil-Initialisierung die pytest-Flags zu parsen und scheitert an `-q`. Die drei betroffenen Module wurden mit initialisierten Flags und unveränderten Tests erneut ausgeführt und bestanden vollständig; dieser Launcherfehler ist kein Nachweis eines falschen Modellergebnisses. Ein zentraler dokumentierter Runner sollte die Backend-Trennung und Flag-Initialisierung herstellen und bei GPU-Prüfungen unerwartete Skips als fehlgeschlagene Abdeckung melden. Ein grünes Discovery-Ergebnis allein ist kein Beleg für den produktiven H100-Pfad.

### T2 — Der Pallas-Gradiententest akzeptiert sogar komplett falsche Nullgradienten

[test_gated_delta_rule_pallas_bwd.py:107](tests/test_gated_delta_rule_pallas_bwd.py) verlangt lediglich, dass der **Referenzgradient** nicht null ist und der maximale absolute Unterschied unter `0.5` bleibt. Der relative Fehler wird berechnet, aber nur in den Fehlertext geschrieben. Für genau die Eingaben dieses Tests sind alle sechs Referenzmaxima kleiner als `0.5`: `dq=0.000870`, `dk=0.025146`, `dv=0.232422`, `dg=0.336940`, `dbeta=0.411076`, `dinitial_state=0.0000872`.

Damit würde ein Kandidat, der für sämtliche Eingaben einschließlich des getragenen States nur Nullgradienten liefert, alle Assertions dieses zustandsbehafteten Vergleichstests bestehen. Das wurde anhand der tatsächlichen Referenzgradienten nachgerechnet. Es ist **kein Nachweis, dass der aktuelle Kernel Nullgradienten liefert**. Die zusätzlichen Integrationstests geben weitere Absicherung, ersetzen aber keine aussagekräftige numerische Prüfung dieses Kernels. Sinnvoll sind skalierte absolute/relative Grenzen und ein Test der tatsächlichen State-Gradienten über mehrere Segmente.

### T3 — Ein wichtiger Gradiententest verwendet unbeabsichtigt BF16-Parameter

`test_gdn_only_full_bptt_matches_single_forward_loss_and_gradients` setzt zwar `dtype=float32`, lässt aber `param_dtype` auf dem Qwen-Default `bfloat16`. Der Test fordert anschließend sehr enge Gradientengleichheit zweier unterschiedlich angeordneter Berechnungen. Im aktuellen Prüflauf scheitert er bei Abweichungen bis etwa `0.00714` relativ. Mit unverändertem Produktionspfad und ausschließlich `param_dtype=float32` als Testkonfiguration besteht derselbe Test.

Belege: [Tiny-Konfiguration:45](tests/test_pretrain_training.py), [Vergleich:826](tests/test_pretrain_training.py), [Parameterdefault:66](omegalax/models/qwen3_5/config.py). Diese konkrete Fehlermeldung ist daher kein hinreichender Beleg für eine fehlerhafte State-Gradientenweitergabe. Ein präziser FP32-Äquivalenztest und ein gesonderter BF16-Test mit begründeten Toleranzen würden die Aussage klarer machen.

### T4 — Realdaten-Tests enthalten Annahmen des alten Korpus

[pretrain_real_data_test_utils.py:24](tests/pretrain_real_data_test_utils.py) verwendet standardmäßig noch `fineweb_edu_dedup_30b_8kto32k`. Für dieses Review wurde bewusst der aktuelle `fineweb_edu_dedup_2048_2kto32k`-Root gewählt. Drei Assertions scheitern dann: Ein Test erwartet Fenster aus jedem Bucket bei C2×4096, obwohl der neue kurze Bucket nicht zwingend ein vollständiges solches Fenster enthält; ein anderer erwartet die ersten beiden IID-Chunks aus demselben Bucket.

Betroffene Tests: [Statepassing:122](tests/test_pretrain_statepassing.py), [IID:107](tests/test_pretrain_iid.py), [IID-Batch:140](tests/test_pretrain_iid.py). Das sind reproduzierbare Fixture-/Erwartungsfehler beim aktuellen Dataset, kein Beleg für kaputte Indexierung. Die Tests sollten ihre Erwartungen aus den gelesenen Dokumentlängen ableiten und die Produktionsgeometrie 2048/C4 gezielt abdecken. Ein vollständiger Scan oder Neubau des kanonischen C4-Indexes wurde nicht durchgeführt.

### Abdeckungsmatrix

| Produktionsverhalten | Vorhandene Absicherung | Konkrete verbleibende Lücke |
|---|---|---|
| GDN-Weitergabe und Gradient durch den Anfangsstate | Echte XLA-Stateful-Tests, explizites Routing mit Modellgradienten, Pallas-Vergleiche | Aussagekraft der Pallas-Toleranz; numerische Drift über sehr lange Ketten |
| Conv-State und Padding | Vergleich mit vollständiger Sequenz, rechte Pads, echte Modellpfade | Kombinierte selektive Layer-/Conv-/BPTT-Varianten auf produktiver GPU-Topologie |
| BPTT-Gruppen und Boundary-Loss | Maskentests, mocked Handoff-Proben, echter Full-BPTT-Test | Äquivalente Flagwerte; partielles BPTT über mehrere Gruppen mit echten Modellgradienten |
| Gradientenakkumulation | Iteratorfortsetzung und Trainingssmokes | Ungleiche Ziel-Tokenzahlen gegen denselben logischen Gesamtbatch |
| HF-Initialisierung | Echter HF-Load/Stop/Resume im Memory-Tail-Pfad | Natürlicher Curriculum-Einstieg mit nachgewiesen geladenen Gewichten |
| Resume und Signale | Echter normaler Save/Resume einschließlich Optimizer und Iterator; Tail-Tests | Echter Curriculum-Signal-Save, Phasenwechsel plus Restart und finaler Signalpfad |
| Natürliche Evals | Echter Tiny-Hybrid-Eval, Donoren, Zero/Shuffle, Resume; Zwei-Device-CPU-Subprozess | `last_to_first`; Full-Context-Vergleich verschiedener Kohorten; Full-Context-Cache nach Codeänderung |
| Synthetic D1/D7 | Generatorinvarianten, echte CLI Training→Manifest→Eval, autoregressive D7-Referenz, eigener/Donor-NLL | D7 mit Hybridmodell und GPU-Train/Eval gemeinsam; längere D7-Ketten als getestete C2-Referenz |
| Train/Eval-Überschneidung | Paar-/Tabellendigest, Vorfahren, Gates, Holdouts, gemeinsame C-Auswahl | Kein neuer vollständiger historischer Trainingsstrom-Audit in diesem Review |
| Assistant-only Trace-Pfad | Ein echter Statepassing-Iterator-Test für `loss_spans`; externe Run-Audits | Regressionssuite für tatsächlichen Trace-Encoder, IID-/Eval-Masken, Maskenidentität und leere Zielgruppen |
| Produktionsmaßstab | Kleine Modelle plus ausgewählte GPU-Kernels; externe Smoke-/Capacity-Runs | Kein automatischer End-to-End-Nachweis für 2B/9B, TP2/FSDP4, lange BPTT-Ketten und 128k+ |

Die genannten Tests sind nicht pauschal „nur Mocks“. Besonders D7-CLI, Checkpoint-Restore, natürliche Hybrid-Evals und Kernelreferenzen prüfen echte Produktionsfunktionen. Die Lücken liegen häufig zwischen diesen einzeln abgesicherten Teilen. Eine Zeilenabdeckungsquote würde diese semantischen Lücken nicht zuverlässig beschreiben; eine solche Quote wurde hier nicht erhoben.

## Abstraktionen, Design und unnötige Komplexität

### D1 — Drei Trainingsschleifen entwickeln sich auseinander

`run_pretrain`, `run_statepassing_curriculum` und `run_tail_pretrain` wiederholen Initialisierung, Optimizeraufbau, Microbatch-Akkumulation, Logging, Save- und Signalbehandlung. F1 und F6 zeigen bereits konkrete Auswirkungen: Eine Reparatur existiert in einem Pfad und fehlt im anderen. Das ist nicht bloß eine Frage der Dateilänge.

Eine kleine gemeinsame Implementierung der wirklich gleichen Save-/Signal- und Initialisierungsregeln wäre sinnvoll, nachdem Regressionstests die Unterschiede festhalten. Eine neue allgemeine Trainerhierarchie wäre dafür unnötig. Phase/Iterator und Tail-Step-Offset sollten ausdrücklich sichtbar bleiben.

### D2 — Die gemeinsame State-Konfiguration endet an mehreren API-Grenzen

Trainerargumente, `StatePassingConfig`, `ManualEvalConfig` und `ResolvedEvalConfig` repräsentieren ähnliche Einstellungen in unterschiedlichem Umfang. Zusätzlich implementieren Trainer, Eval und Chat ihre Layerauswahl getrennt. F2 ist die direkte Folge dieses auseinanderlaufenden Modells. Ein kleiner vollständiger Vertrag für tatsächlich unterstützte State-Übergaben und ein gemeinsamer Selektionspfad wären angemessen; ein generisches Plugin-System für hypothetische Memory-Methoden ist dafür nicht erforderlich.

### D3 — Provenienz ist zugleich zu weit und zu eng gekoppelt

Der Memory-Pollution-Pfad bindet die Rekonstruktion an Byte-Hashes ganzer Trainingsskripte und Trainerdateien und pflegt anschließend historische Hash-Allowlisten ([memory_pollution.py:24](omegalax/evals/memory_pollution.py)). Selbst semantisch irrelevante Änderungen dieser Dateien erzwingen dadurch eine neue Kompatibilitätsentscheidung. Diese Checks schützen reale wissenschaftliche Anforderungen; sie einfach zu entfernen wäre keine Verbesserung. Besser wäre eine explizite, getestete Versionierung von Generator und Pair-Verbrauch plus ein separater Codefingerprint für Provenienz.

Umgekehrt fehlt dem resumierbaren Full-Context-Eval ein Codefingerprint ([full_context.py:272](omegalax/evals/full_context.py)); vorhandene Shards werden anhand ihrer Existenz übersprungen. Nach einer Korrektur an Modell, Loader oder Loss kann derselbe Resultroot deshalb alte und neue Berechnungen mischen. Hier fehlt gerade die Bindung, die andere Evalpfade bereits besitzen. F8 zeigt dieselbe Art von Lücke für Masken. Die mehreren selbst implementierten Identitäts-, Resume- und Schreibprotokolle sollten auf ihre konkreten Invarianten vereinheitlicht werden, statt weitere Sonder-Hashlisten und parallele Stores hinzuzufügen.

### D4 — BPTT-Abbruch begrenzt nicht automatisch alle Trainingsaktivierungen

Der Trainer sammelt sämtliche `hidden_segments` und berechnet danach eine gemeinsame, segmentübergreifende Loss ([pretrain.py:420](omegalax/trainers/pretrain.py), [pretrain.py:479](omegalax/trainers/pretrain.py)). Das erhält die gewollten Boundary-Targets, bedeutet aber: kleineres BPTT stoppt den State-Gradienten, macht den gesamten Schritt nicht automatisch zu einem speicherkonstanten Stream über beliebig viele Segmente. Außerdem wächst die Anzahl der vollständigen Diagnosemasken mit C.

Das ist kein nachgewiesener Rechenfehler der heutigen C2/C4/C8-Experimente und kein Grund für einen sofortigen Umbau. Für das Projektziel 128k+ und sehr große Segmentzahlen sollte diese Grenze jedoch explizit in Capacity-Planung und Messungen auftauchen. Aus „fester weitergereichter State“ folgt keine konstante Trainingsspeichernutzung.

### Wissenschaftliche Interpretation der RoPE-Option

Ohne KV-Carry sehen die Full-Attention-Layer nur das aktuelle Segment. Ein gemeinsamer RoPE-Offset für Q und K kürzt sich mathematisch in ihren relativen Positionen heraus; GDN selbst konsumiert diese Positions-IDs nicht. `pass_rope_positions` ist deshalb in diesem Modus keine eigenständige Quelle früherer Information, abgesehen von numerischen Effekten. Das ist bereits im Forschungs-Worklog erklärt. Die vorhandenen Tests bestätigen das Durchreichen der Positionswerte, aber nicht einen kausalen Gedächtniseffekt. Ein RoPE+Conv-Erfolg darf nicht als isolierter RoPE-Memory-Effekt interpretiert werden.

## Ausgeführte Verifikation und Grenzen

Alle rechenintensiven Prüfungen liefen auf Slurm-Compute-Nodes mit Host-/Job-Prüfung vor den schweren Imports, 24 CPUs und 120G. Keine Dependencies wurden installiert. Realdaten wurden nur über die kleinen bestehenden Testfixtures gelesen; der kanonische C4-Index blieb unverändert.

Die 29 ausgewählten bestehenden Testmodule ergeben nach Bereinigung des Launcherkonflikts **385 bestandene, vier fehlgeschlagene und null übersprungene Testfälle**. Wiederholungen sind nicht doppelt gezählt; zusätzlich gemeldete Subtests werden nicht als eigene JUnit-Testfälle addiert.

| Bereich | Module | Bestanden | Fehlgeschlagen | Übersprungen |
|---|---:|---:|---:|---:|
| Training, Daten, Contracts, Tail/Curriculum und XLA-State | 9 | 173 | 4 | 0 |
| Synthetic Memory, HF-Initialisierung, Holdouts und Memory-Konfiguration | 6 | 129 | 0 | 0 |
| Natürliche Evals einschließlich Full Context, CLI und Submitter | 10 | 72 | 0 | 0 |
| Pallas Forward/Backward, echter GPU-Trainingsschritt und Recomputation | 4 | 11 | 0 | 0 |
| Gesamt | 29 | 385 | 4 | 0 |

Job `154918` lief auf `hai006` für 23:44 Minuten: 25 CPU-Module einzeln mit XLA und vier GPU-Module mit CUDA/Pallas auf einem H100. Der erste Durchlauf ergab 371 bestandene und 18 fehlgeschlagene Tests. Vier Fehler entsprechen T3/T4: ein BF16-Gradientenvergleich und drei nicht zum aktuellen Korpus passende Erwartungen. Die übrigen 14 Fehler entstanden am Abseil-/pytest-Flagkonflikt. Job `154924` wiederholte ausschließlich die drei betroffenen Module `eval_executor`, `eval_experiments` und `eval_runner` mit initialisierten Flags; alle 27 Tests dieser Module bestanden auf `hai001` in 4:17 Minuten. Produktionscode und bestehende Testdateien blieben unverändert. Die vier GPU-Module bestanden bereits im ersten Lauf ohne Skips.

Die Modulübersicht samt verbleibenden Testnamen steht in [verification_summary.json](/fast/home/salan.isaqzoi/review_artifacts/omegalax_20260924/verification_summary.json). Dies war eine gezielte Auswahl für die untersuchten Produktionspfade, kein Gesamtlauf sämtlicher Modell-/Real-Weights-Tests. Die vier Suitefehler sind von den separat nachgewiesenen Produktionsfehlern F1–F9 zu unterscheiden: Viele dieser Fehler besitzen gerade keinen wirksamen bestehenden Regressionstest.

Die ergänzenden Reproduktionen sind bereits abgeschlossen: Job `154919` bestätigt F2/F3/F4/F6/F7; Job `154921` bestätigt F8/F9 und die numerische Einordnung T2/T3. Der erste Trace-Probejob `154920` scheiterte an einem fehlenden Pflichtargument im Review-Hilfsskript; das wurde ausschließlich im Hilfsskript korrigiert. Dieser Fehlversuch zählt nicht als Produktionsfehler.

Alle Wrapper, Reproduktionen, JUnit-Dateien und Modul-Logs liegen unter [review_artifacts/omegalax_20260924](/fast/home/salan.isaqzoi/review_artifacts/omegalax_20260924). Besonders relevant sind [probes.json](/fast/home/salan.isaqzoi/review_artifacts/omegalax_20260924/probes.json), [trace_probe.json](/fast/home/salan.isaqzoi/review_artifacts/omegalax_20260924/trace_probe.json) und [precision_probe.json](/fast/home/salan.isaqzoi/review_artifacts/omegalax_20260924/precision_probe.json). Die Jobs sind zusätzlich im zentralen `test_runs.md` dokumentiert.

Nicht durchgeführt wurden ein vollständiger Audit aller historischen Runpakete und generierten Daten, ein erneutes Training großer Modelle oder eine Prüfung aller archivierten Einzel-Experimentskripte unter `Salan_DONT_PUSH`. Der Trace-Branch wurde gezielt auf seine neuen Masken-/Curriculum-/Eval-Schnittstellen geprüft; sein kompletter dreiquelliger Datenaudit wurde nicht wiederholt. Das Review beweist konkrete Fehler und Abdeckungslücken, keine allgemeine Fehlerfreiheit der übrigen Codebase.
