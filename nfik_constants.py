"""Konstanten der Kopien aus Stück 4 (`zugwahl-lernen-demo`, Präfix zwl_, hier nfik_): nur die Namen, die `nfik_mlp`, `nfik_featnet` und `nfik_policy` brauchen,
mit **denselben Namen und Werten** wie in `zwl_constants.py` (Test `tests/test_copies.py`); einzige Ausnahme ist `MODELL_ORDNER` (hier `nfi_models`).

Das Merkmalsnetz K (sieben Merkmale, 145 Parameter) ist das Netz B dieser Demo; das Zugnetz Z (177 Parameter) wird hier nicht benutzt, seine Namen stehen nur, weil die kopierte
Datei `nfik_mlp` sie beim Laden nennt."""

H = 16                                          # versteckte Einheiten
Z_MERKMALE, K_MERKMALE = 9, 7
Z_PARAMS = H * Z_MERKMALE + H + H + 1           # 177
K_PARAMS = H * K_MERKMALE + H + H + 1           # 145
K_LERNRATE = 0.03
K_CLIP = 5.0
K_INST, K_ROLL = 32, 8                          # Instanzen und Touren (verschiedene Startstädte) je Trainingsschritt
K_BATCH_TOUREN = K_INST * K_ROLL                # 256
K_SCHRITTE_FROZEN = 3750                        # 3 750 x 256 = 960 000 Touren (Rezept von L2 und Stück 4)
K_N_TRAIN = 20
MODELL_FORMAT = "zwl-mlp-1"
MODELL_ORDNER = "nfi_models"                    # in der Quelle zwl_modelle

# Zwecke der Zufalls-Seeds der Quelle (Bereich zwl 100 bis 149): das Training von K benutzt dieselben Ströme wie Stück 4, damit B mit s und B ohne s gepaart sind
ZWECK_K_INIT, ZWECK_K_INST, ZWECK_K_START, ZWECK_K_ZIEHEN = 104, 105, 106, 107
