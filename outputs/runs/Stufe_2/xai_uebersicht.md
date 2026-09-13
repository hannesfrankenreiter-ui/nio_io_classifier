# Versuchsstufe 2 – XAI-Heatmaps je Lauf und Split

Erzeugt: 2026-08-19 21:34  
Je Lauf und Split die laufunabhaengigen Vergleichsbilder (44 zufaellig je Split (Seed 123, laufunabhaengig)) plus bis zu 6 Fehlklassifikationen.  
Fehler bestimmt bei **Schwelle 0.5** – dem Auswahlkriterium des Versuchsplans, nicht der auf val bestimmten Schwelle aus `metrics/threshold.json`.  
Erklaert wird jeweils die **vorhergesagte** Klasse, nicht die wahre.

| Lauf | Backbone | Freeze-Strategie | Split | Vergleichsbilder (Basis) | Fehler gesamt | Fehler gezeigt | Heatmaps | Status |
|---|---|---|---|---|---|---|---|---|
| `S2_01_resnet18_letzterblock` | resnet18 | letzter Block + Kopf | val | 44 | 1 | 1 | 45 | fertig |
| `S2_01_resnet18_letzterblock` | resnet18 | letzter Block + Kopf | test | 44 | 10 | 6 | 50 | fertig |
| `S2_02_resnet50_letzterblock` | resnet50 | letzter Block + Kopf | val | 44 | 2 | 2 | 46 | fertig |
| `S2_02_resnet50_letzterblock` | resnet50 | letzter Block + Kopf | test | 44 | 9 | 6 | 50 | fertig |
| `S2_03_efficientnet_b0_letzterblock` | efficientnet_b0 | letzter Block + Kopf | val | 44 | 0 | 0 | 44 | fertig |
| `S2_03_efficientnet_b0_letzterblock` | efficientnet_b0 | letzter Block + Kopf | test | 44 | 10 | 6 | 50 | fertig |
| `S2_04_convnext_tiny_letzterblock` | convnext_tiny | letzter Block + Kopf | val | 44 | 0 | 0 | 44 | fertig |
| `S2_04_convnext_tiny_letzterblock` | convnext_tiny | letzter Block + Kopf | test | 44 | 2 | 2 | 46 | fertig |
| `S2_05_resnet18_nurkopf` | resnet18 | nur Kopf | val | 44 | 12 | 6 | 50 | fertig |
| `S2_05_resnet18_nurkopf` | resnet18 | nur Kopf | test | 44 | 64 | 6 | 50 | fertig |
| `S2_06_resnet18_vollstaendig` | resnet18 | vollstaendiges Fine-Tuning | val | 44 | 1 | 1 | 45 | fertig |
| `S2_06_resnet18_vollstaendig` | resnet18 | vollstaendiges Fine-Tuning | test | 44 | 3 | 3 | 47 | fertig |
| `S2_07_resnet50_nurkopf` | resnet50 | nur Kopf | val | 44 | 3 | 3 | 47 | fertig |
| `S2_07_resnet50_nurkopf` | resnet50 | nur Kopf | test | 44 | 8 | 6 | 50 | fertig |
| `S2_08_resnet50_vollstaendig` | resnet50 | vollstaendiges Fine-Tuning | val | 44 | 0 | 0 | 44 | fertig |
| `S2_08_resnet50_vollstaendig` | resnet50 | vollstaendiges Fine-Tuning | test | 44 | 1 | 1 | 45 | fertig |
| `S2_09_efficientnet_b0_nurkopf` | efficientnet_b0 | nur Kopf | val | 44 | 3 | 2 | 46 | fertig |
| `S2_09_efficientnet_b0_nurkopf` | efficientnet_b0 | nur Kopf | test | 44 | 25 | 6 | 50 | fertig |
| `S2_10_efficientnet_b0_vollstaendig` | efficientnet_b0 | vollstaendiges Fine-Tuning | val | 44 | 0 | 0 | 44 | fertig |
| `S2_10_efficientnet_b0_vollstaendig` | efficientnet_b0 | vollstaendiges Fine-Tuning | test | 44 | 8 | 6 | 50 | fertig |
| `S2_11_convnext_tiny_nurkopf` | convnext_tiny | nur Kopf | val | 44 | 10 | 6 | 50 | fertig |
| `S2_11_convnext_tiny_nurkopf` | convnext_tiny | nur Kopf | test | 44 | 27 | 6 | 50 | fertig |
| `S2_12_convnext_tiny_vollstaendig` | convnext_tiny | vollstaendiges Fine-Tuning | val | 44 | 0 | 0 | 44 | fertig |
| `S2_12_convnext_tiny_vollstaendig` | convnext_tiny | vollstaendiges Fine-Tuning | test | 44 | 0 | 0 | 44 | fertig |
