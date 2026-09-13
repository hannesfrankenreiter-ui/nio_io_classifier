# Versuchsstufe 1 – XAI-Heatmaps je Lauf und Split

Erzeugt: 2026-08-11 09:47  
Je Lauf und Split die laufunabhaengigen Vergleichsbilder (Anfang und Ende des Splits) plus bis zu 6 Fehlklassifikationen.  
Fehler bestimmt bei **Schwelle 0.5** – dem Auswahlkriterium des Versuchsplans, nicht der auf val bestimmten Schwelle aus `metrics/threshold.json`.  
Erklaert wird jeweils die **vorhergesagte** Klasse, nicht die wahre.

| Lauf | Backbone | Freeze-Strategie | Split | Feste Vergleichsbilder | Fehler gesamt | Fehler gezeigt | Heatmaps | Status |
|---|---|---|---|---|---|---|---|---|
| `S1_01_resnet18_letzterblock` | resnet18 | letzter Block + Kopf | val | 10 | 36 | 6 | 16 | fertig |
| `S1_01_resnet18_letzterblock` | resnet18 | letzter Block + Kopf | test | 10 | 9 | 6 | 16 | fertig |
| `S1_02_resnet50_letzterblock` | resnet50 | letzter Block + Kopf | val | 10 | 16 | 6 | 16 | fertig |
| `S1_02_resnet50_letzterblock` | resnet50 | letzter Block + Kopf | test | 10 | 6 | 6 | 16 | fertig |
| `S1_03_efficientnet_b0_letzterblock` | efficientnet_b0 | letzter Block + Kopf | val | 10 | 39 | 6 | 16 | fertig |
| `S1_03_efficientnet_b0_letzterblock` | efficientnet_b0 | letzter Block + Kopf | test | 10 | 11 | 6 | 16 | fertig |
| `S1_04_convnext_tiny_letzterblock` | convnext_tiny | letzter Block + Kopf | val | 10 | 10 | 6 | 16 | fertig |
| `S1_04_convnext_tiny_letzterblock` | convnext_tiny | letzter Block + Kopf | test | 10 | 9 | 6 | 16 | fertig |
| `S1_05_resnet18_nurkopf` | resnet18 | nur Kopf | val | 10 | 55 | 6 | 16 | fertig |
| `S1_05_resnet18_nurkopf` | resnet18 | nur Kopf | test | 10 | 47 | 6 | 16 | fertig |
| `S1_06_resnet18_vollstaendig` | resnet18 | vollstaendiges Fine-Tuning | val | 10 | 27 | 6 | 16 | fertig |
| `S1_06_resnet18_vollstaendig` | resnet18 | vollstaendiges Fine-Tuning | test | 10 | 5 | 5 | 15 | fertig |
| `S1_07_resnet50_nurkopf` | resnet50 | nur Kopf | val | 10 | 49 | 6 | 16 | fertig |
| `S1_07_resnet50_nurkopf` | resnet50 | nur Kopf | test | 10 | 20 | 6 | 16 | fertig |
| `S1_08_resnet50_vollstaendig` | resnet50 | vollstaendiges Fine-Tuning | val | 10 | 21 | 6 | 16 | fertig |
| `S1_08_resnet50_vollstaendig` | resnet50 | vollstaendiges Fine-Tuning | test | 10 | 66 | 6 | 16 | fertig |
| `S1_09_efficientnet_b0_nurkopf` | efficientnet_b0 | nur Kopf | val | 10 | 47 | 6 | 16 | fertig |
| `S1_09_efficientnet_b0_nurkopf` | efficientnet_b0 | nur Kopf | test | 10 | 8 | 6 | 16 | fertig |
| `S1_10_efficientnet_b0_vollstaendig` | efficientnet_b0 | vollstaendiges Fine-Tuning | val | 10 | 7 | 6 | 16 | fertig |
| `S1_10_efficientnet_b0_vollstaendig` | efficientnet_b0 | vollstaendiges Fine-Tuning | test | 10 | 26 | 6 | 16 | fertig |
| `S1_11_convnext_tiny_nurkopf` | convnext_tiny | nur Kopf | val | 10 | 184 | 6 | 16 | fertig |
| `S1_11_convnext_tiny_nurkopf` | convnext_tiny | nur Kopf | test | 10 | 100 | 6 | 16 | fertig |
| `S1_12_convnext_tiny_vollstaendig` | convnext_tiny | vollstaendiges Fine-Tuning | val | 10 | 3 | 3 | 13 | fertig |
| `S1_12_convnext_tiny_vollstaendig` | convnext_tiny | vollstaendiges Fine-Tuning | test | 10 | 1 | 1 | 11 | fertig |
