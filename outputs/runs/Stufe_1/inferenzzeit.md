# Versuchsstufe 1 – Inferenzzeit je Architektur

Erzeugt: 2026-08-27 11:49  
Median aus 30 Wiederholungen nach 10 Aufwaermdurchlaeufen, 512x512 px (HxB), fp32.  
**Je Architektur gemessen, nicht je Lauf**: unter `model.eval()` und `torch.no_grad()` hat die Freeze-Strategie keinen Einfluss auf die Inferenzzeit.  
Ersetzt die Spalte „Rechenzeit je Bild" der Vergleichstabellen, die beilaeufig waehrend zwoelf verschiedener Trainingslaeufe erhoben wurde und dadurch ueberwiegend den GPU-Lastzustand misst.

| Backbone | Batch 16 [ms/Bild] | Einzelbild [ms] | Streuung Batch [ms] | Schnellster Lauf [ms] | Parameter gesamt |
|---|---|---|---|---|---|
| resnet18 | 24,66 | 16,97 | 0,03 | 24,63 | 11.441.218 |
| efficientnet_b0 | 45,32 | 21,54 | 0,02 | 45,27 | 4.665.470 |
| convnext_tiny | 88,29 | 70,65 | 0,16 | 88,25 | 28.215.906 |
| resnet50 | 93,74 | 44,37 | 0,04 | 93,69 | 24.559.170 |
