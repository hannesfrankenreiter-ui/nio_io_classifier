# Versuchsstufe 1 – XAI-Kennzahlen

Erzeugt: 2026-08-26 13:08  
Lauf: `S1_12_convnext_tiny_vollstaendig`, Checkpoint aus Epoche 14, Split test.  
Umfang `alle`: 311 Aufnahmen (173 i.O. / 138 n.i.O.). Attributionsziel: die vorhergesagte Klasse (argmax, entspricht Schwelle 0,5).  
Bauteilmaske: 311× otsu; Flächenanteil 9,8 % im Mittel (3,0 % … 16,8 %), 0 als unplausibel markiert.

## Konzentrationsfaktor

Massenanteil geteilt durch Flächenanteil der Maske. 1,0 = gleichverteilte Karte.

| Verfahren | Belegte Karten | Median | Q25 – Q75 | Mittel | Min – Max | davon i.O. | davon n.i.O. |
|---|---|---|---|---|---|---|---|
| Integrated Gradients | 311 von 311 | 4,43 | 3,94 – 5,21 | 4,71 | 2,44 – 9,27 | 4,10 (173) | 4,77 (138) |
| Saliency Map | 311 von 311 | 2,03 | 1,69 – 2,50 | 2,12 | 0,97 – 3,49 | 2,33 (173) | 1,73 (138) |
| Occlusion Map | 178 von 311 | 5,19 | 3,79 – 6,62 | 5,62 | 0,00 – 23,69 | 4,78 (126) | 6,44 (52) |

„Belegte Karten“ zählt die Aufnahmen, für die das Verfahren überhaupt Attribution ausweist; eine durchgehend leere Karte trägt keinen Anteil bei und wird nicht mitgemittelt.

## Maskensensitivität

Konzentrationsfaktor (Median) auf erodierter bzw. dilatierter Maske. Zeigt, wie stark das Ergebnis an der Maskengrenze hängt.

| Verfahren | -16 px | -8 px | unverändert | +8 px | +16 px |
|---|---|---|---|---|---|
| Integrated Gradients | 4,53 | 4,68 | 4,43 | 3,76 | 3,26 |
| Saliency Map | 2,30 | 2,27 | 2,03 | 1,82 | 1,69 |
| Occlusion Map | 5,51 | 5,42 | 5,19 | 4,91 | 4,59 |

## Deletion und Insertion

Fläche unter der Wahrscheinlichkeitskurve über 20 Schritten, ohne Bauteilmaske gerechnet. Deletion: **klein** ist gut. Insertion: **groß** ist gut. Bezug ist die zufällige Reihenfolge.

| Reihenfolge | Deletion-AUC | Insertion-AUC |
|---|---|---|
| Integrated Gradients | 0,639 | 0,912 |
| Saliency Map | 0,575 | 0,788 |
| Occlusion Map | 0,582 | 0,669 |
| Zufällige Reihenfolge | 0,595 | 0,598 |

## Completeness der Integrated Gradients

Mittelwert 0,5206, Median 0,0304, Maximum 96,3687 bei 50 Stützpunkten über 311 Aufnahmen. Bei 256 Stützpunkten (Teilmenge von 12): Mittelwert 0,0057, Maximum 0,0158. Toleranz des automatisierten Tests: 0,05.

## Sanity-Check (Adebayo et al. 2018)

Betrag der Rangkorrelation gegen die Karte des trainierten Modells, gemittelt über 12 Aufnahmen. Ein Wert nahe null heißt: die Karte hängt am Modell.

| Verfahren | 1 Schichten | 2 Schichten | 4 Schichten | 8 Schichten | 16 Schichten |
|---|---|---|---|---|---|
| Integrated Gradients | 0,619 | 0,599 | 0,578 | 0,531 | 0,441 |
| Saliency Map | 0,552 | 0,542 | 0,465 | 0,369 | 0,154 |
| Occlusion Map | 0,333 | 0,087 | 0,205 | nan | nan |
