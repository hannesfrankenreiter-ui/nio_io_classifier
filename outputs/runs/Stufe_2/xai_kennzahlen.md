# Versuchsstufe 2 – XAI-Kennzahlen

Erzeugt: 2026-08-26 10:48  
Lauf: `S2_12_convnext_tiny_vollstaendig`, Checkpoint aus Epoche 13, Split test.  
Umfang `alle`: 400 Aufnahmen (200 i.O. / 200 n.i.O.). Attributionsziel: die vorhergesagte Klasse (argmax, entspricht Schwelle 0,5).  
Bauteilmaske: 400× abgelegt; Flächenanteil 16,1 % im Mittel (6,3 % … 31,3 %), 0 als unplausibel markiert.

## Konzentrationsfaktor

Massenanteil geteilt durch Flächenanteil der Maske. 1,0 = gleichverteilte Karte.

| Verfahren | Belegte Karten | Median | Q25 – Q75 | Mittel | Min – Max | davon i.O. | davon n.i.O. |
|---|---|---|---|---|---|---|---|
| Integrated Gradients | 400 von 400 | 2,69 | 2,04 – 3,16 | 2,66 | 1,06 – 4,95 | 2,51 (200) | 2,84 (200) |
| Saliency Map | 400 von 400 | 3,09 | 2,38 – 3,64 | 3,07 | 1,36 – 5,81 | 2,94 (200) | 3,14 (200) |
| Occlusion Map | 227 von 400 | 4,24 | 3,18 – 5,49 | 4,29 | 0,24 – 8,27 | 4,22 (65) | 4,26 (162) |

„Belegte Karten“ zählt die Aufnahmen, für die das Verfahren überhaupt Attribution ausweist; eine durchgehend leere Karte trägt keinen Anteil bei und wird nicht mitgemittelt.

## Maskensensitivität

Konzentrationsfaktor (Median) auf erodierter bzw. dilatierter Maske. Zeigt, wie stark das Ergebnis an der Maskengrenze hängt.

| Verfahren | -16 px | -8 px | unverändert | +8 px | +16 px |
|---|---|---|---|---|---|
| Integrated Gradients | 2,85 | 2,88 | 2,69 | 2,56 | 2,39 |
| Saliency Map | 3,19 | 3,25 | 3,09 | 2,80 | 2,54 |
| Occlusion Map | 4,86 | 4,64 | 4,24 | 3,89 | 3,48 |

## Deletion und Insertion

Fläche unter der Wahrscheinlichkeitskurve über 20 Schritten, ohne Bauteilmaske gerechnet. Deletion: **klein** ist gut. Insertion: **groß** ist gut. Bezug ist die zufällige Reihenfolge.

| Reihenfolge | Deletion-AUC | Insertion-AUC |
|---|---|---|
| Integrated Gradients | 0,883 | 0,878 |
| Saliency Map | 0,849 | 0,898 |
| Occlusion Map | 0,868 | 0,892 |
| Zufällige Reihenfolge | 0,862 | 0,867 |

## Completeness der Integrated Gradients

Mittelwert 1,4649, Median 0,1344, Maximum 67,4476 bei 50 Stützpunkten über 400 Aufnahmen. Bei 256 Stützpunkten (Teilmenge von 12): Mittelwert 0,0452, Maximum 0,0986. Toleranz des automatisierten Tests: 0,05.

## Sanity-Check (Adebayo et al. 2018)

Betrag der Rangkorrelation gegen die Karte des trainierten Modells, gemittelt über 12 Aufnahmen. Ein Wert nahe null heißt: die Karte hängt am Modell.

| Verfahren | 1 Schichten | 2 Schichten | 4 Schichten | 8 Schichten | 16 Schichten |
|---|---|---|---|---|---|
| Integrated Gradients | 0,626 | 0,636 | 0,614 | 0,578 | 0,450 |
| Saliency Map | 0,812 | 0,818 | 0,802 | 0,763 | 0,607 |
| Occlusion Map | 0,184 | 0,008 | 0,023 | nan | nan |
