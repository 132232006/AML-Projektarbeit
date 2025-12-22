# AML‑Projektarbeit: Verkehrszeichenerkennung mit Objektdetektion (YOLOv8) auf GTSDB / FullIJCNN2013

## Zusammenfassung (Abstract)
Diese Projektarbeit implementiert eine reproduzierbare Pipeline zur **Detektion und Klassifikation deutscher Verkehrszeichen** mittels **Single‑Stage Objektdetektion** (Ultralytics **YOLOv8**) auf dem Datensatz **German Traffic Sign Detection Benchmark (GTSDB)**, bereitgestellt als Archiv **FullIJCNN2013**. Der Workflow umfasst (i) die Bereitstellung des Datensatzes inklusive Ground‑Truth‑Annotationen, (ii) die Konvertierung in das **YOLO‑Format** (Train/Val/Test), (iii) Fine‑Tuning eines vortrainierten YOLOv8‑Modells unter Verwendung gezielter Datenaugmentierungen sowie (iv) quantitative Evaluation (mAP, Precision/Recall), qualitative Inferenz und Export nach **ONNX**.

---

## Datensatz & Quellen

### Offizielle Benchmark‑Seite (RUB, Institut für Neuroinformatik)
Die Benchmark‑Seite beschreibt den Datensatz, das Bildformat (PPM), die Variabilität der Schildgrößen und das Annotationformat (CSV, Semikolon‑separiert) inklusive der Felder:
`filename; x1; y1; x2; y2; class_id`

- Übersicht: https://benchmark.ini.rub.de/gtsdb_dataset.html  
  **Download‑Pfad:** Unter *Downloads* bei „The GTSDB dataset is available via this link.“ auf **„this link“** klicken und auf der Folgeseite am Ende **„FullIJCNN2013.zip“** herunterladen.

### Direkter Download (FullIJCNN2013.zip)
- https://sid.erda.dk/public/archives/ff17dc924eba88d5d01a807357d6614c/FullIJCNN2013.zip

---

## Wichtiger manueller Schritt (Ground Truth)
Nach Ausführung von **Schritt 2** muss die Datei **`gt.csv`** in den Ordner **`FullIJCNN2013/`** eingefügt werden.

Die Konvertierung in YOLO‑Labels basiert auf `gt.csv`. Ohne korrekt positionierte Ground‑Truth‑Datei ist keine konsistente Label‑Generierung möglich.

---

## Projektlogik / Pipeline‑Überblick
Die Pipeline erzeugt eine YOLO‑kompatible Struktur unter `BASE_DIR/yolo_data/`:

- `yolo_data/images/{train,val,test}/*.jpg`
- `yolo_data/labels/{train,val,test}/*.txt`
- `data.yaml` (Dataset‑Konfiguration; `nc=43`, `names=[...]`)

Die Annotationen werden von Pixel‑Koordinaten in das YOLO‑Normformat überführt:
`class_id x_center y_center width height` (alle Größen normiert auf Bildbreite/‑höhe).

---

## Reproduzierbarer Workflow (Schritte 1–8)

### 1 — Installation der Abhängigkeiten & Umgebungsprüfung (Python/PyTorch/CUDA)
Ziel ist die Sicherstellung einer lauffähigen Umgebung für Training, Auswertung, Visualisierung sowie ggf. GPU‑Beschleunigung. Die Umgebung wird durch Versionsausgaben (Python, Platform, PyTorch, CUDA‑Status) validiert.

---

### 2 — Datensatzbereitstellung: Entpacken von FullIJCNN2013.zip und Download (optional)
Der Datensatz wird in `BASE_DIR` bereitgestellt. Falls ein Download‑Link konfiguriert ist, wird das ZIP heruntergeladen; andernfalls ist eine manuelle Ablage möglich. Anschließend erfolgt das Entpacken in `BASE_DIR`, sodass ein `FullIJCNN2013/`‑Verzeichnis mit Bilddaten (PPM) verfügbar ist.

**Hinweis:** Der Datensatz liegt in PPM vor; die Pipeline konvertiert nach JPG für Ultralytics‑Kompatibilität.

---

### 3 — Konvertierung des FullIJCNN2013/GTSDB‑Datensatzes in das YOLO‑Format (Train/Val/Test)
Die CSV‑Annotationen (`gt.csv`, Semikolon‑separiert) werden eingelesen, pro Bild aggregiert und in YOLO‑Labels überführt. Parallel werden Bilder von PPM nach JPG konvertiert. Anschließend wird ein Split in Train/Val/Test erzeugt.

#### 3.1 — Dataset‑Sanity‑Check: Label‑Vollständigkeit & Box‑Größenverteilung
Ziel ist die Plausibilisierung der Konvertierung durch:
- 1:1‑Zuordnung von Bild‑ und Label‑Dateien
- Identifikation fehlender Labels
- Analyse der Verteilung relativer Boxflächen (insbesondere kleine Objekte)

#### 3.2 — Interaktive Augmentierungs‑Vorschau (photometrisch)
Eine interaktive Vorschau (Brightness/Saturation/Contrast, optional Flip/Mosaic) dient der qualitativen Einschätzung augmentierungsbedingter Bildveränderungen. Es werden keine Daten persistent verändert.

---

### 4 — `data.yaml` erstellen (43 Klassen)
Es wird eine Ultralytics‑kompatible Dataset‑Konfiguration erzeugt mit:
- `path` auf `BASE_DIR/yolo_data`
- `train/val/test` auf die jeweiligen Bildordner
- `nc=43`
- `names` als Liste der 43 Klassenbezeichnungen (Index entspricht `class_id`)

**Wichtig:** `len(names)` muss exakt `nc` entsprechen und mit der Label‑ID‑Semantik übereinstimmen.

---

### 5 — YOLOv8‑Training (Fine‑Tuning auf GTSDB mit gezielten Augmentierungen)
Ein vortrainiertes YOLOv8‑Modell wird auf dem konvertierten Datensatz feinabgestimmt. Der Trainingslauf nutzt augmentierungsbasierte Strategien zur Robustheitssteigerung und zur Verbesserung der Detektion kleiner Objekte (u. a. Mosaic, leichte MixUp‑Anteile, Copy‑Paste). Die Trainingsartefakte (Weights, Logs, Plots) werden im Run‑Verzeichnis abgelegt.

#### 5.1 — Auswertung des Trainingsverlaufs (`results.png`)
Ultralytics erzeugt standardisierte Learning Curves, u. a.:
- Train/Val: `box_loss`, `cls_loss`, `dfl_loss`
- Metriken: `precision(B)`, `recall(B)`, `mAP50(B)`, `mAP50‑95(B)`

---

### 6 — Validierung und Testauswertung/Metriken (model.val auf val und test)
Die Evaluation erfolgt getrennt auf `val` und `test` (entsprechend `data.yaml`) mittels `model.val`. Berichtet werden mAP‑Metriken (mAP50, mAP50‑95) sowie Precision/Recall. Optional kann eine Confusion‑Matrix zur Fehleranalyse (insbesondere bei 43 Klassen) erzeugt werden.

---

### 7 — Qualitative Inferenz auf Testbildern und Speicherung visualisierter Vorhersagen
Zur qualitativen Validierung werden Inferenzläufe auf Testbildern durchgeführt. Vorhersagen werden mit Visualisierung (BBox + Label + Confidence) gespeichert.

#### 7.1 — Interaktive Visualisierung der Ground‑Truth‑Annotationen (YOLO‑Labels)
Ein interaktiver Viewer ermöglicht das Durchklicken der Splits und die Visualisierung der Ground‑Truth‑Bounding‑Boxes.

#### 7.2 — Interaktiver Inferenz‑Viewer für Testdaten (Parameter: conf, iou, imgsz)
Ein interaktiver Viewer erlaubt die Variation zentraler Inferenzparameter:
- `conf` (Confidence‑Threshold)
- `iou` (NMS‑IoU‑Threshold)
- `imgsz` (Inferenzauflösung)

#### 7.3 — Interaktive Inferenz auf frei wählbarem Pfad/Ordner (Batch‑Predict mit Ergebnisablage)
Ein UI‑Element erlaubt die Inferenz auf einem beliebigen Datei‑/Ordnerpfad (eigene Daten) mit persistenter Ablage der Ergebnisse im Run‑Verzeichnis.

---

### 8 — Export des trainierten YOLO‑Modells nach ONNX
Der Export erfolgt via:
- `model.export(format='onnx', opset=12)`

Ziel ist die Nutzung des trainierten Modells in ONNX‑kompatiblen Inferenzumgebungen (z. B. onnxruntime). Der Parameter `opset` steuert die ONNX‑Operator‑Version und beeinflusst die Kompatibilität mit Ziel‑Toolchains.









---

# AML-Projektarbeit
Verkehrszeichenerkennung mit Objektdetektion (YOLOv8, GTSDB, FullIJCNN2013)

Übersicht auf der Seite des Instituts für Neuroinformatik: https://benchmark.ini.rub.de/gtsdb_dataset.html.
--> hier unter Downloads bei "The GTSDB dataset is available via this link." auf "this link" klicken und anschließend auf der neuen Seite ganz unten die Datei "FullIJCNN2013.zip" runterladen
Direkter Donload Linkhttps://sid.erda.dk/public/archives/ff17dc924eba88d5d01a807357d6614c/FullIJCNN2013.zip

Nach Ausführung von schritt 2 die datei "gt.csv" in den Ordner "FullIJCNN2013" hinzufügen

1 – Installation der Abhängigkeiten, Imports und Umgebungsprüfung (Python/PyTorch/CUDA)
2 – Datensatzbereitstellung: Entpacken von FullIJCNN2013.zip und Download (optional)
3 – Konvertierung des FullIJCNN2013/GTSDB‑Datensatzes in das YOLO‑Format (Train/Val/Test)
  3.1 – Dataset‑Sanity‑Check: Vollständigkeit von Labels und Verteilung der Bounding‑Box‑Größen
  3.2 – Interaktive Augmentierungs‑Vorschau (photometrisch)
4. data.yaml erstellen (43 Klassen) (Dataset‑Konfiguration für Ultralytics/YOLO)
5 – YOLOv8‑Training (Fine‑Tuning auf GTSDB mit gezielten Augmentierungen)
  5.1 – Auswertung des Trainingsverlaufs anhand results.png (Learning Curves: loss, mAP, Precision/Recall)
6 – Validierung und Testauswertung/Metriken (model.val auf val und test)
7 – Qualitative Inferenz auf Testbildern und Speicherung visualisierter Vorhersagen
  7.1 – Interaktive Visualisierung der Ground‑Truth‑Annotationen (YOLO‑Labels)
  7.2 – Interaktiver Inferenz‑Viewer für Testdaten (Parameter: conf, iou, imgsz)
  7.3 – Interaktive Inferenz auf frei wählbarem Pfad/Ordner mit eigenen Daten (Batch‑Predict mit Ergebnisablage)
8 – Export des trainierten YOLO‑Modells nach ONNX (model.export(format='onnx', opset=12))
