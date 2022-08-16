# Bachelorarbeit Alexander Schnerring

In diesem Repo sind Python Skripte mit verschiedenen Funktionen zur Datenverarbeitung sowie Skripte zur Implementierung neuronaler Netze vorhanden. Weiterhin wurden Skripte zum automatischen Training verschiedener Architekturen sowie deren Evaluation implementiert. 

Die Ordnerstruktur sollte beibehalten werden, da die Pfade in den Skripten und Notebooks relativ formuliert wurden.

## Dieser Ordner entält die Ausarbeitung und Abschlusspräsentation als PDFs sowie vier Audiodateien zur Demonstration der Ergebnisse. Weiterhin enthält der Ordner **code** den relevanten Code, der zur Datengenerierung aus Audiodateien, zur Initialisierung und Trainings der Modelle sowie zu deren Evaluation verwendet wurde.

## code Ordner

### Python Skipte

Die Datei **utils.py** enthält Funktionen für die Transformation von (Musik-)Signalen in verschiedene Darstellungen (STFT linear/mel, CQT) sowie deren Inversen. Weiterhin sind kleine Funktionen enthalten, die in den Verarbeitungsschritten nützlich sind, so z.B. Funktionen zum Kürzen von Soundfiles, Überführung von Arrays in Polarkoordinaten, Konvertierung in dB etc.

Der Ordner **Datengenerierung** enthält Skripte zur Generierung von Trainingsdaten aus Soundfiles. Die Soundfiles müssen im Ordner **Daten** gespeichert werden. Dort speichern die Skripte zur Datengenerierung auch die Trainingsdaten ab.

Der Ordner **Training** enthält Skipte zur Implementierung neuronaler Netze. Diese greifen auf die Trainingsdaten in **Daten** zu und speichern die Informationen der die trainierten Modelle im Ordner **Modelle** ab.


### Daten

Hier sind Daten abgelegt, die von Python Skripten und Notebooks genutzt werden. Dies beinhaltet neben den ursprünglichen Soundfiles Numpy Arrays, welche die STFT/CQT dieser Soundfiles beschreiben.


### Ergebnisse

Hier sind verschiedene Ergebnisse abgelegt, bspw. Trainingsverläufe oder Plots der vorhergesagten STFTs.


### Modelle

Hier sind verschiedene Modelle als .json bzw. .h5 Dateien abgelegt, welche mit Python Skripten erstellt wurden und mit Notebooks verwendet werden können.


###Hinweis zu nsgt
Es muss noch der Ordner *nsgt* hier liegen. Dieser kann von GitHub mit *git clone https://github.com/grrrr/nsgt.git* geklont werden. Dann Installationsanweisungen folgen:
In Konsole in Ordner *BA_Schnerring/nsgt* navigieren und
python3.6 setup.py build
sudo python3.6 setup.py install
ausführen. Das Paket kann dann eingebunden werden mit

import sys
sys.path.insert(1, "blabla/BA_Schnerring/nsgt/")
import nsgt

**ACHTUNG:** Das Paket wurde noch nicht vollständig für Python3 umgeschrieben, verwendet man nsgt.backwards() so tritt ein Fehler auf. Dieser kann behoben werden, indem man im Skript *nsgt/nsgt/nsigtf.py* **die Zeile fc = mmap(fft, c) ändert in fc = list(mmap(fft, c))**.
