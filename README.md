# Handwritten-Numbers-Classifier
*** Neural Networks Exam Project ***
Topic: Recognize Handwritten (Long) Numbers
Author: Valerio Felici (valerio.felici@student.unisi.it)

Questa cartella contiene:
	. 'main.py'
	. 'slides.pdf'
	. 'classificatore_cnn1.pth'
	. 'classificatore_cnn2.pth'
	. 'readme.txt'
	. 'pics'
	. 'foto'
	. 'dataset'


Il file 'main.py' contiene il codice sorgente per addestrare e valutare il modello.
I files 'classificatore_cnn1.pth' e 'classificatore_cnn2.pth' sono due modelli addestrati utilizzando due CNN con strutture diverse. Si può specificare da linea di comando quale modello utilizzare nella modalità 'eval' e 'eval_pics'.  
Il dataset MNIST viene scaricato e inserito nella cartella 'dataset' al primo avvio del codice 'main.py' (se non è già presente). Di seguito le modalità di utilizzo di quest'ultimo:

*TRAINING DEL MODELLO:
$ python main.py train

*VALUTAZIONE DEL MODELLO SUL TEST SET MNIST:
$ python main.py eval

*VALUTAZIONE DEL MODELLO SU FOTO:
$ python main.py eval_pics --folder='foto'

*VALUTAZIONE DEL MODELLO SU SCREENSHOTS:
$ python main.py eval_pics --folder='pics'


Ci sono alcuni parametri opzionali che possono essere specificati da linea di comando:

*VISUALIZZAZIONE PARAMETRI:
$python.py main.py -h
