"""
*** Neural Networks Exam Project ***
Topic: Recognize Handwritten (Long) Numbers
Author: Valerio Felici (valerio.felici@student.unisi.it)
"""

import os
import torch
import argparse
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, random_split
from PIL import Image, ImageOps

# Parametri che verranno specificati da linea di comando
LR = None
EPOCHS = None
BATCH_SIZE = None


# FUNZIONE PER IL DOWNLOAD DEI DATI
def download_data(folder):
    """Download del dataset MNIST nella cartella 'folder' (se non è già presente).

    Args:
        folder: cartella per il download

    Returns:
        training_set: dati per il training (50k)
        validation_set: dati per la validazione (10k)
        test_set: dati per il test (10k)
    """

    # Data augmentation sui dati del training RIMUOVERE??? ->  rotazione tra (25,-25) gradi e trasformazione di scala tra (0.95,1.05)

    training_data_transform = transforms.Compose([
        transforms.RandomAffine(degrees=25, scale=(0.95, 1.05)),
        transforms.ToTensor(),  # passaggio da PIL a tensore e normalizzazione tra 0 e 1
    ])

    training_set = MNIST(root=folder,   # cartella di destinazione
                         train=True,    # dati per il train
                         download=True,
                         transform=ToTensor()
                         )

    test_set = MNIST(root=folder,
                     train=False,  # dati per il test
                     download=True,
                     transform=ToTensor()  # i dati di test non sono soggetti a trasformazioni in questa fase
                     )

    # Creazione del validation set (bilanciato rispetto alle classi) prendendo 10k esempi dal training set

    training_set, validation_set = random_split(training_set, [50000, 10000])

    return training_set, validation_set, test_set


class Classifier:
    """Classificatore per la predizione su immagini di numeri scritti a mano."""

    def __init__(self, cnn_structure='cnn1', device='cpu'):
        """Crea un classificatore non addestrato.

        Args:
            cnn_structure: stringa che indica il nome della CNN da addestrare
            data_augmentation: bit che indica se effettuare o meno data augmentation sui 3 set di dati
            device: stringa che indica il device da usare ("cpu", "cuda:0", "cuda:1", ...)
        """

        # Attributi della classe

        self.num_outputs = 10   # 10 classi mutuamente esclusive
        self.device = torch.device(device)  # device in cui saranno spostati i dati

        # Creazione della rete
        if cnn_structure is not None and cnn_structure == 'cnn1':

            # Primo tipo di CNN
            self.net = nn.Sequential(   # SimpleCNN-based network
                nn.Conv2d(1, 32, kernel_size=3, stride=1),  # 1° layer: 32 filtri (3x3) convoluti con stride=1
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, stride=1),  # 2° layer: 64 filtri (3x3) convoluti con stride=1
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=1),  # scelta delle migliori features tramite max pooling
                nn.Dropout(),   # incrementa la convergenza
                nn.Flatten(),   # appiattisce le feature maps
                nn.Linear(64 * 23 * 23, 64),  # due layers di convoluzione e il pooling hanno generato 64 feature maps (23x23)
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(64, self.num_outputs),
            )

        elif cnn_structure is not None and cnn_structure == 'cnn2':

            # Secondo tipo di CNN
            self.net = nn.Sequential(  # SimpleCNN-based network
                nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # 1° layer: 32 filtri (3x3) convoluti con stride=1
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 2° layer: 64 filtri (3x3) convoluti con stride=1
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),  # scelta delle migliori features tramite max pooling
                nn.Dropout(),  # incrementa la convergenza
                nn.Flatten(),  # appiattisce le feature maps
                nn.Linear(64 * 7 * 7, 128),  # due layers di convoluzione e il pooling hanno generato 64 feature maps (23x23)
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(128, self.num_outputs),
            )

        # Data augmentation sui dati per il training -> rotazione random (25,-25) gradi e trasformazione di scala tra (0.95,1.05)
        self.preprocess_train = transforms.Compose([
            transforms.RandomAffine(degrees=25, scale=(0.95, 1.05)),
            transforms.ToTensor(),  # passaggio da PIL a tensore e normalizzazione tra (0,1)
        ])

        # Data augmentation sui dati per il validation e per il test (qua sono trasformazioni fisse)
        self.preprocess_eval = [transforms.ToTensor(),
                                transforms.Compose([
                                    transforms.RandomRotation(degrees=(15, 15)),
                                    transforms.ToTensor(),
                                ]),
                                transforms.Compose([
                                    transforms.RandomRotation(degrees=(-15, -15)),
                                    transforms.ToTensor(),
                                ]),
                                transforms.Compose([
                                    transforms.RandomAffine(degrees=0, scale=(1.05, 1.05)),
                                    transforms.ToTensor(),
                                ])
                                ]

        # Spostamento della rete nella memoria del dispositivo corretto
        self.net.to(self.device)

    # FUNZIONE DI SALVATAGGIO DEL CLASSIFICATORE

    def save(self, nome_file):
        """Salva il il classificatore."""

        torch.save(self.net.state_dict(), nome_file)

    # FUNZIONE DI CARICAMENTO DEL CLASSIFICATORE

    def load(self, nome_file):
        """Carica il classificatore."""

        self.net.load_state_dict(torch.load(nome_file, map_location=self.device))   # Metodo messo a disposizione da PyTorch

    # FUNZIONE PER IL CALCOLO DELL'OUTPUT DELLA RETE

    def forward(self, x):
        """Calcola l'output della rete."""

        output_net_no_act = self.net(x)  # output della rete senza attivazione
        output_net = torch.softmax(output_net_no_act, dim=1)    # output applicando la funzione di attivazione (softmax)

        return output_net_no_act, output_net

    @staticmethod
    def decision(output_net):
        """Restituisce la predizione dato l'output della rete dopo l'applicazione della funzione di attivazione.

        Args:
            output_net: Tensore 2D con righe pari agli outputs generati dalla rete

        Returns:
            predictions: Tensore 1D in cui ogni elemento è la classe predetta dalla rete per ciascun output (ciascuna riga)
        """

        predictions = torch.argmax(output_net, dim=1)  # le predizioni sono gli indici dell'elemento più alto di ciascuna riga

        return predictions

    @staticmethod
    def compute_accuracy(predictions, labels):
        """Calcola la precisione della rete rispetto alle classi corrette.

        Args:
            predictions: Tensore 1D con le predizioni della rete
            labels: Tensore 1D con le predizioni corrette

        Returns:
            accuracy: percentuale di predizioni corrette
        """

        correct = torch.sum(predictions==labels).item()   # conteggio predizioni corrette
        accuracy = correct*100.0/len(labels)  # percentuale predizioni corrette

        return accuracy

    # FUNZIONE DI VALUTAZIONE DEL MODELLO

    def eval_classifier(self, data_set):
        """Valuta le prestazioni del modello su un set di dati. Considera tutte le versioni del set
        che ne derivano dalle diverse trasformazioni valutando le prestazioni sul set composto da queste diverse versioni."""

        # Controllo se il classificatore sia in train mode, se si, passa a eval mode
        is_train = self.net.training  # assume valore True se è in modalità addestramento
        if is_train:
            self.net.eval()  # passaggio a modalità valutazione

        # Inizializzazione variabili utili
        tot_predictions = []  # contenitore di tutte le predizioni fatte dalla rete
        tot_labels = []  # contenitore di tutte le labels corrette

        with torch.no_grad():  # disabilità il calcolo del gradiente in questa fase

            # Ciclo sulla lista contenente le trasformazioni da fare sul set
            for t in range(len(self.preprocess_eval)):

                data_set.dataset.transform = self.preprocess_eval[t]  # applica la trasformazione sui dati

                # Ciclo sui batch di dati
                for i, (images, labels) in enumerate(data_set):
                    # Spostamento dei dati nel dispositivo corretto
                    images = images.to(self.device)
                    labels = labels.to(self.device)  # serve ????????

                    output_net_no_act, output_net = self.forward(images)  # calcolo output della rete sul batch di dati
                    predictions = self.decision(output_net)  # predizioni della rete sul batch
                    tot_predictions.append(predictions)  # aggiunta predizioni sul batch attuale alle altre???? predictions.cpu()??
                    tot_labels.append(labels)  # aggiunta labels sul batch attuale alle altre

        # Calcolo precisione
        dataset_accuracy = self.compute_accuracy(torch.cat(tot_predictions, 0), torch.cat(tot_labels, 0))

        # Ripristino dei dati alle trasformazioni casuali usate per il training
        data_set.dataset.transform = self.preprocess_train

        # Ripristino della modalità alla quale era il classificatore
        if is_train:
            self.net.train()

        return dataset_accuracy

    # FUNZIONE DI ADDESTRAMENTO DEL CLASSIFICATORE

    def train_classifier(self, learning_rate, epochs):
        """"Addestramento del classificatore con i dati per training e validation."""

        # Inizializzazione variabili utili
        best_accuracy = -1  # precisione maggiore ottenuta
        val_accuracy = -1   # memorizza il valore della percentuale di precisione ottenuta
        val_accuracies = []  # memorizza le percentuali di precisione ottenute nell'addestramento

        # Controllo che la rete sia in modalità training
        self.net.train()

        # Creazione ottimizzatore (Adam)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)

        # Loss function  (Cross Entropy Loss)
        criterion = nn.CrossEntropyLoss()

        # Data augmentation
        train_dataloader.dataset.transform = self.preprocess_train

        # Ciclo sulle epoche
        for e in range(epochs):

            # Inizializzazione variabili per le statistiche sull'epoca attuale
            epoch_train_acc = 0.    # precisione sul training
            epoch_train_loss = 0.   # loss sul training
            epoch_train_examples = 0    # contatore esempi presentati durante l'epoca

            # Ciclo sui batch di dati (batch mode)
            for i, (images, labels) in enumerate(train_dataloader):

                # Spostamento dei dati nel dispositivo corretto
                images = images.to(self.device)
                labels = labels.to(self.device)

                batch_train_examples = images.shape[0]  # per l'ultimo batch di dati può essere inferiore a batch_size
                epoch_train_examples += batch_train_examples    # aggiornamento contatore esempi presentati nell'epoca

                # Calcolo delle predizioni della rete
                output_net_no_act, output_net = self.forward(images)

                # Calcolo della loss function
                loss = criterion(output_net_no_act, labels)   # calcolo loss tra output della rete e classi corrette

                # Calcolo gradienti e aggiornamento dei pesi della rete
                optimizer.zero_grad()   # azzeramento aree di memoria in cui erano stati inseriti i gradienti calcolati in precedenza
                loss.backward()  # backpropagation
                optimizer.step()   # aggiornamento pesi

                # Calcolo prestazioni rete sul mini-batch per il training corrente
                with torch.no_grad():

                    self.net.eval()  # passaggio alla modalità di valutazione

                    # Calcolo prestazioni sul batch
                    predictions = self.decision(output_net)  # predizioni sul batch corrente
                    batch_train_acc = self.compute_accuracy(predictions, labels)    # precisione sul batch corrente

                    # Aggiornamento prestazioni sull'epoca
                    epoch_train_acc += batch_train_acc*batch_train_examples
                    epoch_train_loss += loss.item()*batch_train_examples

                    self.net.train()    # ritorno alla modalità di training

            val_accuracy = self.eval_classifier(val_dataloader)  # precisione sul validation set

            epoch_train_loss /= epoch_train_examples
            epoch_train_acc /= epoch_train_examples

            val_accuracies.append(val_accuracy)     # riempimento lista con le accuracies ottenute in ogni epoca

            # Stampa statistiche ottenute sull'epoca
            print('Epoca: ', e + 1, '\nPrecisione sul training: ', epoch_train_acc, '%')
            print('Loss sul training: ', round(epoch_train_loss, 3))
            print('Precisione sul validation: ', val_accuracy, '%')
            #
            # print(self.eval_classifier(val_dataloader))
            # print(self.eval_classifier(val_dataloader))
            # print(self.eval_classifier(val_dataloader))

            # Controllo che sia stato raggiunto il valore max della precisione tra quelli ottenuti fino ad adesso
            if val_accuracy > best_accuracy:
                self.save('classificatore_cnn2.pth')  # salvataggio del modello (è il miglior ottenuto fin qui)
                best_accuracy = val_accuracy  # aggiornamento del valore più alto di precisione ottenuto
                print('Salvataggio del miglior modello... \n')

        # Stampa dei grafici relativi a precisione e loss
        # fig, axs = plt.subplots(2)
        # axs[0].plot(val_accuracy)
        # axs[1].plot(epoch_loss)
        # axs[0].set_title('Accuracy')
        # axs[1].set_title('Loss')
        # #plt.plot(accuracies)
        # #plt.show()
        # #plt.plot(losses)
        # #plt.show()
        # plt.savefig('plots.pdf')    # salvataggio grafici in formato pdf

    # FUNZIONE DI SEGMENTAZIONE IMMAGINI
    def segment_image(self, image):
        """Segmenta l'immagine in tante immagini quante sono le cifre scritte a mano. Ogni immagine rappresenta l'area in cui è contenuta
        ciascuna cifra.

        Args:
            image: nome dell'immagine da segmentare

        Returns:
            sub_images: tensore contenente le sotto-immagini relative a ciascuna cifra
        """

        self.net.eval()     # passaggio alla fase di valutazione

        # Apertura immagine
        immagine = Image.open(image)

        # Operazioni sull'immagine
        immagine = ImageOps.grayscale(immagine)  # passaggio a scala di grigi (1 canale)
        immagine = ImageOps.invert(immagine)  # inversione colori immagine (in accordo con il dataset MNIST)
        # plt.imshow(immagine, cmap='gray')
        # plt.show()

        # Definizione variabili per passaggio da PIL a Tensore e viceversa
        pil_tensor = transforms.ToTensor()  # per la conversione da PIL a tensore (normalizzando tra 0 e 1)
        tensor_pil = transforms.ToPILImage()    # per la conversione da tensore a PIL

        # Trasformazione sull'immagine
        image_tensor = pil_tensor(immagine)

        # Inizializzazione variabili
        column = 0  # rappresenta la colonna relativa all'inizio dell'area in cui è contenuta una cifra (estremo sx area)
        count_digit = 0  # contatore delle cifre individuate nell'immagine
        sub_images = []  # tensore con le sotto-immagini relative a ciascuna cifra
        lim_sup = 0
        lim_inf = 0
        end_number = 0

        # Definizione variabile soglia
        threshold = 0.75  # soglia sotto la quale un pixel viene considerato spento

        # Rimozione parte vuota sopra e sotto l'immagine
        for i in range(image_tensor.size(1)):  # scorre le righe

            for j in range(image_tensor.size(2)):  # scorre le colonne

                if (lim_sup == 0) & (image_tensor[0][i][j] > threshold):  # limite superiore trovato
                    lim_sup = i - 35
                    break

                if (lim_sup != 0) & (image_tensor[0][i][j] > threshold):  # limite inferiore non trovato
                    break

                if (lim_sup != 0) & (j == (image_tensor.size(2) - 1)):  # True se è stata trovata una riga senza pixel accesi
                    lim_inf = i + 35
                    end_number = 1
                    break

            if end_number == 1:  # esce dal ciclo se sono stati trovati i due limiti
                break

        # Eliminazione bordi superiori e inferiori immagine
        image_tensor = image_tensor[:, lim_sup:lim_inf, :]

        # Segmentazione di ogni singolo digit
        for j in range(image_tensor.size(2)):  # scorre le colonne

            for i in range(image_tensor.size(1)):  # scorre le righe

                if (column == 0) & (image_tensor[0][i][j] > threshold):  # inizia il digit
                    # salva la colonna
                    column = j - 35  # lascia un po' di spazio di pixels come bordo sx
                    break  # passa alla colonna successiva

                if (column != 0) & (image_tensor[0][i][j] > threshold):  # non è finito il digit
                    break  # passa alla colonna successiva

                if (column != 0) & (image_tensor[0][i][j] < threshold) & (i == (image_tensor.size(1) - 1)):  # è finito il digit
                    count_digit += 1  # incremento il contatore dei digit trovati
                    # creazione area digit trovato
                    sub_images.append(image_tensor[0, :, column:j + 35])  # salvataggio del tensore relativo all'area trovata
                    column = 0  # si azzera una volta definito l'estremo dx dell'area della cifra

        digit = ''  # stringa che sarà composta dalle predizioni su ogni cifra

        # Loop sulle sotto-immagini identificate )per passare da tensore a PIL, per passare alla dimensione (28x28) e tornare a tensore
        for i in range(len(sub_images)):

            sub_images[i] = tensor_pil(sub_images[i])   # tensore -> PIL
            sub_images[i] = sub_images[i].resize((28, 28))  # ridimensionamento a immagine (28x28)
            plt.imshow(sub_images[i], cmap='gray')     # visualizza immagine digit
            plt.show()
            sub_images[i] = pil_tensor(sub_images[i])   # PIL -> tensore

            if folder_name == 'foto':   # controlla se la cartella è quella con le foto reali (foto scattate da cellulare)
                sub_images[i] = blacken_pixel(sub_images[i])  # annerimento pixels sotto una certa soglia definita nel metodo

            #sub_images[i] = tensor_pil(sub_images[i])
            #plt.imshow(sub_images[i], cmap='gray')
            #plt.show()
            #sub_images[i] = pil_tensor(sub_images[i])
            # plt.imshow(sub_images[i], cmap='gray')
            # plt.show()
            # print(sub_images[i].size())
            # sub_images[i] = tensor_pil(sub_images[i])
            # plt.imshow(sub_images[i], cmap='gray')
            # plt.show()
            # sub_images[i] = pil_tensor(sub_images[i])
            #print(sub_images[i].size())

            sub_images[i] = sub_images[i].to(self.device)  # spostamento nel dispositivo corretto
            output_net_no_act, output_net = self.forward(sub_images[i][None, :, :])
            prediction = self.decision(output_net)
            #d = torch.argmax(b, dim=1).item()
            digit += str(prediction.item())

        # print("\n\nIl numero in foto è: ", int(digit))

        return digit

    # FUNZIONE PER LA PREDIZIONE DI CIFRE IN IMMAGINI PRESENTI IN UNA CARTELLA
    def eval_pics(self, folder):
        """Effettua delle predizioni su immagini contenute in una cartella.

        Args:
            cartella: percorso della cartella

        """

        self.net.eval()    # passaggio alla fase di valutazione

        pics_list = next(os.walk(folder))[2]  # lista contenente i nomi delle immagini nella cartella
        print('La cartella contiene le seguenti immagini:', pics_list)

        # Loop sulle immagini
        for i in range(len(pics_list)):

            digit = self.segment_image(folder + '/' + pics_list[i])
            print('Il numero scritto nella foto', pics_list[i], 'è:', digit)


# FUNZIONE DI ANNERIMENTO PIXEL
def blacken_pixel(image):
    """Annerisce i pixel dell'immagine in input che hanno un livello di luminosità sotto la soglia di 0.5."""

    threshold = 0.5    # soglia sotto la quale un pixel viene considerato spento

    # Loop sui pixels dell'immagine (tensore con elementi compresi tra 0 e 1)
    for i in range(image.size(1)):

        for j in range(image.size(2)):

            if image[0][i][j] < threshold:

                image[0][i][j] = 0

    return image


# ENTRY POINT
if __name__ == '__main__':

    # Istruzioni da linea di comando
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'eval', 'eval_pics'],
                        help='Specificare la modalità tra: addestramento(train), valutazione(eval), predizione su immagini(eval_pics)')
    parser.add_argument('--cnn_structure', type=str, default='cnn1', choices=['cnn1', 'cnn2'],
                        help='Specificare il modello da usare (default: cnn1)')
    parser.add_argument('--lr', type=float, default=0.001, help='Specificare il learning rate per addestramento (default: 0.001)')
    parser.add_argument('--epochs', type=int, default=10, help='Specificare il numero di epoche per addestramento (default: 10)')
    parser.add_argument('--batch_size', type=int, default=64, help='Specificare la dimensione dei mini-batches (default: 64)')
    parser.add_argument('--device', choices=['cpu', 'gpu'], default='cpu', help='Specificare il device da usare (default:cpu)')
    parser.add_argument('--folder', type=str, default=None,
                        help='Specificare il nome della cartella in cui sono contenute le immagini (default: None)')
    parser.add_argument('--classifier', type=str, default='classificatore_cnn1.pth',
                        choices=['classificatore_cnn1.pth', 'classificatore_cnn2.pth'],
                        help='Specificare il modello addestrato da utilizzare (default: classificatore_cnn2.pth)')
    args = parser.parse_args()

    # VALUTAZIONI ISTRUZIONI DA LINEA DI COMANDO
    LR = args.lr
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size

    # DOWNLOAD DATASET MNIST NELLA CARTELLA 'dataset' SE NON GIA' PRESENTE
    train_data, val_data, test_data = download_data('dataset')

    # Conversione del dataset in data loaders
    train_dataloader = DataLoader(train_data,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True)    # rimescola i dati

    val_dataloader = DataLoader(val_data,
                                batch_size=BATCH_SIZE,
                                shuffle=False)

    test_dataloader = DataLoader(test_data,
                                 batch_size=BATCH_SIZE,
                                 shuffle=False)

    # Selezione del device corretto
    if (args.device == 'gpu') & (torch.cuda.is_available()):
        device = 'cuda:0'
        print('\nDevice: GPU')
    else:
        device = 'cpu'
        print('\nDevice: CPU')

    # Modalità training...
    if args.mode == 'train':

        print('Training classifier...')

        # Creazione di un nuovo classificatore
        classificatore = Classifier(args.cnn_structure, device)

        # Addestramento del classificatore
        classificatore.train_classifier(LR, EPOCHS)

        # Caricamento del modello con cui sono stati ottenuti i risultati migliori sul validation_set
        print('\nAddestramento completato, caricamento del miglior modello...')
        classificatore.load('classificatore.pth')

        # Valutazione delle prestazioni del modello sui 3 dataset
        # train_acc = classificatore.eval_classifier(train_dataloader)
        val_acc = classificatore.eval_classifier(val_dataloader)
        test_acc = classificatore.eval_classifier(test_dataloader)

        # Stampa risultati ottenuti sui 3 dataset
        # print('Accuracy sul training set: ', round(train_acc.item(), 2), '%')
        print('Accuracy sul validation set: ', val_acc, '%')
        print('Accuracy sul test set: ', test_acc, '%')

    elif args.mode == 'eval':

        print('Valutazione classificatore...')

        if args.classifier == 'classificatore_cnn1.pth':
            classificatore = Classifier('cnn1', device)     # crea istanza di classificatore con la struttura cnn1
        else:
            classificatore = Classifier('cnn2', device)     # crea istanza di classificatore con la struttura cnn2

        # Caricamento classificatore
        classificatore.load(args.classifier)

        # Valutazione del modello sul test set
        test_acc = classificatore.eval_classifier(test_dataloader)

        # Stampa risultati ottenuti
        print('Accuracy sul test set: ', round(test_acc, 2), '%')

    elif args.mode == 'eval_pics':

        folder_name = args.folder   # nome cartella in cui sono contenuti le immagini

        if args.folder is None:
            print('SPECIFICARE IL NOME DELLA CARTELLA IN CUI SONO CONTENUTE LE IMMAGINI! -> --folder=folder_name in the command line')
        else:
            print('Predizione delle immagini nella cartella: ', args.folder)

            if args.classifier == 'classificatore_cnn1.pth':
                classificatore = Classifier('cnn1', device)  # crea istanza di classificatore con la struttura cnn1
            else:
                classificatore = Classifier('cnn2', device)  # crea istanza di classificatore con la struttura cnn2

            # Caricamento classificatore
            classificatore.load(args.classifier)

            # Predizione sulle immagini
            classificatore.eval_pics(args.folder)
