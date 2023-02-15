"""
*** Neural Networks Exam Project ***
Topic: Recognize Handwritten (Long) Numbers
Author: Valerio Felici (valerio.felici@student.unisi.it)
"""

import copy
import numpy as np
import torch
import torchvision
import argparse
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, random_split, IterableDataset, TensorDataset
from torch import optim
import os
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
                         transform=training_data_transform)

    test_set = MNIST(root=folder,
                     train=False,  # dati per il test
                     download=True,
                     transform=ToTensor(),  # i dati di test non sono soggetti a trasformazioni in questa fase
                     )

    # Creazione del validation set (bilanciato rispetto alle classi) prendendo 10k esempi dal training set

    training_set, validation_set = random_split(training_set, [50000, 10000])

    return training_set, validation_set, test_set


class Classifier:
    """Classificatore per la predizione su immagini di numeri scritti a mano."""

    def __init__(self, cnn_structure='cnn1', data_augmentation=1, device="cpu"):
        """Crea un classificatore non addestrato.

        Args:
            device: stringa che indica il device da usare ("cpu", "cuda:0", "cuda:1", ...)
        """

        # Attributi della classe

        self.num_outputs = 10   # 10 classi mutuamente esclusive
        self.device = torch.device(device)  # device in cui saranno spostati i dati
        self.preprocess_train = None    # operazioni di pre-processamento dei dati (per il training)
        self.preprocess_eval = [None, None, None, None]     # operazioni di pre-processamento dei dati (per validation e testing)

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

        if data_augmentation:

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

        else:   # nessuna trasformazione sui dati oltre alla normalizzazione

            self.preprocess_train = [transforms.ToTensor()]
            self.preprocess_eval = [transforms.ToTensor()]

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
        """Calcola l'output della rete"""

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

    # FUNZIONE DI VALUTAZIONE DEL MODELLO

    def eval_classifier(self, data_set):
        """Valuta le prestazioni del modello su un validation set di (40k esempi) composto da 4 subset a loro volta
        composti da 10k esempi presi dal training set su cui vengono fatte 4 trasformazioni fisse (una per ciascun subset):

        subset 0: Nessuna trasformazione oltre il passaggio da PIL a tensore (con normalizzazione)
        subset 1: Rotazione di 15 gradi e passaggio da PIL a tensore (con normalizzazione)
        subset 2: Rotazione di -15 gradi e passaggio da PIL a tensore (con normalizzazione)
        subset 3: Trasformazione di scala di 0.95 e passaggio da PIL a tensore (con normalizzazione)

        """

        # Inizializzazione elementi
        tot_predictions = []    # contenitore di tutte le predizioni fatte dalla rete
        tot_labels = []     # contenitore di tutte le labels

        # Ciclo sui subset che comporranno il validation set
        for subset in range(len(self.preprocess_eval)):

            data_set.dataset.transform = self.preprocess_eval[subset]  # applica la trasformazione sui dati

            # Ciclo sui batch di dati (batch-mode)
            for i, (images, labels) in enumerate(data_set):

                # Spostamento dei dati nel dispositivo corretto
                images = images.to(self.device)
                # labels = labels.to(self.device)

                output_net_no_act, output_net = self.forward(images)    # calcolo output della rete sul batch di dati
                predictions = self.decision(output_net)   # predizioni della rete sul batch
                tot_predictions.append(predictions.cpu())   # aggiunta predizioni sul batch attuale alle altre
                tot_labels.append(labels)   # aggiunta labels sul batch attuale alle altre

                # predictions = predictions.to(device='cuda:0')
                # correct += torch.sum(predictions == labels)  # incremento del totale delle predizioni corrette
                # tot_examples += output_net_no_act.size(0)  # incremento del totale degli esempi presentati

        # Calcolo precisione
        dataset_accuracy = self.compute_accuracy(torch.cat(tot_predictions, 0), torch.cat(tot_labels, 0))

        # accuracy = correct*100.0/tot_examples calcolo percentuale predizioni corrette sul 40k esempi totali

        # Ripristino dei dati alle trasformazioni casuali usate per il training
        data_set.dataset.transform = self.preprocess_train

        return dataset_accuracy

    # FUNZIONE DI ADDESTRAMENTO DEL CLASSIFICATORE

    def train_classifier(self, learning_rate, epochs):
        """"Addestramento del classificatore con i dati per training e validation."""

        # Inizializzazione variabili utili
        best_epoch = -1     # epoca in cui è stata ottenuta la precisione maggiore
        best_accuracy = -1  # precisione maggiore ottenuta
        accuracy = -1   # memorizza il valore della percentuale di precisione ottenuta
        accuracies = []  # memorizza le percentuali di precisione ottenute nell'addestramento

        #self.net.to(self.device)  # ???????????????????

        # assicura che la rete sia in 'modalità addestramento' (PyTorch)
        self.net.train()

        # Creazione ottimizzatore (Adam)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)

        # Loss function  (Cross Entropy Loss)
        criterion = nn.CrossEntropyLoss()
        batch_loss = []     # loss relativo al batch
        epoch_loss = []     # loss relativo all'epoca

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
                #batch_loss.append(loss.item())

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

                    #print('mini batch:\t LOSS:', loss.item(), 'train accuracy: ', batch_train_acc)

            val_accuracy = self.eval_classifier(val_dataloader)  # precisione sul validation set

            epoch_train_loss /= epoch_train_examples
            epoch_train_acc /= epoch_train_examples

            # Stampa statistiche ottenute sull'epoca
            print("Epoca: ", e + 1, "\nPrecisione on Train: ", epoch_train_acc, "%")
            print('Loss on Train: ', epoch_train_loss)
            print('Precisione sul Validation: ', val_accuracy, '%')

            # Controllo che sia stato raggiunto il valore max della precisione tra quelli ottenuti fino ad adesso
            if val_accuracy > best_accuracy:
                self.save("classificatore.pth")  # salvataggio del modello (è il miglior ottenuto fin qui)
                best_accuracy = val_accuracy  # aggiornamento del valore più alto di precisione ottenuto
                print("Salvataggio del miglior modello: ", val_accuracy, "% \n")

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

    def compute_accuracy(self, predictions, labels):
        """Calcola la precisione della rete rispetto alle classi corrette

        Args:
            predictions: Tensore 1D con le predizioni della rete
            labels: Tensore 1D con le predizioni corrette

        Returns:
            accuracy: percentuale di predizioni corrette
        """
        correct = torch.sum(predictions==labels)   # conteggio predizioni corrette
        accuracy = float(correct*100.0/len(labels))  # percentuale predizioni corrette
        accuracy = round(accuracy, 4)

        return accuracy


    # FUNZIONE DI SEGMENTAZIONE IMMAGINI
    def segment_image(self, image):
        """Segmenta l'immagine in tante immagini quante sono le cifre scritte a mano. Ogni immagine rappresenta l'area in cui è contenuta
        ciascuna cifra.

        Args:
            image: nome dell'immagine da segmentare

        Returns:
            sub_images: tensore contenente le sotto-immagini relative a ciascuna cifra
        """

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

        immagine_tensore = pil_tensor(immagine)
        #immagine_tensore = immagine_tensore / 255  # Normalizzazione tra 0 e 1

        # Inizializzazione variabili
        colonna = 0  # rappresenta la colonna relativa all'inizio dell'area in cui è contenuta una cifra (estremo sx area)
        count_cifre = 0  # contatore delle cifre individuate nell'immagine
        sub_images = []  # tensore con le sotto-immagini relative a ciascuna cifra
        lim_sup = 0
        lim_inf = 0
        end_number = 0

        # Definizione variabile soglia
        soglia = 0.75  # soglia sotto la quale un pixel viene considerato spento

        # Rimozione bianco sopra e sotto
        for i in range(immagine_tensore.size(1)):  # scorre le righe

            for j in range(immagine_tensore.size(2)):  # scorre le colonne

                if (lim_sup == 0) & (immagine_tensore[0][i][j] > soglia):  # limite superiore trovato

                    lim_sup = i - 35
                    break

                if (lim_sup != 0) & (immagine_tensore[0][i][j] > soglia):  # limite inferiore trovato

                    break

                if (lim_sup != 0) & (j == (immagine_tensore.size(2) - 1)):
                    lim_inf = i + 35
                    end_number = 1
                    break

            if end_number == 1:
                break

        # Eliminazione bordi superiori e inferiori immagine
        immagine_tensore = immagine_tensore[:, lim_sup:lim_inf, :]

        # Loop sui pixels dell'immagine, scorrendo per colonne
        for j in range(immagine_tensore.size(2)):  # scorre le colonne

            for i in range(immagine_tensore.size(1)):  # scorre le righe

                # if immagine[0][i][j] < 0.6:     # azzera i pixel poco luminosi (rumore?)
                #     immagine[0][i][j] = 0

                if (colonna == 0) & (immagine_tensore[0][i][j] > soglia):  # inizia il digit
                    # salva la colonna
                    colonna = j - 35  # lascio un po' di spazio di pixels come bordo sx
                    break  # passa alla colonna successiva

                if (colonna != 0) & (immagine_tensore[0][i][j] > soglia):  # non è finito il digit
                    break  # passa alla colonna successiva

                if (colonna != 0) & (immagine_tensore[0][i][j] < soglia) & (i == (immagine_tensore.size(1) - 1)):  # è finito il digit
                    count_cifre += 1  # incremento il contatore dei digit trovati
                    # creazione sezione digit trovato
                    sub_images.append(immagine_tensore[0, :, colonna:j + 35])  # salvataggio del tensore relativo all'area trovata
                    colonna = 0  # si azzera una volta definito l'estremo dx dell'area della cifra

        digit = ''  # stringa che sarà composta dalle predizioni su ogni cifra

        # Loop sulle sotto-immagini per passare da tensore a PIL, per passare alla dimensione (28x28) e tornare a tensore
        for i in range(len(sub_images)):
            sub_images[i] = tensor_pil(sub_images[i])
            sub_images[i] = sub_images[i].resize((28, 28))
            #plt.imshow(sub_images[i], cmap='gray')
            #plt.show()
            sub_images[i] = pil_tensor(sub_images[i])
            # sub_images[i] = sub_images[i] / 255  # normalizzazione tra 0 e 1
            #sub_images[i] = blacken_pixel(sub_images[i])  # annerimento pixels sotto la soglia 0.6
            sub_images[i] = tensor_pil(sub_images[i])
            plt.imshow(sub_images[i], cmap='gray')
            plt.show()
            sub_images[i] = pil_tensor(sub_images[i])
            # plt.imshow(sub_images[i], cmap='gray')
            # plt.show()
            # print(sub_images[i].size())
            # sub_images[i] = tensor_pil(sub_images[i])
            # plt.imshow(sub_images[i], cmap='gray')
            # plt.show()
            # sub_images[i] = pil_tensor(sub_images[i])
            print(sub_images[i].size())

            a, b = self.forward(sub_images[i][None, :, :])
            d = torch.argmax(b, dim=1).item()
            digit += str(d)

        # print("\n\nIl numero in foto è: ", int(digit))

        return digit

    # FUNZIONE PER LA PREDIZIONE DI CIFRE IN IMMAGINI PRESENTI IN UNA CARTELLA
    def eval_pics(self, cartella):
        """Effettua delle predizioni su immagini contenute in una cartella.

        Args:
            cartella: percorso della cartella

        """

        pics_list = next(os.walk(cartella))[2]  # lista contenente i nomi delle immagini nella cartella
        print('La cartella contiene le seguenti immagini:', pics_list)

        # Loop sulle immagini
        for i in range(len(pics_list)):
            digit = self.segment_image(cartella + '/' + pics_list[i])
            print('Il numero scritto nella foto', pics_list[i], 'è:', digit)


# FUNZIONE DI ANNERIMENTO PIXELS ADDESSO SBIANCA!!!!
def blacken_pixel(image):
    """Annerisce i pixel dell'immagine in input che hanno un livello di luminosità sotto la soglia di 0.6."""

    soglia = 0.5    # soglia sotto la quale un pixel viene considerato spento

    # Loop sui pixels dell'immagine (tensore con elementi compresi tra 0 e 1)
    for i in range(image.size(1)):

        for j in range(image.size(2)):

            if image[0][i][j] < soglia:

                image[0][i][j] = 0

    return image

 
# ENTRY POINT
if __name__ == "__main__":

    # Istruzioni da linea di comando
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=['train', 'eval', 'eval_pics'],
                        help="Specificare la modalità tra: addestramento(train), valutazione(eval), predizione su immagini(eval_pics)")
    parser.add_argument('--cnn_structure', type=str, default='cnn1', choices=['cnn1', 'cnn2'],
                        help='Specificare il modello da usare (default: cnn1)')
    parser.add_argument('--data_augmentation', type=int, default=1, choices=[1, 0],
                        help='Specificare se applicare trasformazioni sui dati')
    parser.add_argument("--lr", type=float, default=0.001, help="Specificare il learning rate per l'addestramento (default: 0.001)")
    parser.add_argument("--epochs", type=int, default=10, help="Specificare il numero di epoche per l'addestramento (default: 10)")
    parser.add_argument("--batch_size", type=int, default=64, help='Specificare la dimensione dei mini-batches (default: 64)')
    parser.add_argument("--device", choices=['cpu', 'cuda'], default='cpu', help='Specificare il device da usare (default:cpu)')
    parser.add_argument('--folder', type=str, default=None,
                        help='Specificare il nome della cartella in cui sono contenute le immagini (default: None)')
    args = parser.parse_args()

    # VALUTAZIONI ISTRUZIONI DA LINEA DI COMANDO
    LR = args.lr
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size

    # DOWNLOAD DATASET MNIST NELLA CARTELLA 'dataset' SE NON GIA' PRESENTE
    train_data, val_data, test_data = download_data("dataset")

    # Conversione del dataset in data loaders
    train_dataloader = DataLoader(train_data,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True)    # rimescola i dati

    val_dataloader = DataLoader(val_data,
                                batch_size=BATCH_SIZE,
                                shuffle=True)

    test_dataloader = DataLoader(test_data,
                                 batch_size=10000,
                                 shuffle=False)

    # Selezione del device corretto
    if (args.device == 'cuda') & (torch.cuda.is_available()):
        device = 'cuda:0'
        print('\nDevice: GPU')
    else:
        device = 'cpu'
        print('\nDevice: CPU')

    # Modalità training...
    if args.mode == 'train':

        print('Training classifier...')

        # Creazione di un nuovo classificatore
        classificatore = Classifier(args.cnn_structure, args.data_augmentation, args.device)

        # Addestramento del classificatore
        classificatore.train_classifier(LR, EPOCHS)

        # Caricamento del modello con cui sono stati ottenuti i risultati migliori sul validation_set
        print('Addestramento completato, caricamento del miglior modello...')
        classificatore.load('classificatore.pth')

        # Valutazione delle prestazioni del modello sui 3 dataset
        # train_acc = classificatore.eval_classifier(train_dataloader)
        val_acc = classificatore.eval_classifier(val_dataloader)
        test_acc = classificatore.eval_classifier(test_dataloader)

        # Stampa risultati ottenuti sui 3 dataset
        # print('Accuracy sul training set: ', round(train_acc.item(), 2), '%')
        print('Accuracy sul validation set: ', round(val_acc, 2), '%')
        print('Accuracy sul test set: ', round(test_acc, 2), '%')

    elif args.mode == 'eval':

        print('Valutazione classificatore...')

        # Creazione nuovo classificatore
        classificatore = Classifier(args.cnn_structure, args.data_augmentation, args.device)

        # Caricamento classificatore
        classificatore.load('classificatore.pth')

        # Valutazione del modello sul test set
        test_acc = classificatore.eval_classifier(test_dataloader)

        # Stampa risultati ottenuti
        print('Accuracy sul test set: ', round(test_acc.item(), 2), '%')

    elif args.mode == 'eval_pics':

        if args.folder is None:
            print('SPECIFICARE IL NOME DELLA CARTELLA IN CUI SONO CONTENUTE LE IMMAGINI! -> --folder=folder_name in the command line')

        else:
            print('Predizione delle immagini nella cartella: ', args.folder)

            # Creazione nuovo classificatore
            classificatore = Classifier(args.cnn_structure, args.data_augmentation, args.device)

            # Caricamento classificatore
            classificatore.load('classificatore.pth')

            # Predizione sulle immagini
            classificatore.eval_pics(args.folder)


    # c = Classifier()    # istanza di classificatore
    # #c.train_classifier(0.001, 15)   # lr=0.001 epoche= 10
    #
    # c.load('classificatore.pth')
    # cor = c.eval_classifier(test_dataloader)
    # print(cor)

    # Apertura immagine
    # image = Image.open("difficile.jpeg")
    #
    # # Ridimensionamento immagine in 28x28
    # new_image = image.resize((28, 28))
    #
    # # Conversione da RGB a Grayscale a un solo canale
    # new_image1 = ImageOps.grayscale(new_image)
    # new_image1 = ImageOps.invert(new_image1)
    #
    # new_image1.save(fp="nuovo.jpg")
    # conv = transforms.PILToTensor()
    # new_image2 = conv(new_image1)
    # new_image2 = new_image2 / 255
    # print(new_image2)
    # plt.imshow(new_image1, cmap='gray')
    # plt.show()
    #
    # for i in range(28):
    #     for j in range(28):
    #         if (new_image2[0][i][j] < 0.6) | (i < 2 | i > 25) | (j < 2 | j > 25):
    #             new_image2[0][i][j] = 0
    #
    # conv2 = transforms.ToPILImage()
    # new_image3 = conv2(new_image2)
    # plt.imshow(new_image3, cmap='gray')
    # plt.show()
    # a, b = c.forward(new_image2[None, :, :])
    # print("\nIl numero disegnato è: ", torch.argmax(b, dim=1).item())

    # risultati = segment_image("long6312.jpeg")
    # digit = ""
    #
    # conv = transforms.ToPILImage()
    #
    # for i in range(len(risultati)):
    #     a, b = c.forward(risultati[i][None, :, :])
    #     d = torch.argmax(b, dim=1).item()
    #     digit += str(d)
    #     risultati[i] = conv(risultati[i])
    #     plt.imshow(risultati[i], cmap='gray')
    #     plt.show()
    #
    # print("\n\nIl numero in foto è: ", int(digit))


