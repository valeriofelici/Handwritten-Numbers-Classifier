"""
*** Neural Networks Exam Project ***
Topic: Recognize Handwritten (Long) Numbers
Author: Valerio Felici (valerio.felici@student.unisi.it)
"""

import copy
import numpy as np
import torch
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, random_split, IterableDataset, TensorDataset
from torch import optim
import os
from PIL import Image, ImageOps


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

    def __init__(self, device="cpu"):
        """Crea un classificatore non addestrato.

        Args:
            device: stringa che indica il device da usare ("cpu", "cuda:0", "cuda:1", ...)
        """

        # Attributi della classe

        self.num_outputs = 10   # 10 classi mutuamente esclusive
        self.device = torch.device(device)  # device in cui spostare i dati
        self.preprocess_train = None
        self.preprocess_eval = [None, None, None, None]

        # Creazione della rete

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

        # Data augmentation sui dati per il training ->  rotazione random (25,-25) gradi e trasformazione di scala tra (0.95,1.05)

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
                                    transforms.RandomAffine(degrees=0, scale=(0.95, 0.95)),
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

    # FUNZIONE PER IL CALCOLO DELL'OUTPUT DELLA RETE CON E SENZA FUNZIONE DI ATTIVAZIONE????

    def forward(self, x):
        """Calcola l' output della rete"""

        output_net_no_act = self.net(x)  # Output della rete senza attivazione??? SERVE????
        output_net = torch.softmax(output_net_no_act, dim=1)    # Output dopo la funzione di attivazione (softmax in questo caso)

        return output_net_no_act, output_net

    # def decision(self): # POTREBBE NON SERVIRE!!!!!!!!!!

    # FUNZIONE DI VALUTAZIONE DEL MODELLO

    def eval_classifier(self, validation_set):
        """Valuta le prestazioni del modello su un validation set di (40k esempi) composto da 4 subset a loro volta
        composti da 10k esempi presi dal training set su cui vengono fatte 4 trasformazioni fisse (una per ciascun subset):

        subset 0: Nessuna trasformazione oltre il passaggio da PIL a tensore (con normalizzazione)
        subset 1: Rotazione di 15 gradi e passaggio da PIL a tensore (con normalizzazione)
        subset 2: Rotazione di -15 gradi e passaggio da PIL a tensore (con normalizzazione)
        subset 3: Trasformazione di scala di 0.95 e passaggio da PIL a tensore (con normalizzazione)

        """

        # Inizializzazione elementi
        correct = 0  # contatore predizioni corrette
        tot_esempi = 0  # contatore esempi totali

        # Ciclo sui 4 subset che compongono il validation set

        for subset in range(4):

            validation_set.dataset.transform = self.preprocess_eval[subset]

            # Ciclo sui batch di dati (batch mode)
            for i, (images, labels) in enumerate(validation_set):

                # images = images.cuda()  # ???????????????????
                output_net_no_act, output_net = self.forward(images)    # Calcolo dell' output della rete
                predictions = torch.argmax(output_net, dim=1)   # La predizione sarà relativa all'output più alto della rete
                # predictions = predictions.data.cpu()    # ????????????????????????
                correct += torch.sum(predictions == labels)  # Incrementa il totale delle predizioni corrette
                tot_esempi += output_net_no_act.size(0)  # Incrementa il totale degli esempi presentati

        correct = correct*100./tot_esempi    # Calcolo percentuale

        # Ripristino dei dati alle trasformazioni casuali usate per il training
        validation_set.dataset.transform = self.preprocess_train

        return correct

    # FUNZIONE DI ADDESTRAMENTO DEL CLASSIFICATORE

    def train_classifier(self, learning_rate, epoche):
        """"Addestramento del classificatore con i dati per training e validation."""

        # Inizializzazione elementi
        best_epoch = -1     # Epoca in cui è stata ottenuta la precisione maggiore
        best_accuracy = -1  # Precisione maggiore ottenuta
        accuracy = -1   # Memorizza il valore della percentuale di precisione ottenuta
        accuracies = []  # Memorizza le percentuali di precisione ottenute nell' addestramento (per il grafico finale)

        self.net.to(self.device)  # ???????????????????

        # Controllo che la rete sia in 'modalità addestramento'
        self.net.train()

        # Creazione ottimizzatore (Adam)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)

        # Scelta della loss function (Cross Entropy Loss)
        loss_function = nn.CrossEntropyLoss()
        losses = []     # Memorizza i valori assunti dalla loss function

        # Ciclo sulle epoche
        for e in range(epoche):

            # Ciclo sui batch di dati (batch mode)
            for i, (images, labels) in enumerate(train_dataloader):

                # Spostamento dei dati nel dispositivo corretto
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Calcolo delle predizioni della rete

                predictions, b = self.forward(images)

                # Calcolo della loss function
                loss = loss_function(predictions, labels)   # Calcolo della loss function con classi predette e classi corrette
                losses.append(loss.item())
                # Calcolo gradienti e aggiornamento dei pesi della rete
                optimizer.zero_grad()   # Azzeramento aree di memoria in cui erano stati inseriti i gradienti calcolati in precedenza
                loss.backward()  # Backpropagation
                optimizer.step()   # Aggiornamento pesi (valori contenuti nei filtri di convoluzione)

            # Passaggio alla fase di valutazione della rete
            self.net.eval()

            accuracy = float(self.eval_classifier(val_dataloader))
            accuracies.append(accuracy)

            # Ritorno alla modalità di addestramento
            self.net.train()

            # Stampa statistiche ottenute su ciascuna epoca
            print("Epoca: ", e+1, "Precisione: ", round(accuracy, 4), "%")

            # Controllo che sia stato raggiunto il valore max della precisione tra quelli ottenuti fino ad adesso
            if accuracy > best_accuracy:
                self.save("classificatore.pth")  # Salvataggio del modello (è il miglior ottenuto fin qui)
                best_accuracy = accuracy  # Aggiornamento del valore più alto di precisione ottenuto
                print("Salvataggio del miglior modello: ", round(accuracy, 4), "% \n")

        # Stampa del grafico relativo alla precisione della rete

        fig, axs = plt.subplots(2)
        fig.suptitle("Grafici")
        axs[0].plot(accuracies)
        axs[1].plot(losses)
        #plt.plot(accuracies)
        #plt.show()
        #plt.plot(losses)
        plt.show()

# ENTRY POINT


if __name__ == "__main__":

    # Download dataset MNIST

    train_data, val_data, test_data = download_data("dataset")

    # Conversione del dataset in data loaders

    dim_batch = 1  # dimensione massima dei batch !!! VERRA' DATA DA LINEA DI COMANDO!!!

    train_dataloader = DataLoader(train_data,
                                  batch_size=dim_batch,
                                  shuffle=True)    # rimescola i dati

    val_dataloader = DataLoader(val_data,
                                batch_size=dim_batch,
                                shuffle=True)

    c = Classifier()    # istanza di classificatore
    #c.train_classifier(0.001, 5)   # lr=0.001 epoche= 10

    c.load('classificatore.pth')
    # Apertura immagine
    image = Image.open("prova.jpeg")

    # Ridimensionamento immagine in 28x28
    new_image = image.resize((28, 28))

    # Conversione da RGB a Grayscale ad un solo canale
    new_image1 = ImageOps.grayscale(new_image)
    new_image1 = ImageOps.invert(new_image1)

    new_image1.save(fp="nuovo.jpg")
    conv = transforms.PILToTensor()
    new_image2 = conv(new_image1)
    new_image2 = new_image2 / 255
    print(new_image2)
    plt.imshow(new_image2[0][:][:], cmap='gray')
    #plt.show()
    a, b = c.forward(new_image2[None, :, :])
    print("\nIl numero disegnato è: ", torch.argmax(b, dim=1).item())

