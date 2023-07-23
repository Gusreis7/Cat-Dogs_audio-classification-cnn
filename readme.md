<h2>English</h2>

Hey, this is a basic implementation of a CNN for audio classification purpuses.
The propose of this repository is to study how CNNs works and improve my skills on this field
The implemention was made in Pytorch, using  OOP for better writen code and usage for my experiments.

For this work, I used a public dataset of audio classification, who contains 2 classes, dogs and cats, the dataset is avaible at https://www.kaggle.com/datasets/mmoreaux/audio-cats-and-dogs . 

There is some data preparation on-the-fly for training,the audio is padded to 5s  and resampled to 16Khz if necessary, after that, there is a feature extraction and data augmentation using time masking with a 0.25 probability, that transforms the raw audio into a spectrogram, using n_ffts = 600, and then the spectogram is feeded to the folowing neural network

        self.conv_layer = nn.Sequential(
            nn.Conv2d(1,128, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128,64, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64,32, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32,16, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)

        )
        
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(16*20*18,256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Linear(32, 4),
            nn.ReLU(),
            nn.Linear(4, 2)
        )


The model is trained for 150 epochs, after that the model is reaching a plato, and the network don't learn anymore, even with more epochs.
Feel free to use this code to implement your own audio classifier  

<h2>Português</h2>

Ei, esta é uma implementação básica de uma CNN para fins de classificação de áudio. A proposta deste repositório é estudar como as CNNs funcionam e melhorar minhas habilidades neste campo. A implementação foi feita em Pytorch, usando POO para melhor escrita de código e nos meus experimentos.

Para este trabalho, utilizei um conjunto de dados público de classificação de áudio, que contém 2 classes, cães e gatos, o conjunto de dados está disponível em https://www.kaggle.com/datasets/mmoreaux/audio-cats-and-dogs .

Há alguma preparação de dados on-the-fly para treinamento, o áudio é preenchido/truncado para 5s e reamostrado para 16Khz se necessário, depois disso, há uma etapa de feature extraction e aplicação de dada augmentation usando time masking com 0.25 de probabilidade de ser aplicado , que transformam o áudio bruto em um espectrograma usando n_ffts = 600, e então o espectograma é alimentado para a seguinte rede neural

    self.conv_layer = nn.Sequential(
            nn.Conv2d(1,128, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128,64, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64,32, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32,16, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)

        )
        
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(16*20*18,256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Linear(32, 4),
            nn.ReLU(),
            nn.Linear(4, 2)
        )


O modelo é treinado por 150 épocas, depois disso o modelo atinge um plato e a rede não aprende mais, mesmo com mais épocas Sinta-se à vontade para usar este código para implementar seu próprio classificador de áudio

