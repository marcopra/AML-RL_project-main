# List of TODOs

## Raw notes for possible improvements in the code for point 4

* Using _terminal_ for terminating the episode @marcopra -> Necessario
* Use the real reward (not R = Q(St,At)) to train the _Actor_ @marcopra -> se adottiamo questo, il nostro approccio non è più model-free, da chieder a gabriele per chiarimenti
* Fare training alterno @marcopra -> inutile, forse utile stoppare il critc network learning ad una certa
* Do the thing above and remove the _Critic_ Network @marcopra -> Non provato ma frose è inutile
* Try different activation functions for _Actor_ and _Critic_ classes
* Try different architectures for _Actor_ and _Critic_, such as ResNet-18
* Stack more than 1 images together to use it as input for _Actor_ and _Critic_ @marcopra -> provato con 4 immagini stacked e mooolto meglio
* Try _Batch Normalization_ in _Actor_ and _Critic_ @marcopra -> provato solo con BN da vedere senza BN
* Try to used pretrained networks
* Try different weight inizialization (For example Xavier (Already implemented))  @marcopra -> provato solo con xavier inizialization da vedere senza
* Parameter tuning (! with plots)
* Try loss Actor = (r_t + self.gamma*q_t_1 - q_t)*lp ... DA CAPIRE MEGLIO
* SALVARE I DATI PER OGNI PROVA FATTI! -> CREARE POI UN BENCHMARK CON OGNI TENTATIVO
* Try to add noise - @marcopra ho dubbi che aggiungere noise senza decay sia così utile
* Try to add noise decay
* Cleaning the code
* Save the model wich performs better on the test environment
* Try scaling of figures 0-1 value instead of 0-255
* Restart dell'optimizer (?)

## Must TODOs

* Delete `Agent.py` file
* PLOTS AND DATAs, PLOTS AND DATAs, PLOTS and DATAs!!!!!!!!!!!
* Save the optimizer
* 

## General Coniderations

* Stacked frames (4) funziona molto meglio, il training di solito si svolge con un plateaux iniziale e poi curva inizia a scendere (per single frames si inizia a scendere da 10000). Con stacked frames questo plateaux è più piccolo o non c'è proprio
* La policy loss continua a scendere sempre di più con il training -> c'è un plateaux finale ? potenzialmente più lo alleno meglio è
* Critic network ogni tanto sfaciola di molto, ora è molto più raro di prima ma la mse loss si trova generalmente al di sotto di 1.0, forse più si va avanti più amenta ma non so perchè. Due possibili motivi: overfitting oppure l agent migliora e vede cose mai viste. Overfitting non credo, vedere anche [qui (overfitting in deepRL)](https://www.quora.com/Is-overfitting-a-problem-in-deep-reinforcement-learning).
* 


