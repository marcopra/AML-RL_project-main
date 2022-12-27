# List of TODOs

## Raw notes for possible improvements to the code for point 4

* Try different activation functions for _Actor_ and _Critic_ classes
* Try different architectures for _Actor_ and _Critic_, such as ResNet-18
* Stack more than 1 images together to use it as input for _Actor_ and _Critic_
* Try _Batch Normalization_ in _Actor_ and _Critic_
* Try to used pretrained networks
* Try different weight inizialization (For example Xavier (Already implemented))
* Parameter tuning (! with plots)
* Try loss Actor = (r_t + self.gamma*q_t_1 - q_t)*lp ... DA CAPIRE MEGLIO
* SALVARE I DATI PER OGNI PROVA FATTI! -> CREARE POI UN BENCHMARK CON OGNI TENTATIVO

## Must TODOs

* Cleaning the code and comment the code
* Create two folders: PPO and Pixels
* Delete `Agent.py` file
* Github:
  * Create 1 main branch + 3 branches(Marco, Fabiola, Samu) for experiments
* Create a Plot Folder
* PLOTS AND DATAs, PLOTS AND DATAs, PLOTS and DATAs!!!!!!!!!!!
