# List of TODOs

## Raw notes for possible improvements in the code for point 4

* Using _terminal_ for terminating the episode
* Use the real reward (not R = Q(St,At)) to train the _Actor_
* Fare training alterno
* Do the thing above and remove the _Critic_ Network
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

* Delete `Agent.py` file
* PLOTS AND DATAs, PLOTS AND DATAs, PLOTS and DATAs!!!!!!!!!!!
