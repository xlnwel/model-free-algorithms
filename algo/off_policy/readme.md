
## Experimental Results

### SAC
Using Q-error instead of V-error as priority
Learning rate decay helps stabalize the learning at convergence
Even without bias correction for prioritized sampling, SAC behaves more robust and yield better performance than td3

### TD3
Applying bias correction for prioritized sampling to actor loss improves the performance.