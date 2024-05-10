### Hyperparameters
We determined the optimal hyperparameters through a systematic grid search to ensure the best reconstruction performance. The training is performed with a batch size of 128 for 100 epochs. We utilize a learning rate scheduler that reduces the learning rate by 50% if there is no improvement in validation performance after 10 epochs.

### Hyperparameter Table

Below is a table detailing the specific hyperparameters tailored to each dataset, which includes learning rate (lr), weight decay, dropout, gradient clipping (clip grad), and the number of RBF centers:

| Dataset | Model                | lr    | weight decay | dropout | clip grad | RBF centers |
|---------|----------------------|-------|--------------|---------|-----------|-------------|
| SMD     | RESTAD (R)     | 0.01  | 0.001        | 0.0     | 1.5       | 256         |
| SMD     | RESTAD (K)     | 0.01  | 0.00001      | 0.0     | 3         | 32          |
| MSL     | RESTAD (R)     | 0.01  | 0.00001      | 0.0     | 3         | 128         |
| MSL     | RESTAD (K)     | 0.1   | 0.1          | 0.3     | 3         | 8           |
| PSM     | RESTAD (R)     | 0.01  | 0.001        | 0.5     | 3         | 32          |
| PSM     | RESTAD (K)     | 0.01  | 0.001        | 0.1     | 1.5       | 16          |
