
## Agreement metrics, channel set = all24

| comparison | Spearman | Rank@5 | Feature@5 | Sign | Signed-rank@5 | Signed-rank@5 | Rank@K |
|---|---|---|---|---|---|---|---|
| within: cnn / ig / occlusion | 0.904 | 0.262 | 0.730 | 0.730 | 0.262 | 0.262 | 0.200 |
| within: cnn / ig / lrp | 0.889 | 0.267 | 0.734 | 0.902 | 0.267 | 0.267 | 0.422 |
| within: cnn / occlusion / lrp | 0.819 | 0.134 | 0.626 | 0.657 | 0.133 | 0.133 | 0.135 |
| within: lstm / ig / occlusion | 0.659 | 0.249 | 0.669 | 0.634 | 0.246 | 0.246 | 0.194 |
| within: lstm / ig / lrp | 0.352 | 0.134 | 0.477 | 0.703 | 0.121 | 0.121 | 0.356 |
| within: lstm / occlusion / lrp | 0.328 | 0.146 | 0.432 | 0.534 | 0.132 | 0.132 | 0.139 |
| within: hybrid / ig / occlusion | 0.637 | 0.276 | 0.660 | 0.654 | 0.268 | 0.268 | 0.195 |
| within: hybrid / ig / lrp | -0.335 | 0.128 | 0.451 | 0.476 | 0.014 | 0.014 | 0.360 |
| within: hybrid / occlusion / lrp | -0.063 | 0.146 | 0.495 | 0.442 | 0.059 | 0.059 | 0.149 |
| cross: ig / cnn / lstm | 0.446 | 0.118 | 0.410 | 0.711 | 0.107 | 0.107 | 0.348 |
| cross: ig / cnn / hybrid | 0.503 | 0.088 | 0.336 | 0.770 | 0.079 | 0.079 | 0.344 |
| cross: ig / lstm / hybrid | 0.554 | 0.148 | 0.533 | 0.767 | 0.139 | 0.139 | 0.361 |
| cross: occlusion / cnn / lstm | 0.422 | 0.116 | 0.400 | 0.591 | 0.104 | 0.104 | 0.259 |
| cross: occlusion / cnn / hybrid | 0.426 | 0.091 | 0.397 | 0.590 | 0.085 | 0.085 | 0.254 |
| cross: occlusion / lstm / hybrid | 0.581 | 0.127 | 0.550 | 0.612 | 0.122 | 0.122 | 0.269 |
| cross: lrp / cnn / lstm | 0.219 | 0.072 | 0.317 | 0.666 | 0.054 | 0.054 | 0.334 |
| cross: lrp / cnn / hybrid | -0.183 | 0.059 | 0.238 | 0.551 | 0.030 | 0.030 | 0.331 |
| cross: lrp / lstm / hybrid | -0.137 | 0.096 | 0.397 | 0.562 | 0.030 | 0.030 | 0.354 |

## Agreement metrics, channel set = active17

| comparison | Spearman | Rank@5 | Feature@5 | Sign | Signed-rank@5 | Signed-rank@5 | Rank@K |
|---|---|---|---|---|---|---|---|
| within: cnn / ig / occlusion | 0.928 | 0.262 | 0.730 | 0.904 | 0.262 | 0.262 | 0.181 |
| within: cnn / ig / lrp | 0.934 | 0.267 | 0.734 | 0.875 | 0.267 | 0.267 | 0.186 |
| within: cnn / occlusion / lrp | 0.917 | 0.134 | 0.626 | 0.802 | 0.133 | 0.133 | 0.089 |
| within: lstm / ig / occlusion | 0.697 | 0.249 | 0.669 | 0.771 | 0.246 | 0.246 | 0.166 |
| within: lstm / ig / lrp | 0.375 | 0.134 | 0.477 | 0.644 | 0.121 | 0.121 | 0.091 |
| within: lstm / occlusion / lrp | 0.347 | 0.146 | 0.432 | 0.629 | 0.132 | 0.132 | 0.090 |
| within: hybrid / ig / occlusion | 0.693 | 0.276 | 0.660 | 0.769 | 0.268 | 0.268 | 0.174 |
| within: hybrid / ig / lrp | -0.301 | 0.128 | 0.451 | 0.347 | 0.014 | 0.014 | 0.096 |
| within: hybrid / occlusion / lrp | -0.043 | 0.146 | 0.495 | 0.464 | 0.059 | 0.059 | 0.110 |
| cross: ig / cnn / lstm | 0.513 | 0.118 | 0.410 | 0.651 | 0.107 | 0.107 | 0.080 |
| cross: ig / cnn / hybrid | 0.518 | 0.088 | 0.336 | 0.710 | 0.079 | 0.079 | 0.075 |
| cross: ig / lstm / hybrid | 0.568 | 0.148 | 0.533 | 0.724 | 0.139 | 0.139 | 0.098 |
| cross: occlusion / cnn / lstm | 0.417 | 0.116 | 0.401 | 0.680 | 0.104 | 0.104 | 0.075 |
| cross: occlusion / cnn / hybrid | 0.418 | 0.091 | 0.397 | 0.688 | 0.085 | 0.085 | 0.080 |
| cross: occlusion / lstm / hybrid | 0.585 | 0.127 | 0.551 | 0.735 | 0.122 | 0.122 | 0.096 |
| cross: lrp / cnn / lstm | 0.260 | 0.072 | 0.317 | 0.564 | 0.054 | 0.054 | 0.061 |
| cross: lrp / cnn / hybrid | -0.249 | 0.059 | 0.238 | 0.465 | 0.030 | 0.030 | 0.058 |
| cross: lrp / lstm / hybrid | -0.159 | 0.096 | 0.397 | 0.459 | 0.030 | 0.030 | 0.087 |

## Sensitivity / Infidelity (mean, median, std)

- cnn ig: sens 0.115/0.116/0.022; infid 2.580/2.472/1.767
- cnn occlusion: sens 0.142/0.144/0.024; infid 2.335/2.109/1.761
- cnn lrp: sens 0.175/0.174/0.054; infid 2.625/2.454/1.999
- lstm ig: sens 0.184/0.121/0.160; infid 0.881/0.377/1.151
- lstm occlusion: sens 0.319/0.217/0.348; infid 0.856/0.105/1.359
- lstm lrp: sens 0.178/0.136/0.191; infid 1.502/0.586/2.068
- hybrid ig: sens 0.134/0.120/0.091; infid 0.899/0.488/1.357
- hybrid occlusion: sens 0.230/0.149/0.233; infid 0.838/0.226/1.642
- hybrid lrp: sens 0.242/0.219/0.128; infid 0.925/0.552/1.348

## Activation maximization, Spearman of per-sensor means: unconstrained 17 ch / unconstrained 24 ch / box +-1, +-2, +-3 SD (17 ch)
- cnn vs lstm: 0.265 / 0.352 / 0.473 / 0.358 / 0.350
- cnn vs hybrid: -0.206 / -0.131 / -0.142 / -0.159 / -0.142
- lstm vs hybrid: -0.593 / -0.388 / -0.311 / -0.392 / -0.444
- output robustness cnn: mean 0.979, median 0.800, std 0.788
- output robustness lstm: mean 0.424, median 0.160, std 0.664
- output robustness hybrid: mean 0.500, median 0.193, std 0.728
