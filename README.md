
# May 13

- model save // restore 해서 inference 해보는거 한번 해보고
- batch normalization 직전 tensorflow level 에서 구현해보고 결과 비교해보는 것 정도로 마무리 
- 지금 mnist alphabet 으로 실험하고 있는데, 완성한 후에 utils.mnist_loader 에서 가져온 digits 로 99.x % 정도는 나오는지 실험하고 마무리하는 걸로 


# May 12 

- mnist character classification problem testing
- using a i) deeper/complext convolutional network, ii) applying batch normalization from scratch is a main objective of this run. 

- test accuracy 97% 정도가 나옴 .... 다른 top-tier network 이랑 비교해봤을때 나쁘지 않은 수준..
- learning rate decay 걸어보니까 오히려 결과 나빠짐... epoch 은 10, 처음 1e-4 로 출발해서 낮
