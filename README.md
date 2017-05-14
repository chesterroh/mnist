
# May 13

- code refactored to have more succinct structure / gainged 97% with mnist A~J data, not digit data

- (AI) Batch normalization from-scratch implementation
  ==> reference point is at: http://r2rt.com/implementing-batch-normalization-in-tensorflow.html
- (AI) model save/restore 
- (AI) BN inference after saved model restore + refactoring to inference/train/test module based on TF tutorial

# May 12 

- mnist character classification problem testing
- using a i) deeper/complext convolutional network, ii) applying batch normalization from scratch is a main objective of this run. 

- test accuracy 97% 정도가 나옴 .... 다른 top-tier network 이랑 비교해봤을때 나쁘지 않은 수준..
- learning rate decay 걸어보니까 오히려 결과 나빠짐... epoch 은 10, 처음 1e-4 로 출발해서 낮
