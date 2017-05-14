# May 14
- (AI) change the dataset to normal mnist digit 0~9 dataset.. this would be the final trial with MNIST.. 

# May 13

- code refactored to have more succinct structure / gainged 97% with mnist A~J data, not digit data

- (AI:DONE) Batch normalization from-scratch implementation
  ==> reference point is at: http://r2rt.com/implementing-batch-normalization-in-tensorflow.html
  
- (AI:DONE) model save/restore 
- (AI:DONE) BN inference after saved model restore + I SHOULD test the model with 'is_training=False'
    --- previous test accuracy w/o BN was around 97%, should there be any progress? ==> 97.2% I think that's the limitation of this dataset..
    

- (AI:DONE) refactoring to inference/train/test module based on TF tutorial

# May 12 

- mnist character classification problem testing
- using a i) deeper/complext convolutional network, ii) applying batch normalization from scratch is a main objective of this run. 

- test accuracy 97% 정도가 나옴 .... 다른 top-tier network 이랑 비교해봤을때 나쁘지 않은 수준..
- Adam optimizer is better than SGD ,,,  test accuracy 94% vs 97%

* Dataset I'm using is not conventional mnist digit dataset which is 0~9. What I'm using is mnist character dataset from A~J. After looking through the dataset, it seems that it would have slightly less accuracy than the digits, it seems a little bit more difficult.



