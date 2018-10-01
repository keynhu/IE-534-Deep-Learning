In the first Homework, here's some mistakes I've made.
1. I misunderstood ReLu in some sense. During back propagation, the gradient of ReLu should be taken w.r.t. Z, while I took it w.r.t. the back derivative.

In addition, I have done the following things in order to debug:
1. I’ve set the updated values of each parameter to be 0 in order to see which of the updated process may cause mistakes. 
2. I’ve printed out the parameters and found some of them had NA values. This is because mistakes in updated process let W explode and cause Inf to appear during softmax process, which turns the value into NA.
