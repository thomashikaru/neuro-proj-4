# Project 4: Serial Order Representations in Working Memory
Based on Botvinick and Watanabe (2007)

## Implementation of Neural Network

- An **example** consists of an item vector (length=N) and a rank vector (length=R), both of them one-hot vectors. 
    - The item vector indicates the identity of the sequence element being seen by the model. 
    - The rank vector indicates the position in the sequence of the element being seen by the model. 
- A **trial** consists of a sequential series of examples being passed into the model, followed by a weight update based on comparing the model's output with the target. 

For example, suppose we have the sequence of items [0, 5, 3, 1, 4, 2]. 
We start with the first element, item 0 in position 0. This gives us item vector [1, 0, 0, 0, 0, 0] and rank vector [1, 0, 0, 0, 0, 0]. 
Next, we have item 5 in position 1. This gives us item vector [0, 0, 0, 0, 0, 1] and rank vector [0, 1, 0, 0, 0, 0]. 

## Results


