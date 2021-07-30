# Deep Adversarial Training for Teaching Networks to Reject Unknown Inputs

### A bachelor thesis under the supervision of [Prof. Dr. Manuel Günther](https://github.com/siebenkopf)

![](./images/flowers.PNG)

## ABSTRACT

Modern day machine learning models are becoming omnipresent 
and are required to handle progressively more complex 
environments in their tasks. In classification problems, 
an increasingly popular scenario is called Open Set 
Recognition, which does not require the model to have 
complete knowledge of the world and during which unknown 
classes can be submitted to the algorithm while testing. 
This thesis tackles the challenge to correctly handle and 
reject these unknown inputs by performing adversarial 
training on our classification model. Furthermore, 
we analyze the difference in performance of several 
state-of-the-art adversarial attacks used in our 
adversarial training. The experiments show that our 
approach effectively deals with unknown inputs and 
delivers very promising results. To our knowledge, 
there has been no prior work that used adversarial 
training for Open Set Recognition like in our approach.