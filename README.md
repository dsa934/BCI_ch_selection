# Channel Selection Method of Brain Computer Interfaces(BCIs)

## Background

### **Brain-computer interfaces system**

 - Communication between the human brain and external devices by converting brain signals into commands without body movement
 
### **Electroencephalography (EEG)**

 - Electrophysiological monitoring method to record electrical activity of the brain.

### **Process of EEG-based BCI**
 
 ![img_process](https://user-images.githubusercontent.com/83410590/127595814-529cfb69-2eab-477f-9e17-6bcc3e299506.PNG)
--------------------

## **Why do we need a channel selection algorithm in BCI research?**

 - BCI systems require too many channels (electrodes)

![weakness_cs](https://user-images.githubusercontent.com/83410590/127596798-8313300e-3d16-4c86-8a22-09d7dd1e6192.PNG)

[1] (https://www.researchgate.net/figure/Brain-Computer-Interface-scheme_fig2_261186435)

[2] (https://medium.com/svilenk/bciguide-246a9ca76fcd)

 - The use of many electrodes does not guarantee high accuracy. This is because different subjects have different brain areas of activity for the same task.

 - Therefore, by selecting a relatively important channel through the proposed method, we expect to be able to control real life devices through brain cognitive computing technology. In addition, it is expected to be able to solve the problem that the user intention recognition accuracy is lowered by internal/external factors in a real life environment.
 
 ## LRPCNN
 - In the proposed algorithm, CNN and LRP algorithms are used.
 - Layer-wise Relevance Propagation(LRP) (ref. Explaining Decisions of Neural Networks by LRP. Alexander Binder @ Deep Learning: Theory, Algorithms, and Applications. Berlin, June 2017)
 - [Convolutional Neural Networks(CNNs)](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53)
 
![LRP](https://user-images.githubusercontent.com/83410590/127599192-530e1a11-cddb-427e-b2c8-7a9546205f6d.PNG)
 
 - LRP calculates a relevance score for each pixel of the input data.
 - The trained model outputs a predicted value f(x) for a given x.
 - LRP represents the reason why the trained model predicted f(x) by expressing the input value x as a heat map. 
 - LRP ref codes : https://github.com/moboehle/Pytorch-LRP

 
 ## Concept
 
 ![concept2](https://user-images.githubusercontent.com/83410590/127611291-01d5a58f-12c9-47ee-8be8-fc14ee4eea85.PNG)

  - In order to convert RAW EEG data, which is sequential data, into image data, it is converted into a spectrogram through STFT.
  - By applying LRP to the CNN model that has learned the spectrogram, the contribution of each channel can be calculated.
 
 ## Block diagram of system
 
 ![block_diagram](https://user-images.githubusercontent.com/83410590/127598264-431de67f-0ebb-4321-91c8-1150876bd62e.PNG)
 
 ## Codes for the proposed method and the paper used for comparison
 
  - LRPCNN (Proposed Method )
  - CCS-CSP(RCSP) (Restored Code by JW)
  - LRFCSP (Restored Code by JW)
  - Optimal channel selection using covariance matrix and cross-combining (Restored Code by JW)
  - Optimal Channel Selection Using Correlation Coefficient for CSP Based EEG Classification(Restored Code by JW)
 
 
