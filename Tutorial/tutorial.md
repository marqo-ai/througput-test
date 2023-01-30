# Tutorial

This short tutorial summarizes the efforts we have made on implementing [BentoML](https://www.bentoml.com/).
In general, the whole process can be found in this [video](https://www.youtube.com/watch?v=HHkmfI_yncc&ab_channel=ValerioVelardo-TheSoundofAI).

## Why we use BentoML

BentoML can be used to set up an inference server. It supports most machine learning frameworks that we are 
interested such as ONNX, PyTorch. The most important feature is [adaptive batching](https://docs.bentoml.org/en/latest/guides/batching.html).
We expect that this can cooperate with concurrent image downloading to increase the throughput of Marqo at 
indexing stage. 



## Problems

During the testing, we have two main problems. 

### 1. Flexibility

In BentoML, all machine learning models needs to be
- **a) saved** 
- **b) served**
- **c) containerized**

Although the containerized model can be easily deployed on difference instances and provide inference services, the contradicts with
the flexibility requirements of models. Different models may require different wrappers in the **save** and **serve** processes.
Different configurations are required to containerize the model to achieve the best throughput. Therefore, it is impossible to take any
model from users and make it an inference server in the edge device.
We have to select several widely used models and containerize them beforehand. This will reduce the flexibility of Marqo.

### 2. Data Transmission
To utilize the inference server, Marqo needs to serialize the data and send it to the serve. The server will do the inference and 
send the data back. For the clip model `ViT-L/14`, we need to serialize and send a `(3,224,224)` tensor and receive a `(1,768)`. The serialization 
will take around `100ms`, which is much greater than the inference time (`20ms`). The serialization time is also linear to the `batch_size`,
which makes any batching strategy meaningless.

One possible solution is, instead of sending the processed image (then tensor), we send the path of the image directly to the server. The image 
loading, preprocessing is finished in the server. This largely reduces the cost in data transmission, while increases the complexity in the server.


