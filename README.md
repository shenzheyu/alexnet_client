# Alexnet

## Preperation

Create virtual environment and install packages.
```shell script
python3 -m venv venv
source venv/bin/activate
pip install tensorflow==1.14 tensorflow-serving-api==1.14 numpy==1.15 opencv-python
deactivate
```

Download alexnet weights (from caffee) in .npy format.
```shell script
wget http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy
```

## Save Alexnet in saved_model format

Original implemention is from https://www.cs.toronto.edu/~guerzhoy/tf_alexnet/ .
```shell script
python dump_pb.py
```

## Examination

Serve model via TF Serving
```shell script
sudo docker run -it --rm -p 8900:8500 --gpus all -v /home/shenz/alexnet_client/models/alexnet/:/models/alexnet -e MODEL_NAME=alexnet tensorflow/serving:latest-gpu --enable_batching=true
```

Run client.py
```shell script
python client.py -server=127.0.0.1:8900 -batch_size=1 -img_path=./examples/dog.png
```

Run batch_test.py
```shell script
python batch_test.py -server=127.0.0.1:8900
```