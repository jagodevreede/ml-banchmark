# Build
Call maven with the profile you need, `cpu` or `gpu` are available.

example: `mvn clean package -P cpu`

First go into the folder you need to run (dlj or dl4j)

Then call the shaded jar with the model you want to train. `resnet` and `vgg16` are available.

example dl4j: `java -jar target/dl4j-1.0-SNAPSHOT-shaded.jar resnet 16` where 16 is the batch size.
example djl: `java -jar target/djl-1.0-SNAPSHOT-shaded.jar resnet 16` where 16 is the batch size.

# Run on AWS:
Instance type g4dn.xlarge

To get the cheapesed instance in aws checkout this pricing site:
https://www.instance-pricing.com/provider=aws-ec2/instance=g4dn.xlarge/

# Python
To disable cuda set `CUDA_VISIBLE_DEVICES=""` in your environment.

to change the model change the `base_model` variable in [transfer_learning.py](pyhton/transfer_learning.py)