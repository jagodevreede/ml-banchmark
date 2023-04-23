import os
import requests
import tarfile

# set the URL of the file to download
url = "http://download.tensorflow.org/example_images/flower_photos.tgz"

# set the directory where the file should be extracted
output_dir = os.path.expanduser("~/dl4j-examples-data/dl4j-examples")

# check if the output directory already exists
if not os.path.exists(output_dir):
    # create the output directory
    os.makedirs(output_dir)

    # download the file
    response = requests.get(url, stream=True)

    # extract the file
    with tarfile.open(fileobj=response.raw, mode="r|gz") as tar:
        tar.extractall(output_dir)

    print("File downloaded and extracted to:", output_dir)
else:
    print("Output directory already exists:", output_dir)
