package org.acme;

import ai.djl.Model;
import ai.djl.basicmodelzoo.cv.classification.ResNetV1;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.repository.zoo.Criteria;
import org.slf4j.Logger;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

public class DjlResnetNoTransferLearning extends DjlAbstractLearner {
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(DjlResnetNoTransferLearning.class);

    @Override
    protected Model getModel() {
        Model model = Model.newInstance("my-demo-model");

        Block resNet50 = ResNetV1.builder()
                .setImageShape(new Shape(3, IMAGE_HEIGHT, IMAGE_HEIGHT))
                .setNumLayers(50)
                .setOutSize(NUM_OF_OUTPUT)
                .build();

        model.setBlock(resNet50);

        return model;
    }

    @Override
    protected Criteria.Builder<Image, Classifications> getModelBuilder() {
      throw new NotImplementedException();
    }

    public static void main(String[] args) throws Exception {
        log.info("Starting Clean Resnet model");
        new DjlResnetNoTransferLearning().start(args);
    }

}
