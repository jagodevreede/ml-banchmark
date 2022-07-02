package org.acme;

import ai.djl.Application;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.repository.zoo.Criteria;
import ai.djl.training.util.ProgressBar;
import org.slf4j.Logger;

public class DjlVgg16 extends DjlAbstractLearner {
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(DjlVgg16.class);

    @Override
    protected Criteria.Builder<Image, Classifications> getModelBuilder() {
        Criteria.Builder<Image, Classifications> builder =
                Criteria.builder()
                        .optApplication(Application.CV.IMAGE_CLASSIFICATION)
                        .setTypes(Image.class, Classifications.class)
                        .optProgress(new ProgressBar())
                        .optArtifactId("vgg");
        builder.optGroupId("ai.djl.mxnet");
        builder.optFilter("layers", "16");
        builder.optFilter("dataset", "imagenet");
        return builder;
    }

    public static void main(String[] args) throws Exception {
        log.info("Starting Vgg16 model");
        new DjlVgg16().start(args);
    }

}
