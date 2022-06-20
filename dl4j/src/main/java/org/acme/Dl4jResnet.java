package org.acme;

import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.ResNet50;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;

public class Dl4jResnet {
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(Dl4jResnet.class);

    protected static final int numClasses = 5;
    protected static final long seed = 12345;

    private static final int trainPerc = 80;
    private static final int batchSize = 16;
    private static final int trainEpochs = 3;

    private static ComputationGraph createResNet() throws IOException {
        ZooModel zooModel = ResNet50.builder().build();
        ComputationGraph resnet = (ComputationGraph) zooModel.initPretrained();

        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                .updater(new Adam(1e-3))
                .seed(seed)
                .build();

        return new TransferLearning.GraphBuilder(resnet)
                .fineTuneConfiguration(fineTuneConf)
                .setFeatureExtractor("bn5b_branch2c") //"block5_pool" and below are frozen
                .addLayer("fc", new DenseLayer
                        .Builder().activation(Activation.RELU).nIn(1000).nOut(256).build(), "fc1000") //add in a new dense layer
                .addLayer("newpredictions", new OutputLayer
                        .Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX)
                        .nIn(256)
                        .nOut(numClasses)
                        .build(), "fc") //add in a final output dense layer,
                // configurations on a new layer here will be override the finetune confs.
                // For eg. activation function will be softmax not RELU
                .setOutputs("newpredictions") //since we removed the output vertex and it's connections we need to specify outputs for the graph
                .build();
    }

    public static void main(String[] args) throws IOException {
        final long startTime = System.currentTimeMillis();

        ComputationGraph transferGraph = createResNet();
        log.info("Model setup complete");

        //Dataset iterators
        Dl4jFlowerDataSetIterator.setup(batchSize,trainPerc);
        DataSetIterator trainIter = Dl4jFlowerDataSetIterator.trainIterator();
        DataSetIterator testIter = Dl4jFlowerDataSetIterator.testIterator();

        exportLabels(trainIter);
        log.info("Labels exported");

        Evaluation eval;

        int epoch = 1;
        while (epoch <= trainEpochs) {
            int iter = 0;
            while (trainIter.hasNext()) {
                log.info(epoch + " Start train iter " + iter + ".... score: " + transferGraph.score());
                transferGraph.fit(trainIter.next());
                iter++;
            }
            log.info(epoch + " Evaluate model score " + transferGraph.score() + " at iter " + " ....  in " + (System.currentTimeMillis() - startTime) / 1000 + "sec");
            eval = transferGraph.evaluate(testIter);
            log.info(eval.stats());
            testIter.reset();
            trainIter.reset();
            epoch++;
        }

        log.info("Model build complete in " + (System.currentTimeMillis() - startTime) / 1000 + "sec");
        transferGraph.save(new File("flowerResNet50.zip"));
    }

    private static void exportLabels(DataSetIterator trainIter) throws IOException {
        FileWriter writer = new FileWriter("flowerResNet50.csv");
        List<String> test = trainIter.getLabels();

        String collect = String.join(",", test);
        writer.write(collect);
        writer.close();
    }
}
