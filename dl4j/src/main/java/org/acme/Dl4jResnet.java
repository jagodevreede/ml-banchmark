package org.acme;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.ResNet50;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;

import java.io.IOException;

public class Dl4jResnet extends Dl4jAbstractLearner  {
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(Dl4jResnet.class);

    protected ComputationGraph createComputationGraph() throws IOException {
        ZooModel zooModel = ResNet50.builder().build();
        ComputationGraph resnet = (ComputationGraph) zooModel.initPretrained();

        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
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
        log.info("Starting Resnet model");
        new Dl4jResnet().start(args);
    }
}
