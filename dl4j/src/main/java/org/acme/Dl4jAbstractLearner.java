package org.acme;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;

public abstract class Dl4jAbstractLearner {
    private final Logger log = org.slf4j.LoggerFactory.getLogger(this.getClass());
    private final String MODEL_NAME = "flowers-" + this.getClass().getSimpleName();
    protected static final int numClasses = 5;
    protected static final long seed = 12345;

    private static final int trainPerc = 80;
    private static int batchSize = 16;
    private static final int trainEpochs = 3;

    protected abstract ComputationGraph createComputationGraph() throws IOException;

    void start(String... args) throws IOException {
        if (args.length == 1) {
            batchSize = Integer.parseInt(args[0]);
        }
        log.info("Starting training with batch size: {}", batchSize);
        final long startupTime = System.currentTimeMillis();
        ComputationGraph transferGraph = createComputationGraph();
        log.info("Model setup complete");

        //Dataset iterators
        Dl4jFlowerDataSetIterator.setup(batchSize,trainPerc);
        DataSetIterator trainIter = Dl4jFlowerDataSetIterator.trainIterator();
        DataSetIterator testIter = Dl4jFlowerDataSetIterator.testIterator();

        exportLabels(trainIter);
        log.info("Labels exported, starting training with a batch size of {}. Startup took {}ms", batchSize, (System.currentTimeMillis() - startupTime));

        Evaluation eval;

        final long startTime = System.currentTimeMillis();
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

        log.info("Model build complete in {}sec with batch of {}," , (System.currentTimeMillis() - startTime) / 1000, batchSize);
        transferGraph.save(new File(MODEL_NAME + ".zip"));
    }

    private void exportLabels(DataSetIterator trainIter) throws IOException {
        FileWriter writer = new FileWriter(MODEL_NAME + ".csv");
        List<String> test = trainIter.getLabels();

        String collect = String.join(",", test);
        writer.write(collect);
        writer.close();
    }
}
