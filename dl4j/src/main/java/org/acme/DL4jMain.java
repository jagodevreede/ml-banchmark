package org.acme;

public class DL4jMain {
    public static void main(String[] args) throws Exception {
        if (args.length != 2) {
            System.out.println("Usage: java -jar dl4j-sharded.jar <model> <batchsize>");
            System.exit(1);
        }
        if (args[0].equals("vgg16")) {
            new Dl4jVGG16().start(args[1]);
        } else if (args[0].equals("resnet")) {
            new Dl4jResnet().start(args[1]);
        } else {
            System.out.println("Unknown model: " + args[0]);
            System.exit(1);
        }
    }
}
