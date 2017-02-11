import java.util.ArrayList;
import java.util.List;

/**
 * Created by Khiem on 1/26/2017.
 */
public class NeuralNetwork {
    public List<Node>[] net;

    public Vector getInputVector() {
        return inputVector;
    }

    public void setInputVector(Vector inputVector) {
        this.inputVector = inputVector;
    }

    public Vector inputVector;

    public int getNumberOfLayers() {
        return numberOfLayers;
    }

    private int numberOfLayers;

    public NeuralNetwork(int numLayers) {
        inputVector = new Vector();
        numberOfLayers = numLayers;
        net  = new ArrayList[numLayers];
        for (int i = 0; i < numLayers; i++) {
            net[i] = new ArrayList<>();
        }
    }

    public static void main(String[] args) throws Exception {
        NeuralNetwork n = NeuralNetwork.makeCompleteNetwork(3, 2, 2, 2);
        System.out.println(n.numberOfLayers);
        int numWeight = 0;
        for (int i = 0; i < n.numberOfLayers-1; i++) {
            List<Node> layer = n.getLayer(i);
            for (Node node :
                    layer) {
                numWeight += node.outWeight.size();
            }
        }
        System.out.println(numWeight);
        int num = 0;
        for (int i = 1;i<n.numberOfLayers;i++) {
            List<Node> layer = n.getLayer(i);
            for (Node node : layer) {
                if (!(node instanceof InputNode) && !(node instanceof DummyNode)) {
                    try {
                        num += node.inWeight.size();
                    } catch (Exception e) {
                        System.out.println("this node is: "+node.getClass());
                    }
                }
            }
        }
        System.out.println(num);
    }

    /**
     * for lazy testing
     * @param numLayers number of layers including input and output layers
     * @param args number of nodes value each layer (excluding dummy node)
     * @return input complete network, where 2 nodes are always connected
     */
    public static NeuralNetwork makeCompleteNetwork(int numLayers, int... args) throws Exception {
        NeuralNetwork nn = new NeuralNetwork(numLayers);
        nn.makeCompleteNetWork(args);
        return nn;
    }

    private void makeCompleteNetWork(int... args) throws Exception {
        for (int i = 0; i < numberOfLayers; i++) {
            makeNodesInLayer(i, args[i]);
        }
        for (int i = 0; i < numberOfLayers - 1; i++) {
            makeWeightsBetweenLayers(i, i + 1);
        }
    }

    /**
     * method returns the layer specified
     * @param layer this layer
     * @return the specified layer
     */
    public List<Node> getLayer(int layer) {
        return net[layer];
    }

    /**
     * method updates values for this layer when forwarding
     * @param layer this layer
     */
    public void updateValuesAndInputsForLayer(int layer) {
        int i = (layer == numberOfLayers - 1) ? 0 : 1;
            List<Node> lay = getLayer(layer);
            for(;i<lay.size();i++) {
                Node n = lay.get(i);
            n.updateValue();
            n.updateInput();
        }
    }

    /**
     * method updates delta for layer while back-propagating
     * @param layer this layer
     */
    public void updateDeltasForLayer(int layer){
        for (Node node : net[layer]) {
            node.updateDelta();
        }
    }

    /**
     * method updates weights when back-propagating
     * update the inweight so as to include dummy nodes
     */
    public void updateWeights() {
        for (int i = 1; i < numberOfLayers; i++) {
            List<Node> layer = getLayer(i);
            int j = (numberOfLayers - 1 == i) ? 0 : 1;
            for (; j < layer.size(); j++) {//REMEMBER TO EXCLUDE DUMMY NODE IN MIDDLE LAYERS
                List<Weight> weights = layer.get(j).inWeight;
                for (Weight w :
                        weights) {
                    w.updateWeight();
                }
            }
        }
    }

    /**
     * method initialize input according to input vector
     * note input has dummy node
     * @param vector input vector value sample space
     */
    public void initializeInputAndOutput(Vector vector) {
        setInputVector(vector);
        InputNode inode;
        List<Node> inputLayer = net[0];
        for (int i = 1; i < inputLayer.size(); i++) {
            inode = (InputNode) inputLayer.get(i);
            inode.setInput(vector.x(i-1));
        }
        OutputNode oNode;
        Vector output = vector.getOutput();
        List<Node> outputLayer = net[numberOfLayers - 1];
        for (int i = 0; i < outputLayer.size(); i++) {
            oNode = (OutputNode) outputLayer.get(i);
            oNode.setExpectedOutput(output.x(i));
        }
    }

    /**
     * method creates input complete layer, without connections to other layers yet
     * will create input dummy node first if layer if NOT output layer
     * @param layer this layer
     * @param numNodes number of Nodes to add (exclude the dummy)
     */
    public void makeNodesInLayer(int layer, int numNodes){
        if (0 <= layer && layer <= numberOfLayers-2) {
            Node dummy = new DummyNode();
            net[layer].add(dummy);
        }
            while (numNodes > 0) {
                add(layer);
                numNodes--;
            }
        }

    /**
     * private method to add input node to input layer
     * note that makeNodesInLayer will make the dummy node first therefore add does not need to support adding Dummy node
     * @param layer this specific layer value range(0, numberOfLayer)
     * @return
     */
    private Node add(int layer) {
        //if layer is output layer
        Node node;
        if (layer == numberOfLayers - 1) {
            node = new OutputNode();
            net[numberOfLayers - 1].add(node);
        } else if (layer == 0) {
            node = new InputNode();
            net[0].add(node);
        } else {
            node = new NeuralNode();
            net[layer].add(node);
        }
        return node;
    }

    /**
     * complete every connections possible between 2 layers
     * note output layer does not have dummy variable
     * @param l1 from layer
     * @param l2 to layer
     * @throws Exception
     */
    public void makeWeightsBetweenLayers(int l1, int l2) throws Exception {
        List<Node> layer1 = getLayer(l1);
        List<Node> layer2 = getLayer(l2);
        for (int i = 0; i < layer1.size(); i++) {
            Node node1 = layer1.get(i);
            int j = (numberOfLayers-1 == l2) ? 0 : 1;
            for (;j< layer2.size();j++) {
                Node node2 = layer2.get(j);
                node1.connectsTo(node2);
            }
        }
    }
    public List<List<Double>> getWeights() {
        List<List<Double>> list = new ArrayList<>();
        int sum = 0;
        for (int i = 0; i < numberOfLayers-1; i++) { // from input layer to layer before output layer
            for (Node node :
                    getLayer(i)) {
                list.add(node.getWeights());
            }
        }
         return list;
        }

    public double[] solve(Vector vector) {
        setInputVector(vector);
        InputNode inode;
        List<Node> inputLayer = net[0];
        for (int i = 1; i < inputLayer.size(); i++) {
            inode = (InputNode) inputLayer.get(i);
            inode.setInput(vector.x(i-1));
        }
        for (int i = 1; i < numberOfLayers; i++) {
         updateValuesAndInputsForLayer(i);
        }

        int dim = inputVector.getOutput().getDim();
        double[] a = new double[dim];
        for (int i = 0; i < dim; i++) {
            a[i] = net[numberOfLayers - 1].get(i).getInput();
        }
        return a;
    }

    public double getError() {
        Vector expectedOutputVector = inputVector.getOutput();
        double error = 0;
        List<Node> layer = getLayer(numberOfLayers - 1);
        for (int i = 0; i < expectedOutputVector.getDim(); i++) {

            error += Math.pow(layer.get(i).getInput() - expectedOutputVector.x(i), 2);
        }
        return Math.sqrt(error);
    }
    }

