import java.util.Arrays;
import java.util.List;

/**
 * Created by Khiem on 1/26/2017.
 */
public class BackPropagation {
    private NeuralNetwork network;
    private Vector[] examples;
    private static final double precision = 0.05;
    public BackPropagation(NeuralNetwork net, Vector[] exs) {
        network = net;
        examples = exs;
    }

    public void propagate() {
        double error = 10;
        while (Math.abs(error) > precision) {
            for (Vector ex :
                    examples) {

                network.initializeInputAndOutput(ex);
                for (int i = 1; i < network.getNumberOfLayers(); i++) {
                    network.updateValuesAndInputsForLayer(i);
                }
                error = network.getError();
                System.out.println("Error for this epoch = "+error);

                for (int i = network.getNumberOfLayers() - 1; i >= 0; i--) {
                    network.updateDeltasForLayer(i);
                }
                network.updateWeights();
            }
        }
    }

    public static void main(String[] args) throws Exception {
        NeuralNetwork nw = NeuralNetwork.makeCompleteNetwork(3, 2, 2, 2);
        Vector[] exs = new Vector[4];
        exs[0] = new Vector(new Vector(0, 1), 0, 1);
        exs[1] = new Vector(new Vector(0, 0), 0, 0);
        exs[2] = new Vector(new Vector(0, 1), 1, 0);
        exs[3] = new Vector(new Vector(1, 0), 1, 1);
        for (List l :
                nw.net) {
            System.out.println(l);
        }
        System.out.println(nw.getWeights());
        System.out.println();
        BackPropagation bp = new BackPropagation(nw, exs);
        bp.propagate();
        System.out.println(nw.getWeights());
        System.out.println("testing");
        System.out.println(Arrays.toString(nw.solve(new Vector(new Vector(0,1),1, 0))));
        System.out.println(Arrays.toString(nw.solve(new Vector(new Vector(1,0),1, 1))));
        System.out.println(Arrays.toString(nw.solve(new Vector(new Vector(0,0),0, 0))));
        System.out.println(Arrays.toString(nw.solve(new Vector(new Vector(0,1),0, 1))));
    }
}
