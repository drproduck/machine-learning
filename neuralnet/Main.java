import java.util.Arrays;

/**
 * Created by drproduck on 2/6/17.
 */
public class Main{
    public static void main(String[] args) throws Exception {
        NeuralNetwork n = NeuralNetwork.makeCompleteNetwork(4, 3, 5, 5, 1);
        Vector[] examples = new Vector[1000];
        for (int i = 0; i < 1000; i++) {
            double a = -100 + 200 * Math.random();
            double b = -100 + 200 * Math.random();
            double c = -100 + 200 * Math.random();
            double d = (a*b+c>=0) ? 1 : 0;
            Vector v = new Vector(new Vector(d), a, b, c);
            examples[i] = v;
        }
        BackPropagation bp = new BackPropagation(n, examples);
        bp.propagate();
        System.out.println("Testing: ");
        /**
        double[] result;
        double a;
        double b;
        double c;
        for (int i = 0; i < 100; i++) {
            a = -30 + 60*Math.random();
            b = -30 + 60 * Math.random();
            c = (a+b>=0) ? 1 : 0;
            Vector v = new Vector(new Vector(c), a, b);
            result = n.solve(v);
            System.out.println("Tested input: " +a+" "+b+", ouput: "+Arrays.toString(result) + " expected output: " + Arrays.toString(v.getOutput().getCoordinate()));
        }

         */
        int i = 10;
        while (i > 0) {
            double a = -100 + 200 * Math.random();
            double b = -100 + 200 * Math.random();
            double c = -100 + 200 * Math.random();
            System.out.println("manual test: a*b+c");
            Vector v = new Vector(new Vector((a * b + c >= 0) ? 1 : 0), a, b, c);
            System.out.printf("%f, %f, %f, expected output is: %f\n", a, b, c, v.getOutput().x(0));
            System.out.println(Arrays.toString(n.solve(v)));
            i--;
        }
    }

}
