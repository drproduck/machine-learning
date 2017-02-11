import java.util.ArrayList;
import java.util.List;

/**
 * Created by Khiem on 1/24/2017.
 */
public class top{
    public static void main(String[] args) throws Exception {
        int dim = 11;
        int num = 7;

        List<Vector> samples = new ArrayList();
        for (int i = 0; i < num; i++) {
            Vector v = new Vector(dim, true);
            v.random(-10, 10);
            samples.add(v);
        }
        /**

        List<Vector> samples = new ArrayList<>();
        Vector v1 = new Vector(6, 1, 1, 2, 3,4, 5);
        Vector v2 = new Vector(-3, 3, 4, 5, 6, 7, 8);
        Vector v3 = new Vector(4, 3, 2, -2, -3, -4, -5);
        Vector v4 = new Vector(9, 1, 2, 3, 4, -6, -7);
        Vector v5 = new Vector(-7, - 5, -6, -7, -8, -5, 4);
        samples.add(v1);
        samples.add(v2);
        samples.add(v3);
        samples.add(v4);
        samples.add(v5);
        */

        GradientDescent gd = new GradientDescent(samples,dim);
        Vector weight = gd.standard();
        System.out.println("Checking result: ");
        for (int i = 0; i < num; i++) {
            Vector v = samples.get(i);
            System.out.printf("vector %d, true value: %d, learned value: %f\n", i, v.getValue(), v.dot(weight));
        }
        System.out.println();
        weight = gd.stochastic_unchangedOutput();
        System.out.println("Checking result: ");
        for (int i = 0; i < num; i++) {
            Vector v = samples.get(i);
            System.out.printf("vector %d, true value: %d, learned value: %f\n", i, v.getValue(), v.dot(weight));
        }
        System.out.println();
        weight = gd.stochastic();
        System.out.println("Checking result: ");
        for (int i = 0; i < num; i++) {
            Vector v = samples.get(i);
            System.out.printf("vector %d, true value: %d, learned value: %f\n", i, v.getValue(), v.dot(weight));
        }
    }
}
