import java.util.List;


/**
 * Created by Khiem on 1/23/2017.
 */
public class GradientDescent {
    List<Vector> samples;
    int dim;
    static double gamma = 0.001;
    static double margin = 0.000001;
    private boolean testable = false;

    public GradientDescent(List<Vector> s, int d) {
        samples = s;
        dim = d;
    }

    public GradientDescent(List<Vector> s, int d, boolean b) {
        this(s, d);
        testable = true;
    }

    public Vector standard() {
        Vector delta;
        Vector weight = new Vector(dim);
        weight.fill(1);
        double output = 0;
        int count = 0;
        double error = Double.MAX_VALUE;
        while (error > margin&&count < 1000000) {
            count++;
            delta = new Vector(dim);
            for (Vector v :
                    samples) {

                output = weight.dot(v);
                for (int i = 0; i < dim; i++) {
                    delta.set(i, delta.x(i) + gamma * (v.getValue() - output) * v.x(i));
                }
            }

            for (int i = 0; i < dim; i++) {
                weight.set(i, weight.x(i) + delta.x(i));
            }

            error = error(weight);
            if (testable)
            System.out.printf("count %d: output = %f, error = %f\n", count, output, error);
        }
        System.out.println("Number of iterations: "+count);
        return weight;
    }

    public Vector stochastic_unchangedOutput() {
        Vector delta;
        Vector weight = new Vector(dim);
        weight.fill(1);
        double output = 0;
        int count = 0;
        double error = Double.MAX_VALUE;
        while (error > margin&&count < 1000000) {
            count++;
            for (Vector v :
                    samples) {

                output = weight.dot(v);
                for (int i = 0; i < dim; i++) {
                    weight.set(i, weight.x(i) + gamma * (v.getValue() - output) * v.x(i));
                }
            }

            error = error(weight);
            if (testable)
                System.out.printf("count %d: output = %f, error = %f\n", count, output, error);
        }
        System.out.println("Number of iterations: "+count);
        return weight;
    }

    public Vector stochastic() {
        Vector delta;
        Vector weight = new Vector(dim);
        weight.fill(1);
        double output = 0;
        int count = 0;
        double error = Double.MAX_VALUE;
        while (error > margin&&count < 1000000) {
            count++;
            for (Vector v :
                    samples) {
                for (int i = 0; i < dim; i++) {
                    weight.set(i, linearRegression(v, weight, i));
                }
            }

            error = error(weight);
            if (testable)
                System.out.printf("count %d: output = %f, error = %f\n", count, output, error);
        }
        System.out.println("Number of iterations: "+count);
        return weight;
    }

    private double linearRegression(Vector v, Vector w, int i) {
        return w.x(i) + gamma * (v.getValue() - w.dot(v)) * v.x(i);
    }

    private double logisticRegression(Vector v, Vector w, int i) {
        double h = 1 / (1 + Math.exp(-1 * w.dot(v)));
        return w.x(i) + gamma * (v.getValue() - h) * h * (1 - h) * v.x(i);
    }

    private double error(Vector weight) {
        double result = 0;
        for (Vector v :
                samples) {
            result += 0.5 * Math.pow(v.dot(weight) - v.getValue(), 2);
        }
        return result;
    }
}
