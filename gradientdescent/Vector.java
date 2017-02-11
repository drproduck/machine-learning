import java.util.Arrays;

/**
 * Created by Khiem on 1/19/2017.
 */
public class Vector {
    private double[] co;

    public int getDim() {
        return dim;
    }

    private int dim;
    private int value;
    private boolean testable = false;

    public Vector getOutput() {
        return output;
    }

    private Vector output;

    public int getValue() {
        return value;
    }

    public void setValue(int v) {
        value = v;
    }

    public double x(int t) {
        return co[t];
    }

    public Vector(double... args) {
        dim = args.length;
        co = new double[dim];
        for (int i = 0; i < dim; i++) {
            co[i] = args[i];
        }
    }

    public Vector(int value, double args) {
        this(args);
        this.value = value;
    }

    public Vector(Vector o, double...args) {
        dim = args.length;
        co = new double[dim];
        for (int i = 0; i < dim; i++) {
            co[i] = args[i];
        }
        output = o;
    }



    public Vector(int d, boolean b) {
        this(d);
        testable = b;
    }

    public void fill(double i) {
        for (int j = 0; j < dim; j++) {
            co[j] = i;
        }
    }

    public void set(int i, double j) {
        co[i] = j;
    }

    public double distTo(Vector other) {

        double sum = 0;
        for (int i = 0; i < dim; i++) {
            sum += Math.pow(co[i] - other.co[i], 2);
        }
        return Math.sqrt(sum);
    }

    public double dot(Vector other) {
        double sum = 0;
        for (int i = 0; i < dim; i++) {
            sum += co[i] * other.co[i];
        }
        return sum;
    }

    public void random(int valueFrom, int valueTo) throws Exception{
        if (!testable) {
            throw new UnsupportedOperationException("Not allowed to be tested");
        }
        else {
            for (int i = 0; i < dim; i++) {
                co[i] = (int)(10 * Math.random());
            }
            value = valueFrom + (int) ((valueTo - valueFrom) * Math.random());
            System.out.printf("randomize for testing: value: %d, coordinates: %s\n", value, Arrays.toString(co));
        }
    }

    public static void main(String[] args) {
        Vector v1 = new Vector(2, 4, 5, 7);
        Vector v2 = new Vector(2, 10, 12, 11);
        System.out.println(v1.dot(v2));
    }
}
