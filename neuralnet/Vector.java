/**
 * Created by Khiem on 1/19/2017.
 */
public class Vector {
    public double[] getCoordinate() {
        return coordinate;
    }

    private double[] coordinate;

    public int getDim() {
        return dim;
    }

    private int dim;

    private boolean testable = false;

    public Vector getOutput() {
        return output;
    }

    private Vector output;

    public double x(int t) {
        return coordinate[t];
    }

    public Vector(double... args) {
        dim = args.length;
        coordinate = new double[dim];
        for (int i = 0; i < dim; i++) {
            coordinate[i] = args[i];
        }
    }

    public Vector(Vector o, double...args) {
        dim = args.length;
        coordinate = new double[dim];
        for (int i = 0; i < dim; i++) {
            coordinate[i] = args[i];
        }
        output = o;
    }



    public Vector(int d, boolean b) {
        this(d);
        testable = b;
    }

    public void fill(double i) {
        for (int j = 0; j < dim; j++) {
            coordinate[j] = i;
        }
    }

    public void set(int i, double j) {
        coordinate[i] = j;
    }

    public double distTo(Vector other) {

        double sum = 0;
        for (int i = 0; i < dim; i++) {
            sum += Math.pow(coordinate[i] - other.coordinate[i], 2);
        }
        return Math.sqrt(sum);
    }

    public double dot(Vector other) {
        double sum = 0;
        for (int i = 0; i < dim; i++) {
            sum += coordinate[i] * other.coordinate[i];
        }
        return sum;
    }

    public static void main(String[] args) {
        Vector v1 = new Vector(2, 4, 5, 7);
        Vector v2 = new Vector(2, 10, 12, 11);
        System.out.println(v1.dot(v2));
    }
}
