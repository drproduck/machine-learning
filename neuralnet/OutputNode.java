import java.util.ArrayList;

/**
 * Created by drproduck on 1/29/17.
 */
public class OutputNode extends Node {
    double expectedOutput;
    public OutputNode(){
        inWeight = new ArrayList<>();
        expectedOutput = 0; //have to be manually set for each input vector
    }

    public void setExpectedOutput(double output) {
        expectedOutput = output;
    }

    public void updateDelta(){
        delta  = input * (1 - input) * (expectedOutput - input);
    }
}
