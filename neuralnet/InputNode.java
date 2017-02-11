import java.util.ArrayList;

/**
 * Created by drproduck on 1/29/17.
 */
public class InputNode extends Node {
    public InputNode(){
        outWeight = new ArrayList<>();
    }

    public InputNode(double input) {
        this.input = input;
    }

    public void setInput(double input) {
        this.input = input;
    }
}
