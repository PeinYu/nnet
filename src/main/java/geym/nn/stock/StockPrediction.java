package geym.nn.stock;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.util.TransferFunctionType;

import geym.nn.mlperceptron.MlPerceptron;
import geym.nn.mlperceptron.ch7.MlPerceptronLineOutput;

import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.List;
import java.util.Vector;

/**
 * http://q.stock.sohu.com/zs/000001/lshq.shtml
 * @author Geym
 *
 */
public class StockPrediction  implements LearningEventListener{

    /**
     * @param args
     * @throws IOException
     */
    public static void main(String[] args) throws IOException {

        new StockPrediction().run();
    }

    public List<DataSetRow> getTrainData() throws IOException{
        InputStream in = StockPrediction.class.getResourceAsStream("/geym/nn/mlperceptron/收盘.txt");
        BufferedReader br=new BufferedReader(new InputStreamReader(new DataInputStream(in)));
        List<Double> prices=new Vector<Double>();
        List<DataSetRow> re=new Vector<DataSetRow>();
        String line=null;
        while((line=br.readLine())!=null){
            prices.add(0,Double.parseDouble(line)/10000);
        }
        System.out.println(prices);
        for(int i=0;i<prices.size()-4;i++){

            double[] inputs=new double[]{prices.get(i),prices.get(i+1),prices.get(i+2),prices.get(i+3)};
            double[] outputs=new double[]{prices.get(i+4)};
            re.add(new DataSetRow(inputs,outputs));
        }
        //System.out.println(re);
        return re;
    }

    public void run() throws IOException {

        DataSet trainingSet = new DataSet(4, 1);
        List<DataSetRow> rows=getTrainData();
        for(int i=0;i<rows.size()-1;i++){
            trainingSet.addRow(rows.get(i));
        }

        // create multi layer perceptron
        MlPerceptron myMlPerceptron = new MlPerceptronLineOutput(TransferFunctionType.SIGMOID, 4,30,1);
        myMlPerceptron.setLearningRule(new BackPropagation());
        myMlPerceptron.getLearningRule().setLearningRate(0.2);
        myMlPerceptron.getLearningRule().setMaxError(0.00005d);
        myMlPerceptron.getLearningRule().addListener(this);

        // learn the training set
        System.out.println("Training neural network...");
        myMlPerceptron.learn(trainingSet);

        // test perceptron
        System.out.println("Testing trained neural network");
        testNeuralNetwork(myMlPerceptron, rows.get(rows.size()-1));

    }
    public static void testNeuralNetwork(NeuralNetwork neuralNet, DataSetRow row) {
        neuralNet.setInput(row.getInput());
        neuralNet.calculate();
        double[] networkOutput = neuralNet.getOutput();
        System.out.print("Input: " + Arrays.toString(row.getInput()));
        System.out.println(" Output: " + Arrays.toString( networkOutput) );
    }

    @Override
    public void handleLearningEvent(LearningEvent event) {
        BackPropagation bp = (BackPropagation)event.getSource();
        System.out.println(bp.getCurrentIteration() + ". iteration : "+ bp.getTotalNetworkError());
    }
}