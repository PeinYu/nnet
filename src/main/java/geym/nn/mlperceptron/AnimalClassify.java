package geym.nn.mlperceptron;

import geym.nn.stock.StockPrediction;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.util.TransferFunctionType;

import java.io.*;
import java.util.Arrays;
import java.util.List;
import java.util.Vector;

public class AnimalClassify implements LearningEventListener{

    /**
     * @param args
     * @throws IOException
     */
    public static void main(String[] args) throws IOException {

        new AnimalClassify().run();
    }

    public static double[] int2double(int i){
        double[] re=new double[32];
        for(int j=0;j<32;j++){
            re[j]=(double)((i>>j)&1);
        }
        return re;
    }

    /**
     * @param filepath: like "/geym/nn/mlperceptron/TS9010/zoo.90.percent.txt"
     * @return
     * @throws IOException
     */
    public static List<DataSetRow> getTrainData(String filepath) throws IOException{
        InputStream in = StockPrediction.class.getResourceAsStream(filepath);
        BufferedReader br=new BufferedReader(new InputStreamReader(new DataInputStream(in)));
        List<Double> prices=new Vector<Double>();
        List<DataSetRow> re=new Vector<DataSetRow>();
        String line=null;
        int Attribute_Len=16;
        while((line=br.readLine())!=null){
            String[] item=line.split("\t");
            double[] inputs=new double[Attribute_Len];
            int i=0;
            for(i=0;i<Attribute_Len;i++){
                inputs[i]=Double.parseDouble(item[i]);
            }
            double[] outputs=new double[7];
            for(;i<Attribute_Len+7;i++){
                outputs[i-Attribute_Len]=Double.parseDouble(item[i]);
            }
            re.add(new DataSetRow(inputs,outputs));
        }
        return re;
    }



    public void run() throws IOException {

        DataSet trainingSet = new DataSet(16, 7);
        List<DataSetRow> rows=getTrainData("/geym/nn/mlperceptron/TS9010/zoo.90.percent.txt");
        for(int i=0;i<rows.size();i++){
            trainingSet.addRow(rows.get(i));
        }

        MlPerceptron myMlPerceptron = new MlPerceptron(TransferFunctionType.SIGMOID, 16,6,7);
        myMlPerceptron.getLearningRule().setMaxError(0.01d);
        myMlPerceptron.getLearningRule().addListener(this);

        System.out.println("Training neural network...");
        myMlPerceptron.learn(trainingSet);

        System.out.println("Testing trained neural network");
        testNeuralNetwork(myMlPerceptron);

    }
    public static void testNeuralNetwork(NeuralNetwork neuralNet) throws IOException {
        List<DataSetRow> rows=getTrainData("/geym/nn/mlperceptron/TS9010/zoo.10.percent.txt");
        for(DataSetRow row:rows){
            neuralNet.setInput(row.getInput());
            neuralNet.calculate();
            double[] networkOutput = neuralNet.getOutput();
            System.out.print("Input: " + Arrays.toString(row.getInput()));
            System.out.println(" Output: " + Arrays.toString( networkOutput) );
            System.out.println("Desired Output: " + Arrays.toString( row.getDesiredOutput()) );
        }

    }

    @Override
    public void handleLearningEvent(LearningEvent event) {
        BackPropagation bp = (BackPropagation)event.getSource();
        System.out.println(bp.getCurrentIteration() + ". iteration : "+ bp.getTotalNetworkError());
    }
}