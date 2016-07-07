package geym.nn.mlperceptron.ch6;

import org.neuroph.core.Connection;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.Neuron;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.core.learning.IterativeLearning;
import org.neuroph.core.learning.LearningRule;
import org.neuroph.util.TransferFunctionType;

import geym.nn.mlperceptron.MlPerceptron;

import java.util.Arrays;

public class XorMlPerceptron  implements LearningEventListener{

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		new XorMlPerceptron().run();
	}
	
    public void run() {
        // 训练数据
        DataSet trainingSet = new DataSet(2,1);
        trainingSet.addRow(new DataSetRow(new double[]{0, 0}, new double[]{0}));
        trainingSet.addRow(new DataSetRow(new double[]{0, 1}, new double[]{1}));
        trainingSet.addRow(new DataSetRow(new double[]{1, 0}, new double[]{1}));
        trainingSet.addRow(new DataSetRow(new double[]{1, 1}, new double[]{0}));
        // 2个输入数据，4个神经元隐层，1个输出
        MlPerceptron myPerceptron = new MlPerceptronBinOutput(TransferFunctionType.SIGMOID,2,4,1);
        LearningRule learningRule = myPerceptron.getLearningRule();
        learningRule.addListener(this);
        // 训练神经网络
        System.out.println("Training neural network...");
        myPerceptron.learn(trainingSet);
        // 测试神经网络
        System.out.println("Testing trained neural network");
        testNeuralNetwork(myPerceptron, trainingSet);
    }

	/**
     * Prints network output for the each element from the specified training set.
     * @param neuralNet neural network
     * @param trainingSet training set
     */
    public static void testNeuralNetwork(NeuralNetwork neuralNet, DataSet testSet) {

        for(DataSetRow testSetRow : testSet.getRows()) {
            neuralNet.setInput(testSetRow.getInput());
            neuralNet.calculate();
            double[] networkOutput = neuralNet.getOutput();

            System.out.print("Input: " + Arrays.toString( testSetRow.getInput() ) );
            System.out.println(" Output: " + Arrays.toString( networkOutput) );
        }
    }
    
    @Override
    public void handleLearningEvent(LearningEvent event) {
        System.out.println("============");
        System.out.println(event.getClass().toString());
        IterativeLearning bp = (IterativeLearning)event.getSource();
        System.out.println("iterate:"+bp.getCurrentIteration()); 
        Neuron neuron=(Neuron)bp.getNeuralNetwork().getOutputNeurons()[0];

        for(Connection conn:neuron.getInputConnections()){
            System.out.println(conn.getWeight().value);
        }
    }    

}
