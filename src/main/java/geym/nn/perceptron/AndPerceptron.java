package geym.nn.perceptron;

import java.util.Arrays;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.core.learning.IterativeLearning;
import org.neuroph.nnet.learning.PerceptronLearning;

public class AndPerceptron  implements LearningEventListener{

	public static void main(String[] args) {
		new AndPerceptron().run();
	}
	
    public void run() {
    	
        // 数据集有2个输入 和一个输出
    	// 测试数据是And逻辑运行的结果
        DataSet trainingSet = new DataSet(2, 1);
        trainingSet.addRow(new DataSetRow(new double[]{0, 0}, new double[]{0}));
        trainingSet.addRow(new DataSetRow(new double[]{0, 1}, new double[]{0}));
        trainingSet.addRow(new DataSetRow(new double[]{1, 0}, new double[]{0}));
        trainingSet.addRow(new DataSetRow(new double[]{1, 1}, new double[]{1}));
        // 感知机有2个输入
        SimplePerceptron myPerceptron = new SimplePerceptron(2);

        PerceptronLearning learningRule =(PerceptronLearning) myPerceptron.getLearningRule();
        learningRule.addListener(this);
        
        // 进行学习
        System.out.println("Training neural network...");
        myPerceptron.learn(trainingSet);

        // 测试感知机是否能给出正确输出
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
            System.out.println("Output: " + Arrays.toString( networkOutput) );
        }
    }
    
    @Override
    public void handleLearningEvent(LearningEvent event) {
        IterativeLearning bp = (IterativeLearning)event.getSource();
        System.out.println("iterate:"+bp.getCurrentIteration()); 
        System.out.println(Arrays.toString(bp.getNeuralNetwork().getWeights()));
    }    

}
