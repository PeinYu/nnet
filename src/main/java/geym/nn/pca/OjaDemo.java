package geym.nn.pca;

import java.util.Arrays;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.core.learning.IterativeLearning;


public class OjaDemo  implements LearningEventListener{

	public static void main(String[] args) {
		new OjaDemo().run();
	}
	
    public void run() {
    	
        DataSet trainingSet = new DataSet(3, 1);
        trainingSet.addRow(new DataSetRow(new double[]{0,-2,0}, new double[]{0}));
        trainingSet.addRow(new DataSetRow(new double[]{1,1,0}, new double[]{0}));
        trainingSet.addRow(new DataSetRow(new double[]{-1,-1,0}, new double[]{0}));
        trainingSet.addRow(new DataSetRow(new double[]{0,2,0}, new double[]{0}));
        OjaNetwork oja = new OjaNetwork(3);

        OjaLearning learningRule =(OjaLearning) oja.getLearningRule();
        learningRule.setLearningRate(0.001);
        learningRule.addListener(this);
        
        // 进行学习
        System.out.println("Training neural network...");
        oja.learn(trainingSet);

        // 测试感知机是否能给出正确输出
        System.out.println("Testing trained neural network");
        testNeuralNetwork(oja, trainingSet);
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
