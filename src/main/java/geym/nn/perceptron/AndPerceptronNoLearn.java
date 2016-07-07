
package geym.nn.perceptron;

import java.util.Arrays;

import org.neuroph.core.Layer;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.Neuron;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.nnet.comp.neuron.BiasNeuron;
import org.neuroph.nnet.comp.neuron.InputNeuron;
import org.neuroph.util.ConnectionFactory;
import org.neuroph.util.LayerFactory;
import org.neuroph.util.NeuralNetworkFactory;
import org.neuroph.util.NeuralNetworkType;
import org.neuroph.util.NeuronProperties;
import org.neuroph.util.TransferFunctionType;

public class AndPerceptronNoLearn extends NeuralNetwork {

    /**
     * The class fingerprint that is set to indicate serialization compatibility
     * with a previous version of the class.
     */
    private static final long serialVersionUID = 1L;

    /**
     * Creates new Perceptron with specified number of neurons in input and
     * output layer, with Step trqansfer function
     * 
     * @param inputNeuronsCount number of neurons in input layer
     * @param outputNeuronsCount number of neurons in output layer
     */
    public AndPerceptronNoLearn(int inputNeuronsCount) {
        this.createNetwork(inputNeuronsCount);
    }

    /**
     * Creates perceptron architecture with specified number of neurons in input
     * and output layer, specified transfer function
     * 
     * @param inputNeuronsCount number of neurons in input layer
     * @param outputNeuronsCount number of neurons in output layer
     * @param transferFunctionType neuron transfer function type
     */
    private void createNetwork(int inputNeuronsCount) {
        // 设置网络类别为 感知机
        this.setNetworkType(NeuralNetworkType.PERCEPTRON);

        // 输入神经元建立 ，表示输入的刺激
        NeuronProperties inputNeuronProperties = new NeuronProperties();
        inputNeuronProperties.setProperty("neuronType", InputNeuron.class);

        // 由输入神经元构成的输入层
        Layer inputLayer = LayerFactory.createLayer(inputNeuronsCount, inputNeuronProperties);
        this.addLayer(inputLayer);
        // 在输入层增加BiasNeuron，表示神经元偏置
        inputLayer.addNeuron(new BiasNeuron());
        
        NeuronProperties outputNeuronProperties = new NeuronProperties();
        outputNeuronProperties.setProperty("transferFunction", TransferFunctionType.STEP);
        Layer outputLayer = LayerFactory.createLayer(1, outputNeuronProperties);
        this.addLayer(outputLayer);
        
        ConnectionFactory.fullConnect(inputLayer, outputLayer);
        NeuralNetworkFactory.setDefaultIO(this);
        Neuron n=outputLayer.getNeuronAt(0);
        
        n.getInputConnections()[0].getWeight().setValue(2);
        n.getInputConnections()[1].getWeight().setValue(2);
        n.getInputConnections()[2].getWeight().setValue(-3);
    }
    
    public static void main(String[] args) {
        DataSet trainingSet = new DataSet(2, 1);
        trainingSet.addRow(new DataSetRow(new double[]{0, 0},new double[]{Double.NaN}));
        trainingSet.addRow(new DataSetRow(new double[]{0, 1},new double[]{Double.NaN}));
        trainingSet.addRow(new DataSetRow(new double[]{1, 0},new double[]{Double.NaN}));
        trainingSet.addRow(new DataSetRow(new double[]{1, 1},new double[]{Double.NaN}));
        
    	AndPerceptronNoLearn perceptron=new AndPerceptronNoLearn(2);
    	
    	for(DataSetRow row:trainingSet.getRows()){
    		perceptron.setInput(row.getInput());
    		perceptron.calculate();
    		double[] networkOutput = perceptron.getOutput();
    		System.out.println(Arrays.toString(row.getInput())+"="+Arrays.toString(networkOutput));
    	}
	}
}
