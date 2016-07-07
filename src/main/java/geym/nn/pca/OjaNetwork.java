
package geym.nn.pca;

import org.neuroph.core.Layer;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.nnet.comp.neuron.InputNeuron;
import org.neuroph.util.ConnectionFactory;
import org.neuroph.util.LayerFactory;
import org.neuroph.util.NeuralNetworkFactory;
import org.neuroph.util.NeuralNetworkType;
import org.neuroph.util.NeuronProperties;
import org.neuroph.util.TransferFunctionType;

public class OjaNetwork extends NeuralNetwork {

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
    public OjaNetwork(int inputNeuronsCount) {
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
        this.setNetworkType(NeuralNetworkType.PERCEPTRON);

        // 输入神经元建立 ，表示输入的刺激
        NeuronProperties inputNeuronProperties = new NeuronProperties();
        inputNeuronProperties.setProperty("neuronType", InputNeuron.class);

        // 由输入神经元构成的输入层
        Layer inputLayer = LayerFactory.createLayer(inputNeuronsCount, inputNeuronProperties);
        this.addLayer(inputLayer);
        
        NeuronProperties outputNeuronProperties = new NeuronProperties();
        outputNeuronProperties.setProperty("transferFunction", TransferFunctionType.LINEAR);

        // 输出层，也就是神经元
        Layer outputLayer = LayerFactory.createLayer(1, outputNeuronProperties);
        this.addLayer(outputLayer);

        // 将输入层的输入导向神经元
        ConnectionFactory.fullConnect(inputLayer, outputLayer);
        NeuralNetworkFactory.setDefaultIO(this);
        this.setLearningRule(new OjaLearning());
    }

}
