package geym.nn.adaline;

import org.neuroph.core.Layer;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.nnet.comp.neuron.BiasNeuron;
import org.neuroph.nnet.learning.LMS;
import org.neuroph.util.ConnectionFactory;
import org.neuroph.util.LayerFactory;
import org.neuroph.util.NeuralNetworkFactory;
import org.neuroph.util.NeuralNetworkType;
import org.neuroph.util.NeuronProperties;
import org.neuroph.util.TransferFunctionType;

public class Adaline extends NeuralNetwork {

	/**
	 * The class fingerprint that is set to indicate serialization compatibility
	 * with a previous version of the class.
	 */
	private static final long serialVersionUID = 1L;

	/**
	 * Creates new Adaline network with specified number of neurons in input
	 * layer
	 * 
	 * @param inputNeuronsCount
	 *            number of neurons in input layer
	 */
	public Adaline(int inputNeuronsCount, int outputNeuronsCount, double learnRate, TransferFunctionType transferFunction) {
		this.createNetwork(inputNeuronsCount, outputNeuronsCount, learnRate,transferFunction);
	}

	/**
	 * Creates adaline network architecture with specified number of input
	 * neurons
	 * 
	 * @param inputNeuronsCount
	 *            number of neurons in input layer
	 */
	private void createNetwork(int inputNeuronsCount, int outputNeuronsCount, double learnRate,
			TransferFunctionType transferFunction) {
		// set network type code
		this.setNetworkType(NeuralNetworkType.ADALINE);

		// create input layer neuron settings for this network
		NeuronProperties inNeuronProperties = new NeuronProperties();
		inNeuronProperties.setProperty("transferFunction", TransferFunctionType.LINEAR);

		// createLayer input layer with specified number of neurons
		Layer inputLayer = LayerFactory.createLayer(inputNeuronsCount, inNeuronProperties);
		inputLayer.addNeuron(new BiasNeuron()); // add bias neuron (always 1,
												// and it will act as bias input
												// for output neuron)
		this.addLayer(inputLayer);

		// create output layer neuron settings for this network
		NeuronProperties outNeuronProperties = new NeuronProperties();
		if (transferFunction == TransferFunctionType.LINEAR) {
			outNeuronProperties.setProperty("transferFunction", TransferFunctionType.LINEAR);
		} else {
			outNeuronProperties.setProperty("transferFunction", TransferFunctionType.RAMP);
			outNeuronProperties.setProperty("transferFunction.slope", new Double(1));
			outNeuronProperties.setProperty("transferFunction.yHigh", new Double(1));
			outNeuronProperties.setProperty("transferFunction.xHigh", new Double(1));
			outNeuronProperties.setProperty("transferFunction.yLow", new Double(-1));
			outNeuronProperties.setProperty("transferFunction.xLow", new Double(-1));
		}
		// createLayer output layer (only one neuron)
		Layer outputLayer = LayerFactory.createLayer(outputNeuronsCount, outNeuronProperties);
		this.addLayer(outputLayer);

		// createLayer full conectivity between input and output layer
		ConnectionFactory.fullConnect(inputLayer, outputLayer);

		// set input and output cells for network
		NeuralNetworkFactory.setDefaultIO(this);

		// set LMS learning rule for this network
		LMS l = new LMS();
		l.setLearningRate(learnRate);
		this.setLearningRule(l);
	}

}
