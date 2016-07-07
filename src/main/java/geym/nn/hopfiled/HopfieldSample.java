package geym.nn.hopfiled;

import java.util.Arrays;

import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.nnet.Hopfield;
import org.neuroph.nnet.comp.neuron.InputOutputNeuron;
import org.neuroph.nnet.learning.HopfieldLearning;
import org.neuroph.util.NeuronProperties;
import org.neuroph.util.TransferFunctionType;

public class HopfieldSample {

	public static void main(String args[]) {
		NeuronProperties neuronProperties = new NeuronProperties();
		neuronProperties.setProperty("neuronType", InputOutputNeuron.class);
		neuronProperties.setProperty("bias", new Double(0.0D));
		neuronProperties.setProperty("transferFunction", TransferFunctionType.STEP);
		neuronProperties.setProperty("transferFunction.yHigh", new Double(1.0D));
		neuronProperties.setProperty("transferFunction.yLow", new Double(-1.0D));

		// create training set (H and T letter in 3x3 grid)
		DataSet trainingSet = new DataSet(9);
		trainingSet.addRow(new DataSetRow(new double[] { 1, -1, 1, 1, 1, 1, 1, -1, 1 })); // H

		trainingSet.addRow(new DataSetRow(new double[] { 1, 1, 1, -1, 1, -1, -1, 1, -1 })); // T

		// create hopfield network
		Hopfield myHopfield = new Hopfield(9, neuronProperties);
		myHopfield.setLearningRule(new StandHopfieldLearning());
		// learn the training set
		myHopfield.learn(trainingSet);

		// test hopfield network
		System.out.println("Testing network");

		// add one more 'incomplete' H pattern for testing - it will be
		// recognized as H
		// DataSetRow h=new DataSetRow(new double[] { 1, 0, 0, 1, 0, 1, 1, 0, 1
		// });
		// DataSetRow h=new DataSetRow(new double[] { 1, 0, 0, 1, 0, 1, 1, 0, 1
		// });
		DataSetRow h = new DataSetRow(new double[] { 1, -1, 1, -1, 1, 1, 1, -1, 1 }); // H
		trainingSet.addRow(h); // incomplete
								// H
								// letter


		myHopfield.setInput(h.getInput());

		double[] networkOutput = null;
		double[] preNetworkOutput = null;
		while (true) {
			myHopfield.calculate();
			networkOutput = myHopfield.getOutput();
			if (preNetworkOutput == null) {
				preNetworkOutput = networkOutput;
				continue;
			}
			if (Arrays.equals(networkOutput, preNetworkOutput)) {
				break;
			}
			preNetworkOutput = networkOutput;
		}

		System.out.print("Input: " + Arrays.toString(h.getInput()));
		System.out.println(" Output: " + Arrays.toString(networkOutput));

	}

}
