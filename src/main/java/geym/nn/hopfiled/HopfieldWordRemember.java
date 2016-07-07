package geym.nn.hopfiled;

import java.util.Arrays;

import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.nnet.Hopfield;
import org.neuroph.nnet.comp.neuron.InputOutputNeuron;
import org.neuroph.nnet.learning.HopfieldLearning;
import org.neuroph.util.NeuronProperties;
import org.neuroph.util.TransferFunctionType;

public class HopfieldWordRemember {
	public static char doubles2Char(double[] doubles) {
		int re = 0;
		int flag = 0x80;
		for (int i = 0; i < doubles.length; i++) {
			if (doubles[i] == 1) {
				re |= (flag >> i);
			}
		}
		return (char) re;
	}

	public static double[] char2Doubles(char c) {
		int intChar = (int) c;
		double[] re = new double[8];
		int flag = 0x80;
		for (int i = 0; i < re.length; i++) {
			if ((flag & intChar) != 0) {
				re[i] = 1;
			} else {
				re[i] = -1;
			}
			flag >>= 1;
		}
		return re;
	}

	public static double[] str2Doubles(String str) {
		double[] re = new double[str.length() * 8];
		for (int i = 0; i < str.length(); i++) {
			double[] dChar = char2Doubles(str.charAt(i));
			System.arraycopy(dChar, 0, re, i * 8, 8);
		}
		return re;
	}

	public static String doubles2Str(double[] dstr) {
		double[] dchar = new double[8];
		StringBuffer sb = new StringBuffer();
		for (int i = 0; i < dstr.length / 8; i++) {
			System.arraycopy(dstr, i * 8, dchar, 0, 8);
			char c = doubles2Char(dchar);
			sb.append(c);
		}
		return sb.toString();
	}

	public static void printDoubles(double[] doubles) {
		for (int i = 0; i < doubles.length; i++) {
			System.out.print(doubles[i] + ", ");
		}
	}

	public static void main(String args[]) {
		NeuronProperties neuronProperties = new NeuronProperties();
		neuronProperties.setProperty("neuronType", InputOutputNeuron.class);
		neuronProperties.setProperty("bias", new Double(0.0D));
		neuronProperties.setProperty("transferFunction", TransferFunctionType.STEP);
		neuronProperties.setProperty("transferFunction.yHigh", new Double(1.0D));
		neuronProperties.setProperty("transferFunction.yLow", new Double(-1.0D));

		// remember 3 words
		DataSet trainingSet = new DataSet(32);
		trainingSet.addRow(new DataSetRow(str2Doubles("good")));
		trainingSet.addRow(new DataSetRow(str2Doubles("bad ")));
		trainingSet.addRow(new DataSetRow(str2Doubles("test")));

		// create hopfield network
		Hopfield myHopfield = new Hopfield(32, neuronProperties);
		myHopfield.setLearningRule(new StandHopfieldLearning());
		// learn the training set
		myHopfield.learn(trainingSet);

		// test hopfield network
		System.out.println("I remember good ,bad  and test");

		recallWords("jood", myHopfield);
		recallWords("dad ", myHopfield);
		recallWords("tast", myHopfield);
	}

	public static void recallWords(String word, Hopfield myHopfield) {
		DataSetRow h = new DataSetRow(str2Doubles(word));

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

		System.out.println("Input: " + HopfieldWordRemember.doubles2Str(h.getInput()));
		System.out.println("Output: " + HopfieldWordRemember.doubles2Str(networkOutput));
	}

}
