package geym.nn.mlperceptron;

import geym.nn.perceptron.SimplePerceptron;
import org.neuroph.core.Layer;
import org.neuroph.core.Neuron;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.learning.IterativeLearning;
import org.neuroph.core.transfer.TransferFunction;
import org.neuroph.nnet.comp.neuron.BiasNeuron;
import org.neuroph.nnet.comp.neuron.InputNeuron;
import org.neuroph.nnet.learning.LMS;
import org.neuroph.util.*;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.events.LearningEventListener;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Vector;


public class RedClassify extends NeuralNetwork implements LearningEventListener {

    public static void main(String[] args) throws IOException {
        new RedClassify().run();
    }

    public void run()  {
        //数据集含有2个输入一个输出
        DataSet trainingSet = new DataSet(2, 1);
        trainingSet.addRow(new DataSetRow(new double[]{0, 0}, new double[]{0}));
        trainingSet.addRow(new DataSetRow(new double[]{0, 1}, new double[]{1}));
        trainingSet.addRow(new DataSetRow(new double[]{1, 0}, new double[]{1}));
        trainingSet.addRow(new DataSetRow(new double[]{1, 1}, new double[]{1}));
        //感知机有2个输入
        SimplePerceptron myPerception = new SimplePerceptron(2);
        LMS learningRule = (LMS) myPerception.getLearningRule();
        learningRule.addListener(this);
        //进行学习
        System.out.println("Training netual network...");
        myPerception.learn(trainingSet);

        //测试感知机输出是否正确
        System.out.println("Testing trained neural network");
        testNeuralNetwork(myPerception, trainingSet);

    }

    public static void testNeuralNetwork(NeuralNetwork neuralNet, DataSet row) {

        for(DataSetRow testSetRow : row.getRows()) {
            neuralNet.setInput(testSetRow.getInput());
            neuralNet.calculate();
            double[] networkOutput = neuralNet.getOutput();

            System.out.println("Input: " + Arrays.toString(testSetRow.getInput()));
            System.out.println("Output:" + Arrays.toString(networkOutput));
        }
    }

    private void createNetwork(int inputNeuronsCount) {
        //设置网络类别为 感知机
        this.setNetworkType(NeuralNetworkType.PERCEPTRON);
        //输入神经元建立，表示输入的刺激
        NeuronProperties inputNeuronProperties = new NeuronProperties();
        inputNeuronProperties.setProperty("neuronType", InputNeuron.class);
        //由输入的神经元构成的输入层
        Layer inputLayer =
                LayerFactory.createLayer(inputNeuronsCount, inputNeuronProperties);
        this.addLayer(inputLayer);
        //在输入层增加BiasNeuron, 表示神经元偏置
        inputLayer.addNeuron(new BiasNeuron());
        //传输函数为 step
        NeuronProperties outputNeuronProperties = new NeuronProperties();
        outputNeuronProperties.setProperty("transferFunction",
                TransferFunctionType.STEP);
        //输出层 即神经元
        Layer outputLayer = LayerFactory.createLayer(1, outputNeuronProperties);
        this.addLayer(outputLayer);
        //将输入层的输入导向神经元
        ConnectionFactory.fullConnect(inputLayer, outputLayer);
        NeuralNetworkFactory.setDefaultIO(this);
        //设置感知机为LMS学习算法
        this.setLearningRule(new LMS());
    }

    @Override
    public void handleLearningEvent(LearningEvent event) {
        IterativeLearning bp = (IterativeLearning) event.getSource();
        System.out.println("iterate:" + bp.getCurrentIteration());
        System.out.println("weights:" + Arrays.toString(bp.getNeuralNetwork().getWeights()));
    }
}
