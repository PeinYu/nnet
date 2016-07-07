
package geym.nn.mlperceptron.ch5;

import java.util.List;

import org.neuroph.core.Connection;
import org.neuroph.core.Layer;
import org.neuroph.core.Neuron;
import org.neuroph.core.Weight;
import org.neuroph.core.transfer.Linear;
import org.neuroph.nnet.comp.neuron.BiasNeuron;
import org.neuroph.nnet.comp.neuron.InputNeuron;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.util.ConnectionFactory;
import org.neuroph.util.LayerFactory;
import org.neuroph.util.NeuralNetworkFactory;
import org.neuroph.util.NeuralNetworkType;
import org.neuroph.util.NeuronProperties;
import org.neuroph.util.TransferFunctionType;

import geym.nn.mlperceptron.MlPerceptron;
 
public class MlPerceptronAndOutputNoLearn extends MlPerceptron {
    
    public MlPerceptronAndOutputNoLearn(TransferFunctionType transferFunctionType, int... neuronsInLayers) {
        super(transferFunctionType,neuronsInLayers);
    }

    protected void createNetwork(List<Integer> neuronsInLayers, NeuronProperties neuronProperties) {
        //多层感知机
        this.setNetworkType(NeuralNetworkType.MULTI_LAYER_PERCEPTRON);

        // 输入层，代表每一个输入数据
        NeuronProperties inputNeuronProperties = new NeuronProperties(InputNeuron.class, Linear.class);
        Layer layer = LayerFactory.createLayer(neuronsInLayers.get(0), inputNeuronProperties);
        layer.addNeuron(new BiasNeuron());
        this.addLayer(layer);

        // 建立中间隐层
        Layer prevLayer = layer;
        int layerIdx = 1;
        for (layerIdx = 1; layerIdx < neuronsInLayers.size()-1; layerIdx++) {
            Integer neuronsNum = neuronsInLayers.get(layerIdx);
            layer = LayerFactory.createLayer(neuronsNum, neuronProperties);
            this.addLayer(layer);
            if (prevLayer != null) {
                ConnectionFactory.fullConnect(prevLayer, layer);
            }
            prevLayer = layer;
        }
        
        //设置初始权值
        Neuron n1=layer.getNeuronAt(0);
        Connection[] c1=n1.getInputConnections();
        c1[0].setWeight(new Weight(2));
        c1[1].setWeight(new Weight(2));
        c1[2].setWeight(new Weight(-1));
        
        Neuron n2=layer.getNeuronAt(1);
        Connection[] c2=n2.getInputConnections();
        c2[0].setWeight(new Weight(-1));
        c2[1].setWeight(new Weight(-1));
        c2[2].setWeight(new Weight(1.5));
        
        // 建立输出层
        Integer neuronsNum = neuronsInLayers.get(layerIdx);
        NeuronProperties outProperties=new NeuronProperties();
        // 输入函数为逻辑与
        outProperties.put("inputFunction", org.neuroph.core.input.And.class);
        outProperties.put("transferFunction", neuronProperties.getTransferFunction());
        
        layer = LayerFactory.createLayer(neuronsNum, outProperties);
        this.addLayer(layer);
        ConnectionFactory.fullConnect(prevLayer, layer);
        prevLayer = layer;
       
        NeuralNetworkFactory.setDefaultIO(this);
    }
}
