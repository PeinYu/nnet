package org.neuroph.nnet.learning;

import org.neuroph.core.Connection;
import org.neuroph.core.Neuron;
import org.neuroph.core.Weight;

public class PerceptronLearning extends LMS {
    @Override
    protected void updateNeuronWeights(Neuron neuron) {
        // 取得神经元误差
        double neuronError = neuron.getError();

        // 根据所有的神经元输入 迭代学习
        for (Connection connection : neuron.getInputConnections()) {
            // 神经元的一个输入
            double input = connection.getInput();
            // 计算权值的变更
            double weightChange = neuronError * input;
            // 更新权值
            Weight weight = connection.getWeight();
            weight.weightChange = weightChange;                
            weight.value += weightChange;
        }
    }
}
