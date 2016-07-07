/**
 * Copyright 2010 Neuroph Project http://neuroph.sourceforge.net
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package geym.nn.hopfiled;

import org.neuroph.core.Connection;
import org.neuroph.core.Layer;
import org.neuroph.core.Neuron;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.core.learning.LearningRule;

/**
 * Learning algorithm for the Hopfield neural network.
 * 
 * @author Zoran Sevarac <sevarac@gmail.com>
 */
public class StandHopfieldLearning extends LearningRule {
	
	/**
	 * The class fingerprint that is set to indicate serialization
	 * compatibility with a previous version of the class.
	 */	
	private static final long serialVersionUID = 1L;

	/**
	 * Creates new HopfieldLearning
	 */
	public StandHopfieldLearning() {
		super();
	}


	/**
	 * Calculates weights for the hopfield net to learn the specified training
	 * set
	 * 
	 * @param trainingSet
	 *            training set to learn
	 */
	public void learn(DataSet trainingSet) {
		int M = trainingSet.size();
		int N = neuralNetwork.getLayerAt(0).getNeuronsCount();
		Layer hopfieldLayer = neuralNetwork.getLayerAt(0);

		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				if (j == i)
					continue;
				Neuron ni = hopfieldLayer.getNeuronAt(i);
				Neuron nj = hopfieldLayer.getNeuronAt(j);
				Connection cij = nj.getConnectionFrom(ni);
				Connection cji = ni.getConnectionFrom(nj);
				
				double wij=0;
				for(int k = 0;k < M;k++){
					DataSetRow row=trainingSet.getRowAt(k);
					double[] inputs=row.getInput();
					wij+=inputs[i]*inputs[j];
				}
				cij.getWeight().setValue(wij);
				cji.getWeight().setValue(wij);
			}// j
		} // i

	}

}
