package geym.nn.som;

import org.neuroph.core.Neuron;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.nnet.Kohonen;

import geym.nn.som.ResultFrame.ElementString;

public class KohonenDemo {

	public static double[][] data = {
			{ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0 }, // 鸽子
			{ 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0 }, // 母鸡
			{ 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1 }, // 鸭
			{ 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1 }, // 鹅
			{ 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0 }, // 猫头鹰
			{ 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0 }, // 凖
			{ 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0 }, // 鹰
			{ 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0 }, // 狐狸
			{ 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0 }, // 狗
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0 }, // 狼
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0 }, // 猫
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0 }, // 虎
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0 }, // 狮
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0 }, // 马
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0 }, // 斑马
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0 }  // 牛
	};
	
	public static String[] dataKey={"鸽子","母鸡","鸭","鹅","猫头鹰","凖","鹰","狐狸","狗","狼","猫","虎","狮","马","斑马","牛"};

	public static void main(String[] args) {
		ResultFrame frame = new ResultFrame();
		Kohonen som = new Kohonen(29, 100);
		DataSet ds = new DataSet(29);
		for (double[] row : data) {
			ds.addRow(new DataSetRow(row));
		}

		som.learn(ds);

		for (int i=0;i<data.length;i++) {
			som.setInput(data[i]);
			som.calculate();
			int winnerIndex=getWinnerIndex(som);
			int x=getRowFromIndex(winnerIndex);
			int y=getColFromIndex(winnerIndex);
			System.out.println(dataKey[i]+" "+x+" "+y );
			frame.addElementString(new ResultFrame.ElementString(dataKey[i], x, y));
		}
		frame.showMe();
	}

	// get unit with closetst weight vector
	private static int getWinnerIndex(Kohonen neuralNetwork) {
		Neuron winner = new Neuron();
		double minOutput = 100;
		int winnerIndex=-1;
		Neuron[] neurons=neuralNetwork.getLayerAt(1).getNeurons();
		for (int i=0;i<neurons.length;i++) {
			double out = neurons[i].getOutput();
			if (out < minOutput) {
				minOutput = out;
				winnerIndex = i;
			} // if
		} // while
		return winnerIndex;
	}

	/**
	 * 10行10列中的位置
	 * @param index
	 * @return
	 */
	private static int getRowFromIndex(int index){
		return index/10+1;
	}
	private static int getColFromIndex(int index){
		return index%10+1;
	}
}
