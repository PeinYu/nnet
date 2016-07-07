package geym.nn.som;

import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.util.ArrayList;
import java.util.List;

import javax.swing.JFrame;
import javax.swing.JPanel;

public class ResultFrame extends JFrame {
		private List<ElementString> elements=new ArrayList<ElementString>();
		
	    public ResultFrame() {
	    }

		private void init() {
			setTitle("训练结果");
	        setSize(800, 800);
	        DrawPanel panel = new DrawPanel();
	        add(panel);
		}
		
		public void showMe(){
			if(elements.size()==0)throw new RuntimeException("elements is empty");
	    	init();
	    	normalCood();
	        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
	        setVisible(true);
		}
		
		public void addElementString(ElementString str){
			elements.add(str);
		}
		
		public void normalCood(){
			float minX=Float.MAX_VALUE,maxX=0,minY=Float.MAX_VALUE,maxY=0;
			for(ElementString es:elements){
				if(es.x>maxX)maxX=es.x;
				if(es.y>maxY)maxY=es.y;
				if(es.x<minX)minX=es.x;
				if(es.y<minY)minY=es.y;
			}
			for(ElementString es:elements){
				es.x=(es.x-minX)/(maxX-minX)*700+20;
				es.y=(es.y-minY)/(maxY-minY)*700+20;
			}
		}

	    public static void main(String[] args) {
	    	ResultFrame frame = new ResultFrame();
	    	frame.showMe();
	    }
	    
	    class DrawPanel extends JPanel {
	        public void paintComponent(Graphics g) {
	            super.paintComponent(g);
	            Graphics2D g2 = (Graphics2D) g;//将Graphics对象转换为Graphics2D对象
	            g2.setFont(new Font("TimesRoman", Font.PLAIN, 20));
	            for(ElementString es:elements){
	            	g2.drawString(es.text, es.x, es.y);
	            }
	        }
	    }
	    
	    public static class ElementString{
	    	private String text;
	    	private float x;
	    	private float y;
	    	
	    	
			public ElementString(String text, float x, float y) {
				super();
				this.text = text;
				this.x = x;
				this.y = y;
			}
			public String getText() {
				return text;
			}
			public void setText(String text) {
				this.text = text;
			}
			public float getX() {
				return x;
			}
			public void setX(float x) {
				this.x = x;
			}
			public float getY() {
				return y;
			}
			public void setY(float y) {
				this.y = y;
			}
	    }
}
