package geym.nn.bam;

import org.neuroph.core.data.DataSetRow;

//import dataguru.nn.ch8.BAM;

public class RunBAM {
    public static final String[] NAMES = { "TINA ", "ANTJE", "LISA " };
    public static final String[] NAMES2 = { "TINE ", "ANNJE", "LISE " };
    public static final String[] PHONES = { "6843726", "8034673", "7260915" };
    public static final String[] PHONES2 = { "6843725", "8134673", "7360915" };

    public static final int IN_CHARS    = 5;
    public static final int OUT_CHARS   = 7;
    public static final int BITS_PER_CHAR = 6;
    public static final char FIRST_CHAR = ' ';
    public static final int INPUT_NEURONS = (IN_CHARS  * BITS_PER_CHAR);
    public static final int OUTPUT_NEURONS = (OUT_CHARS * BITS_PER_CHAR);

    public static boolean[] stringToBipolar(String str)
    {
        boolean[] result = new boolean[str.length()*BITS_PER_CHAR];
        int currentIndex = 0;
        for(int i=0;i<str.length();i++)
        {
            char ch = Character.toUpperCase(str.charAt(i));
            int idx = ch-FIRST_CHAR;

            int place = 1;
            for( int j=0;j<BITS_PER_CHAR;j++)
            {
                boolean value = (idx&place)>0;
                result[currentIndex++]=value;
                place*=2;
            }
        }
        return result;
    }

    public static String bipolalToString(boolean[] data)
    {
        StringBuilder result = new StringBuilder();

        int j,a,p;

          for (int i=0; i<(data.length / BITS_PER_CHAR); i++) {
            a = 0;
            p = 1;
            for (j=0; j<BITS_PER_CHAR; j++) {
                //if( data.getBoolean(i*BITS_PER_CHAR+j) )
                if( data[(i*BITS_PER_CHAR+j)] )
                    a+=p;
              p *= 2;
            }
            result.append((char)(a + FIRST_CHAR));
          }
        return result.toString();
    }

    public static boolean[] randomBiPolar(int size)
    {
        boolean[] result = new boolean[size];
        for(int i=0;i<size;i++)
        {
            if(Math.random()>0.5)
                result[i]=false;
            else
                result[i]=true;        
        	}
        return result;
    }

    public static double bipolar2double(final boolean b) {
        if (b) {
            return 1;
        }
        return -1;
    }
    public static double[] bipolar2double(final boolean[] b) {
        final double[] result = new double[b.length];

        for (int i = 0; i < b.length; i++) {
            result[i] = bipolar2double(b[i]);
        }

        return result;
    }
    public static boolean double2bipolar(final double d) {
        if (d > 0) {
            return true;
        }
        return false;
    }

    /**
     * Convert a bipolar array to booleans.
     *
     * @param d
     *            A bipolar array.
     * @return An array of booleans.
     */
    public static boolean[] double2bipolar(final double[] d) {
        final boolean[] result = new boolean[d.length];

        for (int i = 0; i < d.length; i++) {
            result[i] = double2bipolar(d[i]);
        }

        return result;
    }

    public static String mappingToString(DataSetRow row)
    {
        StringBuilder result = new StringBuilder();
        result.append( bipolalToString(double2bipolar(row.getInput()))) ;
        result.append(" -> ");
        result.append( bipolalToString(double2bipolar(row.getDesiredOutput() )));
        return result.toString();
    }
/*
    public static void runBAM(BAM logic, DataSetRow data )
    {

        StringBuilder line = new StringBuilder();
        line.append(mappingToString(data));
        logic.calculate(data);
        line.append("  |  ");
        line.append(mappingToString(data));
        System.out.println(line.toString());
    }


    public static void main(String[] args) {
        BAM logic = new BAM(INPUT_NEURONS, OUTPUT_NEURONS);
        DataSet ds =new DataSet(INPUT_NEURONS,OUTPUT_NEURONS);

        // train
        for(int i=0;i<NAMES.length;i++)
        {
            DataSetRow dr=new DataSetRow();
            dr.setInput(bipolar2double(stringToBipolar(NAMES[i])));
            dr.setDesiredOutput(bipolar2double(stringToBipolar(PHONES[i])));
            ds.addRow(dr);

        }

        logic.learn(ds);

        for(int i=0;i<NAMES.length;i++)
        {
            DataSetRow dr=new DataSetRow();
            dr.setInput(bipolar2double(stringToBipolar(NAMES[i])));
            dr.setDesiredOutput(bipolar2double(randomBiPolar(OUT_CHARS*BITS_PER_CHAR)));
            runBAM(logic, dr);
        }

        System.out.println();

        for(int i=0;i<NAMES2.length;i++)
        {
            DataSetRow dr=new DataSetRow();
            dr.setInput(bipolar2double(stringToBipolar(NAMES2[i])));
            dr.setDesiredOutput(bipolar2double(randomBiPolar(OUT_CHARS*BITS_PER_CHAR)));
            runBAM(logic, dr);
        }

        logic.switchInputOutput();
        System.out.println();
        for(int i=0;i<PHONES.length;i++)
        {
            DataSetRow dr=new DataSetRow();
            dr.setInput(bipolar2double(stringToBipolar(PHONES[i])));
            dr.setDesiredOutput(bipolar2double(randomBiPolar(IN_CHARS*BITS_PER_CHAR)));
            runBAM(logic, dr);
        }

        System.out.println();
        for(int i=0;i<PHONES2.length;i++)
        {
            DataSetRow dr=new DataSetRow();
            dr.setInput(bipolar2double(stringToBipolar(PHONES2[i])));
            dr.setDesiredOutput(bipolar2double(randomBiPolar(IN_CHARS*BITS_PER_CHAR)));
            runBAM(logic, dr);
        }
    }
*/
}
