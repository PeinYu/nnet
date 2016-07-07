package geym.nn.test;

import org.junit.Test;

import geym.nn.hopfiled.HopfieldWordRemember;
import junit.framework.Assert;

public class TestHopfieldSample2 {
	@Test
	public void testChar2Doubles() {
		double[] dn=HopfieldWordRemember.char2Doubles('n');
		HopfieldWordRemember.printDoubles(dn);
		System.out.println();
		System.out.println(HopfieldWordRemember.doubles2Char(dn));
		System.out.println();
		HopfieldWordRemember.printDoubles(HopfieldWordRemember.char2Doubles('m'));
	}
	
	@Test
	public void testDoubles2Str(){
		double[] abc=HopfieldWordRemember.str2Doubles("abc");
		String abc2=HopfieldWordRemember.doubles2Str(abc);
		Assert.assertEquals("abc", abc2);
	}
}
