import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayesSimple;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ArffLoader.ArffReader;

public class Experimenter {

	static String trainFile, testFile;
	static int minNumObj=0, maxNumObj=0, minNumFolds=0, maxNumFolds=0, bestNumObj=0, bestNumFolds=0;
	static double minErrorJ48=100, avgErrorNBS=0;

	
	static Classifier bestModel=null;
	static Instances trainData, testData;
	static J48 j48;
	static NaiveBayesSimple nbs;

	//method to Read from user and set up training and testing data
	public static void readData() throws Exception {
		 
		BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
		
		//training data file
		System.out.println("Enter training data file path:");
		trainFile = br.readLine();
	 
		//test data file name
		System.out.println("\nEnter test data file path:");
		testFile = br.readLine();

		
		System.out.println("\nEnter inclusive range for minNumObj:\nFrom: ");
		minNumObj= Integer.parseInt(br.readLine());
		System.out.println("\nTo: ");
		maxNumObj= Integer.parseInt(br.readLine());
		
		System.out.println("\nEnter inclusive range for numFolds:\nFrom: ");
		minNumFolds= Integer.parseInt(br.readLine());
		System.out.println("\nTo: ");
		maxNumFolds= Integer.parseInt(br.readLine());
		
		br.close();

		//read training data from arff file
		ArffReader arffTrain = new ArffReader(new BufferedReader(new FileReader(trainFile)));
		trainData = arffTrain.getData();
		trainData.setClassIndex(0);

		//read test data from arff file
		ArffReader arffTest = new ArffReader(new BufferedReader(new FileReader(testFile)));
		testData = arffTest.getData();
		testData.setClassIndex(0);


	}
	
	//method to calculate J48 for all the settings given
	public static void calcForJ48() throws Exception{
		System.out.println("\n\n******* J48 Cross validation*******");
		
        // J48 classifier
        j48 = new J48();
        j48.setUnpruned(false);        // using an unpruned J48
        j48.setReducedErrorPruning(true);	//to use with minNumObj
        
        System.out.println("minNumObj \tnumFolds \tMisclassified instances in Percent");
        
        //to check different combinations for better settings
        for(int i=minNumObj;i<=maxNumObj;i++){
        	for(int j=minNumFolds;j<=maxNumFolds;j++){
        		
        		//settings
		        j48.setMinNumObj(i);
		        j48.setNumFolds(j);
		        
		      //evaluate j48 with cross validation
	            Evaluation eval=new Evaluation(trainData);
	
	            //first supply the classifier
	            //then the training data
	            //number of folds
	            //random seed
	            eval.crossValidateModel(j48, trainData, 10, new Random(1));
	            
	            System.out.println(i+"\t\t"+j+"\t\t"+ Double.toString(eval.pctIncorrect()));
	            	            
	            //save the best settings
	            if(minErrorJ48>eval.pctIncorrect()){
	            	minErrorJ48=eval.pctIncorrect();
	            	bestNumObj=i;
	            	bestNumFolds= j;
	            }
	            //System.out.println(eval.toSummaryString());
	
        	}
        	System.out.println();
        }
        
        
        System.out.println("\nMinimum misclassified instances in percent: "+ Double.toString(minErrorJ48));
        System.out.println("\nMinimum misclassified instances given by settings: \nminNumobj: "+ bestNumObj + " minFolds: "+ bestNumFolds);

	}
	
	//method to check with NBS
	public static void calcForNBS() throws Exception{
        System.out.println("\n\n******* NBS Cross validation*******");

        //NaivesBayesSimple
        NaiveBayesSimple nbs= new NaiveBayesSimple();

	    //evaluate NaiveBayesSimple with cross validation
        Evaluation eval=new Evaluation(trainData);

        //first supply the classifier
        //then the training data
        //number of folds
        //random seed
        eval.crossValidateModel(nbs, trainData, 10, new Random(1));
        
        avgErrorNBS=eval.pctIncorrect();
        System.out.println("\nMinimum misclassified instances in percent: "+ Double.toString(avgErrorNBS));

	}
	
	//calculations for best model
	public static void calcForBestModel() throws Exception{
       
		//best model
        System.out.println("\n\n******* Best Model *******\n");
        
        //if J48 set best settings for it
        if(minErrorJ48<= avgErrorNBS){
        	System.out.println("J48 is best model produced on training data.");
        	System.out.println("Best Setting for this: \nMinNumObj: "+bestNumObj + " NumFolds: " + bestNumFolds);
        	j48.setMinNumObj(bestNumObj);
        	j48.setNumFolds(bestNumFolds);
        	bestModel=j48;
    
        }
        else{
        	System.out.println("NaivesBayesSimple is best model produced on training data.");
        	bestModel=nbs;
        }
        
        
        //classification of test instances by best model
        bestModel.buildClassifier(testData);
        
        Evaluation eval1 =new Evaluation(trainData);
		
        //first supply the classifier
        //then the training data
        //number of folds
        //random seed
        eval1.crossValidateModel(bestModel, trainData, 10, new Random(1));
        bestModel.buildClassifier(trainData);

        System.out.println("\n\nTest instance no. \tActual class \tPredicted class");
        
        //classify each instance and print
        for (int i = 0; i < testData.numInstances(); i++) {
        	
    	   double pred = bestModel.classifyInstance(testData.instance(i));
    	   
    	   System.out.println(i + "\t\t\t" + testData.classAttribute().value((int) testData.instance(i).classValue())+ "\t\t" + testData.classAttribute().value((int) pred));

    	 }

        eval1.evaluateModel(bestModel, testData);
        System.out.println("\n\nBest model's misclassified instances on test data: "+eval1.pctIncorrect());

	}
	
	public static void main(String args []){
				
		try{

			/*
			trainFile= "D:/sem3/ML/assign3/trainingdata.arff";
			testFile=  "D:/sem3/ML/assign3/testdata.arff";
			minNumObj=1; maxNumObj=4; minNumFolds=2; maxNumFolds=4;
			*/

			readData();
			
			calcForJ48();
			
	        calcForNBS();
	        
	        calcForBestModel();


		}
		catch(Exception e){
			e.printStackTrace();
			System.exit(-1);
		}

	}
}
