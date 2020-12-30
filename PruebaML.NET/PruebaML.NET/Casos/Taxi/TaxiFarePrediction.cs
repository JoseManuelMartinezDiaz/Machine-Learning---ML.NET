using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace PruebaML.NET.Casos.Taxi
{
  class TaxiFarePrediction
  {

    string TrainDataPath = string.Empty;
    string TestDataPath = string.Empty;

    private static string BaseModelsRelativePath = @"../../../Datos";
    private static string ModelRelativePath = $"{BaseModelsRelativePath}/TaxiFareModel.zip";

    private static string ModelPath = GetAbsolutePath(ModelRelativePath);


    public TaxiFarePrediction()
    {

    }

    public void iniciar(MLContext mlContext)
    {
      BuildModelPipeline(mlContext);
      ConsumeModel(mlContext);
    }

  public static string GetAbsolutePath(string relativePath)
  {
    FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
    string assemblyFolderPath = _dataRoot.Directory.FullName;

    string fullPath = Path.Combine(assemblyFolderPath, relativePath);

    return fullPath;
  }
  

  private ITransformer BuildModelPipeline(MLContext mlContext)
    {

      /*
       1. Carga de 
       */ 
      TrainDataPath = Path.Combine(Environment.CurrentDirectory, "../../../Datos", "taxi-fare-train.csv");
      TestDataPath = Path.Combine(Environment.CurrentDirectory, "../../../Datos", "taxi-fare-test.csv");

      IDataView baseTrainingDataView = mlContext.Data.LoadFromTextFile<TaxiTrip>(TrainDataPath, hasHeader: true, separatorChar: ',');
      IDataView testDataView = mlContext.Data.LoadFromTextFile<TaxiTrip>(TestDataPath, hasHeader: true, separatorChar: ',');

      //Sample code of removing extreme data like "outliers" for FareAmounts higher than $150 and lower than $1 which can be error-data 
      var cnt = baseTrainingDataView.GetColumn<float>(nameof(TaxiTrip.FareAmount)).Count();
      IDataView trainingDataView = mlContext.Data.FilterRowsByColumn(baseTrainingDataView, nameof(TaxiTrip.FareAmount), lowerBound: 1, upperBound: 150);
      var cnt2 = trainingDataView.GetColumn<float>(nameof(TaxiTrip.FareAmount)).Count();

      // STEP 2: Common data process configuration with pipeline data transformations
      var dataProcessPipeline = mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: nameof(TaxiTrip.FareAmount))
                                  .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "VendorIdEncoded", inputColumnName: nameof(TaxiTrip.VendorId)))
                                  .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "RateCodeEncoded", inputColumnName: nameof(TaxiTrip.RateCode)))
                                  .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "PaymentTypeEncoded", inputColumnName: nameof(TaxiTrip.PaymentType)))
                                  .Append(mlContext.Transforms.NormalizeMeanVariance(outputColumnName: nameof(TaxiTrip.PassengerCount)))
                                  .Append(mlContext.Transforms.NormalizeMeanVariance(outputColumnName: nameof(TaxiTrip.TripTime)))
                                  .Append(mlContext.Transforms.NormalizeMeanVariance(outputColumnName: nameof(TaxiTrip.TripDistance)))
                                  .Append(mlContext.Transforms.Concatenate("Features", "VendorIdEncoded", "RateCodeEncoded", "PaymentTypeEncoded", nameof(TaxiTrip.PassengerCount)
                                  , nameof(TaxiTrip.TripTime), nameof(TaxiTrip.TripDistance)));

      ConsoleHelper.PeekDataViewInConsole(mlContext, trainingDataView, dataProcessPipeline, 5);
      ConsoleHelper.PeekVectorColumnDataInConsole(mlContext, "Features", trainingDataView, dataProcessPipeline, 5);

      // STEP 3: Set the training algorithm, then create and config the modelBuilder - Selected Trainer (SDCA Regression algorithm)                            
      var trainer = mlContext.Regression.Trainers.Sdca(labelColumnName: "Label", featureColumnName: "Features");
      var trainingPipeline = dataProcessPipeline.Append(trainer);

      var trainedModel = trainingPipeline.Fit(trainingDataView);

      IDataView predictions = trainedModel.Transform(testDataView);
      var metrics = mlContext.Regression.Evaluate(predictions, labelColumnName: "Label", scoreColumnName: "Score");

      ConsoleHelper.PrintRegressionMetrics(trainer.ToString(), metrics);
      mlContext.Model.Save(trainedModel, trainingDataView.Schema, ModelPath);

      Console.WriteLine("The model is saved to {0}", ModelPath);

      return trainedModel;
    }

    private void ConsumeModel(MLContext mlContext)
    {
      var taxiTripSample = new TaxiTrip()
      {
        VendorId = "VTS",
        RateCode = "1",
        PassengerCount = 1,
        TripTime = 1140,
        TripDistance = 3.75f,
        PaymentType = "CRD",
        FareAmount = 0 // To predict. Actual/Observed = 15.5
      };

      ITransformer trainedModel;
      using (var stream = new FileStream(ModelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
      {
        trainedModel = mlContext.Model.Load(stream, out var modelInputSchema);
      }

      // Create prediction engine related to the loaded trained model
      var predEngine = mlContext.Model.CreatePredictionEngine<TaxiTrip, TaxiTripFarePrediction>(trainedModel);

      //Score
      var resultprediction = predEngine.Predict(taxiTripSample);

      Console.WriteLine($"**********************************************************************");
      Console.WriteLine($"Predicted fare: {resultprediction.FareAmount:0.####}, actual fare: 15.5");
      Console.WriteLine($"**********************************************************************");
    }


  }
}
